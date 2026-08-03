"""
video_processor.py  — Production-Grade Multi-Resolution HLS Processor

Generates:
  - 240p / 480p / 720p / 1080p quality ladder
  - Adaptive master.m3u8 playlist
  - 4-second segments (fast startup)
  - AES-128 per-video encryption (key stored in Redis)
  - Thumbnail (best-quality frame at 5% of duration)
  - Preview sprite sheet + VTT file (every 10 seconds)
  - Optimized FFmpeg settings for speed vs quality balance
"""

import os
import subprocess
import secrets
import hashlib
import time
import json
import math
import shutil
import requests
import redis
from dotenv import load_dotenv
from urllib.parse import urljoin
from pathlib import Path

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
FFMPEG_BIN   = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_BIN  = os.getenv("FFPROBE_PATH", "ffprobe")
BACKEND_URL  = os.getenv("NODE_BACKEND_URL", "http://localhost:5000")
WEBHOOK_URL  = urljoin(BACKEND_URL, "/api/upload/video-processed")

# Redis for storing encryption keys (more secure than plain disk)
REDIS_HOST   = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT   = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASS   = os.getenv("REDIS_PASSWORD", None)
INTERNAL_WEBHOOK_SECRET = os.getenv("INTERNAL_WEBHOOK_SECRET", "")

def internal_headers() -> dict:
    return {"x-internal-secret": INTERNAL_WEBHOOK_SECRET} if INTERNAL_WEBHOOK_SECRET else {}

# Quality ladder templates: (label, width, height, video_bitrate_kbps, audio_bitrate_kbps)
QUALITY_LADDER_TEMPLATES = [
    ("240p",  426,  240,   250,  48),
    ("360p",  640,  360,   450,  64),
    ("480p",  854,  480,   700,  96),
    ("720p",  1280, 720,  1400, 128),
    ("1080p", 1920, 1080, 2500, 128),
]

HLS_SEGMENT_DURATION = 6   # seconds — optimized for mobile networks
SPRITE_INTERVAL      = 10  # seconds between sprite frames
SPRITE_THUMB_W       = 160
SPRITE_THUMB_H       = 90

def even(value: float) -> int:
    """Return a positive even dimension for encoder compatibility."""
    return max(2, int(round(value / 2)) * 2)

def build_quality_ladder(width: int, height: int) -> list[tuple[str, int, int, int, int]]:
    """
    Build output dimensions that preserve the source orientation.
    Portrait reels should become 240x426, 480x854, 720x1280, etc.,
    instead of being padded into landscape canvases.
    """
    if width <= 0 or height <= 0:
        return QUALITY_LADDER_TEMPLATES[:1]

    is_portrait = height > width
    quality_ladder = []

    for label, template_w, template_h, vbr, abr in QUALITY_LADDER_TEMPLATES:
        if is_portrait:
            target_w = template_h
            if target_w > width + 10 and quality_ladder:
                continue
            out_w = min(target_w, width)
            out_h = even(out_w * height / width)
        else:
            target_h = template_h
            if target_h > height + 10 and quality_ladder:
                continue
            out_h = min(target_h, height)
            out_w = even(out_h * width / height)

        quality_ladder.append((label, even(out_w), even(out_h), vbr, abr))

    return quality_ladder or [QUALITY_LADDER_TEMPLATES[0]]

# ── Redis Client ─────────────────────────────────────────────────────────────
def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=False)

# ── FFprobe: Get video metadata ───────────────────────────────────────────────
def get_video_metadata(input_path: str) -> tuple[float, int, int]:
    """Return (duration, width, height) using ffprobe."""
    try:
        result = subprocess.run(
            [FFPROBE_BIN, "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "format=duration:stream=width,height",
             "-of", "json", input_path],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace"
        )
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        
        streams = data.get("streams", [])
        width = 0
        height = 0
        if streams:
            width = int(streams[0].get("width", 0))
            height = int(streams[0].get("height", 0))
            if duration == 0:
                duration = float(streams[0].get("duration", 0))
                
        return duration, width, height
    except Exception as e:
        print(f"[ffprobe] Could not get metadata: {e}")
        return 0.0, 0, 0

def check_has_audio(input_path: str) -> bool:
    """Return True if the video file contains at least one audio stream."""
    try:
        result = subprocess.run(
            [FFPROBE_BIN, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", input_path],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace"
        )
        return "audio" in result.stdout.lower()
    except Exception as e:
        print(f"[ffprobe] Could not check audio stream: {e}")
        return True # Default to True to be safe

# ── Thumbnail generation ──────────────────────────────────────────────────────
def generate_thumbnail(input_path: str, output_path: str, duration: float) -> bool:
    """Extract the sharpest frame at ~5% into the video."""
    seek_time = max(3, duration * 0.05)
    cmd = [
        FFMPEG_BIN,
        "-ss", str(seek_time),
        "-i", input_path,
        "-vframes", "1",
        "-vf", "scale=1280:-2",
        "-q:v", "3",
        "-y", output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60,
                                encoding="utf-8", errors="replace")
        return result.returncode == 0
    except Exception as e:
        print(f"[Thumbnail] Error: {e}")
        return False

# ── Preview sprites ───────────────────────────────────────────────────────────
def generate_sprites(input_path: str, output_dir: str, duration: float, video_id: str) -> str | None:
    """
    Generate a sprite sheet JPEG and accompanying VTT file.
    Each thumbnail is SPRITE_THUMB_W x SPRITE_THUMB_H pixels.
    """
    if duration <= 0:
        return None

    sprite_path = os.path.join(output_dir, "sprites.jpg")
    vtt_path    = os.path.join(output_dir, "sprites.vtt")
    num_frames  = max(1, int(duration / SPRITE_INTERVAL))
    grid_cols   = min(10, num_frames)
    grid_rows   = math.ceil(num_frames / grid_cols)

    # Use FFmpeg tile filter to build the sprite sheet
    fps_expr = f"1/{SPRITE_INTERVAL}"
    vf = (
        f"fps={fps_expr},"
        f"scale={SPRITE_THUMB_W}:{SPRITE_THUMB_H}:force_original_aspect_ratio=decrease,"
        f"pad={SPRITE_THUMB_W}:{SPRITE_THUMB_H}:(ow-iw)/2:(oh-ih)/2,"
        f"tile={grid_cols}x{grid_rows}"
    )

    cmd = [
        FFMPEG_BIN, "-i", input_path,
        "-vf", vf,
        "-frames:v", "1",
        "-q:v", "5",
        "-y", sprite_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"[Sprites] FFmpeg error: {result.stderr[:300]}")
            return None
    except Exception as e:
        print(f"[Sprites] Error: {e}")
        return None

    # Write VTT file
    sprite_url = f"/api/media/stream/{video_id}/sprites.jpg"
    with open(vtt_path, "w") as vtt:
        vtt.write("WEBVTT\n\n")
        for i in range(num_frames):
            start_s  = i * SPRITE_INTERVAL
            end_s    = min(start_s + SPRITE_INTERVAL, duration)
            col      = i % grid_cols
            row      = i // grid_cols
            x        = col * SPRITE_THUMB_W
            y        = row * SPRITE_THUMB_H
            t_start  = _fmt_time(start_s)
            t_end    = _fmt_time(end_s)
            vtt.write(f"{t_start} --> {t_end}\n")
            vtt.write(f"{sprite_url}#xywh={x},{y},{SPRITE_THUMB_W},{SPRITE_THUMB_H}\n\n")

    return f"/api/media/stream/{video_id}/sprites.vtt"

def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

# ── Encryption setup ──────────────────────────────────────────────────────────
def setup_encryption(output_dir: str, video_id: str) -> tuple[str, str]:
    """
    Generate AES-128 key, store in Redis (with 1-year TTL), write keyinfo file.
    Returns (key_id, key_info_path).
    """
    key_id     = secrets.token_hex(16)
    key_bytes  = secrets.token_bytes(16)
    key_path   = os.path.join(output_dir, "enc.key")
    key_info_path = os.path.join(output_dir, "enc.keyinfo")

    # Store key on disk (used by FFmpeg during encoding only)
    with open(key_path, "wb") as f:
        f.write(key_bytes)

    # Also store in Redis for serving (keyed by key_id, not video path)
    try:
        r = get_redis()
        r.setex(f"hlskey:{key_id}", 60 * 60 * 24 * 365, key_bytes)
        print(f"[Encryption] Key {key_id} stored in Redis")
    except Exception as e:
        print(f"[Encryption] Redis unavailable, key on disk only: {e}")

    # Use relative key URI to force players to propagate the playback token query param
    key_uri = f"enc.key?kid={key_id}"
    with open(key_info_path, "w") as f:
        f.write(f"{key_uri}\n{key_path}\n")

    return key_id, key_info_path

# ── Multi-resolution HLS FFmpeg command ───────────────────────────────────────
def build_ffmpeg_cmd(
    input_path: str,
    output_dir: str,
    key_info_path: str,
    video_id: str,
    quality_ladder: list,
    has_audio: bool = True
) -> list[str]:
    """
    Build a single FFmpeg command that produces all quality variants simultaneously.
    This is the most efficient approach — one pass, minimal disk reads.
    """
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel", "warning",
        "-i", input_path,
    ]

    # Output map and encode settings per quality
    filter_chains = []
    for i, (label, w, h, vbr, abr) in enumerate(quality_ladder):
        filter_chains.append(
            f"[v:0]scale={w}:{h},setsar=1,"
            f"format=yuv420p[v{i}]"
        )

    cmd += ["-filter_complex", ";".join(filter_chains)]

    for i, (label, w, h, vbr, abr) in enumerate(quality_ladder):
        rend_dir = os.path.join(output_dir, label)
        os.makedirs(rend_dir, exist_ok=True)
        playlist = os.path.join(rend_dir, "playlist.m3u8")
        segment  = os.path.join(rend_dir, "seg_%04d.m4s")

        cmd += [
            # Video stream
            "-map", f"[v{i}]",
            "-c:v", "libx264",
            "-preset", "faster",   # Faster compression
            "-profile:v", "high",
            "-level", "4.1",
            "-crf", "22",          # Optimized CRF
            "-maxrate:v", f"{vbr}k",
            "-bufsize:v", f"{vbr * 2}k",
            "-x264opts", f"keyint={HLS_SEGMENT_DURATION * 30}:min-keyint={HLS_SEGMENT_DURATION * 30}:no-scenecut",
            "-tune", "film",
        ]

        if has_audio:
            cmd += [
                # Audio stream
                "-map", "0:a:0?",
                "-c:a", "aac",
                "-b:a", f"{abr}k",
                "-ar", "48000",
                "-ac", "2",
            ]

        cmd += [
            # HLS muxer — optimized for fMP4 / CMAF
            "-f", "hls",
            "-hls_time", str(HLS_SEGMENT_DURATION),
            "-hls_playlist_type", "vod",
            "-hls_segment_type", "fmp4",          # Fragmented MP4 (fMP4)
            "-hls_segment_filename", segment,
            "-hls_key_info_file", key_info_path,
            "-hls_flags", "independent_segments",
            "-hls_list_size", "0",
            "-start_number", "0",
            playlist,
        ]

    cmd += ["-threads", "0", "-y"]
    return cmd

def build_ffmpeg_cmd_for_quality(
    input_path: str,
    output_dir: str,
    key_info_path: str,
    quality: tuple,
    has_audio: bool = True
) -> list[str]:
    """Build a reliable single-quality HLS command that writes real .ts chunks."""
    label, w, h, vbr, abr = quality
    rend_dir = os.path.join(output_dir, label)
    os.makedirs(rend_dir, exist_ok=True)

    playlist = os.path.join(rend_dir, "playlist.m3u8")
    segment = os.path.join(rend_dir, "seg_%04d.ts")
    vf = f"scale={w}:{h},setsar=1,format=yuv420p"

    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-y",
        "-progress", "pipe:1",
        "-nostats",
        "-loglevel", "warning",
        "-i", input_path,
        "-map", "0:v:0",
    ]

    if has_audio:
        cmd += ["-map", "0:a:0?"]

    cmd += [
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-profile:v", "main",
        "-crf", "23",
        "-maxrate", f"{vbr}k",
        "-bufsize", f"{vbr * 2}k",
        "-g", str(HLS_SEGMENT_DURATION * 30),
        "-keyint_min", str(HLS_SEGMENT_DURATION * 30),
        "-sc_threshold", "0",
    ]

    if has_audio:
        cmd += [
            "-c:a", "aac",
            "-b:a", f"{abr}k",
            "-ar", "48000",
            "-ac", "2",
        ]

    cmd += [
        "-f", "hls",
        "-hls_time", str(HLS_SEGMENT_DURATION),
        "-hls_playlist_type", "vod",
        "-hls_segment_filename", segment,
        "-hls_key_info_file", key_info_path,
        "-hls_flags", "independent_segments",
        "-hls_list_size", "0",
        "-start_number", "0",
        playlist,
    ]

    return cmd

def count_hls_segments(output_dir: str, label: str) -> int:
    rend_dir = os.path.join(output_dir, label)
    if not os.path.exists(rend_dir):
        return 0
    return len([f for f in os.listdir(rend_dir) if f.endswith(".ts")])

# ── Master playlist ───────────────────────────────────────────────────────────
def write_master_playlist(output_dir: str, video_id: str, quality_ladder: list) -> str:
    """Write the HLS adaptive master.m3u8 referencing all quality playlists."""
    master_path = os.path.join(output_dir, "master.m3u8")
    base_url    = f"/api/media/stream/{video_id}"

    lines = ["#EXTM3U", "#EXT-X-VERSION:3", ""]

    for (label, w, h, vbr, abr) in quality_ladder:
        total_bw = (vbr + abr) * 1000
        lines.append(
            f'#EXT-X-STREAM-INF:BANDWIDTH={total_bw},'
            f'RESOLUTION={w}x{h},'
            f'CODECS="avc1.640028,mp4a.40.2",'
            f'NAME="{label}"'
        )
        lines.append(f"{base_url}/{label}/playlist.m3u8")
        lines.append("")

    with open(master_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[Master] Written: {master_path}")
    return f"{base_url}/master.m3u8"

# ── Notify Node.js backend ────────────────────────────────────────────────────
def notify_backend(payload: dict):
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, headers=internal_headers(), timeout=10)
        print(f"[Webhook] → {WEBHOOK_URL}: {resp.status_code}")
    except Exception as e:
        print(f"[Webhook] Failed: {e}")

# ── Main entry point ──────────────────────────────────────────────────────────
def process_video_hls(
    input_path: str,
    output_dir: str,
    video_id: str,
    media_id: str,
    watermark_path: str | None = None,
) -> bool:
    """
    Full pipeline:
    1. Probe duration, width, height
    2. Setup AES-128 encryption
    3. Generate thumbnail
    4. Run multi-resolution FFmpeg with live progress updates
    5. Write master.m3u8
    6. Generate preview sprites
    7. Delete source file
    8. Notify Node.js via webhook
    """
    start = time.time()
    print(f"\n{'='*60}")
    print(f"[GOG PROCESSOR] Starting: {video_id}")
    print(f"[GOG PROCESSOR] Input:    {input_path}")
    print(f"[GOG PROCESSOR] Output:   {output_dir}")
    print(f"{'='*60}\n")

    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        # ── 1. Probe duration, width, height ──────────────────────
        print("[1/6] Probing video metadata...")
        duration, width, height = get_video_metadata(input_path)
        print(f"      Duration: {duration:.1f}s")
        print(f"      Resolution: {width}x{height}")

        # Build dynamic quality ladder without changing the source orientation.
        quality_ladder = build_quality_ladder(width, height)
        print(f"      Active HLS Quality Ladder: {[q[0] for q in quality_ladder]}")
        print(f"      Output Dimensions: {[f'{q[0]}={q[1]}x{q[2]}' for q in quality_ladder]}")

        # ── 2. Encryption ─────────────────────────────────────────
        print("[2/6] Setting up AES-128 encryption...")
        key_id, key_info_path = setup_encryption(output_dir, video_id)

        # ── 3. Thumbnail ──────────────────────────────────────────
        print("[3/6] Generating thumbnail...")
        thumb_path = os.path.join(output_dir, "thumbnail.jpg")
        thumb_ok   = generate_thumbnail(input_path, thumb_path, duration)
        thumb_url  = f"/api/media/stream/{video_id}/thumbnail.jpg" if thumb_ok else None
        print(f"      Thumbnail: {'OK' if thumb_ok else 'FAILED'}")

        # ── 4. Multi-resolution FFmpeg ────────────────────────────
        print("[4/7] Checking for audio stream...")
        has_audio = check_has_audio(input_path)
        print(f"      Audio Stream: {'YES' if has_audio else 'NO (Silent Video)'}")

        print("[5/7] Running FFmpeg HLS encoding...")
        print(f"      Segment duration: {HLS_SEGMENT_DURATION}s")

        # Setup progress endpoint and state
        progress_url = urljoin(BACKEND_URL, "/api/upload/video-progress")
        def notify_progress(pct: float):
            try:
                requests.post(progress_url, json={
                    "media_id": media_id,
                    "progress": round(pct, 1)
                }, headers=internal_headers(), timeout=5)
            except Exception as pe:
                print(f"[Progress Webhook] Failed: {pe}")

        notify_progress(5)
        ffmpeg_start = time.time()
        last_reported_pct = 5.0
        last_report_time = 0.0
        total_qualities = max(len(quality_ladder), 1)

        for quality_index, quality in enumerate(quality_ladder):
            label = quality[0]
            base_pct = 5 + (quality_index / total_qualities) * 90
            range_pct = 90 / total_qualities
            print(f"      Encoding {label} ({quality_index + 1}/{total_qualities})...")

            ffmpeg_cmd = build_ffmpeg_cmd_for_quality(
                input_path,
                output_dir,
                key_info_path,
                quality,
                has_audio
            )

            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace"
            )

            log_tail = []
            while True:
                line = process.stdout.readline()
                if line:
                    log_tail.append(line.strip())
                    log_tail = log_tail[-80:]

                if not line and process.poll() is not None:
                    break

                if "out_time_us=" in line or "out_time_ms=" in line:
                    try:
                        time_raw = int(line.split("=")[1].strip())
                        time_s = time_raw / 1000000.0
                        if duration > 0:
                            current_quality_pct = min(1.0, time_s / duration)
                            pct = min(99.0, base_pct + current_quality_pct * range_pct)
                            now = time.time()
                            if (pct - last_reported_pct >= 2.0) or (now - last_report_time >= 3.0 and pct > last_reported_pct):
                                notify_progress(pct)
                                last_reported_pct = pct
                                last_report_time = now
                    except Exception:
                        pass

            process.wait()

            if process.returncode != 0:
                err_output = "\n".join(log_tail) or "Unknown FFmpeg error"
                print(f"[FFmpeg ERROR: {label}]\n{err_output[-1500:]}")
                notify_backend({
                    "media_id": media_id,
                    "status": "failed",
                    "error": f"FFmpeg {label} exit {process.returncode}: {err_output[-500:]}"
                })
                return False

            segment_count = count_hls_segments(output_dir, label)
            if segment_count <= 0:
                err = f"No HLS segments were written for {label}"
                print(f"[FFmpeg ERROR: {label}] {err}")
                notify_backend({
                    "media_id": media_id,
                    "status": "failed",
                    "error": err
                })
                return False

            notify_progress(min(99.0, base_pct + range_pct))
            last_reported_pct = min(99.0, base_pct + range_pct)
            print(f"      {label}: {segment_count} chunks written")

        ffmpeg_elapsed = time.time() - ffmpeg_start
        print(f"      FFmpeg finished in {ffmpeg_elapsed:.1f}s")

        # ── 5. Master playlist ────────────────────────────────────
        print("[5/6] Writing master.m3u8...")
        master_url = write_master_playlist(output_dir, video_id, quality_ladder)

        # ── 6. Preview sprites ────────────────────────────────────
        print("[6/6] Generating preview sprites...")
        sprite_vtt_url = generate_sprites(input_path, output_dir, duration, video_id)
        print(f"      Sprites: {sprite_vtt_url or 'SKIPPED'}")

        # ── Cleanup source ────────────────────────────────────────
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
                print(f"[Cleanup] Deleted source: {input_path}")
        except Exception as e:
            print(f"[Cleanup] Could not delete source: {e}")

        # ── Notify backend ────────────────────────────────────────
        resolutions = [
            {"label": lbl, "width": w, "height": h, "bitrate": vbr,
             "playlistPath": f"/api/media/stream/{video_id}/{lbl}/playlist.m3u8"}
            for (lbl, w, h, vbr, _abr) in quality_ladder
        ]
        notify_backend({
            "media_id": media_id,
            "status": "ready",
            "duration": round(duration, 2),
            "thumbnailUrl": thumb_url,
            "spriteUrl": sprite_vtt_url,
            "hlsKeyId": key_id,
            "resolutions": resolutions,
        })

        total = time.time() - start
        print(f"\n✅ [GOG PROCESSOR] Done in {total:.1f}s: {video_id}\n")
        return True

    except Exception as e:
        print(f"[FATAL] Uncaught exception for {video_id}: {e}")
        notify_backend({
            "media_id": media_id,
            "status": "failed",
            "error": str(e)[:500]
        })
        return False
