import os
import subprocess
import secrets
import requests
from dotenv import load_dotenv
from urllib.parse import urljoin

load_dotenv()

def process_video_hls(input_path, output_dir, video_id, media_id, watermark_path=None):
    """
    Converts video to HLS with AES-128 encryption, generates thumbnails, and applies watermark.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 1. HLS Encryption Setup
        key_path = os.path.join(output_dir, "enc.key")
        key_info_path = os.path.join(output_dir, "enc.keyinfo")
        
        # Generate 16-byte random key
        key = secrets.token_bytes(16)
        with open(key_path, "wb") as f:
            f.write(key)
        
        # Key info file for FFmpeg:
        # line 1: Key URI (where the player fetches the key securely from Node.js)
        # line 2: Local path to the key file for FFmpeg to read during conversion
        key_uri = f"/api/media/key/{video_id}"
        with open(key_info_path, "w") as f:
            f.write(f"{key_uri}\n{key_path}")

        playlist_path = os.path.join(output_dir, "index.m3u8")
        
        # 2. Fast Thumbnail Generation
        ffmpeg_bin = os.getenv("FFMPEG_PATH", "ffmpeg")
        print(f"[GOG AI] Generating thumbnail for {video_id}...")
        thumb_path = os.path.join(output_dir, "thumbnail.jpg")
        thumb_cmd = [
            ffmpeg_bin, "-ss", "00:00:01", "-i", input_path,
            "-vframes", "1", "-q:v", "2", thumb_path, "-y"
        ]
        try:
            subprocess.run(thumb_cmd, capture_output=True, timeout=30)
        except Exception as e:
            print(f"[Thumbnail Error] Could not generate thumbnail: {e}")

        # 3. FFmpeg HLS Conversion Command
        # We include watermark if provided
        ffmpeg_cmd = [
            ffmpeg_bin, "-i", input_path
        ]

        if watermark_path and os.path.exists(watermark_path):
            # Apply watermark at bottom-right with padding
            ffmpeg_cmd += [
                "-i", watermark_path,
                "-filter_complex", "overlay=main_w-overlay_w-10:main_h-overlay_h-10"
            ]

        ffmpeg_cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-profile:v", "baseline", "-level", "3.0",
            "-start_number", "0", "-hls_time", "20",
            "-hls_list_size", "0", "-f", "hls",
            "-preset", "ultrafast", "-threads", "0",
            "-tune", "fastdecode", "-c:a", "aac",
            "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-hls_key_info_file", key_info_path,
            "-y",
            playlist_path
        ]

        print(f"[GOG AI] Starting processing for {video_id}...")
        
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if process.returncode != 0:
            print(f"[FFmpeg Error]: {process.stderr}")
            # Notify backend of failure
            try:
                backend_url = os.getenv("NODE_BACKEND_URL")
                requests.post(urljoin(backend_url, "/api/upload/video-processed"), json={"media_id": str(media_id), "status": "failed"}, timeout=5)
            except:
                pass
            return False
            
        # 4. Notify Node.js Backend
        backend_url = os.getenv("NODE_BACKEND_URL")
        webhook_url = urljoin(backend_url, "/api/upload/video-processed")
        try:
            requests.post(webhook_url, json={"media_id": str(media_id), "status": "ready"}, timeout=5)
            print(f"[Webhook] Notified backend for {video_id}")
        except Exception as e:
            print(f"[Webhook Error] Could not notify backend: {e}")
            
        # 5. Delete original MP4 to secure content
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
                print(f"[Cleanup] Deleted original MP4 file: {input_path}")
        except Exception as e:
            print(f"[Cleanup Error] Could not delete {input_path}: {e}")

        print(f"[GOG AI] Successfully processed {video_id}")
        return True
            
    except Exception as e:
        print(f"[Processing Exception]: {str(e)}")
        try:
            backend_url = os.getenv("NODE_BACKEND_URL")
            requests.post(urljoin(backend_url, "/api/upload/video-processed"), json={"media_id": str(media_id), "status": "failed"}, timeout=5)
        except:
            pass
        return False
