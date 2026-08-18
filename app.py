"""
app.py — God of Graphics AI & Video Service
Handles:
  - Video processing (multi-resolution HLS)
  - Face recognition (attendance)
  - Encryption key retrieval
"""
import os
import time
import base64
import io
import json
import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

from video_processor import process_video_hls

load_dotenv()

app = FastAPI(title="God of Graphics — AI & Video Service", version="3.0.0")

# ── Redis ────────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASS = os.getenv("REDIS_PASSWORD", None)

def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=False)

# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    r_ok = False
    try:
        r = get_redis()
        r.ping()
        r_ok = True
    except Exception:
        pass
    return {
        "message": "God of Graphics — AI & Video Service",
        "status": "online",
        "version": "3.0.0",
        "redis": "connected" if r_ok else "unavailable",
        "backend_url": os.getenv("NODE_BACKEND_URL", "NOT_SET"),
    }

# ── Video Processing ─────────────────────────────────────────────────────────
class VideoProcessRequest(BaseModel):
    input_path: str
    output_dir: str
    video_id: str
    media_id: str
    watermark_path: Optional[str] = None

@app.post("/process-video")
async def process_video(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """
    Offload multi-resolution HLS processing to a background task.
    Returns immediately — Node.js BullMQ worker waits for this response
    then Python calls the webhook when done.
    """
    print(f"\n[DEBUG] /process-video → {request.video_id}")

    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail=f"Input file not found: {request.input_path}")

    background_tasks.add_task(
        process_video_hls,
        request.input_path,
        request.output_dir,
        request.video_id,
        request.media_id,
        request.watermark_path,
    )

    return {
        "success": True,
        "message": "Processing started in background",
        "video_id": request.video_id,
    }

@app.get("/video-status/{video_id}")
async def video_status(video_id: str, output_dir: str = ""):
    """Check if video processing is complete."""
    if output_dir and os.path.exists(os.path.join(output_dir, "master.m3u8")):
        return {"status": "ready", "video_id": video_id}
    if output_dir and os.path.exists(output_dir):
        return {"status": "processing", "video_id": video_id}
    return {"status": "not_found", "video_id": video_id}

# ── Encryption Key Retrieval ─────────────────────────────────────────────────
@app.get("/hls-key/{key_id}")
async def get_hls_key(key_id: str, request: Request):
    """
    Return the AES-128 encryption key bytes from Redis.
    Called by the Nginx auth_request chain — not directly by players.
    """
    try:
        r = get_redis()
        key_bytes = r.get(f"hlskey:{key_id}")
        if not key_bytes:
            raise HTTPException(status_code=404, detail="Key not found")
        return Response(content=key_bytes, media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis error: {str(e)}")

# ── Face Recognition (lazy import to avoid startup RAM waste) ─────────────────
_face_recognition = None
_cv2 = None
_np = None

def _load_face_libs():
    global _face_recognition, _cv2, _np
    if _face_recognition is None:
        import face_recognition
        import cv2
        import numpy as np
        from PIL import Image
        _face_recognition = face_recognition
        _cv2 = cv2
        _np = np

class KnownFace(BaseModel):
    id: str
    descriptor: List[float]

class RecognitionRequest(BaseModel):
    image_base64: str
    known_faces: List[KnownFace]
    threshold: float = 0.6

class RecognitionResult(BaseModel):
    student_id: str
    confidence: float

@app.post("/recognize", response_model=List[RecognitionResult])
async def recognize_faces(request: RecognitionRequest):
    _load_face_libs()
    fr = _face_recognition
    np = _np

    try:
        from PIL import Image
        _, encoded = request.image_base64.split(",", 1) if "," in request.image_base64 else (None, request.image_base64)
        
        # Robust Base64 Cleaning: strip newlines, spaces, fix URL-encoding '+' sign, and add missing padding
        clean_encoded = encoded.replace(" ", "+").replace("\n", "").replace("\r", "")
        missing_padding = len(clean_encoded) % 4
        if missing_padding:
            clean_encoded += "=" * (4 - missing_padding)

        image = Image.open(io.BytesIO(base64.b64decode(clean_encoded))).convert("RGB")
        rgb   = np.array(image)

        locations = fr.face_locations(rgb, model="hog")
        encodings = fr.face_encodings(rgb, locations)
        print(f"[DEBUG] /recognize: faces_detected={len(encodings)} known_faces={len(request.known_faces)} threshold={request.threshold}")
        if not encodings:
            print("[DEBUG] /recognize: no faces detected in frame")
            return []

        known_ids  = [kf.id for kf in request.known_faces]
        known_encs = [np.array(kf.descriptor) for kf in request.known_faces]

        results = []
        for i, enc in enumerate(encodings):
            if not known_encs:
                continue
            distances = fr.face_distance(known_encs, enc)
            best = int(np.argmin(distances))
            print(f"[DEBUG] /recognize: face#{i} closest known_id={known_ids[best]} distance={distances[best]:.4f}")
            if distances[best] <= request.threshold:
                results.append(RecognitionResult(
                    student_id=known_ids[best],
                    confidence=float(1.0 - distances[best])
                ))
            else:
                print(f"[DEBUG] /recognize: face#{i} rejected — distance {distances[best]:.4f} > threshold {request.threshold}")
        print(f"[DEBUG] /recognize: returning {len(results)} match(es): {[(r.student_id, round(r.confidence, 3)) for r in results]}")
        return results
    except Exception as e:
        print(f"[DEBUG] /recognize: ERROR {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-descriptor")
async def get_descriptor(request: dict):
    _load_face_libs()
    fr = _face_recognition
    np = _np

    try:
        from PIL import Image
        img_b64 = request.get("image_base64", "")
        _, encoded = img_b64.split(",", 1) if "," in img_b64 else (None, img_b64)
        
        # Robust Base64 Cleaning: strip newlines, spaces, fix URL-encoding '+' sign, and add missing padding
        clean_encoded = encoded.replace(" ", "+").replace("\n", "").replace("\r", "")
        missing_padding = len(clean_encoded) % 4
        if missing_padding:
            clean_encoded += "=" * (4 - missing_padding)

        image = Image.open(io.BytesIO(base64.b64decode(clean_encoded))).convert("RGB")
        arr = np.array(image)
        print(f"[DEBUG] /get-descriptor: image_shape={arr.shape} b64_len={len(clean_encoded)}")
        encs  = fr.face_encodings(arr)
        print(f"[DEBUG] /get-descriptor: faces_detected={len(encs)}")
        if not encs:
            print("[DEBUG] /get-descriptor: no face found — returning success=False")
            return {"success": False, "message": "No face detected"}
        print(f"[DEBUG] /get-descriptor: success — descriptor_len={len(encs[0])}")
        return {"success": True, "descriptor": encs[0].tolist()}
    except Exception as e:
        print(f"[DEBUG] /get-descriptor: ERROR {e}")
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # On Windows, multiple workers via multiprocessing/spawn can raise WinError 10022. Force 1 worker.
    workers = 1 if os.name == 'nt' else int(os.getenv("UVICORN_WORKERS", 2))
    print(f"Starting GOG AI Service on port {port} with {workers} worker(s)...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=workers)
