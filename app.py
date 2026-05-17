import face_recognition
import cv2
import numpy as np
import base64
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import io
from PIL import Image
import time
import os
from dotenv import load_dotenv
from video_processor import process_video_hls

# Load Environment Variables
load_dotenv()

app = FastAPI(title="God of Graphics - AI Service")

@app.get("/")
async def root():
    return {
        "message": "God of Graphics - AI & Video Service is Running",
        "status": "online",
        "version": "2.0.0",
        "active_backend_url": os.getenv("NODE_BACKEND_URL", "NOT_SET_IN_ENV")
    }

# --- Video Processing Models & Endpoints ---

class VideoProcessRequest(BaseModel):
    input_path: str
    output_dir: str
    video_id: str
    media_id: str
    watermark_path: Optional[str] = None

@app.post("/process-video")
async def process_video(request: VideoProcessRequest, background_tasks: BackgroundTasks):
    """
    Offload heavy video processing (HLS + Encryption + Watermark) to background.
    """
    print(f"\n[DEBUG] Incoming /process-video for {request.video_id}")
    
    # 1. Check if input file exists
    if not os.path.exists(request.input_path):
        raise HTTPException(status_code=404, detail="Input video file not found")

    # 2. Add to background tasks
    background_tasks.add_task(
        process_video_hls,
        request.input_path,
        request.output_dir,
        request.video_id,
        request.media_id,
        request.watermark_path
    )
    
    return {
        "success": True, 
        "message": "Video processing offloaded successfully",
        "video_id": request.video_id
    }

@app.get("/video-status/{video_id}")
async def video_status(video_id: str, output_dir: str):
    """
    Check if video processing is complete.
    """
    playlist_path = os.path.join(output_dir, "index.m3u8")
    if os.path.exists(playlist_path):
        return {"status": "ready", "videoId": video_id}
    
    if os.path.exists(output_dir):
        return {"status": "processing", "videoId": video_id}
        
    return {"status": "not_found", "videoId": video_id}

# --- Face Recognition (Existing Logic) ---

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
    # ... (existing recognize_faces logic)
    start_time = time.time()
    try:
        header, encoded = request.image_base64.split(",", 1) if "," in request.image_base64 else (None, request.image_base64)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))
        rgb_image = np.array(image.convert("RGB"))
        
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if not face_encodings:
            return []

        known_ids = [kf.id for kf in request.known_faces]
        known_encodings = [np.array(kf.descriptor) for kf in request.known_faces]

        results = []
        for i, face_encoding in enumerate(face_encodings):
            if not known_encodings: continue
                
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            distance = face_distances[best_match_index]
            
            if distance <= request.threshold:
                results.append(RecognitionResult(
                    student_id=known_ids[best_match_index],
                    confidence=float(1.0 - distance)
                ))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-descriptor")
async def get_descriptor(request: dict):
    try:
        image_base64 = request.get("image_base64")
        header, encoded = image_base64.split(",", 1) if "," in image_base64 else (None, image_base64)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))
        rgb_image = np.array(image.convert("RGB"))
        
        encodings = face_recognition.face_encodings(rgb_image)
        if not encodings:
            return {"success": False, "message": "No face detected"}
            
        return {"success": True, "descriptor": encodings[0].tolist()}
    except Exception as e:
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting God of Graphics AI Service on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
