import face_recognition
import cv2
import numpy as np
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import io
from PIL import Image
import time

app = FastAPI(title="Face Recognition Service")

@app.get("/")
async def root():
    return {"message": "God of Graphics - Face AI Service is Running 🐍", "status": "online"}

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
    start_time = time.time()
    print(f"\n[DEBUG] 📩 Incoming /recognize request. Known faces: {len(request.known_faces)}")
    
    try:
        # 1. Decode image
        header, encoded = request.image_base64.split(",", 1) if "," in request.image_base64 else (None, request.image_base64)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))
        rgb_image = np.array(image.convert("RGB"))
        print(f"[DEBUG] 🖼️ Image decoded. Size: {image.size}")

        # 2. Detect all faces and their encodings
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        print(f"[DEBUG] 🔍 Detected {len(face_locations)} face(s) in image.")
        
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if not face_encodings:
            print("[DEBUG] ❓ No face encodings found.")
            return []

        # 3. Prepare known faces
        known_ids = [kf.id for kf in request.known_faces]
        known_encodings = [np.array(kf.descriptor) for kf in request.known_faces]

        results = []
        
        # 4. Match each detected face against all known faces
        for i, face_encoding in enumerate(face_encodings):
            if not known_encodings:
                continue
                
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            distance = face_distances[best_match_index]
            
            print(f"[DEBUG] Face {i+1}: Best match distance = {distance:.4f} (Threshold: {request.threshold})")
            
            if distance <= request.threshold:
                matched_id = known_ids[best_match_index]
                results.append(RecognitionResult(
                    student_id=matched_id,
                    confidence=float(1.0 - distance)
                ))
                print(f"[DEBUG] ✅ Match Found! ID: {matched_id}")
            else:
                print(f"[DEBUG] ❌ No match within threshold.")

        end_time = time.time()
        print(f"[DEBUG] ⏱️ Processing complete in {end_time - start_time:.2f} seconds.")
        return results

    except Exception as e:
        print(f"[DEBUG] 💥 ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-descriptor")
async def get_descriptor(request: dict):
    print(f"\n[DEBUG] 📩 Incoming /get-descriptor request.")
    try:
        image_base64 = request.get("image_base64")
        header, encoded = image_base64.split(",", 1) if "," in image_base64 else (None, image_base64)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))
        rgb_image = np.array(image.convert("RGB"))
        
        encodings = face_recognition.face_encodings(rgb_image)
        if not encodings:
            print("[DEBUG] ❌ No face detected for descriptor.")
            return {"success": False, "message": "No face detected"}
            
        print("[DEBUG] ✅ Descriptor generated successfully.")
        return {"success": True, "descriptor": encodings[0].tolist()}
    except Exception as e:
        print(f"[DEBUG] 💥 ERROR: {str(e)}")
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Face AI Service Debug Mode...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
