"""
Fire & Smoke Detection Service
-------------------------------
Wraps the trained YOLOv8 fire/smoke model in a FastAPI service.
Exposes POST /detect which accepts an image and returns a detection
result matching the team's shared JSON contract.

Run locally with:
    uvicorn app:app --reload --port 8001

Test with:
    curl -X POST "http://127.0.0.1:8001/detect" \
         -F "file=@sample_fire_image.jpg" \
         -F "camera_id=CAM-04"
"""

from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import cv2

app = FastAPI(title="Fire & Smoke Detection Service")

# Load the trained model once at startup (not on every request — that would be slow)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Class index -> hazard name, based on the data.yaml used during training (['smoke', 'fire'])
CLASS_NAMES = {0: "smoke", 1: "fire"}

# Confidence threshold below which we don't report a detection at all
CONF_THRESHOLD = 0.4


def confidence_to_severity(confidence: float) -> str:
    """Simple rule-based severity mapping. Can be refined later
    (e.g. factor in bounding box size relative to frame)."""
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    else:
        return "Low"


@app.get("/")
def health_check():
    """Quick endpoint to confirm the service is running."""
    return {"status": "ok", "service": "fire-smoke-detection"}


@app.post("/detect")
async def detect(file: UploadFile = File(...), camera_id: str = Form(default="UNKNOWN")):
    """
    Accepts an image file and a camera_id, runs the fire/smoke model,
    and returns the highest-confidence detection in the team's shared
    JSON contract format. Returns hazard_type: "none" if nothing found.
    """
    # Read uploaded image bytes into an OpenCV-readable array
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not decode image. Make sure it's a valid image file."}

    # Run inference
    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    # No detections above threshold
    if len(boxes) == 0:
        return {
            "hazard_type": "none",
            "confidence": 0.0,
            "severity": "Low",
            "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            "bbox": None
        }

    # Pick the single highest-confidence detection to report
    confidences = boxes.conf.tolist()
    best_idx = int(np.argmax(confidences))
    best_box = boxes[best_idx]

    class_id = int(best_box.cls[0])
    confidence = float(best_box.conf[0])
    hazard_type = CLASS_NAMES.get(class_id, "unknown")
    severity = confidence_to_severity(confidence)

    # xywh = center_x, center_y, width, height in pixel coordinates
    bbox = [round(v, 1) for v in best_box.xywh[0].tolist()]

    return {
        "hazard_type": hazard_type,
        "confidence": round(confidence, 3),
        "severity": severity,
        "camera_id": camera_id,
        "timestamp": datetime.now().isoformat(),
        "bbox": bbox
    }
