"""
Violence Detection Service
---------------------------
Wraps the trained violence/fight detection model (MobileNetV2 transfer
learning, fine-tuned on RLVS) in a FastAPI service. Accepts a video file,
runs frame-by-frame inference with temporal smoothing, and returns
consolidated incidents matching the team's shared JSON contract.

Run locally with:
    uvicorn violence_service:app --reload --port 8001

Test with:
    curl -X POST "http://127.0.0.1:8001/detect/violence" -F "file=@sample_video.mp4"

Or open http://127.0.0.1:8001/docs in a browser for an interactive test page.
"""

from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import cv2
from collections import deque
import tempfile, os

app = FastAPI(title="Violence Detection Service")

# Load the trained model once at startup (not on every request — that would be slow)
MODEL_PATH = "violence_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

WINDOW_SIZE = 15       # number of sampled frames averaged for smoothing
THRESHOLD = 0.8        # smoothed confidence above which an alert fires
SAMPLE_EVERY = 5       # check every Nth frame, for speed


@app.get("/")
def health_check():
    """Quick endpoint to confirm the service is running."""
    return {"status": "ok", "service": "violence-detection"}


def predict_video(video_path, camera_id="CAM-02"):
    cap = cv2.VideoCapture(video_path)
    window = deque(maxlen=WINDOW_SIZE)
    frame_count = 0
    alerts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % SAMPLE_EVERY == 0:
            img = cv2.resize(frame, (224, 224))
            img = np.expand_dims(img, axis=0).astype(np.float32)
            pred = model.predict(img, verbose=0)[0][0]  # 0=NonViolence, 1=Violence
            window.append(pred)
            smoothed_conf = sum(window) / len(window)

            if smoothed_conf > THRESHOLD:
                timestamp_sec = frame_count / cap.get(cv2.CAP_PROP_FPS)
                alerts.append({
                    "hazard_type": "violence",
                    "confidence": round(float(smoothed_conf), 2),
                    "severity": "High" if smoothed_conf > 0.85 else "Medium",
                    "camera_id": camera_id,
                    "timestamp": round(timestamp_sec, 1)
                })
        frame_count += 1

    cap.release()
    return alerts


def consolidate_alerts(alerts, gap_threshold=2.0):
    """Collapses consecutive alerts from the same violent scene into a
    single incident with a start/end time, instead of one row per frame."""
    if not alerts:
        return []

    incidents = []
    current = dict(
        alerts[0],
        start_time=alerts[0]["timestamp"],
        end_time=alerts[0]["timestamp"],
        max_confidence=alerts[0]["confidence"]
    )

    for a in alerts[1:]:
        if a["timestamp"] - current["end_time"] <= gap_threshold:
            current["end_time"] = a["timestamp"]
            current["max_confidence"] = max(current["max_confidence"], a["confidence"])
            if a["severity"] == "High":
                current["severity"] = "High"
        else:
            incidents.append(current)
            current = dict(
                a,
                start_time=a["timestamp"],
                end_time=a["timestamp"],
                max_confidence=a["confidence"]
            )
    incidents.append(current)
    return incidents


@app.post("/detect/violence")
async def detect_violence(file: UploadFile = File(...)):
    """
    Accepts a video file, runs the violence detection model frame-by-frame
    with temporal smoothing, and returns consolidated incidents.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    raw_alerts = predict_video(tmp_path)
    incidents = consolidate_alerts(raw_alerts)
    os.remove(tmp_path)

    return {"incidents": incidents}
