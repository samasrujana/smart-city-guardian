# Violence Detection Module

Frame-based violence/fight detection for the Smart City Guardian system, using transfer learning on the RLVS (Real Life Violence Situations) dataset.

## Approach

- Sampled frames from video clips (every ~15th frame) and trained an image classifier rather than a full video/action-recognition model, for speed and simplicity.
- Fine-tuned **MobileNetV2** (pretrained on ImageNet) with a small classification head on top.
- At inference time, predictions across a rolling window of recent frames are averaged (**temporal smoothing**) before triggering an alert, so a single misclassified frame doesn't cause a false alarm.
- Consecutive alerts are consolidated into a single **incident** with a start/end time, matching the team's shared alert JSON contract (hazard_type, confidence, severity, camera_id, timestamp).

## Dataset

- **RLVS (Real Life Violence Situations)** — Kaggle
- ~9,568 Violence frames / ~7,217 NonViolence frames (train split)

## Results

- **Validation accuracy: 89.8%** after 10 epochs
- **ROC AUC: 0.961**
- Confusion matrix, ROC curve, and training curves included in this folder

## Files

- `violence_model.keras` — trained model
- `violence_detection_training.ipynb` — full training notebook (Colab)
- `violence_service.py` — FastAPI service wrapping the model for the backend to call
- `training_curves.png` — accuracy/loss over epochs
- `confusion_matrix.png` — validation set confusion matrix
- `roc_curve.png` — ROC curve, AUC = 0.961
- `dataset_distribution.png` — train/val split by class
- `sample_frames_large.png` — sample training frames

## How to run inference

```python
import tensorflow as tf
model = tf.keras.models.load_model('violence_model.keras')
```

## How to run the service

```
pip install fastapi uvicorn tensorflow opencv-python python-multipart
uvicorn violence_service:app --reload --port 8001
```

Then POST a video file to `http://127.0.0.1:8001/detect/violence`, or open `http://127.0.0.1:8001/docs` for an interactive test page.
