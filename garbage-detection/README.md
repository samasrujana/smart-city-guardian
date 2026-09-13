# Garbage Overflow Detection Module

Part of the Smart City Guardian multi-hazard CCTV monitoring system.

## Dataset
**TACO** (Trash Annotations in Context) — `kneroma/tacotrashdataset` on Kaggle.
Original dataset has ~60 fine-grained trash categories; collapsed to a single
class (`garbage`) since the use case is overflow detection, not item
classification.

## Model
- Architecture: YOLOv8n (nano — fastest to train/iterate on)
- Training: 15 epochs, image size 640, batch size 16
- Split: 1200 train images / 300 val images (80/20)
- Trained on Google Colab (free T4 GPU tier)

## Accuracy log

| Date | Epochs | mAP@0.5 | Precision | Recall | Notes |
|---|---|---|---|---|---|
| 2026-09-13 | 15 | [fill in from training output] | [fill in] | [fill in] | First working version, reduced epochs due to Colab session timeouts during longer runs |

*(Earlier 50-epoch run reached mAP@0.5 = 0.443, precision = 0.592, recall = 0.415, but was lost to a Colab disconnect before weights could be downloaded. Plan to retrain at higher epoch count once time allows.)*

## Files in this folder

- `best.pt` — trained YOLOv8 weights
- `service.py` — FastAPI wrapper exposing `POST /detect`, matching the team's shared output contract
- `data.yaml` — YOLO dataset config used for training
- Training notebook (`.ipynb`) — full training pipeline (Kaggle download → COCO-to-YOLO conversion → train/val split → training)

## Running the service