* &#x20;Smart City Guardian: AI-Powered Multi-Hazard Detection



An AI-based surveillance system that monitors CCTV camera feeds in real time to detect multiple public safety and infrastructure hazards, verifies them through a human-in-the-loop control room, and automatically routes confirmed alerts to the correct city department.



* &#x20;Problem Statement



Current smart city surveillance systems either detect only a single hazard, depend on manual monitoring, generate false alarms, or delay emergency response. There is no unified AI system that detects multiple hazards, verifies them, and routes them to the correct authority automatically.



* &#x20;Our Approach



Unlike single-hazard systems, Smart City Guardian:

\- Detects \*\*four hazard types\*\* from one platform

\- Performs \*\*severity analysis\*\* (Low / Medium / High) on every detection

\- Sends every alert to a \*\*Control Room for human verification\*\* before action — this is the key differentiator that reduces false alarms

\- \*\*Automatically routes\*\* verified alerts to the correct department

\- Provides a \*\*centralized live dashboard\*\* for monitoring



* &#x20;Modules



| Module | Detects | Routed To | Status |

|---|---|---|---|

| 🔥 Fire \& Smoke Detection | Fire, smoke | Fire Department | ✅ In progress |

| 🚨 Violence Detection | Fights, weapon-based assault | Police | 🔲 In progress |

| 🛣️ Road Infrastructure Monitoring | Potholes, road damage, waterlogging | Municipal Corporation | 🔲 In progress |

| 🗑️ Garbage/Waste Overflow Detection | Overflowing bins, roadside waste | Sanitation Department | 🔲 Planned |



\## Workflow



```

CCTV Camera

&#x20;   ↓

AI Detection (YOLOv8 + OpenCV)

&#x20;   ↓

Severity Analysis

&#x20;   ↓

Smart City Control Room

&#x20;   ↓

Officer Verification

&#x20;   ↓

Alert to Concerned Department

&#x20;   ↓

Action Taken

```



\## Tech Stack



\- \*\*AI / Computer Vision\*\*: Python, YOLOv8 (Ultralytics), OpenCV, Deep Learning

\- \*\*Backend\*\*: Spring Boot (Java), REST APIs

\- \*\*Frontend\*\*: React

\- \*\*Database\*\*: MySQL

\- \*\*Detection service layer\*\*: FastAPI (wraps each trained model, communicates with the backend)



\## Datasets Used



| Hazard | Dataset |

|---|---|

| Fire \& Smoke | \[D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) |

| Violence | RLVS (Real Life Violence Situations) |

| Potholes \& Road Damage | RDD2022 |

| Waterlogging | FloodNet |

| Garbage | TACO, TrashNet |



\## Repository Structure



```

smart-city-guardian/

├── fire-detection/          # YOLOv8 fire \& smoke detection model + training notebook

├── violence-detection/      # Violence/fight detection model

├── road-garbage-detection/  # Pothole, road damage, waterlogging, garbage detection

├── backend/                 # Spring Boot REST API, database models

├── frontend/                # React dashboard

└── docs/                    # Reports, diagrams, presentation materials

```



\## Current Progress



\- \[x] Fire \& Smoke detection — YOLOv8 model trained on D-Fire dataset (toy run: mAP50 = 0.67)

\- \[ ] Violence detection module

\- \[ ] Road infrastructure detection module

\- \[ ] Garbage overflow detection module

\- \[ ] Spring Boot backend + MySQL schema

\- \[ ] React control room dashboard

\- \[ ] FastAPI services wrapping each detector

\- \[ ] End-to-end integration



\## Team



| Name | Module |

|---|---|

| Sama Srujana | Fire \& Smoke Detection |

| Sushma Chintham | Violence Detection |

| Varshitha penthala | Road Infrastructure + Garbage Detection |



\## How to Run (Fire \& Smoke module)



1\. Open `fire-detection/fire\_smoke\_detection\_training.ipynb` in Google Colab

2\. Install dependencies: `pip install ultralytics kagglehub`

3\. Download the dataset via `kagglehub` (see notebook for exact steps)

4\. Run training cells, or load the pretrained `best.pt` directly for inference:

```python

from ultralytics import YOLO

model = YOLO('fire-detection/best.pt')

results = model.predict('path/to/image\_or\_video', conf=0.4)

```



\---

\*This project is being developed as a B.Tech major project.\*

