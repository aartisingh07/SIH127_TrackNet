# TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine

TrackNet is a high-precision Automatic Number Plate Recognition (ANPR), multi-camera spatial-temporal trajectory tracking, and macro traffic analytics engine designed for modern urban smart cities.

---

## 📁 Repository & Dataset Structure

```text
SIH127-I/
├── dataset_yolo/                  # Structured YOLOv8 Training Dataset
│   ├── data.yaml                  # Dataset configuration (classes & paths)
│   ├── images/                    # Image splits (1,441 train, 339 val)
│   └── labels/                    # Bounding box labels (.txt)
├── train_dataset/                 # Raw & Multi-Source Training Datasets
│   ├── google_images/             # Scraped traffic images (883 samples)
│   ├── manual_annotations/        # Manually annotated datasets 1 & 2
│   ├── olx_statewise/             # 36 Indian States/UTs sample plates
│   ├── roboflow_yolov8/           # Roboflow Indian Cars License Plate Dataset
│   └── video_frames/              # CCTV video frame extractions (1,308 samples)
├── test_dataset/                  # Web App Test Images & Online Test Set
│   ├── test1/                     # Online dataset (images & labels)
│   └── test1.jpg .. test51.jpeg   # Benchmark test images
├── models/                        # Deep Learning Model Weights
│   └── anpr_yolo_best.pt          # Fine-tuned YOLOv8 License Plate Detector
├── analytics_engine/              # Trajectory reconstruction & PDF reports
├── anpr_engine/                   # 2-Stage Hierarchical ANPR & OCR Engine
├── static/                        # CSS styles & JS Zoom Engine
├── templates/                     # Web Dashboard (index.html)
├── app.py                         # Flask Web App Server & REST API
└── train_yolo.py                  # One-Click YOLOv8 Model Training Script
```

---

## 🚀 Quick Start & How to Train Model

### 1. Install Dependencies
```bash
pip install ultralytics easyocr opencv-python flask reportlab pillow numpy
```

### 2. Train / Fine-tune YOLOv8 Model
To train or re-train the plate detector on the complete dataset:
```bash
python train_yolo.py
```
The script will fine-tune YOLOv8 on `dataset_yolo/data.yaml` and save the updated weights to `models/anpr_yolo_best.pt`.

### 3. Run Web Dashboard
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000` to access:
- **ANPR & OCR Test Hub**: 2-Stage detection, zoom controls, and image inspection viewer.
- **Multi-Camera Trajectory Tracking**: Leaflet GIS spatial-temporal route reconstruction & PDF reporting.
- **Macro Traffic Analytics**: City heatmaps, speed monitoring, and origin-destination matrix.
- **Hotlist Watchdog**: Real-time stolen/blacklisted vehicle alert registry.
