# 🛰️ TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine

**TrackNet AI** is an enterprise-grade, high-precision Automatic Number Plate Recognition (ANPR), multi-camera spatial-temporal trajectory tracking, and urban traffic analytics platform built for modern smart city surveillance infrastructure.

---

## 🌟 Key Features

1. **2-Stage Hierarchical ANPR Engine**:
   - **Stage 1 (Detection)**: Fine-tuned YOLOv8 model for real-time license plate detection & localized bounding box extraction.
   - **Stage 2 (Recognition & Enhancement)**: OpenCV image preprocessor (CLAHE, BlackHat filter, Adaptive Thresholding) + EasyOCR high-precision optical character recognition with Fallback Super-Resolution and Indian syntax rules.
2. **OpenStreetMap (OSM) Live Camera Synchronizer**:
   - Dynamic real-world surveillance node discovery via OpenStreetMap Overpass API for major Indian metropolitan regions (*Mumbai, Pune, Ahmedabad, Gandhinagar, Surat, Vadodara, Rajkot*).
3. **Multi-Camera Spatial-Temporal Trajectory Tracking**:
   - Reconstructs vehicle movement chronologically across camera nodes, calculates segment speeds & segment distances using Haversine formulas, and renders interactive routes on Leaflet GIS maps.
4. **Automated PDF Trajectory Reporting**:
   - Generates official, downloadable PDF reports (powered by ReportLab) containing spatial-temporal timeline steps, GIS coordinates, vehicle metrics, and QR signatures.
5. **Macro Urban Traffic Analytics & Hotlist Watchdog**:
   - City-wide traffic volume heatmaps, peak-hour hourly flow breakdowns, origin-destination matrix, and real-time blacklisted vehicle alert registry.

---

## 📁 Repository Structure

```text
SIH127-I/
├── backend/                           # Python Flask REST API & AI Engine
│   ├── analytics_engine/              # Trajectory tracking, OSM sync, PDF reports & analytics
│   │   ├── macro_analytics.py         # Traffic volume, speed & origin-destination matrix
│   │   ├── osm_camera_sync.py         # OpenStreetMap Overpass API camera discoverer
│   │   ├── pdf_generator.py           # ReportLab spatial-temporal PDF exporter
│   │   └── trajectory_tracker.py      # Spatial-temporal sequence & route reconstruction
│   ├── anpr_engine/                   # Deep Learning & Image Preprocessing Pipeline
│   │   ├── dataset_converter.py       # Dataset format converter
│   │   └── train_yolo.py              # YOLOv8 license plate detector fine-tuning script
│   ├── config/                        # Multi-city bounding box & geospatial configurations
│   │   └── city_config.py             # Supported Indian cities & Overpass queries
│   ├── database/                      # SQLite / PostgreSQL Relational Database
│   │   ├── db_engine.py               # SQLAlchemy database session & engine
│   │   └── models.py                  # Camera, CameraConnection & ANPREvent schemas
│   ├── models/                        # Deep Learning Model Weights
│   │   └── anpr_yolo_best.pt          # Fine-tuned YOLOv8 Indian License Plate Detector
│   ├── reports/                       # Generated PDF Trajectory Audit Reports
│   ├── routes/                        # Flask API Blueprints & REST Endpoints
│   │   └── camera_routes.py           # Camera discovery, sync, trajectory & report routes
│   ├── test_dataset/                  # Benchmark test images & dataset files
│   ├── train_dataset/                 # Multi-source raw training image collections
│   ├── app.py                         # Main Flask REST API server (Port 5000)
│   └── tracknet.db                    # Synchronized camera & trajectory SQLite database
│
├── frontend/                          # Decoupled React 18 + Vite Web Application
│   ├── public/                        # Static public assets
│   ├── src/
│   │   ├── components/                # Modular React UI Components
│   │   │   ├── ANPRHub.jsx            # ANPR test hub, bounding box overlay & preprocessors
│   │   │   ├── BlacklistAlerts.jsx    # Hotlist vehicle watchlist & real-time alert feed
│   │   │   ├── Header.jsx             # Top bar, region selector & live OSM sync trigger
│   │   │   ├── MacroAnalytics.jsx     # Traffic heatmaps & hourly volume analytics
│   │   │   ├── MultiCameraTrajectory.jsx # GIS Leaflet route map & step-by-step timeline
│   │   │   └── Navbar.jsx             # Primary navigation tab bar
│   │   ├── styles/
│   │   │   └── main.css               # Glassmorphic CSS design system & utility classes
│   │   ├── App.jsx                    # Root React application wrapper
│   │   └── main.jsx                   # React DOM entrypoint
│   ├── index.html                     # Single-Page Application HTML host
│   ├── package.json                   # Frontend dependencies & scripts
│   └── vite.config.js                 # Vite bundler & API proxy configuration
├── .gitignore                         # Git exclusion rules
└── README.md                          # Comprehensive project documentation
```

---

## 🛠️ Tools & Technologies Used

### **Frontend & UI Stack**
- **React 18 & Vite**: Component-driven SPA architecture with high-frequency rendering and lightning-fast HMR dev server.
- **Vanilla CSS Design System**: Custom glassmorphism aesthetic (`--bg-primary: #0b0f19`, `--accent-blue: #0284c7`), responsive CSS grid/flexbox layouts, and custom utility classes.
- **Leaflet.js GIS Mapping**: Interactive city map visualization rendered with OpenStreetMap HOT map tiles (`https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png`).
- **FontAwesome 6**: Modern vector icons for UI state indicators, tab controls, and metrics headers.

### **Backend & Database**
- **Python 3.10+ & Flask**: Modular REST API built with Flask Blueprints and `Flask-CORS` for asynchronous client-server communication.
- **SQLAlchemy & SQLite**: ORM relational database schema managing `Camera`, `CameraConnection`, and `ANPREvent` records.
- **OpenStreetMap Overpass API**: Live Overpass QL queries (`[out:json]; node["man_made"="surveillance"]...`) for automated surveillance camera discovery.

### **Computer Vision & AI/ML**
- **Ultralytics YOLOv8**: Object detection model fine-tuned on custom Indian vehicle license plate annotations (`models/anpr_yolo_best.pt`).
- **EasyOCR Engine**: PyTorch-backed optical character recognition (OCR) engine for multi-character license plate text extraction.
- **OpenCV (cv2)**: Digital image processing pipeline featuring Contrast Limited Adaptive Histogram Equalization (CLAHE), BlackHat morphological filtering, Adaptive Thresholding, and Sobel edge detection.

### **Analytics & PDF Reporting**
- **ReportLab**: Programmatic PDF report generator featuring flowables, dynamic canvas styling, spatial-temporal timeline tables, and QR code verification signatures.
- **Geospatial Analytics**: Custom Haversine spherical distance calculation algorithms for inter-camera segment distance and vehicle speed metrics.

---

## 📊 Datasets Used So Far

1. **YOLOv8 Indian License Plate Training Dataset**:
   - **Annotated Samples**: 1,441 training images & 339 validation images with precise bounding box coordinates formatted for YOLOv8 (`data.yaml`).
2. **Multi-Source Raw Training Image Collections**:
   - **Roboflow Indian License Plates Dataset**: Standardized Indian vehicle license plate annotations.
   - **Scraped Indian Traffic Images**: 883 real-world traffic camera snapshots captured under varied lighting and angles.
   - **Manually Annotated Plate Datasets (Sets 1 & 2)**: Custom annotated high-resolution vehicle front/rear plates.
   - **OLX Statewise Sample Plates**: License plate samples representing 36 Indian States & Union Territories.
   - **CCTV Video Frame Extractions**: 1,308 frame captures from city traffic surveillance cameras.
3. **Benchmark Test Dataset**:
   - 50+ benchmark test images (`test1.jpg` .. `test51.jpeg`) covering daylight, night IR, rain, multi-plate, and angled vehicle shots.
4. **Geospatial OpenStreetMap Camera Datasets**:
   - Real-world surveillance camera nodes fetched dynamically from OpenStreetMap across Indian metropolitan regions (*Mumbai, Pune, Ahmedabad, Gandhinagar, Surat, Vadodara, Rajkot*).

---

## 🚀 Setup & Quick Start Guide

### 1. Backend Setup
```bash
# Navigate to backend folder
cd backend

# Create & activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required Python dependencies
pip install flask flask-cors ultralytics easyocr opencv-python sqlalchemy reportlab requests pillow numpy

# Start Flask Backend REST Server (Runs on http://127.0.0.1:5000)
python app.py
```

### 2. Frontend Setup
```bash
# Open a new terminal and navigate to frontend folder
cd frontend

# Install Node modules
npm install

# Start Vite Development Server (Runs on http://localhost:5173)
npm run dev
```

### 3. Build Frontend for Production
```bash
cd frontend
npm run build
```

---

## 📄 License & Attribution
Developed for smart city traffic management, multi-camera trajectory tracking, and law enforcement vehicle audit workflows. Powered by OpenStreetMap, Ultralytics YOLOv8, and EasyOCR.
