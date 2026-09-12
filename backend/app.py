"""
================================================================================
File: app.py
Project: TrackNet AI - City-Wide Multi-Camera ANPR & Urban Traffic Analytics Engine
Purpose: Main Flask web application server, REST API router, and engine orchestrator.
Why this file was made:
  To serve the index dashboard template, expose 2-Stage ANPR detection APIs, register
  the modular OpenStreetMap camera discovery blueprint, and orchestrate analytics engines.
================================================================================
"""

import os
import re
import glob
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file

from database.db_engine import init_db
from database.models import Camera
from anpr_engine.anpr_ocr import ANPROCREngine
from analytics_engine.trajectory_tracker import TrajectoryTracker
from analytics_engine.macro_analytics import MacroTrafficAnalytics
from analytics_engine.pdf_generator import generate_trajectory_pdf
from routes.camera_routes import camera_api

app = Flask(__name__)

try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    pass

# Register OpenStreetMap Camera API Blueprint
app.register_blueprint(camera_api)

@app.after_request
def add_cache_control_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Initialize Database Schema & Engines
init_db()
anpr_engine = ANPROCREngine()
trajectory_tracker = TrajectoryTracker()
macro_analytics = MacroTrafficAnalytics(trajectory_tracker=trajectory_tracker)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATASET_DIR = os.path.join(BASE_DIR, "test_dataset")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def image_to_base64(img_np):
    if img_np is None or img_np.size == 0:
        return ""
    _, buffer = cv2.imencode('.jpg', img_np)
    return base64.b64encode(buffer).decode('utf-8')


def draw_2stage_annotations(img, results):
    """Draws Stage-1 Vehicle Box (Blue) and Stage-2 License Plate Box (Green)."""
    annotated = img.copy()
    for res in results:
        if 'vehicle_bbox' in res:
            vx1, vy1, vx2, vy2 = res['vehicle_bbox']
            vtype = res.get('vehicle_type', 'vehicle').upper()
            vconf = res.get('vehicle_confidence', 0.8)
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 144, 30), 2)
            cv2.putText(annotated, f"VEHICLE: {vtype} ({int(vconf*100)}%)", (vx1 + 5, vy1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 144, 30), 2)

        px1, py1, px2, py2 = res['bbox']
        plate = res['plate_text']
        conf = res['confidence']
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.rectangle(annotated, (px1, py1 - 22), (px1 + len(plate)*11 + 45, py1), (0, 255, 0), -1)
        cv2.putText(annotated, f"{plate} ({int(conf*100)}%)", (px1 + 4, py1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return annotated


@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "service": "TrackNet AI ANPR Backend REST Engine",
        "version": "2.0.0"
    })


@app.route('/api/anpr/detect', methods=['POST'])
def api_anpr_detect():
    """Runs 2-Stage Hierarchical ANPR (Vehicle Detection -> Plate Detection -> OCR)."""
    try:
        req_b64 = request.get_json(force=True, silent=True) or {}
        if 'image' in request.files:
            file = request.files['image']
            np_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        elif 'image_base64' in req_b64:
            b64_data = req_b64['image_base64'].split(',')[-1]
            img_bytes = base64.b64decode(b64_data)
            np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            return jsonify({'success': False, 'error': 'No valid image provided'}), 400

        if np_img is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400

        camera_id = request.form.get('camera_id', 'CAM-01')

        # Lookup camera location metadata for geospatial state prediction
        cam_rec = trajectory_tracker.session.query(Camera).filter(Camera.camera_id == camera_id).first()
        cam_context = f"{cam_rec.location_description}, {cam_rec.city}, {cam_rec.state}" if cam_rec else camera_id

        # Run 2-Stage ANPR Detection with Geospatial Camera Context
        results = anpr_engine.detect_and_recognize(np_img, camera_id=camera_id, camera_context=cam_context)
        annotated_img = draw_2stage_annotations(np_img, results)

        for res in results:
            x1, y1, x2, y2 = res['bbox']
            plate = res['plate_text']
            conf = res['confidence']

            crop_img = np_img[y1:y2, x1:x2]
            res['crop_b64'] = image_to_base64(crop_img)

            trajectory_tracker.add_detection_record(plate, camera_id, confidence=conf)
            res['alert'] = macro_analytics.check_blacklist_and_alerts(plate, camera_id)

        _, prep_dict = anpr_engine.preprocess_image(np_img)
        preprocessed_previews = {
            'clahe': image_to_base64(cv2.cvtColor(prep_dict.get('clahe', np_img), cv2.COLOR_GRAY2BGR)),
            'blackhat': image_to_base64(cv2.cvtColor(prep_dict.get('blackhat', np_img), cv2.COLOR_GRAY2BGR)),
            'thresh': image_to_base64(cv2.cvtColor(prep_dict.get('thresh', np_img), cv2.COLOR_GRAY2BGR))
        }

        return jsonify({
            'success': True,
            'annotated_image_b64': image_to_base64(annotated_img),
            'detections': results,
            'preprocessing_previews': preprocessed_previews
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/test_dataset', methods=['GET'])
def list_test_dataset():
    files = sorted(os.listdir(TEST_DATASET_DIR))
    images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return jsonify({'success': True, 'test_images': images})


@app.route('/api/test_dataset/run/<filename>', methods=['POST'])
def run_test_dataset_image(filename):
    try:
        filepath = os.path.join(TEST_DATASET_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Test image not found'}), 404

        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'success': False, 'error': 'Failed to read test image'}), 500

        # Determine Regional City based on test image index
        num_match = re.search(r'test(\d+)', filename, re.IGNORECASE)
        img_idx = int(num_match.group(1)) if num_match else 1
        
        if img_idx <= 7:
            target_city = "Mumbai"
        elif img_idx <= 14:
            target_city = "Pune"
        else:
            target_city = "Ahmedabad"

        req_json = request.get_json(force=True, silent=True) or {}
        
        # Fetch a representative camera node in target_city for this test image
        try:
            city_cams = trajectory_tracker.session.query(Camera).filter(Camera.city == target_city).all()
            if city_cams:
                cam_rec = city_cams[(img_idx * 3) % len(city_cams)]
                camera_id = cam_rec.camera_id
                cam_context = f"{cam_rec.location_description}, {cam_rec.city}, {cam_rec.state}"
            else:
                camera_id = req_json.get('camera_id', 'CAM-01')
                cam_rec = trajectory_tracker.session.query(Camera).filter(Camera.camera_id == camera_id).first()
                cam_context = f"{cam_rec.location_description}, {cam_rec.city}, {cam_rec.state}" if cam_rec else f"{target_city} Camera Grid"
        except Exception:
            trajectory_tracker.session.rollback()
            camera_id = req_json.get('camera_id', 'CAM-01')
            cam_context = f"{target_city} Camera Grid"

        results = anpr_engine.detect_and_recognize(filepath, camera_id=camera_id, camera_context=cam_context)
        annotated_img = draw_2stage_annotations(img, results)

        for res in results:
            x1, y1, x2, y2 = res['bbox']
            plate = res['plate_text']
            conf = res['confidence']

            crop_img = img[y1:y2, x1:x2]
            res['crop_b64'] = image_to_base64(crop_img)

            trajectory_tracker.add_detection_record(plate, camera_id, confidence=conf)
            res['alert'] = macro_analytics.check_blacklist_and_alerts(plate, camera_id)

        _, prep_dict = anpr_engine.preprocess_image(img)
        preprocessed_previews = {
            'clahe': image_to_base64(cv2.cvtColor(prep_dict.get('clahe', img), cv2.COLOR_GRAY2BGR)),
            'blackhat': image_to_base64(cv2.cvtColor(prep_dict.get('blackhat', img), cv2.COLOR_GRAY2BGR)),
            'thresh': image_to_base64(cv2.cvtColor(prep_dict.get('thresh', img), cv2.COLOR_GRAY2BGR))
        }

        return jsonify({
            'success': True,
            'filename': filename,
            'target_city': target_city,
            'camera_id': camera_id,
            'original_image_b64': image_to_base64(img),
            'annotated_image_b64': image_to_base64(annotated_img),
            'detections': results,
            'preprocessing_previews': preprocessed_previews
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/trajectory/search', methods=['GET'])
def get_trajectory():
    try:
        plate = request.args.get('plate', 'MH12AB1234')
        res = trajectory_tracker.reconstruct_trajectory(plate)
        return jsonify({'success': True, 'trajectory': res})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics/macro', methods=['GET'])
def get_macro_analytics():
    try:
        city = request.args.get('city', 'Mumbai')
        summary = macro_analytics.get_city_traffic_summary(city)
        return jsonify({'success': True, 'analytics': summary})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/alerts/blacklist', methods=['GET', 'POST'])
def manage_blacklist():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True, silent=True) or {}
            plate = data.get('plate', '')
            reason = data.get('reason', 'Security Alert')
            risk = data.get('risk', 'HIGH')
            if plate:
                rec = macro_analytics.add_to_blacklist(plate, reason, risk)
                return jsonify({'success': True, 'record': rec})
            return jsonify({'success': False, 'error': 'Plate is required'}), 400
        else:
            return jsonify({'success': True, 'blacklist': macro_analytics.get_all_blacklist()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reports/pdf', methods=['GET'])
def download_pdf_report():
    try:
        plate = request.args.get('plate', 'MH12AB1234')
        traj = trajectory_tracker.reconstruct_trajectory(plate)
        filename = f"trajectory_{plate}_{int(np.random.randint(1000, 9999))}.pdf"
        filepath = os.path.join(REPORTS_DIR, filename)
        generate_trajectory_pdf(traj, filepath)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("Launching 2-Stage Hierarchical ANPR Platform with OpenStreetMap Camera Sync on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
