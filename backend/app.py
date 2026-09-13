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
import requests
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

@app.teardown_appcontext
def shutdown_session(exception=None):
    from database.db_engine import _session_factory
    if _session_factory:
        _session_factory.remove()

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


def safe_image_to_base64_bgr(arr, fallback_img):
    target = arr if (arr is not None and getattr(arr, 'size', 0) > 0) else fallback_img
    if target is None or getattr(target, 'size', 0) == 0:
        return ""
    if len(target.shape) == 2:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    return image_to_base64(target)


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
    """Runs 2-Stage Hierarchical ANPR (Vehicle Detection -> Plate Detection -> OCR) on File, URL, or Base64."""
    try:
        req_json = request.get_json(force=True, silent=True) or {}
        np_img = None
        filename = "External_Image.jpg"
        target_city = "Mumbai"

        if 'image' in request.files:
            file = request.files['image']
            filename = file.filename or "Uploaded_Image.jpg"
            np_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        elif 'image_url' in req_json or 'url' in req_json:
            url = req_json.get('image_url') or req_json.get('url')
            if not url or not str(url).startswith(('http://', 'https://')):
                return jsonify({'success': False, 'error': 'Invalid image URL provided. URL must start with http:// or https://'}), 400

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            resp = requests.get(url, timeout=12, headers=headers)
            if resp.status_code != 200:
                return jsonify({'success': False, 'error': f'Failed to fetch image from URL (HTTP Status {resp.status_code}). Please verify the link is publicly accessible.'}), 400

            np_img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            parsed_name = url.split('/')[-1].split('?')[0]
            filename = parsed_name if parsed_name and len(parsed_name) <= 30 else "URL_Image.jpg"
        elif 'image_base64' in req_json:
            b64_data = req_json['image_base64'].split(',')[-1]
            img_bytes = base64.b64decode(b64_data)
            np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        else:
            return jsonify({'success': False, 'error': 'No valid image file, URL, or image data provided'}), 400

        if np_img is None:
            return jsonify({'success': False, 'error': 'Failed to decode image data. Please ensure the link or file is a valid image (JPG, PNG, WEBP).'}), 400

        camera_id = request.form.get('camera_id') or req_json.get('camera_id') or 'CAM-01'
        target_city = request.form.get('target_city') or req_json.get('target_city') or 'Mumbai'

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

            # Stage 2 Super-Resolution Zoomed License Plate Crop (conditional: < 100px tall)
            if crop_img is not None and crop_img.size > 0:
                h_c, w_c = crop_img.shape[:2]
                if h_c < 100:
                    zoom_scale = min(3.5, 120.0 / float(max(1, h_c)))
                    zoomed_plate = cv2.resize(crop_img, (int(w_c * zoom_scale), int(h_c * zoom_scale)), interpolation=cv2.INTER_CUBIC)
                    z_gray = cv2.cvtColor(zoomed_plate, cv2.COLOR_BGR2GRAY)
                    z_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(z_gray)
                    z_clahe_bgr = cv2.cvtColor(z_clahe, cv2.COLOR_GRAY2BGR)
                    res['zoomed_plate_b64'] = image_to_base64(z_clahe_bgr)
                else:
                    res['zoomed_plate_b64'] = res['crop_b64']
            else:
                res['zoomed_plate_b64'] = res['crop_b64']

            # Stage 1 Zoomed Vehicle Crop
            if 'vehicle_bbox' in res:
                vx1, vy1, vx2, vy2 = res['vehicle_bbox']
                v_crop = np_img[vy1:vy2, vx1:vx2]
                if v_crop is not None and v_crop.size > 0:
                    vh_c, vw_c = v_crop.shape[:2]
                    v_scale = max(1.5, 300.0 / float(max(1, vh_c)))
                    zoomed_vehicle = cv2.resize(v_crop, (int(vw_c * v_scale), int(vh_c * v_scale)), interpolation=cv2.INTER_CUBIC)
                    res['zoomed_vehicle_b64'] = image_to_base64(zoomed_vehicle)
                else:
                    res['zoomed_vehicle_b64'] = ""
            else:
                res['zoomed_vehicle_b64'] = ""

            trajectory_tracker.add_detection_record(plate, camera_id, confidence=conf, confidence_flag=res.get('confidence_flag', True))
            res['alert'] = macro_analytics.check_blacklist_and_alerts(plate, camera_id)

        _, prep_dict = anpr_engine.preprocess_image(np_img)
        preprocessed_previews = {
            'clahe': safe_image_to_base64_bgr(prep_dict.get('clahe'), np_img),
            'blackhat': safe_image_to_base64_bgr(prep_dict.get('blackhat'), np_img),
            'thresh': safe_image_to_base64_bgr(prep_dict.get('thresh'), np_img)
        }

        return jsonify({
            'success': True,
            'annotated_image_b64': image_to_base64(annotated_img),
            'detections': results,
            'preprocessing_previews': preprocessed_previews,
            'filename': filename,
            'target_city': target_city
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

            # Stage 2 Super-Resolution Zoomed License Plate Crop (conditional: < 100px tall)
            if crop_img is not None and crop_img.size > 0:
                h_c, w_c = crop_img.shape[:2]
                if h_c < 100:
                    zoom_scale = min(3.5, 120.0 / float(max(1, h_c)))
                    zoomed_plate = cv2.resize(crop_img, (int(w_c * zoom_scale), int(h_c * zoom_scale)), interpolation=cv2.INTER_CUBIC)
                    z_gray = cv2.cvtColor(zoomed_plate, cv2.COLOR_BGR2GRAY)
                    z_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(z_gray)
                    z_clahe_bgr = cv2.cvtColor(z_clahe, cv2.COLOR_GRAY2BGR)
                    res['zoomed_plate_b64'] = image_to_base64(z_clahe_bgr)
                else:
                    res['zoomed_plate_b64'] = res['crop_b64']
            else:
                res['zoomed_plate_b64'] = res['crop_b64']

            # Stage 1 Zoomed Vehicle Crop
            if 'vehicle_bbox' in res:
                vx1, vy1, vx2, vy2 = res['vehicle_bbox']
                v_crop = img[vy1:vy2, vx1:vx2]
                if v_crop is not None and v_crop.size > 0:
                    vh_c, vw_c = v_crop.shape[:2]
                    v_scale = max(1.5, 300.0 / float(max(1, vh_c)))
                    zoomed_vehicle = cv2.resize(v_crop, (int(vw_c * v_scale), int(vh_c * v_scale)), interpolation=cv2.INTER_CUBIC)
                    res['zoomed_vehicle_b64'] = image_to_base64(zoomed_vehicle)
                else:
                    res['zoomed_vehicle_b64'] = ""
            else:
                res['zoomed_vehicle_b64'] = ""

            trajectory_tracker.add_detection_record(plate, camera_id, confidence=conf, confidence_flag=res.get('confidence_flag', True))
            res['alert'] = macro_analytics.check_blacklist_and_alerts(plate, camera_id)

        _, prep_dict = anpr_engine.preprocess_image(img)
        preprocessed_previews = {
            'clahe': safe_image_to_base64_bgr(prep_dict.get('clahe'), img),
            'blackhat': safe_image_to_base64_bgr(prep_dict.get('blackhat'), img),
            'thresh': safe_image_to_base64_bgr(prep_dict.get('thresh'), img)
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
