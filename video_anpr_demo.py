"""
ANPR Video Processing Module
Task: Process MP4 video file frame-by-frame, detect license plates, extract plate text, and render annotated video.
"""

import cv2
import os
import time
import numpy as np
from anpr_model import ANPRModel

def generate_sample_video(output_video_path: str = "samples/traffic_sample.mp4", num_frames: int = 60):
    """
    Generate a 60-frame simulated traffic MP4 video clip for demonstration.
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fps = 20
    w, h = 800, 500
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    plate_texts = ["MH12AB1234", "KA05NB9876", "DL01CA1001"]

    for f in range(num_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:300] = [80, 75, 70]
        img[300:] = [40, 40, 40]

        # Vehicle moving across frame
        shift_x = int((f / num_frames) * 200)
        car_x1 = 150 + shift_x
        car_x2 = 650 + shift_x

        cv2.fillPoly(img, [np.array([[car_x1 + 50, 300], [car_x1 + 100, 180], [car_x2 - 100, 180], [car_x2 - 50, 300]])], (120, 40, 30))
        cv2.rectangle(img, (car_x1, 300), (car_x2, 420), (140, 50, 40), -1)
        cv2.rectangle(img, (car_x1 - 10, 400), (car_x2 + 10, 435), (30, 30, 30), -1)

        # License Plate
        px1, py1, px2, py2 = car_x1 + 150, 375, car_x1 + 350, 430
        cv2.rectangle(img, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), (0, 0, 0), -1)
        cv2.rectangle(img, (px1, py1), (px2, py2), (255, 255, 255), -1)
        
        cv2.rectangle(img, (px1, py1), (px1 + 25, py2), (180, 80, 20), -1)
        cv2.putText(img, "IND", (px1 + 2, py1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(
            img, "MH12AB1234", (px1 + 32, py1 + 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA
        )

        out.write(img)

    out.release()
    print(f"[Video Generator] Sample traffic video created at: {output_video_path}")
    return output_video_path


def process_video(video_path: str, output_video_path: str = "output_results/annotated_traffic.mp4"):
    """
    Process input MP4 video through ANPR Model frame-by-frame.
    """
    if not os.path.exists(video_path):
        print(f"Generating sample video first...")
        video_path = generate_sample_video(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video file {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    model = ANPRModel()
    frame_idx = 0
    detections_summary = []

    print(f"\n[ANPR Video Processing] Reading '{video_path}'...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        res = model.process(frame)

        annotated_frame = res["annotated_image"]
        out.write(annotated_frame)

        # Log frame detection
        if frame_idx % 10 == 0 or frame_idx == 1:
            print(f"  Frame {frame_idx:03d} -> Detected Plate: {res['plate_number']} | Accuracy: {res['accuracy_percentage']}% | Latency: {res['latency_ms']}ms")
            detections_summary.append(res)

    cap.release()
    out.release()

    print(f"\n[ANPR Video Processing] Done. Processed {frame_idx} frames.")
    print(f"Annotated Output Video Saved To: {output_video_path}")


if __name__ == "__main__":
    process_video("samples/traffic_sample.mp4")
