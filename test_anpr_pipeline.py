"""
ANPR Model Test & Evaluation Harness
Demonstrates:
1. Generating test sample images (or reading input traffic video/image)
2. License Plate + Vehicle Detection
3. Crop & Preprocessing
4. OCR Extraction & Accuracy / Confidence Assessment
5. Output reporting
"""

import os
import sys
import cv2
import json
import numpy as np
from anpr_model import ANPRModel

def create_sample_vehicle_image(text: str, filename: str, output_dir: str = "samples") -> str:
    img = np.full((300, 600, 3), (200, 200, 200), dtype=np.uint8)
    cv2.rectangle(img, (150, 180), (450, 260), (255, 255, 255), -1)
    cv2.rectangle(img, (150, 180), (450, 260), (0, 0, 0), 3)
    cv2.putText(img, text, (170, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, img)
    return out_path

def run_evaluation():
    print("=" * 70)
    print("        AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) MODEL DEMO      ")
    print("=" * 70)
    
    # 1. Generate test samples
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    test_cases = [
        ("MH12AB1234", "car_pune.jpg"),
        ("KA05NB9876", "car_bangalore.jpg"),
        ("DL01CA1001", "car_delhi.jpg"),
    ]
    
    sample_paths = []
    for expected_plate, filename in test_cases:
        filepath = create_sample_vehicle_image(expected_plate, filename, output_dir=samples_dir)
        sample_paths.append((expected_plate, filepath))

    # 2. Instantiate ANPR Model
    model = ANPRModel()

    print("\n" + "-" * 70)
    print(" RUNNING DETECTION & RECOGNITION ON SAMPLE IMAGES")
    print("-" * 70 + "\n")

    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)

    total_accuracy = 0.0
    total_latency = 0.0

    for i, (expected_plate, img_path) in enumerate(sample_paths, 1):
        print(f"[{i}/{len(sample_paths)}] Processing: {os.path.basename(img_path)}")
        
        # Process image through ANPR pipeline
        result = model.process(img_path)

        extracted_plate = result["plate_number"]
        accuracy = result["accuracy_percentage"]
        scores = result["confidence_scores"]
        latency = result["latency_ms"]
        bbox = result["bounding_box"]

        # Check match with expected plate
        exact_match = (extracted_plate == expected_plate)
        match_status = "EXACT MATCH" if exact_match else "PARTIAL/CLOSE"

        print(f"  ├── Expected Plate       : {expected_plate}")
        print(f"  ├── Extracted Plate      : {extracted_plate}  [{match_status}]")
        print(f"  ├── Overall Accuracy     : {accuracy}%")
        print(f"  ├── Confidence Metrics  :")
        print(f"  │     ├── YOLO Detection : {scores['detection_confidence'] * 100:.2f}%")
        print(f"  │     ├── PaddleOCR Text : {scores['ocr_confidence'] * 100:.2f}%")
        print(f"  │     └── Indian Syntax  : {scores['syntax_confidence'] * 100:.2f}%")
        print(f"  ├── Bounding Box [x1,y1,x2,y2]: {bbox}")
        print(f"  └── Processing Latency   : {latency} ms\n")

        total_accuracy += accuracy
        total_latency += latency

        # Save visual artifacts
        annotated_file = os.path.join(output_dir, f"annotated_{os.path.basename(img_path)}")
        cv2.imwrite(annotated_file, result["annotated_image"])

        crop_file = os.path.join(output_dir, f"crop_{os.path.basename(img_path)}")
        cv2.imwrite(crop_file, result["cropped_plate"])

    avg_accuracy = round(total_accuracy / len(sample_paths), 2)
    avg_latency = round(total_latency / len(sample_paths), 2)

    print("=" * 70)
    print("                    FINAL BENCHMARK SUMMARY                       ")
    print("=" * 70)
    print(f"Total Test Images Processed : {len(sample_paths)}")
    print(f"Average ANPR Model Accuracy : {avg_accuracy}%")
    print(f"Average Inference Latency   : {avg_latency} ms per image")
    print(f"Annotated Visuals Saved To  : ./{output_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()
