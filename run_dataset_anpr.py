"""
Unified ANPR Pipeline Interface for Indian License Plate Datasets.

Usage Examples:
1. Prepare dataset from raw images & annotations:
   python run_dataset_anpr.py --prepare-data --raw-dir path/to/raw_dataset

2. Fine-tune YOLO model on dataset:
   python run_dataset_anpr.py --train --epochs 30 --batch 16

3. Run ANPR inference on an image or video:
   python run_dataset_anpr.py --infer --input path/to/car.jpg
"""

import argparse
import os
import cv2
import json
from dataset_loader import IndianPlateDatasetLoader
from train_anpr import train_indian_plate_model
from anpr_model import ANPRModel

def main():
    parser = argparse.ArgumentParser(description="ANPR Dataset Pipeline: Prepare, Train, Infer")
    parser.add_argument("--prepare-data", action="store_true", help="Convert raw dataset to YOLO format")
    parser.add_argument("--raw-dir", type=str, default="raw_dataset", help="Path to raw dataset folder containing images and annotations")
    parser.add_argument("--train", action="store_true", help="Fine-tune YOLO model on dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image resolution")
    parser.add_argument("--infer", action="store_true", help="Run ANPR inference on image/video")
    parser.add_argument("--input", type=str, help="Path to image or video file for inference")

    args = parser.parse_args()

    # 1. Prepare Data
    if args.prepare_data:
        print(f"[ANPR Pipeline] Converting dataset annotations from '{args.raw_dir}'...")
        loader = IndianPlateDatasetLoader(raw_dataset_dir=args.raw_dir)
        loader.prepare_dataset()

    # 2. Train Model
    elif args.train:
        print("[ANPR Pipeline] Starting custom YOLO model training on Indian License Plate dataset...")
        train_indian_plate_model(epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz)

    # 3. Run Inference
    elif args.infer:
        if not args.input or not os.path.exists(args.input):
            print("[ANPR Pipeline Error] Please specify a valid input file with '--input path/to/image.jpg'")
            return

        model = ANPRModel()
        ext = os.path.splitext(args.input)[1].lower()

        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            print(f"[ANPR Pipeline] Running video ANPR on '{args.input}'...")
            cap = cv2.VideoCapture(args.input)
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                res = model.process(frame)
                if frame_idx % 10 == 0 or frame_idx == 1:
                    print(f"  Frame {frame_idx:03d} -> Plate: {res['plate_number']} | Accuracy: {res['accuracy_percentage']}% | Latency: {res['latency_ms']}ms")
            cap.release()
        else:
            print(f"[ANPR Pipeline] Running image ANPR on '{args.input}'...")
            res = model.process(args.input)
            print("\n" + "=" * 60)
            print("                 ANPR DETECTION RESULT                    ")
            print("=" * 60)
            print(f" Extracted Plate  : {res['plate_number']}")
            print(f" Raw OCR Output   : {res['raw_ocr_text']}")
            print(f" Overall Accuracy : {res['accuracy_percentage']}%")
            print(f" Bounding Box     : {res['bounding_box']}")
            print(f" Detection Conf   : {res['confidence_scores']['detection_confidence'] * 100:.2f}%")
            print(f" OCR Confidence   : {res['confidence_scores']['ocr_confidence'] * 100:.2f}%")
            print(f" Syntax Match     : {res['confidence_scores']['syntax_confidence'] * 100:.2f}%")
            print(f" Latency          : {res['latency_ms']} ms")
            print("=" * 60)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
