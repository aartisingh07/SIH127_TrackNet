"""
ANPR YOLO Model Fine-Tuning Module
Fine-tunes pre-trained YOLO object detector on Indian License Plate Datasets 
(Kaggle Indian Vehicle Dataset & DataCluster Indian License Plate Dataset).
"""

import os
import sys
import shutil
from typing import Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from dataset_loader import IndianPlateDatasetLoader


def train_indian_plate_model(
    data_yaml_path: str = "dataset/data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    output_weights_dir: str = "weights"
):
    """
    Train custom YOLO license plate detector model on Indian dataset annotations.
    """
    if not YOLO_AVAILABLE:
        print("[ANPR Trainer Error] Ultralytics package is not installed. Please run: pip install ultralytics")
        return

    # Check if dataset yaml exists; if not, generate structure
    if not os.path.exists(data_yaml_path):
        print(f"[ANPR Trainer] Dataset config '{data_yaml_path}' not found. Preparing dataset structure...")
        loader = IndianPlateDatasetLoader()
        loader.prepare_dataset()

    print("\n" + "=" * 70)
    print("      FINE-TUNING ANPR MODEL ON INDIAN LICENSE PLATE DATASET     ")
    print("=" * 70)
    print(f" Base Model     : {base_model}")
    print(f" Dataset Config : {data_yaml_path}")
    print(f" Epochs         : {epochs}")
    print(f" Batch Size     : {batch_size}")
    print(f" Image Size     : {imgsz}x{imgsz}")
    print("=" * 70 + "\n")

    # Load pre-trained model
    model = YOLO(base_model)

    # Train model with license-plate specific data augmentations
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        degrees=10.0,      # Handle tilted plates up to 10 degrees
        scale=0.5,         # Scale variation
        hsv_h=0.015,       # Hue variation
        hsv_s=0.7,         # Saturation variation
        hsv_v=0.4,         # Brightness variation for night/sunlight conditions
        mosaic=1.0,        # Mosaic augmentation for small plate objects
        name="indian_plate_detector",
        exist_ok=True
    )

    # Export & save best weights
    os.makedirs(output_weights_dir, exist_ok=True)
    best_weights_src = os.path.join("runs", "detect", "indian_plate_detector", "weights", "best.pt")
    target_weights_path = os.path.join(output_weights_dir, "best_indian_plate.pt")

    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, target_weights_path)
        print(f"\n[ANPR Trainer Success] Custom Indian Plate weights saved to: '{target_weights_path}'")
    else:
        print(f"\n[ANPR Trainer] Training finished. Check runs directory for exported model.")

    return target_weights_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ANPR License Plate Detector Model on Indian Dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base pre-trained YOLO model")

    args = parser.parse_args()

    train_indian_plate_model(
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz
    )
