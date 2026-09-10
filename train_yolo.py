import os
import sys
from ultralytics import YOLO

def train_anpr_yolo(epochs=30, batch_size=16, imgsz=640):
    """
    Fine-tunes YOLOv8 License Plate Detection model on dataset_yolo.
    Saves trained weights to models/anpr_yolo_best.pt.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(project_root, "dataset_yolo", "data.yaml")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=========================================================")
    print("TrackNet ANPR - Fine-tuning YOLOv8 License Plate Detector")
    print(f"Dataset config: {yaml_path}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Image size: {imgsz}")
    print("=========================================================")

    # Initialize base YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Start Training
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        name="anpr_plate_train",
        project="runs/detect",
        exist_ok=True
    )

    # Copy best weights to models/anpr_yolo_best.pt
    best_weights = os.path.join("runs", "detect", "anpr_plate_train", "weights", "best.pt")
    target_weights = os.path.join(models_dir, "anpr_yolo_best.pt")

    if os.path.exists(best_weights):
        import shutil
        shutil.copy2(best_weights, target_weights)
        print(f"\nTraining Complete! Fine-tuned weights saved to: {target_weights}")
    else:
        print("\nTraining complete. Check runs/detect/ for output weights.")

if __name__ == "__main__":
    train_anpr_yolo()
