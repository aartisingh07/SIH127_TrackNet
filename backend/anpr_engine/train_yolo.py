import os
import torch
from ultralytics import YOLO

def train_yolo_model(epochs=30, batch=16, imgsz=640):
    print("==================================================")
    print("     ANPR YOLO License Plate Detector Training    ")
    print("==================================================")
    
    models_dir = r"c:\SIH127-I\models"
    os.makedirs(models_dir, exist_ok=True)
    yaml_config = r"c:\SIH127-I\dataset_yolo\dataset.yaml"
    
    # Check PyTorch CUDA GPU availability
    if torch.cuda.is_available():
        device_arg = 0  # NVIDIA GPU index 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU DETECTED] Using NVIDIA CUDA GPU: {gpu_name}")
    else:
        device_arg = 'cpu'
        print("[INFO] CUDA GPU not detected by PyTorch. Training on CPU.")
        print("[INFO] To enable NVIDIA CUDA GPU training, install CUDA PyTorch via:")
        print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")

    print(f"Loading base YOLOv8 model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")
    
    print(f"Starting training for {epochs} epochs on device: '{device_arg}'...")
    results = model.train(
        data=yaml_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device_arg,
        project=r"c:\SIH127-I\models\yolo_runs",
        name="anpr_yolo",
        exist_ok=True,
        verbose=True
    )
    
    # Save best weights
    best_weights_path = os.path.join(r"c:\SIH127-I\models\yolo_runs", "anpr_yolo", "weights", "best.pt")
    target_weights_path = os.path.join(models_dir, "anpr_yolo_best.pt")
    
    if os.path.exists(best_weights_path):
        import shutil
        shutil.copy(best_weights_path, target_weights_path)
        print(f"[SUCCESS] Training completed! Model saved to: {target_weights_path}")
    else:
        model.save(target_weights_path)
        print(f"[SUCCESS] Model saved to: {target_weights_path}")

if __name__ == "__main__":
    train_yolo_model()
