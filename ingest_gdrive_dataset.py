import os
import cv2
import glob
import shutil
from ultralytics import YOLO

def process_and_ingest_gdrive_dataset():
    gdrive_dir = os.path.join("raw_dataset", "gdrive_dataset")
    dataset_yolo_dir = "dataset_yolo"
    
    img_train_dir = os.path.join(dataset_yolo_dir, "images", "train")
    lbl_train_dir = os.path.join(dataset_yolo_dir, "labels", "train")
    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)
    
    model_path = os.path.join("models", "anpr_yolo_best.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
        
    model = YOLO(model_path)
    
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(gdrive_dir, ext)))
        
    print(f"[GDrive Dataset Ingest] Found {len(image_files)} Google Drive images to auto-annotate and merge.")
    
    added_count = 0
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Run inference to detect license plates/vehicles
        results = model.predict(img, conf=0.15, verbose=False)
        yolo_labels = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                # Normalize bounding box for YOLO format (cls x_center y_center width height)
                xywh = box.xywhn[0].cpu().numpy()
                xc, yc, bw, bh = xywh
                yolo_labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                
        # Copy image to dataset_yolo
        target_img_path = os.path.join(img_train_dir, os.path.basename(img_path))
        shutil.copy2(img_path, target_img_path)
        
        # Write label txt file
        target_lbl_path = os.path.join(lbl_train_dir, base_name + ".txt")
        with open(target_lbl_path, "w") as f:
            f.write("\n".join(yolo_labels))
            
        added_count += 1
        
    print(f"[GDrive Dataset Ingest] Successfully ingested {added_count} new images into dataset_yolo!")

if __name__ == "__main__":
    process_and_ingest_gdrive_dataset()
