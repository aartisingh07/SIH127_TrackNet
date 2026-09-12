import os
import glob
import shutil
import random
import xml.etree.ElementTree as ET
import cv2

RAW_DATASET_DIR = r"c:\SIH127-I\raw_dataset"
OUTPUT_YOLO_DIR = r"c:\SIH127-I\dataset_yolo"

def convert_voc_to_yolo():
    print(f"Starting Pascal VOC to YOLO conversion from: {RAW_DATASET_DIR}")
    
    # Clean previous dataset folder to ensure fresh rebuild
    if os.path.exists(OUTPUT_YOLO_DIR):
        try:
            shutil.rmtree(OUTPUT_YOLO_DIR)
        except Exception as e:
            print(f"Warning cleaning output directory: {e}")
            
    train_img_dir = os.path.join(OUTPUT_YOLO_DIR, "images", "train")
    val_img_dir = os.path.join(OUTPUT_YOLO_DIR, "images", "val")
    train_lbl_dir = os.path.join(OUTPUT_YOLO_DIR, "labels", "train")
    val_lbl_dir = os.path.join(OUTPUT_YOLO_DIR, "labels", "val")
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    
    # Search all XML files recursively inside raw_dataset
    xml_files = glob.glob(os.path.join(RAW_DATASET_DIR, "**", "*.xml"), recursive=True)
    
    samples = []
    for xf in xml_files:
        dir_name = os.path.basename(os.path.dirname(xf))
        base_name = os.path.splitext(os.path.basename(xf))[0]
        parent_dir = os.path.dirname(xf)
        
        img_p = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            cand = os.path.join(parent_dir, base_name + ext)
            if os.path.exists(cand):
                img_p = cand
                break
        if img_p:
            samples.append((img_p, xf, dir_name, base_name))

    print(f"Found {len(samples)} valid image-annotation pairs across all subfolders.")
    
    random.seed(42)
    random.shuffle(samples)
    
    split_idx = int(0.85 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    def process_set(dataset_items, img_target, lbl_target):
        count = 0
        for idx, (img_p, xf, dir_name, base_name) in enumerate(dataset_items):
            img = cv2.imread(img_p)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            try:
                tree = ET.parse(xf)
                root = tree.getroot()
            except Exception as e:
                continue
                
            yolo_lines = []
            for obj in root.findall("object"):
                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                xmin = float(bnd.find("xmin").text)
                ymin = float(bnd.find("ymin").text)
                xmax = float(bnd.find("xmax").text)
                ymax = float(bnd.find("ymax").text)
                
                # Clip coordinates to image boundaries
                xmin = max(0.0, min(float(w), xmin))
                ymin = max(0.0, min(float(h), ymin))
                xmax = max(0.0, min(float(w), xmax))
                ymax = max(0.0, min(float(h), ymax))
                
                box_w = xmax - xmin
                box_h = ymax - ymin
                if box_w <= 0 or box_h <= 0:
                    continue
                    
                x_center = (xmin + xmax) / 2.0 / w
                y_center = (ymin + ymax) / 2.0 / h
                norm_w = box_w / w
                norm_h = box_h / h
                
                # Class 0: license_plate
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            
            if not yolo_lines:
                continue
                
            out_name = f"{dir_name}_{base_name}_{idx}"
            dst_img_path = os.path.join(img_target, f"{out_name}.jpg")
            dst_lbl_path = os.path.join(lbl_target, f"{out_name}.txt")
            
            cv2.imwrite(dst_img_path, img)
            with open(dst_lbl_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")
            count += 1
        return count

    n_train = process_set(train_samples, train_img_dir, train_lbl_dir)
    n_val = process_set(val_samples, val_img_dir, val_lbl_dir)
    
    clean_output_dir = OUTPUT_YOLO_DIR.replace('\\', '/')
    yaml_content = f"""path: {clean_output_dir}
train: images/train
val: images/val
names:
  0: license_plate
"""
    yaml_path = os.path.join(OUTPUT_YOLO_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"Conversion Complete! Processed {n_train} Train images & {n_val} Validation images.")
    print(f"Config generated at: {yaml_path}")

if __name__ == "__main__":
    convert_voc_to_yolo()
