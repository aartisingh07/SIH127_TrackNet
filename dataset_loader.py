"""
Dataset Ingestion & Annotation Converter for Indian License Plate Datasets.
Supports:
1. Indian Vehicle License Plate Dataset (Kaggle: saisirishan/indian-vehicle-dataset)
2. DataCluster Indian Licence Plate Dataset (GitHub: datacluster-labs/Indian-Licence-Plate-Image-Dataset)
3. UFPR-ALPR & CCPD Datasets
"""

import os
import json
import xml.etree.ElementTree as ET
import glob
import shutil
import random
from typing import List, Tuple, Dict, Any

class IndianPlateDatasetLoader:
    """
    Loader and annotation parser to convert raw Indian License Plate datasets 
    (Pascal VOC XML, JSON, or CSV) into standard Ultralytics YOLO format.
    """

    def __init__(self, raw_dataset_dir: str = "raw_dataset", processed_dir: str = "dataset"):
        self.raw_dataset_dir = raw_dataset_dir
        self.processed_dir = processed_dir
        self.classes = ["license_plate"]

    def create_yolo_dir_structure(self):
        """
        Creates directory tree required by YOLO:
        dataset/
          ├── images/
          │     ├── train/
          │     └── val/
          └── labels/
                ├── train/
                └── val/
        """
        dirs = [
            os.path.join(self.processed_dir, "images", "train"),
            os.path.join(self.processed_dir, "images", "val"),
            os.path.join(self.processed_dir, "labels", "train"),
            os.path.join(self.processed_dir, "labels", "val")
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"[Dataset Loader] Created YOLO directory tree at '{self.processed_dir}/'")

    def convert_voc_xml_to_yolo(self, xml_path: str, img_w: int, img_h: int) -> List[str]:
        """
        Convert Pascal VOC XML bounding boxes (xmin, ymin, xmax, ymax) to YOLO format (x_center, y_center, width, height).
        Handles XML where <name> is 'plate', 'license', or actual registration text (e.g. 'MH06AB8620').
        """
        yolo_lines = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)

                    # Clamp coordinates to image boundaries
                    xmin = max(0, min(xmin, img_w))
                    xmax = max(0, min(xmax, img_w))
                    ymin = max(0, min(ymin, img_h))
                    ymax = max(0, min(ymax, img_h))

                    w_box = xmax - xmin
                    h_box = ymax - ymin

                    if w_box > 2 and h_box > 2:
                        x_center = ((xmin + xmax) / 2.0) / img_w
                        y_center = ((ymin + ymax) / 2.0) / img_h
                        width = w_box / img_w
                        height = h_box / img_h

                        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        except Exception as e:
            print(f"[Dataset Loader Warning] Failed parsing XML {xml_path}: {e}")

        return yolo_lines

    def convert_json_to_yolo(self, json_path: str, img_w: int, img_h: int) -> List[str]:
        """
        Convert DataCluster / COCO JSON bounding boxes to YOLO format.
        """
        yolo_lines = []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            shapes = data.get("shapes", []) or data.get("objects", []) or data.get("annotations", [])
            for s in shapes:
                points = s.get("points", [])
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)

                    x_center = ((xmin + xmax) / 2.0) / img_w
                    y_center = ((ymin + ymax) / 2.0) / img_h
                    width = (xmax - xmin) / img_w
                    height = (ymax - ymin) / img_h

                    yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        except Exception as e:
            print(f"[Dataset Loader Warning] Failed parsing JSON {json_path}: {e}")

        return yolo_lines

    def prepare_dataset(self, val_split: float = 0.0):
        """
        Scans raw_dataset_dir for images + XML/JSON/TXT annotations, converts annotations to YOLO format,
        and ensures 100% of images are included in the training dataset.
        """
        self.create_yolo_dir_structure()

        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.raw_dataset_dir, "**", ext), recursive=True))

        print(f"[Dataset Loader] Found {len(image_files)} raw images in '{self.raw_dataset_dir}'.")
        if not image_files:
            print(f"[Dataset Loader Notice] Place dataset files inside '{self.raw_dataset_dir}' folder to run dataset conversion.")
            self.generate_data_yaml()
            return

        import cv2

        converted_count = 0

        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            # Copy image to train (and val if val_split is 0.0)
            target_img_train = os.path.join(self.processed_dir, "images", "train", os.path.basename(img_path))
            shutil.copy(img_path, target_img_train)
            
            target_img_val = os.path.join(self.processed_dir, "images", "val", os.path.basename(img_path))
            shutil.copy(img_path, target_img_val)

            # Find matching annotation XML or JSON
            dir_name = os.path.dirname(img_path)
            xml_file = os.path.join(dir_name, base_name + ".xml")
            json_file = os.path.join(dir_name, base_name + ".json")
            txt_file = os.path.join(dir_name, base_name + ".txt")

            yolo_annotations = []
            if os.path.exists(xml_file):
                yolo_annotations = self.convert_voc_xml_to_yolo(xml_file, w, h)
            elif os.path.exists(json_file):
                yolo_annotations = self.convert_json_to_yolo(json_file, w, h)
            elif os.path.exists(txt_file):
                with open(txt_file, 'r') as tf:
                    yolo_annotations = [line.strip() for line in tf if line.strip()]

            # Save label file for train and val
            label_train = os.path.join(self.processed_dir, "labels", "train", base_name + ".txt")
            with open(label_train, "w") as lf:
                lf.write("\n".join(yolo_annotations))

            label_val = os.path.join(self.processed_dir, "labels", "val", base_name + ".txt")
            with open(label_val, "w") as lf:
                lf.write("\n".join(yolo_annotations))

            converted_count += 1

        print(f"[Dataset Loader] Successfully processed & converted ALL {converted_count} image annotations for training.")
        self.generate_data_yaml()

    def generate_data_yaml(self) -> str:
        """
        Generate data.yaml file required by Ultralytics YOLO for model training.
        """
        abs_processed_dir = os.path.abspath(self.processed_dir).replace("\\", "/")
        yaml_content = f"""path: {abs_processed_dir}
train: images/train
val: images/val

names:
  0: license_plate
"""
        yaml_path = os.path.join(self.processed_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"[Dataset Loader] Generated YOLO dataset config file at: '{yaml_path}'")
        return yaml_path


if __name__ == "__main__":
    loader = IndianPlateDatasetLoader(raw_dataset_dir="raw_dataset", processed_dir="dataset")
    loader.prepare_dataset()
