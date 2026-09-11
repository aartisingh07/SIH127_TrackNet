import os
import glob
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

CLASS_NAMES = {0: "vehicle", 1: "license-plate"}

def yolo_to_bbox(x_center, y_center, width, height, img_w, img_h):
    """Converts normalized YOLO (x_center, y_center, width, height) to absolute (xmin, ymin, xmax, ymax)."""
    w_abs = width * img_w
    h_abs = height * img_h
    xmin = int(max(0, (x_center * img_w) - (w_abs / 2.0)))
    ymin = int(max(0, (y_center * img_h) - (h_abs / 2.0)))
    xmax = int(min(img_w, (x_center * img_w) + (w_abs / 2.0)))
    ymax = int(min(img_h, (y_center * img_h) + (h_abs / 2.0)))
    return xmin, ymin, xmax, ymax

def create_pascal_voc_xml(img_path, img_w, img_h, bboxes, output_xml_path):
    """Generates Pascal VOC XML format file for an image."""
    annotation = ET.Element("annotation")
    
    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(os.path.dirname(img_path))
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(img_path)
    
    path = ET.SubElement(annotation, "path")
    path.text = os.path.abspath(img_path)
    
    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(img_w)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(img_h)
    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = "3"
    
    for cls_id, xmin, ymin, xmax, ymax in bboxes:
        obj = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj, "name")
        name.text = CLASS_NAMES.get(cls_id, "license-plate")
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin_elem = ET.SubElement(bndbox, "xmin")
        xmin_elem.text = str(xmin)
        ymin_elem = ET.SubElement(bndbox, "ymin")
        ymin_elem.text = str(ymin)
        xmax_elem = ET.SubElement(bndbox, "xmax")
        xmax_elem.text = str(xmax)
        ymax_elem = ET.SubElement(bndbox, "ymax")
        ymax_elem.text = str(ymax)
        
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
    with open(output_xml_path, "w") as f:
        f.write(xml_str)

def convert_all_trained_images():
    dataset_dir = "dataset_yolo"
    img_dirs = [
        os.path.join(dataset_dir, "images", "train"),
        os.path.join(dataset_dir, "images", "val")
    ]
    
    xml_out_dir = os.path.join(dataset_dir, "annotations_xml")
    json_out_dir = os.path.join(dataset_dir, "annotations_json")
    os.makedirs(xml_out_dir, exist_ok=True)
    os.makedirs(json_out_dir, exist_ok=True)
    
    coco_dataset = {
        "info": {"description": "SIH127 ANPR Trained Dataset Annotations"},
        "categories": [{"id": 0, "name": "vehicle"}, {"id": 1, "name": "license-plate"}],
        "images": [],
        "annotations": []
    }
    
    ann_id = 1
    img_id = 1
    total_images = 0
    total_xml_created = 0
    total_json_created = 0
    
    for img_dir in img_dirs:
        if not os.path.exists(img_dir):
            continue
            
        subfolder = os.path.basename(img_dir)
        lbl_dir = os.path.join(os.path.dirname(img_dir), "..", "labels", subfolder)
        
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        img_files = []
        for ext in image_extensions:
            img_files.extend(glob.glob(os.path.join(img_dir, ext)))
            
        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(lbl_dir, base_name + ".txt")
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_h, img_w = img.shape[:2]
            total_images += 1
            
            bboxes = []
            if os.path.exists(txt_path):
                with open(txt_path, "r") as tf:
                    for line in tf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])
                            xmin, ymin, xmax, ymax = yolo_to_bbox(xc, yc, bw, bh, img_w, img_h)
                            bboxes.append((cls_id, xmin, ymin, xmax, ymax))
                            
            # 1. Create Pascal VOC XML
            xml_file_path = os.path.join(xml_out_dir, base_name + ".xml")
            create_pascal_voc_xml(img_path, img_w, img_h, bboxes, xml_file_path)
            total_xml_created += 1
            
            # 2. Create Individual JSON
            single_json_data = {
                "filename": os.path.basename(img_path),
                "width": img_w,
                "height": img_h,
                "objects": [
                    {
                        "label": CLASS_NAMES.get(cls_id, "license-plate"),
                        "bbox": [xmin, ymin, xmax, ymax]
                    }
                    for cls_id, xmin, ymin, xmax, ymax in bboxes
                ]
            }
            json_file_path = os.path.join(json_out_dir, base_name + ".json")
            with open(json_file_path, "w") as jf:
                json.dump(single_json_data, jf, indent=2)
            total_json_created += 1
            
            # 3. Add to combined COCO dataset
            coco_dataset["images"].append({
                "id": img_id,
                "file_name": os.path.basename(img_path),
                "width": img_w,
                "height": img_h
            })
            for cls_id, xmin, ymin, xmax, ymax in bboxes:
                coco_dataset["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0
                })
                ann_id += 1
            img_id += 1

    combined_json_path = os.path.join(dataset_dir, "dataset_coco_annotations.json")
    with open(combined_json_path, "w") as cjf:
        json.dump(coco_dataset, cjf, indent=2)

    print(f"\n[XML & JSON Generator Complete]")
    print(f"Total Processed Images    : {total_images}")
    print(f"Pascal VOC XML Files Saved : {total_xml_created} -> '{xml_out_dir}/'")
    print(f"Individual JSON Files Saved: {total_json_created} -> '{json_out_dir}/'")
    print(f"Combined COCO JSON Saved   : '{combined_json_path}'")

if __name__ == "__main__":
    convert_all_trained_images()
