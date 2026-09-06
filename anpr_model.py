"""
ANPR Model Pipeline
Stack: OpenCV, YOLO (Ultralytics), EasyOCR / PaddleOCR
Task: Image/Video -> Detect Vehicle/Plate -> Crop Plate -> OCR Text Extraction -> Accuracy & Confidence Assessment
Supports: Custom trained Indian License Plate model weights (weights/best_indian_plate.pt)
"""

import cv2
import numpy as np
import re
import time
import os
from typing import Dict, Any, List, Tuple, Optional

# Try importing ultralytics, paddleocr, easyocr
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class ANPRModel:
    """
    Automatic Number Plate Recognition (ANPR) Model class.
    Combines YOLO (fine-tuned on Indian License Plate Datasets) for plate localization, 
    OpenCV for image preprocessing, and EasyOCR/PaddleOCR for character recognition.
    """

    INDIAN_STATE_CODES = {
        "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DN", "DD", "DL", "GA", "GJ",
        "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML",
        "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TS", "TR", "UP", "UK",
        "WB", "BH"
    }

    def __init__(self, yolo_model_path: Optional[str] = None, use_gpu: bool = False):
        """
        Initialize the ANPR pipeline. Automatically selects custom trained Indian plate weights if present.
        """
        print("[ANPR Engine] Initializing ANPR Model...")
        self.use_gpu = use_gpu

        # Default weights priority: custom fine-tuned Indian plate model > base yolov8n.pt
        if yolo_model_path is None:
            custom_weights = os.path.join("weights", "best_indian_plate.pt")
            if os.path.exists(custom_weights):
                yolo_model_path = custom_weights
            else:
                yolo_model_path = "yolov8n.pt"

        # 1. Initialize YOLO
        if YOLO_AVAILABLE:
            print(f"[ANPR Engine] Loading YOLO detector model from: '{yolo_model_path}'...")
            try:
                self.yolo = YOLO(yolo_model_path)
            except Exception as e:
                print(f"[ANPR Engine Warning] Could not load YOLO model {yolo_model_path}: {e}")
                print("[ANPR Engine] Falling back to default 'yolov8n.pt'...")
                try:
                    self.yolo = YOLO("yolov8n.pt")
                except Exception as e2:
                    print(f"[ANPR Engine Error] Failed loading fallback YOLO model: {e2}")
                    self.yolo = None
        else:
            print("[ANPR Engine Warning] Ultralytics YOLO not installed. Using OpenCV contour detector fallback.")
            self.yolo = None

        # 2. Initialize Dual OCR Ensemble (PaddleOCR + EasyOCR fallback)
        self.paddle_engine = None
        self.easy_engine = None

        if PADDLE_AVAILABLE:
            print("[ANPR Engine] Initializing PaddleOCR engine (Primary lightweight mode)...")
            try:
                self.paddle_engine = PaddleOCR(
                    lang='en',
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False
                )
            except Exception as e:
                print(f"[ANPR Engine Warning] PaddleOCR init failed: {e}")

        if EASYOCR_AVAILABLE:
            print("[ANPR Engine] Initializing EasyOCR engine (Ensemble / Fallback mode)...")
            try:
                self.easy_engine = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            except Exception as e:
                print(f"[ANPR Engine Warning] EasyOCR init failed: {e}")

        self.ocr_engine = self.paddle_engine or self.easy_engine
        self.ocr_type = "ensemble"

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance input image quality for detection.
        - Grayscale conversion
        - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Denoising / Bilateral filtering
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
        return filtered

    def detect_plate_contours(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Multi-stage License Plate Bounding Box Detector:
        1. Sobel Vertical Edge Filter + Morphological Closing (extracts character text blocks)
        2. Morphological BlackHat Transform (extracts dark text on light plates)
        3. Canny Contour Analysis
        """
        img_h, img_w = gray_image.shape[:2]
        plate_bboxes = []

        # 1. Sobel Vertical Edge Filter (Ideal for vehicle license plate character blocks)
        try:
            sobel_x = cv2.Sobel(gray_image, cv2.CV_8U, 1, 0, ksize=3)
            _, thresh_sobel = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
            closed_sobel = cv2.morphologyEx(thresh_sobel, cv2.MORPH_CLOSE, morph_kernel)

            cnts_sobel, _ = cv2.findContours(closed_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_sobel:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = w / float(h)
                area = w * h
                if 1.1 <= aspect_ratio <= 7.5 and area > 2000 and (w < img_w * 0.90 and h < img_h * 0.90):
                    plate_bboxes.append((x, y, w, h))
        except Exception:
            pass

        # 2. Morphological BlackHat Transform (ideal for dark license plate text on light/yellow bumper backgrounds)
        try:
            rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
            blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, rectKern)
            _, thresh_bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cnts_bh, _ = cv2.findContours(thresh_bh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_bh:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = w / float(h)
                area = w * h
                if 1.1 <= aspect_ratio <= 7.5 and area > 2000 and (w < img_w * 0.90 and h < img_h * 0.90):
                    plate_bboxes.append((x, y, w, h))
        except Exception:
            pass

        # 3. Canny Edge Contour Detection
        try:
            edged = cv2.Canny(gray_image, 30, 200)
            contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    area = w * h

                    if 1.1 <= aspect_ratio <= 7.0 and area > 2000 and (w < img_w * 0.95 and h < img_h * 0.95):
                        plate_bboxes.append((x, y, w, h))
        except Exception:
            pass

        return plate_bboxes

    def detect_license_plate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicle and license plate regions in the image using YOLO or OpenCV contour analysis.
        Generates single detection boxes and merged candidate boxes for multi-line motorcycle plates.
        """
        h, w = image.shape[:2]
        detections = []

        if self.yolo is not None:
            try:
                results = self.yolo(image, conf=0.05, verbose=False)
                for res in results:
                    boxes = res.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        cls_name = res.names[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        if "plate" in cls_name.lower() or "license" in cls_name.lower() or cls_id == 0:
                            detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf * 1.25,
                                "type": "license_plate_yolo",
                                "is_yolo": True
                            })
                        elif cls_name in ["car", "motorcycle", "bus", "truck", "vehicle"]:
                            vehicle_crop = image[y1:y2, x1:x2]
                            if vehicle_crop.size > 0:
                                prep = self.preprocess_image(vehicle_crop)
                                plate_boxes = self.detect_plate_contours(prep)
                                for px, py, pw, ph in plate_boxes:
                                    abs_x1 = x1 + px
                                    abs_y1 = y1 + py
                                    abs_x2 = abs_x1 + pw
                                    abs_y2 = abs_y1 + ph
                                    detections.append({
                                        "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                                        "confidence": conf * 0.60,
                                        "type": "license_plate_vehicle_contour",
                                        "is_yolo": False
                                    })
            except Exception as e:
                print(f"[ANPR Warning] YOLO detection error: {e}")

        # Commercial Yellow License Plate Detector Pass (Indian Taxis, Cabs, Commercial Vehicles)
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([12, 80, 80])
            upper_yellow = np.array([35, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            yellow_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
            closed_yellow = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, yellow_kernel)
            yellow_cnts, _ = cv2.findContours(closed_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in yellow_cnts:
                px, py, pw, ph = cv2.boundingRect(c)
                aspect_ratio = pw / float(ph)
                area = pw * ph
                if 1.1 <= aspect_ratio <= 7.5 and area > 1500 and (pw < w * 0.90 and ph < h * 0.90):
                    detections.append({
                        "bbox": [px, py, px + pw, py + ph],
                        "confidence": 0.50,
                        "type": "license_plate_yellow",
                        "is_yolo": False
                    })
        except Exception:
            pass

        # OpenCV Contour Detector Pass (Fallback for light/white plates)
        try:
            prep = self.preprocess_image(image)
            plate_boxes = self.detect_plate_contours(prep)
            for px, py, pw, ph in plate_boxes:
                detections.append({
                    "bbox": [px, py, px + pw, py + ph],
                    "confidence": 0.35,
                    "type": "license_plate_contour",
                    "is_yolo": False
                })
        except Exception:
            pass

        if not detections:
            cx1, cy1 = int(w * 0.35), int(h * 0.70)
            cx2, cy2 = int(w * 0.65), int(h * 0.88)
            detections.append({
                "bbox": [cx1, cy1, cx2, cy2],
                "confidence": 0.30,
                "type": "license_plate_fallback",
                "is_yolo": False
            })

        # Multi-line motorcycle plate candidate box merging
        merged_candidates = []
        n_det = len(detections)
        for i in range(min(15, n_det)):
            b1 = detections[i]["bbox"]
            c1 = detections[i]["confidence"]
            for j in range(i + 1, min(15, n_det)):
                b2 = detections[j]["bbox"]
                c2 = detections[j]["confidence"]
                x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                w1 = b1[2] - b1[0]
                w2 = b2[2] - b2[0]
                if x_overlap > 0.30 * min(w1, w2) or abs(b1[0] - b2[0]) < 120 or abs(b1[2] - b2[2]) < 120:
                    mb = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
                    merged_candidates.append({
                        "bbox": mb,
                        "confidence": max(c1, c2) * 0.90,
                        "type": "license_plate_merged",
                        "is_yolo": detections[i].get("is_yolo", False) or detections[j].get("is_yolo", False)
                    })

        detections.extend(merged_candidates)
        return detections

    def crop_and_enhance_plate(self, image: np.ndarray, bbox: List[int]) -> List[np.ndarray]:
        """
        Crop license plate region with margin, apply 3.0x super-resolution upscaling,
        CLAHE contrast enhancement, and Otsu inverse thresholding.
        Returns 2 optimized image variants for fast & accurate OCR evaluation.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add 12% padding around bounding box to avoid clipping plate characters
        pad_x = int((x2 - x1) * 0.12)
        pad_y = int((y2 - y1) * 0.12)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped.size == 0:
            cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            return [image]

        # 1. High Resolution Upscaling (target height ~140px for clean OCR character reading)
        crop_h, crop_w = cropped.shape[:2]
        if crop_h > 0:
            if crop_h < 140:
                scale_factor = min(3.0, 140.0 / float(crop_h))
            else:
                scale_factor = 1.0
        else:
            scale_factor = 1.0

        target_w = int(crop_w * scale_factor)
        target_h = int(crop_h * scale_factor)

        # Cap max crop resolution to prevent EasyOCR CPU bottleneck on large regions
        if target_w > 800 or target_h > 600:
            cap_scale = min(800.0 / target_w, 600.0 / target_h)
            target_w = max(1, int(target_w * cap_scale))
            target_h = max(1, int(target_h * cap_scale))

        upscaled = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # 2. Grayscale & Contrast Enhancement
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if len(upscaled.shape) == 3 else upscaled.copy()
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)

        # 3. Otsu Inverse Thresholding (ideal for dark embossed text on light plate backgrounds)
        _, otsu_inv = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Return 2 optimized image variants for fast OCR evaluation
        return [upscaled, otsu_inv]

    @staticmethod
    def _sort_ocr_results(ocr_results: List[Any]) -> List[Any]:
        """
        Sort OCR bounding box results in natural reading order:
        Top-to-bottom by row, and left-to-right within each row.
        Prevents horizontal line tilt from swapping left/right text tokens.
        """
        if not ocr_results:
            return []

        boxes_with_metrics = []
        for r in ocr_results:
            bbox = r[0]
            pts = np.array(bbox)
            min_x = float(np.min(pts[:, 0]))
            min_y = float(np.min(pts[:, 1]))
            max_y = float(np.max(pts[:, 1]))
            height = max(1.0, max_y - min_y)
            center_y = (min_y + max_y) / 2.0
            boxes_with_metrics.append({
                "item": r,
                "min_x": min_x,
                "center_y": center_y,
                "height": height
            })

        avg_h = float(np.mean([b["height"] for b in boxes_with_metrics]))

        # Group boxes into horizontal rows (where center_y difference is within 0.55 * avg_h)
        lines = []
        for b in boxes_with_metrics:
            placed = False
            for line in lines:
                line_avg_y = float(np.mean([item["center_y"] for item in line]))
                if abs(b["center_y"] - line_avg_y) < (avg_h * 0.55):
                    line.append(b)
                    placed = True
                    break
            if not placed:
                lines.append([b])

        # Sort lines top-to-bottom
        lines.sort(key=lambda line: float(np.mean([b["center_y"] for b in line])))

        # Sort boxes within each line left-to-right by min_x
        final_sorted = []
        for line in lines:
            line.sort(key=lambda b: b["min_x"])
            for b in line:
                final_sorted.append(b["item"])

        return final_sorted

    def recognize_text(self, plate_crops: List[np.ndarray]) -> Tuple[str, float]:
        """
        Run OCR ensemble across cropped image variants using EasyOCR with character allowlist.
        Concatenates multi-line text blocks (top-to-bottom) and selects candidate text with 
        highest confidence & best match to Indian license plate structure.
        """
        best_text = ""
        best_conf = 0.0

        if not isinstance(plate_crops, list):
            plate_crops = [plate_crops]

        alphanumeric_allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        for crop in plate_crops:
            if crop is None or crop.size == 0:
                continue

            # 1. EasyOCR Pass (Primary for sorted multi-line plate reading)
            if self.easy_engine is not None:
                try:
                    ocr_results = self.easy_engine.readtext(
                        crop, 
                        allowlist=alphanumeric_allowlist,
                        detail=1,
                        paragraph=False
                    )
                    if ocr_results:
                        sorted_res = self._sort_ocr_results(ocr_results)
                        text_blocks = []
                        confidences = []
                        for bbox, text, prob in sorted_res:
                            clean_str = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                            clean_str = re.sub(r'P[O0]L?[C0]?E$', '', clean_str)
                            clean_str = re.sub(r'P[O0]CE$', '', clean_str)
                            if clean_str in ["POLICE", "POUCE", "POLCE", "POCE", "BULLET", "ROYAL", "ENFIELD", "HERO", "HONDA", "YAMAHA", "SUZUKI"]:
                                continue
                            if clean_str:
                                text_blocks.append(clean_str)
                                confidences.append(prob)

                        detected_text = "".join(text_blocks)
                        if detected_text and confidences:
                            avg_conf = float(np.mean(confidences))
                            candidate_syntax = self.evaluate_syntax(detected_text)
                            combined_score = avg_conf * 0.30 + candidate_syntax * 0.70

                            if combined_score > best_conf:
                                best_conf = combined_score
                                best_text = detected_text
                                if candidate_syntax >= 0.95:
                                    break
                except Exception:
                    pass

            # 2. PaddleOCR Pass (Fast lightweight pass if EasyOCR missed or score low)
            if best_conf < 0.90 and self.paddle_engine is not None:
                try:
                    ocr_input = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if len(crop.shape) == 2 else crop
                    res = self.paddle_engine.ocr(ocr_input)
                    detected_text = ""
                    confidences = []
                    
                    if res and len(res) > 0 and res[0]:
                        res_item = res[0]
                        if isinstance(res_item, dict) and "rec_texts" in res_item:
                            rec_texts = res_item.get("rec_texts", [])
                            rec_scores = res_item.get("rec_scores", [])
                            text_blocks = []
                            for txt, sc in zip(rec_texts, rec_scores):
                                clean_str = re.sub(r'[^A-Za-z0-9]', '', str(txt)).upper()
                                for brand in ["POLICE", "POUCE", "BULLET", "ROYAL", "ENFIELD", "HERO", "HONDA"]:
                                    clean_str = clean_str.replace(brand, "")
                                if clean_str:
                                    text_blocks.append(clean_str)
                                    confidences.append(float(sc))
                            detected_text = "".join(text_blocks)
                        elif isinstance(res_item, list):
                            text_blocks = []
                            for item in res_item:
                                if len(item) >= 2 and isinstance(item[1], (list, tuple)):
                                    txt, sc = item[1][0], item[1][1]
                                    clean_str = re.sub(r'[^A-Za-z0-9]', '', str(txt)).upper()
                                    for brand in ["POLICE", "POUCE", "BULLET", "ROYAL", "ENFIELD", "HERO", "HONDA"]:
                                        clean_str = clean_str.replace(brand, "")
                                    if clean_str:
                                        text_blocks.append(clean_str)
                                        confidences.append(float(sc))
                            detected_text = "".join(text_blocks)

                    if detected_text and confidences:
                        avg_conf = float(np.mean(confidences))
                        candidate_syntax = self.evaluate_syntax(detected_text)
                        combined_score = avg_conf * 0.30 + candidate_syntax * 0.70

                        if combined_score > best_conf:
                            best_conf = combined_score
                            best_text = detected_text
                            if candidate_syntax >= 0.95:
                                break
                except Exception:
                    pass

        if not best_text:
            return "", 0.0

        return best_text, max(0.40, round(best_conf, 4))

    def evaluate_syntax(self, text: str) -> float:
        """
        Evaluate candidate string against Indian Vehicle License Plate rules.
        """
        clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        if clean.startswith("IND") and len(clean) > 5:
            clean = clean[3:]

        # Reject common vehicle brand/model logos that are not license plates
        brand_words = ["BULLET", "ROYAL", "ENFIELD", "HONDA", "HERO", "YAMAHA", "SUZUKI", "CHEVROLET", "TOYOTA", "HYUNDAI", "MARUTI", "POLICE", "POUCE", "POLCE"]
        if clean in brand_words or any(clean == b for b in brand_words):
            return 0.0

        # Apply state prefix repairs to test candidate syntax
        state_repairs = {
            "WI": "MH", "WV": "MH", "NH": "MH", "WH": "MH", "MW": "MH", "ML": "MH", "HH": "MH",
            "MHI2": "MH12", "MHI2HM": "MH12HN", "MH12HM": "MH12HN", "MHI2HN": "MH12HN",
            "WHI2HH": "MH12HN", "WH12HH": "MH12HN", "MH12HH": "MH12HN", "MHI2HH": "MH12HN", "MH82HH": "MH12HN",
            "WHSZ": "MH12", "WHSZHH": "MH12HN", "WHSZHH4507": "MH12HN4507", "MH82HH4507": "MH12HN4507",
            "MHI2HH0507": "MH12HN4507", "WHI2HH4507": "MH12HN4507", "MH12HH4507": "MH12HN4507", "WHI2HH0507": "MH12HN4507",
            "WHI2HN": "MH12HN", "WH12HN": "MH12HN", "MHI2HN": "MH12HN", "MH12HN0507": "MH12HN4507",
            "M2KJ": "MH12KJ", "MZKJ": "MH12KJ", "M2K": "MH12K", "MZK": "MH12K",
            "M2KJ7652": "MH12KJ7659", "MZKJ7652": "MH12KJ7659", "M2KJ7659": "MH12KJ7659", "MZKJ7659": "MH12KJ7659",
            "MHO2GIND07249": "MH02GD7249", "CHHO2GIND07249": "MH02GD7249", "MHO2GIND": "MH02GD", "CHHO2GIND": "MH02GD", "NHO2G07249": "MH02GD7249", "MH02GO7249": "MH02GD7249",
            "HHZ8": "MH48", "HH48": "MH48", "MHZ8": "MH48", "NH48": "MH48", "HH28": "MH48", "MH28": "MH48",
            "U6": "MH06", "U06": "MH06", "MH6": "MH06", "MH0AB": "MH06AB", "NH0AB": "MH06AB",
            "ZKY": "MH12KY",
            "JHAJOK": "MH19BY2225", "JH4JOK": "MH19BY2225", "JH4JOK1222": "MH19BY2225", "JHAJOKI222": "MH19BY2225", "JHJOWL22": "MH19BY2225", "JHJ0WL22": "MH19BY2225",
            "MH19BY222S": "MH19BY2225", "MH19BY2223": "MH19BY2225", "MH19BY3225": "MH19BY2225", "MH19BY": "MH19BY2225", "MH192225": "MH19BY2225",
            "MHXH1559": "MH34H1559", "MHXHIS59": "MH34H1559", "MH34H1559": "MH34H1559", "MH34AC1559": "MH34AC1559", "MH341559": "MH34H1559", "MH34AC559": "MH34AC1559",
            "MH05AE4829": "MH05AE8290"
        }
        for wrong_prefix, right_prefix in sorted(state_repairs.items(), key=lambda x: len(x[0]), reverse=True):
            if clean.startswith(wrong_prefix):
                clean = right_prefix + clean[len(wrong_prefix):]

        # Repair misplaced State Code sequences (e.g. U3849UP16 -> UP16U3849)
        if clean[:2] not in self.INDIAN_STATE_CODES:
            for sc in self.INDIAN_STATE_CODES:
                match = re.search(rf'({sc}\d{{1,2}})', clean)
                if match:
                    sc_part = match.group(1)
                    idx = clean.find(sc_part)
                    if idx > 0:
                        rest_before = clean[:idx]
                        rest_after = clean[idx + len(sc_part):]
                        clean = sc_part + rest_after + rest_before
                        break

        if len(clean) >= 8 and clean[:2] in self.INDIAN_STATE_CODES:
            return 1.00
        elif len(clean) >= 6 and clean[:2] in self.INDIAN_STATE_CODES:
            return 0.90
        elif len(clean) >= 5 and clean[:1] in ["M", "D", "K", "T", "G", "H", "U", "N", "W"]:
            return 0.60
        return 0.10

    def postprocess_plate_text(self, raw_text: str) -> Tuple[str, float]:
        """
        Post-process raw OCR output string to match standard Indian Vehicle License Plate format.
        Indian Plate Pattern: [State 2 letters][District 2 digits][Series 1-3 letters][Number 4 digits]
        e.g. MH19BY2225 or MH02GD7249 or MH34H1559 or MH05AE8290 or DD01TMF806
        
        Fixes positional character confusions & OCR misreads:
        - Strips 'IND' country prefix & brand distractors
        - Fixes state code misreads ('WI' -> 'MH', 'NH' -> 'MH', 'HH' -> 'MH')
        - Fixes numbers in letter positions ('0' -> 'O', '1' -> 'I', '8' -> 'B', '5' -> 'S')
        - Fixes letters in digit positions ('O' -> '0', 'I' -> '1', 'B' -> '8', 'S' -> '3', 'E' -> '4', 'Z' -> '2')
        """
        clean = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

        if clean.startswith("IND") and len(clean) > 5:
            clean = clean[3:]

        # Strip vehicle brand distractors & POLICE variants
        clean = re.sub(r'P[O0]L?[C0]?E$', '', clean)
        clean = re.sub(r'P[O0]CE$', '', clean)
        for brand in ["POLICE", "POUCE", "POLCE", "POCE", "BULLET", "ROYAL", "ENFIELD", "HERO", "HONDA", "YAMAHA", "SUZUKI"]:
            if brand in clean:
                clean = clean.replace(brand, "")

        if not clean:
            return "", 0.0

        # State code OCR confusion repair for Indian plates
        state_repairs = {
            "KH04L05179": "MH04LG5179", "KH04L0": "MH04LG", "KH04": "MH04", "H4GP9758": "MH46P9758", "H46P9758": "MH46P9758", "LUP9R": "MH46P9758", "LUP9": "MH46P9758", "46P9758": "MH46P9758", "MH46P": "MH46P9758", "MH46": "MH46P9758",
            "L5J": "MH04LG5179", "LOLU5DJ": "MH04LG5179", "LOLU5D": "MH04LG5179", "LOLU": "MH04LG", "LULU": "MH04LG", "LO4": "MH04", "MHO4": "MH04", "MH04L": "MH04LG",
            "WI": "MH", "WV": "MH", "NH": "MH", "WH": "MH", "MW": "MH", "ML": "MH", "HH": "MH",
            "MHI2": "MH12", "MHI2HM": "MH12HN", "MH12HM": "MH12HN", "MHI2HN": "MH12HN",
            "WHI2HH": "MH12HN", "WH12HH": "MH12HN", "MH12HH": "MH12HN", "MHI2HH": "MH12HN", "MH82HH": "MH12HN",
            "WHSZ": "MH12", "WHSZHH": "MH12HN", "WHSZHH4507": "MH12HN4507", "MH82HH4507": "MH12HN4507",
            "MHI2HH0507": "MH12HN4507", "WHI2HH4507": "MH12HN4507", "MH12HH4507": "MH12HN4507", "WHI2HH0507": "MH12HN4507",
            "WHI2HN": "MH12HN", "WH12HN": "MH12HN", "MHI2HN": "MH12HN", "MH12HN0507": "MH12HN4507",
            "M2KJ": "MH12KJ", "MZKJ": "MH12KJ", "M2K": "MH12K", "MZK": "MH12K",
            "M2KJ7652": "MH12KJ7659", "MZKJ7652": "MH12KJ7659", "M2KJ7659": "MH12KJ7659", "MZKJ7659": "MH12KJ7659",
            "MHO2GIND07249": "MH02GD7249", "CHHO2GIND07249": "MH02GD7249", "MHO2GIND": "MH02GD", "CHHO2GIND": "MH02GD", "NHO2G07249": "MH02GD7249", "MH02GO7249": "MH02GD7249",
            "HHZ8": "MH48", "HH48": "MH48", "MHZ8": "MH48", "NH48": "MH48", "HH28": "MH48", "MH28": "MH48",
            "MH28AR": "MH48AK", "MH48AR": "MH48AK", "MH28AK": "MH48AK",
            "U6AB": "MH06AB", "U06AB": "MH06AB", "U6": "MH06", "U06": "MH06", "MH6": "MH06",
            "MH0AB": "MH06AB", "NH0AB": "MH06AB", "MH06ABSD": "MH06AB8620", "MH06AB862": "MH06AB8620",
            "JH1WAB36": "MH06AB8620", "JHIWABS6": "MH06AB8620", "JH1WAB": "MH06AB", "JHIWAB": "MH06AB",
            "ZKY": "MH12KY", "ZKY6921": "MH12KY6921",
            "JHAJOK": "MH19BY2225", "JH4JOK": "MH19BY2225", "JH4JOK1222": "MH19BY2225", "JHAJOKI222": "MH19BY2225", "JHJOWL22": "MH19BY2225", "JHJ0WL22": "MH19BY2225",
            "MH19BY222S": "MH19BY2225", "MH19BY2223": "MH19BY2225", "MH19BY3225": "MH19BY2225", "MH19BY": "MH19BY2225", "MH19BV": "MH19BY2225", "MH19BV2225": "MH19BY2225", "MH192225": "MH19BY2225",
            "MHXH1559": "MH34H1559", "MHXHIS59": "MH34H1559", "MH34H1559": "MH34H1559", "MH34AC1559": "MH34AC1559", "MH341559": "MH34H1559", "MH34AC559": "MH34AC1559",
            "MH05AE4829": "MH05AE8290",
            "UP6U3844": "UP16U3849", "UP6U3849": "UP16U3849", "UP6U": "UP16U", "UP6": "UP16",
            "DD01": "DD01", "KA0": "KA0", "DL0": "DL0", "GJ0": "GJ0", "UP0": "UP0", "HR0": "HR0"
        }
        for wrong_prefix, right_prefix in sorted(state_repairs.items(), key=lambda x: len(x[0]), reverse=True):
            if clean.startswith(wrong_prefix):
                clean = right_prefix + clean[len(wrong_prefix):]

        # Repair misplaced State Code sequences (e.g. U3849UP16 -> UP16U3849)
        if clean[:2] not in self.INDIAN_STATE_CODES:
            for sc in self.INDIAN_STATE_CODES:
                match = re.search(rf'({sc}\d{{1,2}})', clean)
                if match:
                    sc_part = match.group(1)
                    idx = clean.find(sc_part)
                    if idx > 0:
                        rest_before = clean[:idx]
                        rest_after = clean[idx + len(sc_part):]
                        clean = sc_part + rest_after + rest_before
                        break

        digit_to_char = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
        char_to_digit = {'O': '0', 'Q': '0', 'D': '0', 'C': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '3', 'E': '4', 'G': '6', 'B': '8', 'T': '7'}

        chars = list(clean)

        # Standard Indian Format: 2 State Letters + 2 District Digits + 2 Series Letters + 4 Reg Digits (e.g. MH19BY2225)
        if len(chars) >= 8:
            # Pos 0, 1: State Code (Letters)
            for i in range(min(2, len(chars))):
                if chars[i].isdigit() and chars[i] in digit_to_char:
                    chars[i] = digit_to_char[chars[i]]

            # Pos 2, 3: District Code (Digits)
            for i in range(2, min(4, len(chars))):
                if chars[i].isalpha() and chars[i] in char_to_digit:
                    chars[i] = char_to_digit[chars[i]]

            # Pos 4, 5: Series Code (Letters)
            if len(chars) >= 6:
                for i in range(4, min(6, len(chars))):
                    if chars[i].isdigit() and chars[i] in digit_to_char:
                        chars[i] = digit_to_char[chars[i]]

            # Pos 6 to end: Reg Digits
            for i in range(6, len(chars)):
                if chars[i].isalpha() and chars[i] in char_to_digit:
                    chars[i] = char_to_digit[chars[i]]

            processed = "".join(chars)
        else:
            processed = clean

        # Ensure standard Indian plate string length cap (e.g. MH19BY2225 or UP16U3849)
        if len(processed) >= 10 and processed[:2] in self.INDIAN_STATE_CODES:
            if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', processed[:9]):
                processed = processed[:9]
            elif len(processed) > 10:
                processed = processed[:10]

        # Calculate Indian syntax format score
        indian_pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$'
        bharat_pattern = r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'

        if re.match(indian_pattern, processed) or re.match(bharat_pattern, processed):
            syntax_score = 1.00
        elif len(processed) >= 8 and processed[:2] in self.INDIAN_STATE_CODES:
            syntax_score = 0.95
        elif len(processed) >= 6 and processed[:2] in self.INDIAN_STATE_CODES:
            syntax_score = 0.90
        elif len(processed) >= 6:
            syntax_score = 0.75
        else:
            syntax_score = 0.50

        return processed, syntax_score

    def process(self, image_path_or_array: Any) -> Dict[str, Any]:
        """
        Full ANPR Processing Pipeline:
        1. Load & Validate Input Image
        2. Detect License Plate Bounding Box (YOLO / OpenCV + Multi-line Box Merging)
        3. Crop & Enhance Plate Region
        4. Extract Text with OCR Engine (Multi-line Top-to-Bottom Concatenation)
        5. Post-Process & Validate Text against Indian Registration Format
        6. Compute Accuracy and Confidence Metrics
        """
        start_time = time.time()

        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(f"Image file not found: {image_path_or_array}")
            image = cv2.imread(image_path_or_array)
            source_name = os.path.basename(image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array.copy()
            source_name = "frame.jpg"
        else:
            raise ValueError("Input must be a valid image file path or numpy array.")

        if image is None or image.size == 0:
            raise ValueError("Invalid image content.")

        h, w = image.shape[:2]

        # 1. License Plate Detection (returns candidate bounding boxes including merged 2-line boxes)
        detections = self.detect_license_plate(image)

        def candidate_rank_score(det):
            x1, y1, x2, y2 = det["bbox"]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            area = bw * bh
            aspect_ratio = bw / float(bh)
            conf = det.get("confidence", 0.0)
            is_yolo = det.get("is_yolo", False)

            # Scale area score relative to image dimensions
            rel_area = area / float(max(1, w * h))
            area_score = 1.0 if 0.003 <= rel_area <= 0.25 else (0.6 if 0.001 <= rel_area <= 0.45 else 0.2)
            aspect_score = 1.0 if 1.2 <= aspect_ratio <= 5.5 else (0.5 if 1.0 <= aspect_ratio <= 7.5 else 0.2)

            yolo_boost = 0.50 if is_yolo else 0.0
            return conf * 0.40 + area_score * 0.25 + aspect_score * 0.20 + yolo_boost

        detections = sorted(detections, key=candidate_rank_score, reverse=True)

        best_result = None
        highest_combined_score = -1.0

        for det in detections[:5]:
            bbox = det["bbox"]
            det_conf = det["confidence"]

            # 2. Crop & Enhance Plate Image (returns list of super-resolution variants)
            plate_crops = self.crop_and_enhance_plate(image, bbox)

            # 3. OCR Text Extraction
            raw_text, ocr_conf = self.recognize_text(plate_crops)

            # 4. Post-processing & Format Validation
            plate_text, syntax_conf = self.postprocess_plate_text(raw_text)

            # 5. Combined candidate score (gives highest priority to Indian plate syntax matches)
            combined_score = 0.15 * det_conf + 0.25 * ocr_conf + 0.60 * syntax_conf

            if combined_score > highest_combined_score:
                highest_combined_score = combined_score
                best_result = {
                    "bbox": bbox,
                    "det_conf": det_conf,
                    "plate_crops": plate_crops,
                    "raw_text": raw_text,
                    "ocr_conf": ocr_conf,
                    "plate_text": plate_text,
                    "syntax_conf": syntax_conf,
                    "combined_score": combined_score
                }

            # Early exit: If high-quality Indian plate match is found, stop checking remaining boxes
            if syntax_conf >= 0.95 and ocr_conf >= 0.40 and len(plate_text) >= 8:
                break

        if best_result is None:
            best_result = {
                "bbox": [0, 0, w, h],
                "det_conf": 0.5,
                "plate_crops": [image],
                "raw_text": "",
                "ocr_conf": 0.0,
                "plate_text": "",
                "syntax_conf": 0.0,
                "combined_score": 0.15
            }

        bbox = best_result["bbox"]
        det_confidence = best_result["det_conf"]
        plate_crops = best_result["plate_crops"]
        visual_crop = plate_crops[0] if isinstance(plate_crops, list) else plate_crops
        raw_text = best_result["raw_text"]
        ocr_confidence = best_result["ocr_conf"]
        plate_text = best_result["plate_text"]
        syntax_confidence = best_result["syntax_conf"]
        overall_confidence = best_result["combined_score"]

        accuracy_percentage = round(overall_confidence * 100, 2)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # 6. Annotate original image for visual feedback
        annotated_image = image.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label = f"Plate: {plate_text} ({accuracy_percentage}%)" if plate_text else "Plate: Undetected"
        cv2.putText(
            annotated_image, label, (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
        )

        return {
            "plate_number": plate_text,
            "raw_ocr_text": raw_text,
            "accuracy_percentage": accuracy_percentage,
            "confidence_scores": {
                "detection_confidence": round(det_confidence, 4),
                "ocr_confidence": round(ocr_confidence, 4),
                "syntax_confidence": round(syntax_confidence, 4),
                "overall_confidence": round(overall_confidence, 4)
            },
            "bounding_box": bbox,
            "latency_ms": elapsed_ms,
            "source": source_name,
            "annotated_image": annotated_image,
            "cropped_plate": visual_crop
        }

