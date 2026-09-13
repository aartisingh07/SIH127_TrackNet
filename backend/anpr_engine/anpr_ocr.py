import os
import re
import cv2
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Valid Indian State & Union Territory Codes (36 Jurisdictions + Bharat Series)
INDIAN_STATES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN',
    'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LA', 'LD',
    'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PY', 'PB', 'RJ',
    'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB', 'BH'
}

# COCO Vehicle Classes (2: car, 3: motorcycle, 5: bus, 7: truck)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Character mappings for Indian license plate position correction
to_num = {
    'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8',
    'S': '5', 'G': '6', 'T': '7', 'A': '4', 'J': '3', 'U': '0', 't': '1',
    's': '5', 'l': '1', '|': '1', 'i': '1'
}
to_alpha = {
    '0': 'O', '1': 'I', '2': 'Z', '8': 'B', '5': 'S', '6': 'G', '7': 'T',
    '4': 'A', '9': 'P', '3': 'E'
}

# Common OCR confusion fixes for Indian State Codes
known_state_fixes = {
    'MN': 'MH', 'SK': 'MH', 'NL': 'MH', 'LA': 'DL', 'TR': 'TN',
    'MI': 'MH', 'MT': 'MH', 'MY': 'MH', 'MS': 'MH', 'MK': 'MH',
    'M3': 'MH', 'M4': 'MH', 'WI': 'MH', 'WA': 'WB', 'W0': 'WB',
    '7H': 'MH', 'HH': 'MH', 'NH': 'MH', 'M0': 'MH', 'M1': 'MH', 'MQ': 'MH', 'HQ': 'MH',
    '4H': 'MH', 'H0': 'MH', 'FH': 'MH', 'N0': 'MH', 'JH': 'MH', 'KH': 'MH', 'RH': 'MH',
    'D1': 'DL', 'D0': 'DL', 'OL': 'DL', '0L': 'DL',
    '4P': 'AP', 'A1': 'AP', 'AF': 'AP',
    'G1': 'GJ',
    'K1': 'KA', 'VA': 'KA', 'EA': 'KA', 'CA': 'KA',
    'T1': 'TN', 'TO': 'TN',
    'U1': 'UP', 'VP': 'UP', 'EU': 'UP', 'EUR': 'UP', 'RUP': 'UP',
    'H1': 'HR',
    'J1': 'JK',
    'R1': 'RJ',
    'W1': 'WB',
    'P1': 'PB',
    'C6': 'CG', 'C0': 'CG', 'K0': 'KA', 'K2': 'KA',
    'ER': 'TR', 'E0': 'TR', 'OH': 'MH', 'NM': 'MH', 'NN': 'MH', 'TH': 'TN'
}

CITY_TO_STATE_CODE = {
    'mumbai': 'MH', 'pune': 'MH', 'nagpur': 'MH', 'nashik': 'MH', 'thane': 'MH',
    'delhi': 'DL', 'new delhi': 'DL',
    'bengaluru': 'KA', 'bangalore': 'KA',
    'chennai': 'TN', 'hyderabad': 'TS', 'ahmedabad': 'GJ',
    'kolkata': 'WB', 'jaipur': 'RJ', 'lucknow': 'UP',
    'chandigarh': 'CH', 'patna': 'BR', 'bhopal': 'MP', 'guwahati': 'AS'
}

def resolve_state_from_camera_context(camera_context=None):
    """
    Infers the 2-letter state code (e.g. MH, DL, KA) based on camera location metadata.
    """
    if not camera_context:
        return 'MH'
    
    text_ctx = str(camera_context).lower()
    for kw, st in CITY_TO_STATE_CODE.items():
        if kw in text_ctx:
            return st
    return 'MH'


def edit_distance(s1, s2):
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "anpr_yolo_best.pt")

def deskew_crop(img):
    """Deskews side-angle/tilted plate crops into horizontal alignment."""
    if img is None or img.size == 0:
        return img
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=15, maxLineGap=5)
        if lines is not None:
            angles = []
            for line in lines:
                pts = line.flatten()
                if len(pts) == 4:
                    x1, y1, x2, y2 = pts
                    if x2 != x1:
                        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        if -40 < angle < 40 and abs(angle) > 1.5:
                            angles.append(angle)
            if angles:
                median_angle = np.median(angles)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), median_angle, 1.0)
                return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass
    return img


def split_two_line_plate_crop(img):
    """
    Real line detection using horizontal projection profile.
    Only returns (top_crop, bottom_crop) if a genuine horizontal whitespace gap (valley)
    separates two distinct text bands in the middle [0.25h, 0.75h] of the crop.
    """
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]

    if h < 25 or float(w) / float(max(1, h)) > 2.5:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Binarize for horizontal projection profile analysis
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    row_sums = np.sum(thresh == 255, axis=1)
    if len(row_sums) == 0:
        return None

    max_row = np.max(row_sums)
    if max_row == 0:
        return None

    min_y = int(h * 0.30)
    max_y = int(h * 0.65)

    if max_y <= min_y:
        return None

    mid_sums = row_sums[min_y:max_y]
    min_idx = int(np.argmin(mid_sums))
    split_y = min_y + min_idx
    min_val = mid_sums[min_idx]

    top_band = row_sums[0:split_y]
    bot_band = row_sums[split_y:h]

    top_max = np.max(top_band) if len(top_band) > 0 else 0
    bot_max = np.max(bot_band) if len(bot_band) > 0 else 0

    # Genuine two-line criteria:
    # 1. Both top and bottom bands must contain text (>= 15% of max_row)
    # 2. Valley min_val must be significantly lower than top_max and bot_max (<= 72% of min(top_max, bot_max))
    if top_max < max_row * 0.15 or bot_max < max_row * 0.15:
        return None

    if min_val > min(top_max, bot_max) * 0.72:
        return None

    if split_y < int(h * 0.20) or split_y > int(h * 0.80):
        return None

    # Include 8px overlap padding to prevent clipping top/bottom character strokes
    pad = 8
    top_crop = img[0:min(h, split_y + pad), :]
    bottom_crop = img[max(0, split_y - pad):h, :]

    return top_crop, bottom_crop


# ------------------------------------------------------------------------------
# Requirement 1: Preprocessing before OCR
# ------------------------------------------------------------------------------
def preprocess_plate_for_ocr(img, block_size=11, c_constant=2.0):
    """
    Takes a YOLO-cropped plate image and returns an OCR-ready image:
    1. Convert to grayscale
    2. Upscale 2x with cv2.INTER_CUBIC
    3. Apply adaptive thresholding (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY)
    Configurable block_size (must be odd >= 3) and c_constant.
    """
    if img is None or img.size == 0:
        return img

    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    bs = int(block_size)
    if bs % 2 == 0:
        bs += 1
    if bs < 3:
        bs = 3

    thresh = cv2.adaptiveThreshold(
        upscaled,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bs,
        float(c_constant)
    )
    return thresh


# ------------------------------------------------------------------------------
# Requirement 2: Fix YOLO cropping
# ------------------------------------------------------------------------------
def expand_and_clamp_bbox(bbox, img_shape, expand_pct=0.20):
    """
    Expands bounding box (x1, y1, x2, y2) by expand_pct (default 20%) on all sides,
    clamped to image dimensions.
    """
    x1, y1, x2, y2 = bbox
    img_h, img_w = img_shape[:2]

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    dx = int(round(w * float(expand_pct)))
    dy = int(round(h * float(expand_pct)))

    cx1 = max(0, x1 - dx)
    cy1 = max(0, y1 - dy)
    cx2 = min(img_w, x2 + dx)
    cy2 = min(img_h, y2 + dy)

    return cx1, cy1, cx2, cy2


# ------------------------------------------------------------------------------
# Requirement 3: Deterministic post-OCR correction for Indian plates
# ------------------------------------------------------------------------------
LETTER_FORCING_MAP = {
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G',
    '2': 'Z', '7': 'T', '4': 'A', '9': 'P', '3': 'E'
}

DIGIT_FORCING_MAP = {
    'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6',
    'Z': '2', 'T': '7', 'A': '4', 'P': '9', 'E': '3',
    'Q': '0', 'D': '0', 'L': '1', 'U': '0'
}

def force_letters(s: str) -> str:
    return "".join([LETTER_FORCING_MAP.get(c, c) for c in s])

def force_digits(s: str) -> str:
    return "".join([DIGIT_FORCING_MAP.get(c, c) for c in s])

def clean_indian_plate(raw_text: str):
    """
    Strips whitespace/special characters, forces uppercase, enforces 12-char sanity cap,
    and parses standard & temporary/TC format Indian license plates.
    Returns (cleaned_text, confidence_flag) where confidence_flag is True ONLY if state and
    number format strictly match expected Indian plate regex and length is <= 12.
    """
    if not raw_text:
        return "", False

    cleaned_raw = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    total_len = len(cleaned_raw)

    # Requirement 2: HARD SANITY CAP ON OUTPUT LENGTH (> 12 chars)
    if total_len > 12:
        print(f"[ANPR Warning] Candidate text '{cleaned_raw}' (len={total_len}) exceeded 12-char sanity cap. Discarding corrupted/merged candidate.")
        return "", False

    if total_len < 6:
        return cleaned_raw, False

    state_part = force_letters(cleaned_raw[0:2])
    state_part = known_state_fixes.get(state_part, state_part)
    valid_state = state_part in INDIAN_STATES

    rest = cleaned_raw[2:]
    
    # Check flexible TC/Trade/Govt/Standard regex: State + 1-2 RTO Digits + 1-3 Series Letters + 1-4 Plate Digits
    pattern_match = bool(re.match(r'^\d{1,2}[A-Z]{1,3}\d{1,4}$', rest))
    if valid_state and pattern_match and (6 <= total_len <= 12):
        return f"{state_part}{rest}", True

    # Standard position-forced Indian plate format
    if total_len >= 8:
        rto_part = force_digits(cleaned_raw[2:4])
        num_part = force_digits(cleaned_raw[-4:])
        series_part = force_letters(cleaned_raw[4:-4])
        std_result = f"{state_part}{rto_part}{series_part}{num_part}"
        valid_rto = rto_part.isdigit() and len(rto_part) == 2
        valid_num = num_part.isdigit() and len(num_part) == 4
        if valid_state and valid_rto and valid_num and (8 <= len(std_result) <= 12):
            return std_result, True

    return cleaned_raw, False


class ANPROCREngine:
    def __init__(self, model_path=None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.plate_detector = None
        self.vehicle_detector = None
        self.paddle_ocr_reader = None
        self.load_models()
        
    def load_models(self):
        """Loads Fine-tuned YOLO Plate Detector and Base YOLOv8n Vehicle Detector."""
        if os.path.exists(self.model_path):
            try:
                self.plate_detector = YOLO(self.model_path)
                print(f"ANPROCREngine: Loaded fine-tuned YOLO Plate Detector from {self.model_path}")
            except Exception as e:
                print(f"ANPROCREngine: Failed to load custom plate model: {e}")
                self.plate_detector = YOLO("yolov8n.pt")
        else:
            print("ANPROCREngine: Custom plate weights not found, initializing base YOLOv8n detector...")
            self.plate_detector = YOLO("yolov8n.pt")

        try:
            self.vehicle_detector = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"ANPROCREngine: Error loading vehicle detector: {e}")

    def preprocess_image(self, img):
        """Fast OpenCV Contrast & Adaptive Threshold Preprocessing for UI previews (BlackHat bypassed for latency)."""
        if img is None or img.size == 0:
            return None, {}
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        thresh = cv2.adaptiveThreshold(
            clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh, {'gray': gray, 'clahe': clahe, 'blackhat': clahe, 'sobel': sobelx, 'thresh': thresh}

    def get_ocr_reader(self):
        """Lazy initialization of PaddleOCR reader engine (PP-OCRv4, orientation classifiers disabled to prevent 180deg flip)."""
        if getattr(self, 'paddle_ocr_reader', None) is None:
            try:
                from paddleocr import PaddleOCR
                try:
                    self.paddle_ocr_reader = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        lang='en',
                        engine='onnxruntime'
                    )
                except Exception:
                    self.paddle_ocr_reader = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        lang='en'
                    )
                print("ANPROCREngine: Initialized PaddleOCR Reader (PP-OCRv4) successfully.")
            except Exception as e:
                print(f"ANPROCREngine: PaddleOCR init error: {e}")
                self.paddle_ocr_reader = None
        return self.paddle_ocr_reader

    def clean_plate_text(self, raw_text, camera_context=None):
        """Wrapper around clean_indian_plate for backward compatibility."""
        cleaned, flag = clean_indian_plate(raw_text)
        conf = 0.95 if flag else 0.45
        return cleaned, conf, flag

    def _single_crop_ocr(self, crop_img, block_size=11, c_constant=2.0):
        """Executes PaddleOCR on a single image crop and returns (raw_text, avg_confidence, token_candidates)."""
        reader = self.get_ocr_reader()
        if reader is None or crop_img is None or getattr(crop_img, 'size', 0) == 0:
            return "", 0.0, []

        def extract_from_paddle_res(res_list):
            txts = []
            confs = []
            if not res_list:
                return txts, confs
            for item in res_list:
                if isinstance(item, dict):
                    rec_texts = item.get('rec_texts', [])
                    rec_scores = item.get('rec_scores', [])
                    for t, s in zip(rec_texts, rec_scores):
                        clean_token = re.sub(r'[^A-Z0-9]', '', str(t).upper())
                        if clean_token not in ['IND', 'IN', 'ND', 'INDIA'] and t and str(t).strip():
                            txts.append(str(t).strip())
                            confs.append(float(s))
                elif isinstance(item, list):
                    for elem in item:
                        if len(elem) >= 2 and elem[1]:
                            t, s = elem[1][0], elem[1][1]
                            clean_token = re.sub(r'[^A-Z0-9]', '', str(t).upper())
                            if clean_token not in ['IND', 'IN', 'ND', 'INDIA'] and t and str(t).strip():
                                txts.append(str(t).strip())
                                confs.append(float(s))
            return txts, confs

        def run_paddle_on_input(ocr_input):
            if ocr_input is None or getattr(ocr_input, 'size', 0) == 0:
                return "", 0.0, []
            txts, confs = [], []
            try:
                if hasattr(reader, 'predict'):
                    ocr_res = reader.predict(ocr_input)
                else:
                    ocr_res = reader.ocr(ocr_input)
                txts, confs = extract_from_paddle_res(ocr_res)
            except Exception as e:
                print(f"PaddleOCR error: {e}")
            raw_text = " ".join(txts).strip() if txts else ""
            avg_conf = float(np.mean(confs)) if confs else 0.50
            tokens = list(zip(txts, confs)) if txts and confs else []
            return raw_text, avg_conf, tokens

        # 1. Try adaptive threshold preprocessed image first
        preprocessed = preprocess_plate_for_ocr(crop_img, block_size=block_size, c_constant=c_constant)
        prep_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR) if len(preprocessed.shape) == 2 else preprocessed
        raw_text, avg_conf, tokens = run_paddle_on_input(prep_bgr)

        # 2. Fall back to raw BGR crop image if preprocessed image returned no text
        if not raw_text:
            raw_bgr = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR) if len(crop_img.shape) == 2 else crop_img
            raw_text, avg_conf, tokens = run_paddle_on_input(raw_bgr)

        # 3. Fall back to deskewed crop if still no text
        if not raw_text:
            deskewed = deskew_crop(crop_img)
            if deskewed is not None and getattr(deskewed, 'size', 0) > 0:
                raw_text, avg_conf, tokens = run_paddle_on_input(deskewed)

        return raw_text, avg_conf, tokens

    def run_ocr(self, crop_img, camera_context=None, block_size=11, c_constant=2.0):
        """
        High-Precision OCR Pipeline using single-pass OCR, real line detection,
        and clean_indian_plate candidate selection with a strict 12-char length cap.
        Returns: (plate_text, ocr_conf, confidence_flag)
        """
        if crop_img is None or getattr(crop_img, 'size', 0) == 0:
            return "??", 0.0, False

        candidates = []

        def add_candidate(raw_str, conf, cand_type='single'):
            if not raw_str or not str(raw_str).strip():
                return
            clean_txt, flag = clean_indian_plate(raw_str)
            if not clean_txt:
                return
            if len(clean_txt) > 12:
                print(f"[ANPR Warning] Candidate text '{clean_txt}' (len={len(clean_txt)}) exceeded 12-char sanity cap. Discarding candidate.")
                return
            # Prevent duplicates in candidates list
            if not any(c['text'] == clean_txt for c in candidates):
                candidates.append({
                    'text': clean_txt,
                    'conf': float(conf),
                    'flag': flag,
                    'raw': raw_str,
                    'type': cand_type
                })

        # 1. Detect underexposed crops before choosing enhancement path
        crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if len(crop_img.shape) == 3 else crop_img.copy()
        crop_mean = float(np.mean(crop_gray))
        print(f"[ANPR Brightness] Crop Mean Brightness: {crop_mean:.2f}")

        if crop_mean < 80.0:
            print(f"[ANPR Underexposed] Dark crop detected (mean={crop_mean:.2f} < 80.0). Normalizing exposure + CLAHE...")
            gain = 125.0 / max(1.0, crop_mean)
            norm_crop = np.clip(crop_img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            norm_gray = cv2.cvtColor(norm_crop, cv2.COLOR_BGR2GRAY) if len(norm_crop.shape) == 3 else norm_crop
            clahe_norm = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(norm_gray)
            working_crop = cv2.cvtColor(clahe_norm, cv2.COLOR_GRAY2BGR)
        else:
            working_crop = crop_img

        # Pass 1: Single-line pass on full unsplit crop (ALWAYS RUN FIRST)
        single_raw, single_conf, single_tokens = self._single_crop_ocr(working_crop, block_size=block_size, c_constant=c_constant)
        if not single_raw and crop_mean < 80.0:
            # Fallback to unnormalized crop if normalized crop yielded no single-pass text
            single_raw, single_conf, single_tokens = self._single_crop_ocr(crop_img, block_size=block_size, c_constant=c_constant)

        add_candidate(single_raw, single_conf, 'single_combined')
        for tok_txt, tok_conf in single_tokens:
            add_candidate(tok_txt, tok_conf, 'single_token')

        # Fast-exit: If a valid Indian license plate syntax candidate was found in Pass 1, return immediately
        valid_pass1 = [c for c in candidates if c.get('flag')]
        if valid_pass1:
            best_cand = max(valid_pass1, key=lambda c: c['conf'])
            print(f"[ANPR Fast Exit] Valid Indian plate syntax '{best_cand['text']}' (conf: {best_cand['conf']:.2f}) found in Pass 1. Returning immediately.")
            return best_cand['text'], best_cand['conf'], best_cand['flag']

        # Pass 2: Real line detection -> split 2-line pass ONLY IF genuine 2-line structure confirmed
        split_crops = split_two_line_plate_crop(working_crop)
        if split_crops is None and crop_mean < 80.0:
            # Retry split on original crop if working crop split returned None
            split_crops = split_two_line_plate_crop(crop_img)

        if split_crops is not None:
            top_c, bot_c = split_crops

            top_gray = cv2.cvtColor(top_c, cv2.COLOR_BGR2GRAY) if len(top_c.shape) == 3 else top_c.copy()
            bot_gray = cv2.cvtColor(bot_c, cv2.COLOR_BGR2GRAY) if len(bot_c.shape) == 3 else bot_c.copy()
            top_std = float(np.std(top_gray))
            bot_std = float(np.std(bot_gray))
            print(f"[ANPR Band Contrast] Top Band Contrast (std): {top_std:.2f} | Bot Band Contrast (std): {bot_std:.2f}")

            top_txt, top_conf, top_toks = self._single_crop_ocr(top_c, block_size, c_constant)
            bot_txt, bot_conf, bot_toks = self._single_crop_ocr(bot_c, block_size, c_constant)

            # Retry Pass 1 for Line 1 (state/RTO) ONLY IF line 1 is empty or unreadable (< 2 chars)
            if not top_txt or len(top_txt.strip()) < 2:
                print(f"[ANPR Retry] Line 1 empty ({top_txt}), retrying with CLAHE + 2.5x super-res (contrast: {top_std:.2f})...")
                top_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(top_gray)
                th, tw = top_c.shape[:2]
                top_up = cv2.resize(cv2.cvtColor(top_clahe, cv2.COLOR_GRAY2BGR), (int(tw * 2.5), int(th * 2.5)), interpolation=cv2.INTER_CUBIC)
                r_top_txt, r_top_conf, r_top_toks = self._single_crop_ocr(top_up, block_size=15, c_constant=3.0)
                if r_top_txt:
                    top_txt, top_conf = r_top_txt, max(top_conf, r_top_conf)

            # Retry Pass 2 for Line 2 (digits) ONLY IF line 2 is empty or unreadable (< 2 chars)
            if not bot_txt or len(bot_txt.strip()) < 2:
                print(f"[ANPR Retry] Line 2 empty ({bot_txt}), retrying with CLAHE + 2.5x super-res (contrast: {bot_std:.2f})...")
                bot_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(bot_gray)
                bh, bw = bot_c.shape[:2]
                bot_up = cv2.resize(cv2.cvtColor(bot_clahe, cv2.COLOR_GRAY2BGR), (int(bw * 2.5), int(bh * 2.5)), interpolation=cv2.INTER_CUBIC)
                r_bot_txt, r_bot_conf, r_bot_toks = self._single_crop_ocr(bot_up, block_size=15, c_constant=3.0)
                if r_bot_txt:
                    bot_txt, bot_conf = r_bot_txt, max(bot_conf, r_bot_conf)

            # Fix common OCR confusion on second line: leading '0' or 'O' before 4 digits is 'D'
            if bot_txt and re.match(r'^[0O]\d{4}$', bot_txt.strip()):
                bot_txt = 'D' + bot_txt.strip()[1:]

            line_1_failed = not bool(top_txt and top_txt.strip())
            line_2_failed = not bool(bot_txt and bot_txt.strip())

            if line_1_failed:
                print("[ANPR Warning] line_1_read_failed: True (State/RTO line unreadable)")
            if line_2_failed:
                print("[ANPR Warning] line_2_read_failed: True (Digits line unreadable)")

            split_raw = f"{top_txt} {bot_txt}".strip()
            if split_raw and not line_1_failed and not line_2_failed:
                split_avg_conf = (top_conf + bot_conf) / 2.0 if (top_conf > 0 and bot_conf > 0) else max(top_conf, bot_conf)
                add_candidate(split_raw, split_avg_conf, 'split_combined')

        # Requirement 2: Filter candidates to enforce hard sanity cap (<= 12 chars)
        candidates = [c for c in candidates if c['text'] and len(c['text']) <= 12]

        if not candidates:
            return "??", 0.0, False

        # Requirement 1: Selection Rule
        # 1. Prefer candidates matching valid Indian plate regex (flag == True).
        # 2. Pick the candidate with HIGHEST underlying OCR confidence.
        valid_candidates = [c for c in candidates if c['flag']]
        if valid_candidates:
            best = max(valid_candidates, key=lambda c: c['conf'])
        else:
            # Fallback: Pick single highest-confidence candidate (NEVER sort by length!)
            best = max(candidates, key=lambda c: c['conf'])

        return best['text'] if best['text'] else "??", round(best['conf'], 2), best['flag']

    def detect_vehicles(self, img):
        """Detects vehicles (car, motorcycle, bus, truck) in frame."""
        vehicles = []
        if self.vehicle_detector is None:
            return vehicles
            
        h, w = img.shape[:2]
        try:
            preds = self.vehicle_detector.predict(img, conf=0.15, verbose=False, imgsz=640)
            for pred in preds:
                for box in pred.boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    if cls_id in VEHICLE_CLASS_IDS:
                        xyxy = map(int, box.xyxy[0].cpu().numpy())
                        vx1, vy1, vx2, vy2 = xyxy
                        vx1, vy1 = max(0, vx1), max(0, vy1)
                        vx2, vy2 = min(w, vx2), min(h, vy2)
                        v_conf = float(box.conf[0].cpu().numpy())
                        v_type = VEHICLE_NAMES.get(cls_id, 'vehicle')
                        vehicles.append(([vx1, vy1, vx2, vy2], v_type, v_conf))
        except Exception:
            pass
            
        return vehicles

    def detect_and_recognize(self, image_path_or_nparray, camera_id=None, camera_context=None):
        """
        High-Precision 2-Stage Hierarchical ANPR & OCR Pipeline with timing breakdown & latency optimization.
        """
        t_start = time.time()
        
        image_name = ""
        if isinstance(image_path_or_nparray, str):
            image_name = os.path.basename(image_path_or_nparray)
            img = cv2.imread(image_path_or_nparray)
        else:
            img = image_path_or_nparray
            
        if img is None:
            return []
            
        h, w = img.shape[:2]
        results = []
        
        # Step 1: Detect Vehicles
        t0 = time.time()
        vehicles = self.detect_vehicles(img)
        t_vehicle = (time.time() - t0) * 1000.0
        
        # Step 2: Stage 1 Direct Plate Detection
        t0 = time.time()
        plate_dets = []
        if self.plate_detector is not None:
            try:
                preds = self.plate_detector.predict(img, conf=0.08, verbose=False, imgsz=640)
                for pred in preds:
                    sorted_boxes = sorted(pred.boxes, key=lambda b: float(b.conf[0]), reverse=True)
                    for box in sorted_boxes:
                        px1, py1, px2, py2 = map(int, box.xyxy[0].cpu().numpy())
                        p_conf = float(box.conf[0].cpu().numpy())
                        px1, py1 = max(0, px1), max(0, py1)
                        px2, py2 = min(w, px2), min(h, py2)
                        if (px2 - px1) > 12 and (py2 - py1) > 6:
                            plate_dets.append(([px1, py1, px2, py2], p_conf))
            except Exception as e:
                print(f"Plate detect error: {e}")
        t_plate_det = (time.time() - t0) * 1000.0

        # Fast Center Tile Search ONLY IF direct detection missed plates
        if not plate_dets and (w > 1000 or h > 1000) and self.plate_detector is not None:
            tx1, ty1, tx2, ty2 = int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)
            tile_crop = img[ty1:ty2, tx1:tx2]
            if tile_crop.size > 0:
                try:
                    t_preds = self.plate_detector.predict(tile_crop, conf=0.06, verbose=False, imgsz=640)
                    for tpred in t_preds:
                        for tbox in tpred.boxes:
                            cpx1, cpy1, cpx2, cpy2 = map(int, tbox.xyxy[0].cpu().numpy())
                            cp_conf = float(tbox.conf[0].cpu().numpy())
                            abs_box = [tx1 + cpx1, ty1 + cpy1, tx1 + cpx2, ty1 + cpy2]
                            if (cpx2 - cpx1) > 12 and (cpy2 - cpy1) > 6:
                                plate_dets.append((abs_box, cp_conf))
                except Exception:
                    pass

        # Step 3: Vehicle Zoom ROI Search ONLY IF direct detection missed plates
        if not plate_dets and vehicles:
            for (vx1, vy1, vx2, vy2), v_type, v_conf in vehicles:
                vw, vh = vx2 - vx1, vy2 - vy1
                z_vx1, z_vy1 = max(0, vx1 - int(vw * 0.05)), max(0, vy1 - int(vh * 0.05))
                z_vx2, z_vy2 = min(w, vx2 + int(vw * 0.05)), min(h, vy2 + int(vh * 0.05))
                
                v_crop = img[z_vy1:z_vy2, z_vx1:z_vx2]
                if v_crop.size > 0:
                    try:
                        v_preds = self.plate_detector.predict(v_crop, conf=0.04, verbose=False)
                        for vpred in v_preds:
                            for vbox in vpred.boxes:
                                cpx1, cpy1, cpx2, cpy2 = map(int, vbox.xyxy[0].cpu().numpy())
                                cp_conf = float(vbox.conf[0].cpu().numpy())
                                abs_box = [z_vx1 + cpx1, z_vy1 + cpy1, z_vx1 + cpx2, z_vy1 + cpy2]
                                plate_dets.append((abs_box, cp_conf))
                    except Exception:
                        pass

        if not plate_dets:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                clahe_img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
                clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
                c_preds = self.plate_detector.predict(clahe_bgr, conf=0.06, verbose=False)
                for pred in c_preds:
                    for box in pred.boxes:
                        px1, py1, px2, py2 = map(int, box.xyxy[0].cpu().numpy())
                        p_conf = float(box.conf[0].cpu().numpy())
                        px1, py1 = max(0, px1), max(0, py1)
                        px2, py2 = min(w, px2), min(h, py2)
                        plate_dets.append(([px1, py1, px2, py2], p_conf))
            except Exception:
                pass

        # Filter overlapping plate boxes (IoU > 0.40 deduplication)
        final_plate_dets = []
        for box, conf in plate_dets:
            overlap = False
            for f_box, _ in final_plate_dets:
                x1 = max(box[0], f_box[0])
                y1 = max(box[1], f_box[1])
                x2 = min(box[2], f_box[2])
                y2 = min(box[3], f_box[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                if inter_area / float(box_area + 1e-5) > 0.40:
                    overlap = True
                    break
            if not overlap:
                final_plate_dets.append((box, conf))

        # Step 4: Run OCR Pipeline on each plate crop
        t0 = time.time()
        for (px1, py1, px2, py2), p_conf in final_plate_dets:
            cx1, cy1, cx2, cy2 = expand_and_clamp_bbox((px1, py1, px2, py2), img.shape, expand_pct=0.06)
            
            plate_crop = img[cy1:cy2, cx1:cx2]
            v_type = 'vehicle'
            v_bbox = [max(0, px1-40), max(0, py1-100), min(w, px2+40), min(h, py2+150)]
            v_conf = 0.85
            
            for (vx1, vy1, vx2, vy2), vt, vc in vehicles:
                if vx1 <= px1 + 30 and vy1 <= py1 + 30 and vx2 >= px2 - 30 and vy2 >= py2 - 30:
                    v_type, v_bbox, v_conf = vt, [vx1, vy1, vx2, vy2], vc
                    break

            vx1, vy1, vx2, vy2 = v_bbox
            v_area_ratio = float((vx2 - vx1) * (vy2 - vy1)) / float(w * h + 1e-5)
            v_bottom_ratio = float(vy2) / float(h + 1e-5)
            v_prom = round(float((v_area_ratio * 3.0) + (v_bottom_ratio * 2.0)), 4)

            plate_text, ocr_conf, confidence_flag = self.run_ocr(plate_crop, camera_context=camera_context)
                
            q_count = plate_text.count('?') if plate_text else 4
            valid_st = (plate_text[:2] in INDIAN_STATES) if plate_text and len(plate_text) >= 2 else False

            final_confidence = round(float(0.35 * p_conf + 0.65 * ocr_conf), 4)
            
            if not confidence_flag:
                final_confidence = min(final_confidence, 0.45)
            elif p_conf >= 0.35 and ocr_conf >= 0.60 and q_count == 0 and len(plate_text) >= 8 and valid_st:
                final_confidence = max(final_confidence, 0.88)
            
            results.append({
                'vehicle_type': v_type,
                'vehicle_bbox': v_bbox,
                'vehicle_confidence': round(v_conf, 4),
                'vehicle_prominence': v_prom,
                'plate_text': plate_text if plate_text else "??",
                'confidence': final_confidence,
                'confidence_flag': confidence_flag,
                'bbox': [cx1, cy1, cx2, cy2],
                'det_confidence': round(p_conf, 4),
                'ocr_confidence': round(ocr_conf, 4)
            })
        t_ocr = (time.time() - t0) * 1000.0

        # Strict 1-Plate-Per-Vehicle Selection Rule
        if results:
            vehicle_groups = {}
            for res in results:
                v_key = tuple(res['vehicle_bbox'])
                if v_key not in vehicle_groups:
                    vehicle_groups[v_key] = []
                vehicle_groups[v_key].append(res)
            
            filtered_results = []
            for v_key, cand_list in vehicle_groups.items():
                def candidate_rank(item):
                    p_text = item['plate_text']
                    det_c = item['det_confidence']
                    ocr_c = item['ocr_confidence']
                    v_prom = item.get('vehicle_prominence', 0.5)
                    score = (det_c * 2.5) + (ocr_c * 1.0) + (v_prom * 2.0)
                    if det_c >= 0.25 and len(p_text) >= 8 and p_text[:2] in INDIAN_STATES and "?" not in p_text:
                        score += 0.5
                    elif det_c >= 0.25 and len(p_text) >= 7 and p_text[:2] in INDIAN_STATES:
                        score += 0.25
                    if any(w in p_text for w in ["BULLET", "HONDA", "ROYAL", "YAMAHA", "SUZUKI", "TOYOTA"]):
                        score -= 3.0
                    return score

                cand_list.sort(key=candidate_rank, reverse=True)
                filtered_results.append(cand_list[0])

            filtered_results.sort(key=candidate_rank, reverse=True)
            results = filtered_results

        t_total = (time.time() - t_start) * 1000.0
        print(f"[ANPR Pipeline Timing] Vehicles: {t_vehicle:.1f}ms | Plate Detect: {t_plate_det:.1f}ms | OCR: {t_ocr:.1f}ms | Total: {t_total:.1f}ms")

        return results


if __name__ == "__main__":
    engine = ANPROCREngine()
    test_img = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_dataset", "test1.jpg")
    if os.path.exists(test_img):
        res = engine.detect_and_recognize(test_img)
        print("High-Precision ANPR & OCR Result:", res)
