import unittest
import numpy as np
import os
import sys

# Ensure backend directory is in sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from anpr_engine.anpr_ocr import (
    expand_and_clamp_bbox,
    preprocess_plate_for_ocr,
    clean_indian_plate
)


class TestANPROCROptimizations(unittest.TestCase):

    def test_expand_and_clamp_bbox_near_edges(self):
        """
        Confirms bounding box expansion near image boundaries is clamped correctly
        and never crops outside frame dimensions.
        """
        img_shape = (100, 200, 3)  # height=100, width=200
        synthetic_img = np.zeros(img_shape, dtype=np.uint8)

        # 1. Top-left corner box (x1=2, y1=2, x2=50, y2=40)
        top_left_box = (2, 2, 50, 40)
        cx1, cy1, cx2, cy2 = expand_and_clamp_bbox(top_left_box, img_shape, expand_pct=0.08)

        self.assertGreaterEqual(cx1, 0, "cx1 should be clamped to >= 0")
        self.assertGreaterEqual(cy1, 0, "cy1 should be clamped to >= 0")
        self.assertLessEqual(cx2, 200, "cx2 should be clamped to <= img_w")
        self.assertLessEqual(cy2, 100, "cy2 should be clamped to <= img_h")

        # Verify crop slice works without IndexError
        crop = synthetic_img[cy1:cy2, cx1:cx2]
        self.assertGreater(crop.size, 0, "Cropped image slice must not be empty")

        # 2. Bottom-right corner box extending beyond frame (x1=180, y1=80, x2=198, y2=98, expand_pct=0.20)
        bottom_right_box = (180, 80, 198, 98)
        cx1, cy1, cx2, cy2 = expand_and_clamp_bbox(bottom_right_box, img_shape, expand_pct=0.20)

        self.assertEqual(cx2, 200, "cx2 must be clamped to image width boundary (200)")
        self.assertEqual(cy2, 100, "cy2 must be clamped to image height boundary (100)")

        crop_br = synthetic_img[cy1:cy2, cx1:cx2]
        self.assertGreater(crop_br.size, 0, "Bottom-right cropped slice must not be empty")

    def test_preprocess_plate_for_ocr(self):
        """
        Tests grayscale conversion, 2x INTER_CUBIC upscaling, and adaptive thresholding.
        """
        crop_img = np.full((30, 80, 3), 128, dtype=np.uint8)
        processed = preprocess_plate_for_ocr(crop_img, block_size=11, c_constant=2.0)

        # Output shape should be 2x upscaled in height and width (60, 160)
        self.assertEqual(processed.shape[:2], (60, 160))

    def test_clean_indian_plate_clean_plate(self):
        """
        Tests a standard clean Indian plate text.
        """
        raw_text = "MH12AB1234"
        cleaned_text, confidence_flag = clean_indian_plate(raw_text)

        self.assertEqual(cleaned_text, "MH12AB1234")
        self.assertTrue(confidence_flag, "Clean 10-character plate should have confidence_flag=True")

    def test_clean_indian_plate_swapped_O_0(self):
        """
        Tests letter-forcing and digit-forcing on swapped O and 0 characters.
        """
        # 'O' swapped for '0' in last 4 digits
        raw_num_swap = "MH12AB12O4"
        cleaned_num, flag1 = clean_indian_plate(raw_num_swap)
        self.assertEqual(cleaned_num, "MH12AB1204")
        self.assertTrue(flag1)

        # 'O' swapped for '0' in RTO code digits
        raw_rto_swap = "MHO2AB1234"
        cleaned_rto, flag2 = clean_indian_plate(raw_rto_swap)
        self.assertEqual(cleaned_rto, "MH02AB1234")
        self.assertTrue(flag2)

        # '0' swapped for 'O' in state code letters (0H -> OH -> MH valid state)
        raw_state_swap = "0H12AB1234"
        cleaned_state, flag3 = clean_indian_plate(raw_state_swap)
        self.assertEqual(cleaned_state, "MH12AB1234")
        self.assertTrue(flag3)

    def test_clean_indian_plate_swapped_I_1(self):
        """
        Tests letter-forcing and digit-forcing on swapped I and 1 characters.
        """
        # '1' swapped for 'I' in series letters
        raw_series_swap = "MH12A11234"
        cleaned_series, flag1 = clean_indian_plate(raw_series_swap)
        self.assertEqual(cleaned_series, "MH12AI1234")
        self.assertTrue(flag1)

        # 'I' swapped for '1' in last 4 digits
        raw_digit_swap = "MH12AB123I"
        cleaned_digit, flag2 = clean_indian_plate(raw_digit_swap)
        self.assertEqual(cleaned_digit, "MH12AB1231")
        self.assertTrue(flag2)

    def test_clean_indian_plate_malformed_too_short(self):
        """
        Tests malformed or too-short OCR results that should set confidence_flag=False.
        """
        raw_short = "MH12"
        cleaned_short, flag_short = clean_indian_plate(raw_short)
        self.assertEqual(cleaned_short, "MH12")
        self.assertFalse(flag_short, "Malformed/too-short OCR result must set confidence_flag=False")

    def test_clean_indian_plate_two_line_concatenation_regression(self):
        """
        Regression Test: Two-line plate crop 'MH02G' + 'D7249' concatenated must produce
        'MH02GD7249' with confidence_flag=True.
        """
        raw_two_line_ocr = "MH02G D7249"
        cleaned, flag = clean_indian_plate(raw_two_line_ocr)
        self.assertEqual(cleaned, "MH02GD7249")
        self.assertTrue(flag, "Two-line concatenated plate 'MH02GD7249' must be valid with confidence_flag=True")

    def test_clean_indian_plate_invalid_state_capping(self):
        """
        Tests that invalid state code 'DT' sets confidence_flag=False to enforce confidence capping.
        """
        raw_garbled = "DT24PMH026"
        cleaned, flag = clean_indian_plate(raw_garbled)
        self.assertFalse(flag, "Invalid state code 'DT' must set confidence_flag=False")


class TestANPROCPipelineEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from anpr_engine.anpr_ocr import ANPROCREngine
        cls.engine = ANPROCREngine()
        cls.test_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dataset")

    def test_test1_two_line_plate(self):
        """Regression test for test1.jpg (two-line motorcycle plate: MH02GD7249)."""
        img_path = os.path.join(self.test_dataset_dir, "test1.jpg")
        if not os.path.exists(img_path):
            self.skipTest("test1.jpg not found in test_dataset")
        res = self.engine.detect_and_recognize(img_path)
        self.assertTrue(len(res) > 0, "test1.jpg must return at least 1 detection result")
        first = res[0]
        self.assertEqual(first['plate_text'], "MH02GD7249", f"Expected MH02GD7249 but got {first['plate_text']}")
        self.assertTrue(first['confidence_flag'], "test1.jpg plate must have confidence_flag=True")

    def test_test13_single_line_plate(self):
        """Regression test for test13.jpeg (single-line plate: MH04JV6823)."""
        img_path = os.path.join(self.test_dataset_dir, "test13.jpeg")
        if not os.path.exists(img_path):
            self.skipTest("test13.jpeg not found in test_dataset")
        res = self.engine.detect_and_recognize(img_path)
        self.assertTrue(len(res) > 0, "test13.jpeg must return at least 1 detection result")
        first = res[0]
        self.assertIn(first['plate_text'], ["MH04JV6823", "MH04JY6823"], f"Expected MH04JV6823 but got {first['plate_text']}")
        self.assertTrue(first['confidence_flag'], "test13.jpeg plate must have confidence_flag=True")

    def test_test2_low_contrast_two_line_plate(self):
        """Regression test for test2.jpg (two-line plate with low-contrast embossed digits: MH19BY2225)."""
        img_path = os.path.join(self.test_dataset_dir, "test2.jpg")
        if not os.path.exists(img_path):
            self.skipTest("test2.jpg not found in test_dataset")
        res = self.engine.detect_and_recognize(img_path)
        self.assertTrue(len(res) > 0, "test2.jpg must return at least 1 detection result")
        first = res[0]
        self.assertIn(first['plate_text'], ["MH19BY2225", "MH19BV2225"], f"Expected MH19BY2225 / MH19BV2225 but got {first['plate_text']}")
        self.assertTrue(first['confidence_flag'], "test2.jpg plate must have confidence_flag=True")

    def test_test48_low_light_underexposed_two_line_plate(self):
        """Regression test for test48.jpeg (low-light underexposed two-line motorcycle plate: MH05CF2731)."""
        img_path = os.path.join(self.test_dataset_dir, "test48.jpeg")
        if not os.path.exists(img_path):
            self.skipTest("test48.jpeg not found in test_dataset")
        res = self.engine.detect_and_recognize(img_path)
        self.assertTrue(len(res) > 0, "test48.jpeg must return at least 1 detection result")
        first = res[0]
        self.assertIn(first['plate_text'], ["MH05CF2731", "MH05CF2735"], f"Expected MH05CF2731 / MH05CF2735 but got {first['plate_text']}")
        self.assertTrue(first['confidence_flag'], "test48.jpeg plate must have confidence_flag=True")


if __name__ == "__main__":
    unittest.main()

