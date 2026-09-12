import os
import sys
import glob
import json
import time

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from anpr_engine.anpr_ocr import ANPROCREngine

def run_benchmark():
    engine = ANPROCREngine()
    test_dir = os.path.join(backend_dir, "test_dataset")
    test_images = sorted([
        f for f in os.listdir(test_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print("=========================================================")
    print(f"  TrackNet ANPR Accuracy Benchmark - {len(test_images)} Test Images  ")
    print("=========================================================")

    total_images = len(test_images)
    detected_images_count = 0
    total_detections = 0
    clean_plates_count = 0
    partial_plates_count = 0
    no_detection_files = []
    
    start_time = time.time()

    for idx, fname in enumerate(test_images, 1):
        fpath = os.path.join(test_dir, fname)
        results = engine.detect_and_recognize(fpath)
        
        if results:
            detected_images_count += 1
            total_detections += len(results)
            for res in results:
                plate = res['plate_text']
                conf = res['confidence']
                if '?' not in plate and len(plate) >= 8:
                    clean_plates_count += 1
                else:
                    partial_plates_count += 1
                print(f"[{idx:02d}/{total_images}] {fname} -> Plate: {plate:<12} | Conf: {conf:.2f} | DetConf: {res['det_confidence']:.2f}")
        else:
            no_detection_files.append(fname)
            print(f"[{idx:02d}/{total_images}] {fname} -> NO DETECTION")

    elapsed = time.time() - start_time
    print("=========================================================")
    print(f"Total Test Images       : {total_images}")
    print(f"Images with Detections  : {detected_images_count} ({(detected_images_count/total_images)*100:.1f}%)")
    print(f"Total Plates Detected   : {total_detections}")
    print(f"Clean Full Plates (no ?): {clean_plates_count}")
    print(f"Partial/Masked Plates   : {partial_plates_count}")
    print(f"No Detection Images     : {len(no_detection_files)}")
    if no_detection_files:
        print("  Missing files:", no_detection_files)
    print(f"Time Taken              : {elapsed:.2f} seconds")
    print("=========================================================")

if __name__ == "__main__":
    run_benchmark()
