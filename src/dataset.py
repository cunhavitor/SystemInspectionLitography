import cv2
import time
import os
import numpy as np
from .camera import Camera
from .utils import save_image, create_directory
from .can_process_img.detect_corner import CornerDetector
from .can_process_img.rectified_sheet import SheetRectifier
from .can_process_img.crop_cans import CanCropper
from .can_process_img.align_can import CanAligner
from .can_process_img.resize_can import CanResizer

def run_dataset_mode(config):
    print("Starting Dataset Preparation Mode...")
    print("Press 's' to save ALL cans in current frame, 'q' to quit.")
    
    data_path = config['paths']['data_raw']
    create_directory(data_path)
    
    # Initialize Camera
    cam = Camera(
        camera_index=config['camera']['index'],
        width=config['camera']['width'],
        height=config['camera']['height'],
        fps=config['camera']['fps']
    )
    
    # Initialize Pipeline Components
    detector = CornerDetector()
    detector.load_params()
    
    rectifier = SheetRectifier()
    rectifier.load_params()
    
    cropper = CanCropper()
    cropper.load_params()
    
    # Force 448x448 for dataset
    final_size = (448, 448)
    resizer = CanResizer(size=final_size)
    
    aligner = None
    ref_path = 'models/can_reference/aligned_can_reference448.png'
    if os.path.exists(ref_path):
        try:
            aligner = CanAligner(ref_path, target_size=final_size)
            print(f"✓ Can Aligner loaded for dataset creation")
        except:
            print(f"⚠ Aligner failed to load. Saving unaligned cans.")
    
    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                print("Failed to capture frame")
                break
                
            # Show live feed with corners
            display_frame = frame.copy()
            corners = detector.detect(display_frame)
            if corners is not None:
                for pt in corners:
                    cv2.circle(display_frame, tuple(pt.astype(int).flatten()), 10, (0, 255, 0), -1)
            
            display_small = cv2.resize(display_frame, (1024, 768))
            cv2.imshow("Dataset Mode - 's' to capture", display_small)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("Capturing batch...")
                
                # 1. Detect & Rectify
                corners = detector.detect(frame)
                if corners is None:
                    print("✗ Corners not found! Cannot process.")
                    continue
                    
                rectified = rectifier.get_warped(frame, corners)
                if rectified is None:
                    print("✗ Rectification failed!")
                    continue
                    
                # 2. Crop
                cans = cropper.crop_cans(rectified)
                if not cans:
                    print("✗ No cans found!")
                    continue
                
                print(f"Processing {len(cans)} cans...")
                saved_count = 0
                
                for item in cans:
                    can_img = item['image']
                    
                    # 3. Resize -> 448x448
                    resized = resizer.process(can_img)
                    
                    # 4. Align (if available) -> 448x448
                    if aligner:
                        processed = aligner.align(resized)
                    else:
                        processed = resized
                        
                    # Save
                    save_image(processed, data_path, prefix="good")
                    saved_count += 1
                
                print(f"✓ Saved {saved_count} images to {data_path}")
                print("Size check: ", processed.shape if saved_count > 0 else "N/A")
                
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cam.release()
        cv2.destroyAllWindows()
