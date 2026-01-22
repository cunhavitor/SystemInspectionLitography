
import cv2
import sys
import os
import yaml
import numpy as np

# Add src to path
sys.path.append(os.getcwd())

from src.camera import Camera
from src.can_process_img.detect_corner import CornerDetector
from src.can_process_img.rectified_sheet import SheetRectifier
from src.can_process_img.crop_cans import CanCropper
from src.can_process_img.resize_can import CanResizer

def load_config(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_reference():
    print("=== Creating New 320x320 Reference Image ===")
    config = load_config()
    
    # Initialize components
    cam = Camera(
        camera_index=config['camera']['index'],
        width=config['camera']['width'],
        height=config['camera']['height'],
        fps=config['camera']['fps']
    )
    
    detector = CornerDetector()
    detector.load_params()
    rectifier = SheetRectifier()
    rectifier.load_params()
    cropper = CanCropper()
    cropper.load_params()
    resizer = CanResizer(size=(320, 320))
    
    print("Press 'SPACE' to capture reference can.")
    print("Press 'q' to quit.")
    
    try:
        while True:
            frame = cam.get_frame(stream_name='main') # Use high res stream
            if frame is None: continue
            
            display = cv2.resize(frame, (1024, 768))
            cv2.putText(display, "Press SPACE to capture reference", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Capture Reference", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                print("Capturing...")
                corners = detector.detect(frame)
                if corners is None:
                    print("Corners not found!")
                    continue
                    
                rectified = rectifier.get_warped(frame, corners)
                cans = cropper.crop_cans(rectified)
                
                if not cans:
                    print("No cans found!")
                    continue
                
                # Pick a central can (e.g., middle of the list)
                target_can = cans[len(cans)//2] 
                
                # Resize to 320x320
                ref_img = resizer.process(target_can['image'])
                
                # Calculate brightness to warn user if it's too dark
                mean_val = np.mean(ref_img)
                print(f"Captured Can Mean Brightness: {mean_val:.2f}")
                
                if mean_val < 50:
                    print("WARNING: Image is very dark! Check lighting.")
                
                # Create mask (Circle)
                h, w = ref_img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (w//2, h//2), int(w/2 * 0.98), 255, -1)
                
                # Apply mask to reference to ensure clean edges
                ref_masked = cv2.bitwise_and(ref_img, ref_img, mask=mask)
                
                # Save
                save_path = "models/can_reference/aligned_can_reference320.png"
                cv2.imwrite(save_path, ref_masked)
                print(f"SUCCESS: New reference saved to {save_path}")
                print(f"Dimensions: {ref_masked.shape}")
                
                cv2.imshow("New Reference", ref_masked)
                cv2.waitKey(0)
                break
                
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    create_reference()
