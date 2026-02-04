import cv2
import numpy as np
import os
import sys

def analyze_image(path):
    if not os.path.exists(path):
        return None
    
    img = cv2.imread(path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Calculate sharpness using variance of Laplacian
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate mean brightness
    mean_brightness = np.mean(gray)
    
    return {
        'path': path,
        'resolution': f"{w}x{h}",
        'sharpness': laplacian_var,
        'brightness': mean_brightness,
        'size_bytes': os.path.getsize(path)
    }

def main():
    path1 = "/home/cunhav/projects/InspectionVisionCamera/data/dataset/batch_20260204_151439/train/can002.png"
    path2 = "data/defects/2026/02/NOK_20260204_132114_can11_score0.02.png"
    
    img1 = analyze_image(path1)
    img2 = analyze_image(path2)
    
    print(f"{'Metric':<20} | {'New Dataset (Train)':<30} | {'Defect (Reference)':<30}")
    print("-" * 88)
    
    if img1:
        print(f"{'Resolution':<20} | {img1['resolution']:<30} | {img2['resolution'] if img2 else 'N/A':<30}")
        print(f"{'Sharpness (Var)':<20} | {img1['sharpness']:<30.2f} | {img2['sharpness'] if img2 else 'N/A':<30.2f}")
        print(f"{'Brightness':<20} | {img1['brightness']:<30.2f} | {img2['brightness'] if img2 else 'N/A':<30.2f}")
        print(f"{'File Size':<20} | {img1['size_bytes']:<30} | {img2['size_bytes'] if img2 else 'N/A':<30}")
    else:
        print("Could not load dataset image.")

if __name__ == "__main__":
    main()
