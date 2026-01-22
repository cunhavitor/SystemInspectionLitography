#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from src.camera import Camera
import cv2
import time

print("Testing camera capture...")
cam = Camera(width=640, height=480)

print("\nCapturing 5 frames...")
for i in range(5):
    frame = cam.get_frame()
    if frame is not None:
        print(f"Frame {i+1}: shape={frame.shape}, min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
        
        # Save first frame
        if i == 0:
            cv2.imwrite("test_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print("  Saved to test_frame.jpg")
    else:
        print(f"Frame {i+1}: None")
    time.sleep(0.5)

cam.release()
print("\nTest complete!")
