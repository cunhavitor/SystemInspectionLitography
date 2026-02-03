
import time
import cv2
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from inference.patchcore_inference_v2 import PatchCoreInferencer

def benchmark():
    # Create a dummy image (448x448x3)
    img = np.zeros((448, 448, 3), dtype=np.uint8)
    
    print("Loading model...")
    inferencer = PatchCoreInferencer()
    
    print("Warming up...")
    # Warmup
    for _ in range(3):
        inferencer.infer(img)
        
    print("Running benchmark (10 iterations)...")
    times = []
    for i in range(10):
        start = time.time()
        inferencer.infer(img)
        dt = time.time() - start
        times.append(dt)
        print(f"Iter {i+1}: {dt*1000:.2f} ms")
        
    avg_time = sum(times) / len(times)
    print(f"\nAverage Inference Time: {avg_time*1000:.2f} ms")
    print(f"Average FPS: {1.0/avg_time:.2f} FPS")

if __name__ == "__main__":
    benchmark()
