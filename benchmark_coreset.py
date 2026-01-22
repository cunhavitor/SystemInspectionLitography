
import cv2
import numpy as np
import sys
import os
import time

sys.path.append(os.getcwd())
from src.ml.patchcore_inference import PatchCoreInferencer

def test_coreset_speed():
    print("Initializing PatchCore with Coreset Sampling (Target: 10k)...")
    t_start = time.time()
    
    # Init with default params -> Triggers _patch_memory_bank
    inferencer = PatchCoreInferencer(use_imagenet_norm=False)
    
    t_load = time.time() - t_start
    print(f"Total Load Time: {t_load:.2f}s (includes Coreset calculation)")
    
    # Dummy Inference
    print("Benchmarking Inference (10 runs)...")
    dummy_img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    
    times = []
    # Warmup
    inferencer.predict(dummy_img)
    
    for _ in range(10):
        t0 = time.time()
        inferencer.predict(dummy_img)
        times.append(time.time() - t0)
        
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    print(f"Average Inference Time: {avg_time*1000:.2f} ms ({fps:.2f} FPS)")
    
    if fps > 1.0:
        print("✓ Speed Goal Met (>1 FPS)")
    else:
        print("⚠ Still slow. Check vector count.")

if __name__ == "__main__":
    test_coreset_speed()
