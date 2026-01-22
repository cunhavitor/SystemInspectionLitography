
import cv2
import numpy as np
import sys
import os

# Ensure src in path
sys.path.append(os.getcwd())

from src.ml.padim_inference import PadimInferencer

def test_padim():
    print("Testing PaDiM Inferencer...")
    
    try:
        inferencer = PadimInferencer()
        print(f"Loaded PaDiM. Input: {inferencer.input_size}")
    except Exception as e:
        print(f"FAILED to init: {e}")
        return

    # Dummy image (448, 448, 3)
    dummy_img = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    
    # Predict 1: Random Noise
    print("\n--- Test 1: Random Noise ---")
    try:
        score, is_normal, heatmap = inferencer.predict(dummy_img)
        print(f"Score: {score}")
    except Exception as e:
        print(f"Error: {e}")

    # Predict 2: Gray Image
    print("\n--- Test 2: Gray Image (128) ---")
    gray_img = np.ones((448, 448, 3), dtype=np.uint8) * 128
    try:
        score, is_normal, heatmap = inferencer.predict(gray_img)
        print(f"Score: {score}")
    except Exception as e:
        print(f"Error: {e}")
        
    # Predict 3: Black Image
    print("\n--- Test 3: Black Image (0) ---")
    black_img = np.zeros((448, 448, 3), dtype=np.uint8)
    try:
        score, is_normal, heatmap = inferencer.predict(black_img)
        print(f"Score: {score}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_padim()
