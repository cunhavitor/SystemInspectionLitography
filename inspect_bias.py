
import numpy as np
import os

path = "models/can_reference/patchcore448/bias_map_final.npy"

if os.path.exists(path):
    try:
        bias_map = np.load(path)
        print(f"Loaded {path}")
        print(f"Shape: {bias_map.shape}")
        print(f"Dtype: {bias_map.dtype}")
        print(f"Min: {bias_map.min()}, Max: {bias_map.max()}, Mean: {bias_map.mean()}")
    except Exception as e:
        print(f"Error loading {path}: {e}")
else:
    print(f"File not found: {path}")
