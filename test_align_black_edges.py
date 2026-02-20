import cv2
import numpy as np

# Create a dummy image (e.g. 448x448 grey can)
img = np.full((448, 448, 3), 116, dtype=np.uint8)

# Simulate SIFT translation/scale pulling in black background
M = np.float32([[1.05, 0, 20], [0, 1.05, -15]])
aligned = cv2.warpAffine(img, M, (448, 448))

print("Min value in aligned image (should be 0 for black border):", aligned.min())
print("Max value in aligned image:", aligned.max())
