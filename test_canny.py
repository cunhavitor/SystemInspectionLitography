import cv2
import numpy as np

# Create a dummy image
img = np.ones((100, 100), dtype=np.uint8) * 128

# Create a mask
mask = np.zeros((100, 100), dtype=np.uint8)
cv2.circle(mask, (50, 50), 40, 255, -1)

# Mask before Canny
masked_img = cv2.bitwise_and(img, img, mask=mask)
edges1 = cv2.Canny(masked_img, 50, 150)

# Mask after Canny
edges2 = cv2.Canny(img, 50, 150)
edges2 = cv2.bitwise_and(edges2, edges2, mask=mask)

print("Edges if masked before Canny:", np.sum(edges1 > 0))
print("Edges if masked after Canny:", np.sum(edges2 > 0))
