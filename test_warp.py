import cv2
import numpy as np

# Create a template image (blank with a square)
template = np.zeros((100, 100), dtype=np.float32)
template[40:60, 40:60] = 1.0

# Create an input image shifted by dx=10, dy=5
# Square will be at [45:65, 50:70]
input_img = np.zeros((100, 100), dtype=np.float32)
input_img[45:65, 50:70] = 1.0

warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)

# Find transform
cc, M = cv2.findTransformECC(template, input_img, warp_matrix, cv2.MOTION_TRANSLATION, criteria)

print("Mapping Matrix M:")
print(M)
print("dx (M[0,2]):", M[0, 2])
print("dy (M[1,2]):", M[1, 2])

# Try warping With and Without INVERSE_MAP
warp_default = cv2.warpAffine(input_img, M, (100, 100))
warp_inverse = cv2.warpAffine(input_img, M, (100, 100), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

def check_alignment(img, name):
    diff = np.abs(img - template).sum()
    print(f"{name} diff from template:", diff)

check_alignment(input_img, "Original input")
check_alignment(warp_default, "warp_default")
check_alignment(warp_inverse, "warp_inverse")
