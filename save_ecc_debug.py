import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from can_process_img.align_can import CanAligner

project_root = "/home/cunhav/projects/InspectionVisionCamera"
ref_path = os.path.join(project_root, "models/can_reference/aligned_can_reference448_bpo-rr125.png")

aligner = CanAligner(ref_path)

# Load a test can (assume 8)
image_folder = os.path.join(project_root, "data/raw_sheet_crops")

import glob
files = sorted(glob.glob(os.path.join(image_folder, "*_can8_*.png")))
if not files:
    print("Could not find Can 8 crop.")
    sys.exit(1)

test_img_path = files[0]
can_crop = cv2.imread(test_img_path)

print(f"Testing with {test_img_path}")

# Run SIFT
coarse_aligned, sift_conf = aligner._sift_align(can_crop)

# Run ECC manually to save edges
input_gray = cv2.cvtColor(coarse_aligned, cv2.COLOR_BGR2GRAY)
input_blurred = cv2.GaussianBlur(input_gray, (aligner.BLUR_SIZE, aligner.BLUR_SIZE), 0)
input_edges = cv2.Canny(input_blurred, aligner.CANNY_LOW, aligner.CANNY_HIGH)

input_edges_masked = cv2.bitwise_and(input_edges, input_edges, mask=aligner.ecc_mask)

# Save reference edges and input edges for visual inspection
cv2.imwrite("debug_ref_edges.png", (aligner.ref_edges_f * 255).astype(np.uint8))
cv2.imwrite("debug_input_edges.png", input_edges_masked)
cv2.imwrite("debug_input_blurred.png", input_blurred)
cv2.imwrite("debug_coarse_aligned.png", coarse_aligned)

print("Saved debug_*.png files.")
