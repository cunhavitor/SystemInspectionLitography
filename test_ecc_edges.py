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

import glob
files = sorted(glob.glob(os.path.join(project_root, "data/raw_sheet_crops/*.png")))

test_img_path = files[7] if len(files) >= 8 else files[0]
can_crop = cv2.imread(test_img_path)

# Initialize last_align_info manually for the standalone script
aligner.last_align_info = {'sift': None, 'ecc': None, 'ecc_rejected': False}

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

print("Saved debug_*.png files. Run ECC to see matrix.")
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, aligner.ECC_ITERATIONS, aligner.ECC_EPSILON)
try:
    cc, warp_matrix = cv2.findTransformECC(
        aligner.ref_edges_f, (input_edges_masked.astype(np.float32)/255.0), warp_matrix,
        cv2.MOTION_EUCLIDEAN, criteria
    )
    print(f"ECC CC: {cc:.4f} DX: {warp_matrix[0, 2]:.2f} DY: {warp_matrix[1, 2]:.2f}")
    
    fine_aligned = cv2.warpAffine(
        coarse_aligned, warp_matrix, (aligner.ref_w, aligner.ref_h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE
    )
    cv2.imwrite("debug_fine_aligned.png", fine_aligned)
except Exception as e:
    print(f"ECC Failed: {e}")
