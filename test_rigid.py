import numpy as np
import cv2

# Create some dummy points
theta = 0.5
s = 1.15 # Hallucinated scale
R_true = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

src_pts = np.random.rand(100, 2) * 100
# True transform: just rotate and translate
dst_pts = np.dot(src_pts, R_true.T) + np.array([10.0, -20.0])

# Now corrupt dst_pts with scaling from SIFT (simulating the SIFT hallucination)
# Or wait, SIFT calculates the matrix FROM the points.
# Let's say SIFT returns A.
A, inliers = cv2.estimateAffinePartial2D(np.float32(src_pts), np.float32(dst_pts))

print("Original A:")
print(A)

# Extract pure rigid transform
src_inliers = src_pts[inliers.ravel() == 1]
dst_inliers = dst_pts[inliers.ravel() == 1]

# 1. Get raw scale and pure rotation
raw_scale = np.sqrt(A[0,0]**2 + A[1,0]**2)
R = A[:, :2] / raw_scale

# 2. Calculate pure translation using centroids
src_mean = np.mean(src_inliers, axis=0)
dst_mean = np.mean(dst_inliers, axis=0)

# 3. tx, ty = dst_mean - R * src_mean
t = dst_mean - np.dot(R, src_mean)

A_rigid = np.zeros((2, 3), dtype=np.float32)
A_rigid[:, :2] = R
A_rigid[:, 2] = t

print("\nRigid A:")
print(A_rigid)
