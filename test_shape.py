import cv2
import numpy as np

mask_img = np.zeros((448, 448), dtype=np.uint8)
erode_kernel = np.ones((19, 19), np.uint8)
mask_img = cv2.erode(mask_img, erode_kernel, iterations=1)
print(mask_img.shape)

anomaly_map = np.zeros((448, 448), float)
noise_floor = 0.6
anomaly_map = np.where(anomaly_map < noise_floor, 0, anomaly_map - noise_floor)
print(anomaly_map.shape)

anomaly_map *= mask_img
print(anomaly_map.shape)
