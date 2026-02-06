
import cv2
import numpy as np
import openvino.runtime as ov
import os
import sys

# CONFIG
MODEL_DIR = "models/bpo_rr125_patchcore_resnet50"
REF_IMG = "models/can_reference/aligned_can_reference448_bpo-rr125.png"

def main():
    print(f"--- DEBUGGING RESNET50 PATCHCORE ---")
    
    # 1. Load Model
    core = ov.Core()
    model_xml = os.path.join(MODEL_DIR, "model.xml")
    if not os.path.exists(model_xml):
        print(f"Error: {model_xml} not found")
        return
        
    print(f"Loading {model_xml}...")
    model = core.read_model(model=model_xml)
    compiled_model = core.compile_model(model=model, device_name="CPU")
    output_layer = compiled_model.output(0)
    
    # 2. Load Threshold Map
    bias_path = os.path.join(MODEL_DIR, "threshold_map.npy")
    bias_map = None
    if os.path.exists(bias_path):
        bias_map = np.load(bias_path)
        print(f"Loaded bias map: {bias_map.shape}, Min: {np.min(bias_map):.4f}, Max: {np.max(bias_map):.4f}, Mean: {np.mean(bias_map):.4f}")
    else:
        print("Warning: No threshold map found")

    # 3. Load & Preprocess Image
    if not os.path.exists(REF_IMG):
        print(f"Error: Ref image {REF_IMG} not found")
        return
        
    img = cv2.imread(REF_IMG)
    print(f"Loaded Image: {img.shape}")
    
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Resize (448)
    if img_clahe.shape[:2] != (448, 448):
        resized = cv2.resize(img_clahe, (448, 448))
    else:
        resized = img_clahe
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - mean) / std
    
    # Transpose
    input_tensor = np.transpose(rgb, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # 4. Inference
    print("Running Inference...")
    results = compiled_model([input_tensor])[output_layer]
    raw_map = results.squeeze()
    
    print(f"Raw Map Stats: Shape {raw_map.shape}, Min: {np.min(raw_map):.4f}, Max: {np.max(raw_map):.4f}, Mean: {np.mean(raw_map):.4f}")
    
    # 5. Clean
    if bias_map is not None:
        if raw_map.shape != bias_map.shape:
            print(f"Resizing bias map from {bias_map.shape} to {raw_map.shape}")
            bias_map = cv2.resize(bias_map, (raw_map.shape[1], raw_map.shape[0]))
            
        clean_map = np.maximum(raw_map - bias_map, 0)
        score = np.max(clean_map)
        print(f"Cleaning Stats: Max (Score): {score:.4f}")
    else:
        print(f"Score (No bias): {np.max(raw_map):.4f}")

if __name__ == "__main__":
    main()
