import cv2
import numpy as np
import openvino.runtime as ov
import os
import time

class PatchCoreInferencer:
    def __init__(self, model_dir="models/bpo_rr125_patchcore_v2", device="CPU"):
        self.model_dir = model_dir
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.threshold = 10.0 # Default threshold
        
        # Load OpenVINO model
        core = ov.Core()
        model_xml = os.path.join(model_dir, "model.xml")
        model = core.read_model(model=model_xml)
        self.compiled_model = core.compile_model(model=model, device_name=device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # Load Bias Map
        bias_path = os.path.join(model_dir, "threshold_map.npy")
        if os.path.exists(bias_path):
            self.bias_map = np.load(bias_path)
            print(f"Loaded bias map from {bias_path} with shape {self.bias_map.shape}")
        else:
            print(f"WARNING: Bias map not found at {bias_path}. processing without it.")
            self.bias_map = None

    def apply_clahe(self, image):
        """
        Applies CLAHE to the L channel of the LAB image.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def preprocess(self, image):
        """
        CLAHE -> Resize -> Normalize -> Transpose
        """
        # 1. Apply CLAHE (Centralized here)
        image = self.apply_clahe(image)

        # 2. Resize to 448x448
        # If image is already 448x448, resize is no-op or safe
        if image.shape[:2] != (448, 448):
            resized = cv2.resize(image, (448, 448))
        else:
            resized = image.copy()
        
        # 3. Convert to RGB (OpenCV uses BGR by default)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 4. Normalize
        rgb_image = rgb_image.astype(np.float32) / 255.0
        rgb_image = (rgb_image - self.mean) / self.std
        
        # 5. Transpose to [1, 3, 448, 448]
        input_tensor = np.transpose(rgb_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, resized

    def infer(self, image):
        """
        Full inference pipeline.
        Returns: score, clean_map, anomaly_map_viz
        """
        start_time = time.time()
        
        # Preprocess
        input_tensor, resized_image = self.preprocess(image)
        
        # Inference
        results = self.compiled_model([input_tensor])[self.output_layer]
        
        # Post-process (based on Anomalib output structure usually [1, 1, 448, 448] or similar)
        # Squeeze to remove batch dim if necessary: [1, H, W] -> [H, W]
        raw_map = results.squeeze() 
        
        # Apply Bias Map
        # Apply Bias Map (Threshold)
        # DEBUG DIAGNÓSTICO
        raw_max = np.max(raw_map)
        bias_max = np.max(self.bias_map) if self.bias_map is not None else 0
        bias_min = np.min(self.bias_map) if self.bias_map is not None else 0
        
        #print(f"--- DEBUG ---", flush=True)
        #print(f"RAW Map Max: {raw_max:.4f}", flush=True)   # Deve ser ~18 a 30
        #print(f"BIAS Map Max: {bias_max:.4f}", flush=True) # Deve ser ~18 a 22 (perto do raw)
        #print(f"BIAS Map Min: {bias_min:.4f}", flush=True) # Deve ser > 15
        
        if self.bias_map is not None:
             # 1. Garantir que as dimensões batem certo (segurança)
             if raw_map.shape != self.bias_map.shape:
                  # Se necessário, faz resize do bias para bater com o raw
                  self.bias_map = cv2.resize(self.bias_map, (raw_map.shape[1], raw_map.shape[0]))
             
             clean_map = np.maximum(raw_map - self.bias_map, 0)
             print(f"CLEAN Map Max (Score): {np.max(clean_map):.4f}", flush=True)
        else:
             clean_map = raw_map

        # Logic for Anomaly Score
        score = np.max(clean_map)
        
        # Visualization
        anomaly_map_viz = self.visualize(clean_map, resized_image)
        
        infer_time = time.time() - start_time
        # print(f"Inference time: {infer_time:.4f}s")
        
        return score, clean_map, anomaly_map_viz

    def visualize(self, anomaly_map, original_image):
        """
        Overlay heatmap on image.
        """
        # Normalize anomaly map to 0-255 for visualization
        # We need a robust normalization. 
        # For visualization, we often want to map [0, max] -> [0, 255] or [0, threshold*factor]
        
        # Simple min-max normalization for viz
        am_min = np.min(anomaly_map)
        am_max = np.max(anomaly_map)
        
        if am_max - am_min > 0:
            am_norm = (anomaly_map - am_min) / (am_max - am_min)
        else:
            am_norm = np.zeros_like(anomaly_map)
            
        am_norm = (am_norm * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(am_norm, cv2.COLORMAP_JET)
        
        # Resize heatmap to match original image (if not already 448x448, but here it is)
        if heatmap.shape[:2] != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
        # Superimpose
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        return overlay


    def predict(self, image):
        """
        API wrapper for inspection.py compatibility.
        """
        score, clean_map, viz = self.infer(image)
        # Usar o threshold configurado dinamicamente
        is_normal = score < self.threshold
        return score, is_normal, viz, clean_map

if __name__ == "__main__":
    # Test block
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python patchcore_inference_v2.py <image_path>")
    else:
        img_path = sys.argv[1]
        
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found.")
            sys.exit(1)
            
        img = cv2.imread(img_path)
        inferencer = PatchCoreInferencer()
        score, _, viz = inferencer.infer(img)
        
        print(f"Anomaly Score: {score:.4f}")
        result = "ANOMALY" if score > 0.5 else "NORMAL"
        print(f"Result: {result}")
        
        cv2.imwrite("output_viz.jpg", viz)
        print("Saved visualization to output_viz.jpg")
