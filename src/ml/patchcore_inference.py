import cv2
import numpy as np
import openvino.runtime as ov
import os
import time

class PatchCoreInferencer:
    def __init__(self, 
                 model_dir="models/bpo_rr125_patchcore_v2", 
                 device="CPU",
                 use_imagenet_norm=True, # Argument kept for compatibility, but logic is fixed as per prompt
                 normalize_scores=True, # Argument kept for compatibility
                 **kwargs):
        
        # Determine model path
        # If model_dir points to a file (xml), use it. If dir, append model.xml
        if model_dir.endswith('.xml'):
            self.model_xml = model_dir
            self.model_dir = os.path.dirname(model_dir)
        else:
            self.model_dir = model_dir
            self.model_xml = os.path.join(model_dir, "model.xml")
            
        self.device = device
        self.threshold = 0.5 # Default as requested
        
        # Hardcoded params as per prompt
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = (448, 448)

        print(f"Initializing PatchCoreInferencer from {self.model_dir}...")

        # Load OpenVINO model
        if not os.path.exists(self.model_xml):
            # Fallback to discover if existing code used different path
            # But we are instructed to use specific model. 
            print(f"Warning: {self.model_xml} not found. Checking absolute path...")
        
        try:
            core = ov.Core()
            print(f"Reading model: {self.model_xml}")
            model = core.read_model(model=self.model_xml)
            self.compiled_model = core.compile_model(model=model, device_name=device)
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            print("✓ OpenVINO model compiled")
        except Exception as e:
            print(f"✗ Failed to load OpenVINO model: {e}")
            raise e
        
        # Load Bias Map
        bias_path = os.path.join(self.model_dir, "bias_map_final.npy")
        if os.path.exists(bias_path):
            try:
                self.bias_map = np.load(bias_path)
                print(f"✓ Loaded bias map from {bias_path} with shape {self.bias_map.shape}")
            except Exception as e:
                print(f"⚠ Failed to load bias map: {e}")
                self.bias_map = None
        else:
            print(f"WARNING: Bias map not found at {bias_path}. processing without it.")
            self.bias_map = None

        # Load Threshold Map Zones (New Feature)
        threshold_map_path = os.path.join(self.model_dir, "threshold_map_zones.npy")
        if os.path.exists(threshold_map_path):
            try:
                self.threshold_map = np.load(threshold_map_path)
                print(f"✓ Loaded threshold map from {threshold_map_path} with shape {self.threshold_map.shape}")
                # Set default threshold/margin to 10% if map exists
                self.threshold = 10.0 
            except Exception as e:
                print(f"⚠ Failed to load threshold map: {e}")
                self.threshold_map = None
        else:
            print(f"Info: Threshold map not found at {threshold_map_path}. Using global scalar threshold.")
            self.threshold_map = None

    def preprocess(self, image):
        """
        Resize, normalize and transpose image.
        """
        # Resize to 448x448
        # If image is already 448x448, resize is no-op or safe
        if image.shape[:2] != self.input_size:
            resized = cv2.resize(image, self.input_size)
        else:
            resized = image.copy()
        
        # Convert to RGB (OpenCV uses BGR by default)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        rgb_image = rgb_image.astype(np.float32) / 255.0
        rgb_image = (rgb_image - self.mean) / self.std
        
        # Transpose to [1, 3, 448, 448]
        input_tensor = np.transpose(rgb_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, resized

    def predict(self, image):
        """
        Compatible API for inspection.py
        Returns: score, is_normal, heatmap
        """
        score_raw, clean_map, _ = self.infer(image)
        
        # Get resized image for visualization (needed since we generate a new heatmap)
        _, resized_image = self.preprocess(image)

        if self.threshold_map is not None:
             # Zone-based thresholding: Percentage Excess
             # Logic: Score = Max % that the image exceeds the threshold map
             # 0% = perfect (or below map), 20% = 20% brighter than map
             
             if clean_map.shape == self.threshold_map.shape:
                 # 1. Calculate Absolute Excess
                 excess_map = clean_map - self.threshold_map
                 
                 # Clip negative values (we only care about excess)
                 valid_excess = np.maximum(excess_map, 0)
                 
                 # 2. Calculate Percentage Excess Map
                 # Avoid division by zero (though map min is ~28)
                 pct_excess_map = (valid_excess / (self.threshold_map + 1e-6)) * 100.0
                 
                 # 3. Score = Max Percentage Excess
                 score = np.max(pct_excess_map)
                 
                 is_normal = score < self.threshold
                 
                 # 4. Visualization
                 # visualize the VALID EXCESS (absolute or percentage?)
                 # Visualizing absolute excess might be cleaner for the eye, 
                 # but percentage is what the score is based on.
                 # Let's visualize the valid_excess (absolute) so it looks like a standard heatmap
                 # but only showing the "bad" parts.
                 heatmap_viz = self.visualize(valid_excess, resized_image)
                 
             else:
                 print(f"⚠ Shape mismatch: clean_map {clean_map.shape} != threshold_map {self.threshold_map.shape}")
                 score = score_raw
                 is_normal = score < self.threshold
                 heatmap_viz = self.visualize(clean_map, resized_image)
        else:
            # Legacy scalar threshold
            score = score_raw
            is_normal = score < self.threshold
            heatmap_viz = self.visualize(clean_map, resized_image)

        return score, is_normal, heatmap_viz

    def infer(self, image):
        """
        Full inference pipeline.
        Returns: score, clean_map, anomaly_map_viz
        """
        # Check input type
        if image.dtype != np.uint8:
             # Assuming it might be already normalized?
             # For safety, if user passes float, maybe we should warn or handle.
             # But let's assume I fix `inspection.py`.
             pass

        start_time = time.time()
        
        # Preprocess
        input_tensor, resized_image = self.preprocess(image)
        
        # Inference
        results = self.compiled_model([input_tensor])[self.output_layer]
        
        # Post-process
        raw_map = results.squeeze() 
        
        # Apply Bias Map
        if self.bias_map is not None:
            if raw_map.shape != self.bias_map.shape:
                 # Resize bias map if needed
                 # This can happen if output is 56x56 but bias is 448x448?
                 # Bias map `models/bpo_rr125_patchcore_v2/bias_map_final.npy` shape was (56, 56) in my test output.
                 # Raw map likely (56, 56) or similar.
                 if raw_map.shape != self.bias_map.shape:
                      print(f"Warning: Map shape mismatch. Raw: {raw_map.shape}, Bias: {self.bias_map.shape}")
                      # Try resize bias to match raw
                      # Skipping for now to avoid errors, or resize bias
                      pass
            
            # clean_map = raw_map - bias_map
            clean_map = raw_map - self.bias_map
        else:
            clean_map = raw_map

        # Logic for Anomaly Score
        score = np.max(clean_map)
        
        # Visualization
        anomaly_map_viz = self.visualize(clean_map, resized_image)
        
        return score, clean_map, anomaly_map_viz

    def visualize(self, anomaly_map, original_image):
        """
        Overlay heatmap on image.
        """
        # Normalize anomaly map to 0-255 for visualization
        am_min = np.min(anomaly_map)
        am_max = np.max(anomaly_map)
        
        if am_max - am_min > 0:
            am_norm = (anomaly_map - am_min) / (am_max - am_min)
        else:
            am_norm = np.zeros_like(anomaly_map)
            
        am_norm = (am_norm * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(am_norm, cv2.COLORMAP_JET)
        
        # Resize heatmap to match original image (448x448)
        if heatmap.shape[:2] != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
        # Superimpose
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        return overlay
