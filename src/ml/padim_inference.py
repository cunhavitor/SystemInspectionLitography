
import onnxruntime as ort
import numpy as np
import cv2
import os

class PadimInferencer:
    def __init__(self, model_path='models/padim448/model.onnx', threshold=None, use_imagenet_norm=False):
        """
        Inference class for PaDiM model (ONNX format).
        
        Args:
            model_path: Path to the .onnx model file.
            threshold: Anomaly score threshold. If None, tries to load from metadata or defaults to 0.5.
            use_imagenet_norm: Whether to apply ImageNet mean/std normalization.
        """
        self.model_path = model_path
        self.use_imagenet_norm = use_imagenet_norm
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PaDiM model not found at {model_path}")

        print(f"Initializing PaDiM from {model_path}...")
        
        # Load ONNX Session
        try:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print("✓ PaDiM ONNX session loaded.")
        except Exception as e:
            print(f"✗ Failed to load PaDiM session: {e}")
            raise

        # Input/Output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_score_name = self.session.get_outputs()[0].name
        self.output_mask_name = self.session.get_outputs()[1].name
        
        # Standard configs
        self.input_size = (448, 448) # Fixed for this specific model export
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Threshold priority: Arg > Config > Default
        self.threshold = threshold if threshold is not None else 0.5
        
        # Try to load threshold from metadata/config if available (optional)
        # (Not implemented here as PaDiM export often self-contained, but good practice)

    def predict(self, image_bgr):
        """
        Run inference on a single image.
        Args:
            image_bgr: Input image (H, W, 3) in BGR format.
        Returns:
            score (float): Anomaly score.
            is_normal (bool): True if score < threshold.
            heatmap (np.ndarray): Anomaly map (H, W).
        """
        # 1. Preprocess
        # Resize to model input
        img = cv2.resize(image_bgr, self.input_size)
        
        # Convert BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize (0-1)
        img = img.astype(np.float32) / 255.0
        
        # Standardize (ImageNet) - Optional
        if self.use_imagenet_norm:
            img = (img - self.mean) / self.std
        
        # CHW format
        img = img.transpose(2, 0, 1)
        
        # Batch dimension (1, C, H, W)
        img = np.expand_dims(img, axis=0)
        
        # 2. Inference
        inputs = {self.input_name: img}
        outputs = self.session.run(None, inputs)
        
        # 3. Parse Outputs
        # Output 0: Score [1, 1]
        # Output 1: Map [1, 1, 448, 448] (Name: add_405 per inspection)
        
        raw_score = outputs[0]
        raw_map = outputs[1]
        
        score = float(raw_score.item())
        
        # Heatmap: Squeeze to (H, W)
        heatmap = raw_map.squeeze() 
        
        # 4. Post-process
        is_normal = score < self.threshold
        
        # Ensure heatmap matches original image size if needed (usually 448x448 anyway)
        if heatmap.shape != image_bgr.shape[:2]:
            heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
            
        return score, is_normal, heatmap
