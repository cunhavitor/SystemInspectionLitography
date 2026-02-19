import cv2
import numpy as np
import openvino.runtime as ov
import os
import time

class PadimInferencer:
    def __init__(self, model_dir="models/bpo_rr125_padimResnet18", device="CPU"):
        self.model_dir = model_dir
        self.device = device
        
        # ImageNet Normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # --- INDUSTRIAL CONFIG ---
        self.threshold = 10.0 # Default threshold, adjust based on validation
        
        # Load OpenVINO model
        print(f"📦 Loading PaDiM model from {model_dir}...")
        core = ov.Core()
        model_xml = os.path.join(model_dir, "model.xml")
        
        if not os.path.exists(model_xml):
            raise FileNotFoundError(f"Model file not found: {model_xml}")

        # Cache config
        core.set_property({'CACHE_DIR': os.path.join(model_dir, 'cache')})
        
        model = core.read_model(model=model_xml)

        # Raspberry Pi 5 Optimization
        config = {
            "INFERENCE_NUM_THREADS": "4",
            "NUM_STREAMS": "1",
            "PERFORMANCE_HINT": "LATENCY"
        }
        self.compiled_model = core.compile_model(model=model, device_name=device, config=config)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # Load Statistics (Mean & Inv Covariance)
        mean_path = os.path.join(model_dir, "mean.npy")
        inv_cov_path = os.path.join(model_dir, "inv_cov.npy")
        
        if os.path.exists(mean_path) and os.path.exists(inv_cov_path):
            print("⏳ Loading statistical model (this may take a moment)...")
            self.stats_mean = np.load(mean_path)     # (100, 56, 56)
            self.stats_inv_cov = np.load(inv_cov_path) # (100, 100, 56, 56)
            
            # Reshape for vectorized calculation: (H*W, C) and (H*W, C, C)
            C, H, W = self.stats_mean.shape
            self.H, self.W = H, W
            self.C = C
            
            # (C, H, W) -> (C, H*W) -> (H*W, C)
            self.stats_mean = self.stats_mean.reshape(C, -1).T.astype(np.float32)
            
            # (C, C, H, W) -> (C, C, H*W) -> (H*W, C, C)
            self.stats_inv_cov = self.stats_inv_cov.reshape(C, C, -1).transpose(2, 0, 1).astype(np.float32)

            # Pre-compute Einsum optimization path
            # This significantly speeds up the Mahalanobis calculation (4x faster in benchmarks)
            print("⚡ Optimizing Mahalanobis calculation path...")
            dummy_delta = np.zeros((H * W, C), dtype=np.float32)
            # Using 'ij,ijk,ik->i': (N, C) * (N, C, C) * (N, C) -> (N,)
            self.einsum_path = np.einsum_path('ij,ijk,ik->i', dummy_delta, self.stats_inv_cov, dummy_delta, optimize='optimal')[0]
            
            print(f"✅ Loaded statistical model. Grid: {H}x{W}, Channels: {C}")
        else:
            raise FileNotFoundError("PaDiM statistics (mean.npy, inv_cov.npy) not found!")

    def preprocess(self, image):
        # Resize -> Normalize -> Transpose
        t0 = time.time()
        
        # Resize to 448x448
        if image.shape[:2] != (448, 448):
            resized = cv2.resize(image, (448, 448))
        else:
            resized = image
            
        # Normalize
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb_image /= 255.0
        rgb_image -= self.mean
        rgb_image /= self.std
        
        # Transpose to [1, 3, 448, 448]
        input_tensor = np.transpose(rgb_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        timings = {'preprocess': (time.time() - t0) * 1000}
        return input_tensor, resized, timings

    def compute_mahalanobis(self, embeddings):
        """
        Versão Ultra-Otimizada para 448x448 (Grelha 56x56)
        """
        # 1. Reshape sem cópia de memória
        B, C, H, W = embeddings.shape # C=100, H=56, W=56
        
        # Transformamos (100, 56, 56) -> (3136, 100)
        flat_embeddings = embeddings.reshape(C, -1).T
        
        # Diferença para a média (Operação vectorizada rápida)
        delta = (flat_embeddings - self.stats_mean) # (3136, 100)

        # 2. Preparar para Batch Matrix Multiplication (Matmul)
        # Adicionamos uma dimensão para o cálculo: (3136, 1, 100)
        delta_batch = delta[:, np.newaxis, :]
        
        # 3. O SEGREDO DA VELOCIDADE: Matmul em vez de Einsum
        # No Pi 5, o matmul aproveita melhor a cache do processador
        # (3136, 1, 100) @ (3136, 100, 100) @ (3136, 100, 1) -> (3136, 1, 1)
        dist_sq = np.matmul(np.matmul(delta_batch, self.stats_inv_cov), 
                            delta_batch.transpose(0, 2, 1))
        
        # 4. Finalização
        dist = np.sqrt(np.maximum(dist_sq.squeeze(), 0))
        return dist.reshape(H, W)

    def infer(self, image):
        start_time = time.time()
        
        # 1. Preprocess
        input_tensor, resized_image, timings = self.preprocess(image)
        
        # 2. Inference (Backbone)
        t_start_inf = time.time()
        results = self.compiled_model([input_tensor])[self.output_layer]
        embeddings = results # (1, 100, 56, 56)
        timings['openvino'] = (time.time() - t_start_inf) * 1000
        
        # 3. Post-Process (Mahalanobis)
        t_start_dist = time.time()
        anomaly_map_small = self.compute_mahalanobis(embeddings)
        timings['mahalanobis'] = (time.time() - t_start_dist) * 1000
        
        # Resize to 448x448
        anomaly_map = cv2.resize(anomaly_map_small, (448, 448))
        
        # Gaussian Blur (Optimized kernel)
        anomaly_map = cv2.GaussianBlur(anomaly_map, (7, 7), 0)
        
        # Score
        score = np.max(anomaly_map)
        
        timings['total'] = (time.time() - start_time) * 1000
        
        return score, anomaly_map, resized_image, timings

    def visualize(self, anomaly_map, bg_image):
        """
        Overlay heatmap on image.
        """
        # Normalize anomaly map to 0-255
        # For PaDiM, distances are unbounded. We should normalize dynamically or using a fixed robust max.
        # Let's use dynamic for now, or fixed if we knew the range (usually 0-50 for Mahalanobis).
        
        # Robust min/max
        am_min = np.min(anomaly_map)
        am_max = np.max(anomaly_map)
        
        div = am_max - am_min
        if div == 0: div = 1.0
        
        am_norm = ((anomaly_map - am_min) / div * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(am_norm, cv2.COLORMAP_JET)
        
        if heatmap.shape[:2] != bg_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (bg_image.shape[1], bg_image.shape[0]))
            
        overlay = cv2.addWeighted(bg_image, 0.6, heatmap, 0.4, 0)
        return overlay

    def predict(self, image):
        score, amap, resized, timings = self.infer(image)
        is_normal = score < self.threshold
        viz = self.visualize(amap, resized)
        return score, is_normal, viz, amap, timings

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python padim_inference.py <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    inferencer = PadimInferencer()
    
    # Warmup
    print("🔥 Warming up...")
    inferencer.infer(np.zeros((448, 448, 3), dtype=np.uint8))
    
    # Run
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image")
        sys.exit(1)
        
    score, amap, rez, times = inferencer.infer(img)
    
    print(f"\nScore: {score:.4f}")
    print(f"Timings: {times}")
    
    viz = inferencer.visualize(amap, rez)
    cv2.imwrite("padim_result.jpg", viz)
    print("Saved padim_result.jpg")
