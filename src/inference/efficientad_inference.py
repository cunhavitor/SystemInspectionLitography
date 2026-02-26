import cv2
import numpy as np
import openvino.runtime as ov
import os
import time


class EfficientAdInferencer:
    """
    OpenVINO-based inferencer for EfficientAD anomaly detection models.
    """

    def __init__(
        self,
        model_dir: str = "models/bpo_rr125_efficientAD_M",
        device: str = "CPU",
        threshold: float = 0.5,
    ):
        self.model_dir = model_dir
        self.device = device
        self.threshold = threshold

        # ------------------------------------------------------------------ #
        # Configurações de Post-Processing (Partilhadas entre score e visualização)
        # ------------------------------------------------------------------ #
        self.noise_floor = 0.08
        self.amplify_factor = 3.0
        self.power_factor = 3.0 # Fator exponencial para suprimir ruído (x^3)

        self.DEBUG_INFER = True

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print(f"📦 Loading EfficientAD model from '{model_dir}'...")
        model_xml = os.path.join(model_dir, "model.xml")

        if not os.path.exists(model_xml):
            raise FileNotFoundError(f"Model XML not found: {model_xml}")

        core = ov.Core()
        core.set_property({"CACHE_DIR": os.path.join(model_dir, "cache")})
        model = core.read_model(model=model_xml)

        config = {
            "INFERENCE_PRECISION_HINT": "f32",
            "INFERENCE_NUM_THREADS": "4",
            "NUM_STREAMS": "1",
            "PERFORMANCE_HINT": "LATENCY",
        }
        self.compiled_model = core.compile_model(
            model=model, device_name=device, config=config
        )
        self.input_layer  = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        print(
            f"✅ EfficientAD compiled | "
            f"input={self.input_layer.partial_shape} | "
            f"output={self.output_layer.partial_shape}"
        )

        mask_path = os.path.join(model_dir, "mask.png")
        if not os.path.exists(mask_path):
            mask_path = "data/can_mask_448x448.png"
        
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (448, 448))
            erode_kernel = np.ones((13, 13), np.uint8)
            mask_img = cv2.erode(mask_img, erode_kernel, iterations=1)
            self.can_mask = mask_img.astype(np.float32) / 255.0
            print(f"✅ Can mask loaded from '{mask_path}' and eroded inwards by 1%.")
        else:
            self.can_mask = None
            print(f"⚠️  No mask found in '{model_dir}' or 'data/'. Inspecting full image.")

    def preprocess(self, image: np.ndarray):
        t0 = time.time()
        if image.shape[:2] != (448, 448):
            resized = cv2.resize(image, (448, 448), interpolation=cv2.INTER_AREA)
        else:
            resized = image.copy()
        t_resize = time.time()
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb /= 255.0
        rgb -= self.mean
        rgb /= self.std
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...] 

        t_norm = time.time()
        timings = {
            "resize": (t_resize - t0) * 1000,
            "norm":   (t_norm - t_resize) * 1000,
        }
        return tensor, resized, timings

    def infer(self, image: np.ndarray, can_id: int = -1, align_info: dict = None):
        t_start = time.time()
        tag = f"[CAN {can_id}]" if can_id >= 0 else "[DBG]"

        tensor, resized_img, timings = self.preprocess(image)

        t_ov = time.time()
        result      = self.compiled_model([tensor])[self.output_layer]
        anomaly_map = result[0, 0]
        timings["openvino"] = (time.time() - t_ov) * 1000

        # ---------------------------------------------------------
        # A) SCORE CALCULATION ON ORIGINAL (SMALL) MAP
        # ---------------------------------------------------------
        score_map = np.copy(anomaly_map)
        
        # 1. Static Noise Floor
        score_map = np.where(score_map < self.noise_floor, 0, score_map - self.noise_floor)
        
        # 2. Apply Mask
        if self.can_mask is not None:
            h_sm, w_sm = score_map.shape[:2]
            small_mask = cv2.resize(self.can_mask, (w_sm, h_sm), interpolation=cv2.INTER_AREA)
            score_map *= small_mask

        # 3. Gaussian Blur (Aglomerar densidade)
        blurred_score_map = cv2.GaussianBlur(score_map, (7, 7), 0)
        
        # 4. Amplificação Exponencial (Alarga o fosso entre ruído e defeito)
        powered_score_map = blurred_score_map ** self.power_factor
        
        # O score agora deriva diretamente do mapa elevado à potência.
        score = float(np.max(powered_score_map))

        # ---------------------------------------------------------
        # B) VISUALIZATION MAP PREPARATION
        # ---------------------------------------------------------
        # Redimensiona o mapa QUE GEROU O SCORE (powered_score_map) para 448x448
        vis_map = cv2.resize(powered_score_map, (448, 448), interpolation=cv2.INTER_LINEAR)

        # Multiplica pelo fator linear apenas para facilitar a visualização (cores)
        vis_map *= self.amplify_factor
        anomaly_map = vis_map

        align_dbg = ""
        if align_info:
            if align_info.get('sift'):
                s = align_info['sift']
                align_dbg += f" | SIFT Inliers: {s['inliers']}"
            if align_info.get('ecc'):
                e = align_info['ecc']
                rej = " [REJ]" if align_info.get('ecc_rejected') else ""
                align_dbg += f" | ECC CC: {e['cc']:.4f}{rej}"

        print(f"{tag} AD Score: {score:.4f}{align_dbg}")
        timings["total_infer"] = (time.time() - t_start) * 1000

        return score, anomaly_map, resized_img, timings

    def visualize(self, anomaly_map: np.ndarray, bg_image: np.ndarray) -> np.ndarray:
        # ---------------------------------------------------------
        # Ajuste do Color Ceiling para Sincronização
        # ---------------------------------------------------------
        # Como o vis_map foi multiplicado por self.amplify_factor, temos de subir o
        # teto proporcionalmente para que as cores não saturem precocemente.
        adjusted_threshold = self.threshold * self.amplify_factor
        ceiling = max(adjusted_threshold * 1.5, 1.0)

        heatmap_u8 = np.clip(anomaly_map / ceiling * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        if heatmap_bgr.shape[:2] != bg_image.shape[:2]:
            heatmap_bgr = cv2.resize(
                heatmap_bgr, (bg_image.shape[1], bg_image.shape[0])
            )

        overlay = cv2.addWeighted(bg_image, 0.6, heatmap_bgr, 0.4, 0)
        return overlay

    def predict(self, image: np.ndarray, can_id: int = -1, align_info: dict = None):
        score, anomaly_map, resized, timings = self.infer(image, can_id=can_id, align_info=align_info)
        is_normal = score < self.threshold
        viz = self.visualize(anomaly_map, resized)
        return score, is_normal, viz, anomaly_map, timings


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python efficientad_inference.py <image_path> [threshold]")
        sys.exit(1)

    img_path  = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    inferencer = EfficientAdInferencer(threshold=threshold)

    print("🔥 Warming up...", end="", flush=True)
    inferencer.infer(np.zeros((448, 448, 3), dtype=np.uint8))
    print(" Done.")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        sys.exit(1)

    score, is_normal, viz, amap, times = inferencer.predict(img)

    status = "✅ OK" if is_normal else "❌ NG"
    print(f"\nResult : {status}")
    print(f"Score  : {score:.4f}  (threshold={threshold})")
    print(f"Timings (ms): resize={times['resize']:.1f}  norm={times['norm']:.1f}  "
          f"openvino={times['openvino']:.1f}  total={times['total_infer']:.1f}")

    out_path = "efficientad_result.jpg"
    cv2.imwrite(out_path, viz)
    print(f"Saved  : {out_path}")