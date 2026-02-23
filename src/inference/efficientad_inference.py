import cv2
import numpy as np
import openvino.runtime as ov
import os
import time


class EfficientAdInferencer:
    """
    OpenVINO-based inferencer for EfficientAD anomaly detection models.

    The model outputs an anomaly map directly (result[0, 0]) at its native
    resolution.  Post-processing consists of:
      1. Resize to 448x448
      2. Gaussian blur (noise suppression)
      3. Mask (zero-out pixels outside the can)
      4. Peak score computation

    API is identical to PatchCoreInferencer / PadimInferencer so that
    inspection_window.py requires no changes beyond swapping the inferencer
    instance:
        score, is_normal, viz, heatmap, timings = inferencer.predict(image)
    """

    # ------------------------------------------------------------------ #
    #  Constructor                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        model_dir: str = "models/bpo_rr125_efficientAD_M",
        device: str = "CPU",
        threshold: float = 0.5,
    ):
        self.model_dir = model_dir
        self.device = device
        self.threshold = threshold

        # Set True to print per-stage statistics for every can (useful for debugging
        # score variability between cans). Set False in production.
        self.DEBUG_INFER = True

        # ImageNet normalisation (matching training pre-processing)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # ---- Load OpenVINO model ----------------------------------------
        print(f"📦 Loading EfficientAD model from '{model_dir}'...")
        model_xml = os.path.join(model_dir, "model.xml")

        if not os.path.exists(model_xml):
            raise FileNotFoundError(f"Model XML not found: {model_xml}")

        core = ov.Core()
        # Enable caching for faster cold-start on subsequent runs
        core.set_property({"CACHE_DIR": os.path.join(model_dir, "cache")})

        model = core.read_model(model=model_xml)

        # Raspberry Pi 5 / embedded CPU optimisation
        config = {
            "INFERENCE_PRECISION_HINT": "f32",  # Fix for EfficientAD fp16 spikes on ARM
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

        # ---- Can mask (optional) ----------------------------------------
        # Use the mask that ships WITH the model (same one used in Colab reference code)
        mask_path = os.path.join(model_dir, "mask.png")
        # Fallback to legacy global mask
        if not os.path.exists(mask_path):
            mask_path = "data/can_mask_448x448.png"
        
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # CRITICAL: Always resize the mask to match the 448x448 model output
            mask_img = cv2.resize(mask_img, (448, 448))
            
            # Erode the mask inwards to ignore the outermost 5% of the radius 
            # (224px radius * 5% = ~12px per side).
            # A 25x25 kernel erodes exactly 12 pixels from every direction.
            erode_kernel = np.ones((25, 25), np.uint8)
            mask_img = cv2.erode(mask_img, erode_kernel, iterations=1)
            
            # Float mask in [0, 1] so arith. multiplication suppresses background
            self.can_mask = mask_img.astype(np.float32) / 255.0
            print(f"✅ Can mask loaded from '{mask_path}' and eroded inwards by 5%.")
        else:
            self.can_mask = None
            print(f"⚠️  No mask found in '{model_dir}' or 'data/'. Inspecting full image.")

    # ------------------------------------------------------------------ #
    #  Pre-processing                                                      #
    # ------------------------------------------------------------------ #

    def preprocess(self, image: np.ndarray):
        """BGR uint8 image -> normalised (1, 3, 448, 448) float32 tensor."""
        t0 = time.time()

        # Resize to 448x448 (model input)
        # The Colab simulation script explicitly uses INTER_AREA for all resizing
        # to ensure consistency between x86 and ARM.
        if image.shape[:2] != (448, 448):
            resized = cv2.resize(image, (448, 448), interpolation=cv2.INTER_AREA)
        else:
            resized = image.copy()

        t_resize = time.time()
        
        print(f"DEBUG PRE-PROCESS - Resized Image: Min={resized.min()}, Max={resized.max()}, Mean={resized.mean():.4f}")

        # BGR -> RGB, [0, 1], ImageNet normalisation
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb /= 255.0
        rgb -= self.mean
        rgb /= self.std

        # HWC -> NCHW
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 448, 448)

        t_norm = time.time()
        print(f"DEBUG PRE-PROCESS - Tensor: Min={tensor.min():.4f}, Max={tensor.max():.4f}, Mean={tensor.mean():.4f}")

        timings = {
            "resize": (t_resize - t0) * 1000,
            "norm":   (t_norm - t_resize) * 1000,
        }
        return tensor, resized, timings

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def infer(self, image: np.ndarray, can_id: int = -1):
        """
        Full forward pass.

        Returns
        -------
        score        : float  – peak anomaly score (after mask)
        anomaly_map  : ndarray (448, 448) float32 – cleaned anomaly map
        resized_img  : ndarray (448, 448, 3) uint8 – preprocessed BGR image
        timings      : dict   – per-stage millisecond timings
        """
        t_start = time.time()
        tag = f"[CAN {can_id}]" if can_id >= 0 else "[DBG]"

        # ---- Stage 0: Input image ----------------------------------------
        if self.DEBUG_INFER:
            print(f"{tag} INPUT  shape={image.shape} dtype={image.dtype} "
                  f"mean={image.mean():.2f} min={image.min()} max={image.max()}")

        # 1. Preprocess
        tensor, resized_img, timings = self.preprocess(image)

        if self.DEBUG_INFER:
            print(f"{tag} TENSOR shape={tensor.shape} "
                  f"mean={tensor.mean():.4f} std={tensor.std():.4f} "
                  f"min={tensor.min():.4f} max={tensor.max():.4f}")

        # 2. OpenVINO inference
        t_ov = time.time()
        result      = self.compiled_model([tensor])[self.output_layer]
        anomaly_map = result[0, 0]
        timings["openvino"] = (time.time() - t_ov) * 1000
        
        print(f"DEBUG INFER - RAW_MAP: Min={anomaly_map.min():.6f}, Max={anomaly_map.max():.6f}, Mean={anomaly_map.mean():.6f}")

        if self.DEBUG_INFER:
            print(f"{tag} RAW_MAP shape={anomaly_map.shape} "
                  f"mean={anomaly_map.mean():.4f} max={anomaly_map.max():.4f} "
                  f"min={anomaly_map.min():.4f} std={anomaly_map.std():.4f}")

        # 3. Post-processing
        # a) Resize anomaly map to display resolution
        anomaly_map = cv2.resize(anomaly_map, (448, 448), interpolation=cv2.INTER_LINEAR)
        print(f"DEBUG POST-PROCESS - After Resize: Min={anomaly_map.min():.6f}, Max={anomaly_map.max():.6f}, Mean={anomaly_map.mean():.6f}")

        if self.DEBUG_INFER:
            print(f"{tag} AFTER_RESIZE max={anomaly_map.max():.4f} mean={anomaly_map.mean():.4f}")

        # Static Noise Floor:
        # ResNet18 creates a baseline of structural "ghosts" on the can body.
        # Zero out anything below this floor to create a clean background.
        # We SUBTRACT the floor from the survivors so that a pixel of 0.81 becomes a faint 0.01 
        # instead of jumping abruptly on the heatmap at full 0.81 force.
        noise_floor = 0.8
        anomaly_map = np.where(anomaly_map < noise_floor, 0, anomaly_map - noise_floor)

        # d) Gaussian blur – suppress isolated noise pixels
        anomaly_map = cv2.GaussianBlur(anomaly_map, (11, 11), 0)
        print(f"DEBUG POST-PROCESS - After Blur: Min={anomaly_map.min():.6f}, Max={anomaly_map.max():.6f}, Mean={anomaly_map.mean():.6f}")

        if self.DEBUG_INFER:
            print(f"{tag} AFTER_BLUR  max={anomaly_map.max():.4f} mean={anomaly_map.mean():.4f}")

        # c) Apply can mask (zero out everything outside the can)
        if self.can_mask is not None:
            anomaly_map *= self.can_mask

            print(f"DEBUG POST-PROCESS - After Mask: Min={anomaly_map.min():.6f}, Max={anomaly_map.max():.6f}, Mean={anomaly_map.mean():.6f}")

            if self.DEBUG_INFER:
                print(f"{tag} AFTER_MASK  max={anomaly_map.max():.4f} mean={anomaly_map.mean():.4f} "
                      f"mask_coverage={self.can_mask.mean()*100:.1f}%")

        # 5. Amp up the score/heat!
        # Multiply the final processed anomaly map by a factor to make defects "hotter"
        AMPLIFY_FACTOR = 2.5
        anomaly_map *= AMPLIFY_FACTOR

        # 6. Peak score — use the average of the top N pixels.
        # np.max is too sensitive to 1-pixel jitter from alignment interpolation, causing
        # the same physical defect to score 1.6 in one frame and 2.4 in the exact next frame.
        # Averaging the top 30 pixels provides a robust score for small 50-pixel defects
        # while absorbing any single pixel anomalies caused by digital noise.
        flat_map = anomaly_map.flatten()
        top_pixels = np.sort(flat_map)[-30:]
        score = float(np.mean(top_pixels))
        
        # No Pi e no Colab
        print(f"DEBUG - Map Stats: Min={anomaly_map.min():.6f}, Max={anomaly_map.max():.6f}, Mean={anomaly_map.mean():.6f}")
        
        if self.DEBUG_INFER:
            print(f"{tag} FINAL SCORE = {score:.4f}  "
                  f"[max={float(np.max(anomaly_map)):.4f}, p99.9={score:.4f}]  "
                  f"(threshold={self.threshold})")

        timings["total_infer"] = (time.time() - t_start) * 1000

        return score, anomaly_map, resized_img, timings


    # ------------------------------------------------------------------ #
    #  Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def visualize(self, anomaly_map: np.ndarray, bg_image: np.ndarray) -> np.ndarray:
        """
        Overlay a JET heatmap on *bg_image*.

        The normalisation uses a fixed ceiling so the colour scale is stable
        between frames (no per-frame flicker).
        """
        # Fixed ceiling: values above it go full-red.
        # Adjust together with threshold for visual balance. 
        # Making the ceiling slightly lower relative to the threshold makes defects hotter.
        ceiling = max(self.threshold * 1.5, 1.0)

        heatmap_u8 = np.clip(anomaly_map / ceiling * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        if heatmap_bgr.shape[:2] != bg_image.shape[:2]:
            heatmap_bgr = cv2.resize(
                heatmap_bgr, (bg_image.shape[1], bg_image.shape[0])
            )

        overlay = cv2.addWeighted(bg_image, 0.6, heatmap_bgr, 0.4, 0)
        return overlay

    # ------------------------------------------------------------------ #
    #  Public API (matches PatchCoreInferencer / PadimInferencer)         #
    # ------------------------------------------------------------------ #

    def predict(self, image: np.ndarray, can_id: int = -1):
        """
        Parameters
        ----------
        image  : ndarray (any size, BGR uint8)
        can_id : int, optional – printed in debug output to identify which can

        Returns
        -------
        score      : float
        is_normal  : bool   (True = OK, False = NG)
        viz        : ndarray – heatmap overlay (448×448 BGR)
        heatmap    : ndarray – raw anomaly map (448×448 float32)
        timings    : dict
        """
        score, anomaly_map, resized, timings = self.infer(image, can_id=can_id)
        is_normal = score < self.threshold
        viz = self.visualize(anomaly_map, resized)
        return score, is_normal, viz, anomaly_map, timings


# --------------------------------------------------------------------------- #
#  CLI test                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python efficientad_inference.py <image_path> [threshold]")
        sys.exit(1)

    img_path  = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    inferencer = EfficientAdInferencer(threshold=threshold)

    # Warmup (important on Pi – JIT / cache warm-up)
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
