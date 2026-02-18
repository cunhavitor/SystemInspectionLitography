import cv2
import numpy as np
import os
from datetime import datetime
from .camera import Camera
from .can_process_img.detect_corner import CornerDetector
from .can_process_img.rectified_sheet import SheetRectifier
from .can_process_img.crop_cans import CanCropper
from .can_process_img.align_can import CanAligner
from .inference.patchcore_inference_v2 import PatchCoreInferencer


def apply_clahe(image):
    """
    Applies CLAHE to the L channel of the LAB image.
    Standard optimization for local contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def run_inspection_mode(config):
    print("Starting Inspection Mode...")
    print("Press 'SPACE' to inspect current frame")
    print("Press 'q' to quit.")
    
    # Initialize Camera
    cam = Camera(
        camera_index=config['camera']['index'],
        width=config['camera']['width'],
        height=config['camera']['height'],
        fps=config['camera']['fps']
    )
    
    # ===== INITIALIZE INSPECTION PIPELINE (ONCE) =====
    print("\n=== Initializing Inspection Pipeline ===")
    
    # 1. Corner Detector
    detector = CornerDetector()
    detector.load_params()
    print("✓ Corner Detector loaded")
    
    # 2. Sheet Rectifier
    rectifier = SheetRectifier()
    rectifier.load_params()
    print("✓ Sheet Rectifier loaded")
    
    # 3. Can Cropper
    cropper = CanCropper()
    cropper.load_params()
    print("✓ Can Cropper loaded")
    
    # 4. Can Aligner
    aligner = None
    ref_path = 'models/can_reference/aligned_can_reference448.png'
    if os.path.exists(ref_path):
        try:
            # FIX: Set target_size to 448x448 to match PatchCore model
            aligner = CanAligner(ref_path, target_size=(448, 448))
            print(f"✓ Can Aligner loaded with reference: {ref_path}")
        except Exception as e:
            print(f"⚠ Failed to load aligner: {e}")
    else:
        print(f"⚠ Reference image not found at {ref_path}")
    
    # 5. PatchCore Inferencer
    try:
        # Load PatchCore model (OpenVINO + Bias Map + Coreset automatically handled)
        inferencer = PatchCoreInferencer()
        print(f"✓ PatchCore Inferencer loaded. Threshold: {inferencer.threshold}")
    except Exception as e:
        print(f"✗ Failed to load PatchCore: {e}")
        print("Inspection will not work without model!")
        cam.release()
        return
    
    print(f"\n=== Pipeline Ready | Threshold: {inferencer.threshold:.1f} ===\n")
    
    try:
        frame_count = 0
        
        while True:
            frame = cam.get_frame(stream_name='main') # Use high res stream
            if frame is None:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            
            # Display live feed with overlay
            display_frame = frame.copy()
            cv2.putText(display_frame, "INSPECTION MODE - Press SPACE to inspect", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Inspection Mode", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # SPACE key - trigger inspection
                print(f"\n=== INSPECTING FRAME {frame_count} ===")
                inspect_frame(frame, detector, rectifier, cropper, aligner, inferencer)
                
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cam.release()
        cv2.destroyAllWindows()


def inspect_frame(frame, detector, rectifier, cropper, aligner, inferencer):
    """
    Perform full inspection pipeline on a single frame.
    """
    import time
    
    t_trigger = time.time()
    timings = {}

    try:
        # Step 1: Detect Corners
        print("  [1/5] Detecting Corners...")
        t0 = time.time()
        corners = detector.detect(frame)
        timings['detect_corners'] = (time.time() - t0) * 1000

        if corners is None:
            print(" ⚠ No corners found.")
            return

        # Step 2: Rectify Sheet
        print("  [2/5] Rectifying Sheet...")
        t0 = time.time()
        rectified, pixels_per_mm = rectifier.rectify(frame, corners)
        timings['rectify'] = (time.time() - t0) * 1000
        
        if rectified is None:
             print(" ⚠ Rectification failed.")
             return
             
        # Debug dynamic scale
        if isinstance(pixels_per_mm, tuple):
             print(f"  [2.1] Dynamic Scale: X={pixels_per_mm[0]:.2f}, Y={pixels_per_mm[1]:.2f} px/mm")
        else:
             print(f"  [2.1] Dynamic Scale: {pixels_per_mm:.2f} px/mm")

        # Step 3: Crop Cans
        print(f"  [3/5] Cropping Cans...")
        t0 = time.time()
        cans = cropper.crop(rectified, pixels_per_mm=pixels_per_mm)
        timings['crop'] = (time.time() - t0) * 1000

        if not cans:
            print(" ⚠ No cans found.")
            return

        # Step 4: Inspect Each Can
        print(f"  [4/5] Running PatchCore on {len(cans)} cans...")
        
        annotated_sheet = rectified.copy()
        ok_count = 0
        ng_count = 0
        
        t_cans_start = time.time()
        
        # Accumulators for average
        acc_align = 0
        acc_infer = 0
        acc_clahe = 0
        acc_resize = 0
        acc_norm = 0
        acc_ov = 0

        for i, item in enumerate(cans): 
            can_id = item['id']
            can_img = item['image'] # Imagem cortada (BGR Raw)
            bbox = item['bbox']
            
            # 1. ALINHAMENTO
            t_align_0 = time.time()
            if aligner:
                aligned_can = aligner.align(can_img)
            else:
                aligned_can = can_img
            acc_align += (time.time() - t_align_0) * 1000
            
            # 3. INFERÊNCIA
            # Passamos a imagem RAW ALINHADA. O inferencer faz CLAHE + Resize.
            score, is_normal, heatmap, clean_map, infer_metrics = inferencer.predict(aligned_can)
            
            acc_infer += infer_metrics['total_infer']
            acc_clahe += infer_metrics['clahe']
            acc_resize += infer_metrics['resize']
            acc_norm += infer_metrics['norm']
            acc_ov += infer_metrics['openvino']
            
            # DEBUG: Ver se o score baixou para valores normais (0 a 5)
            # print(f"    DEBUG: Can #{can_id} Score: {score:.4f}")
            
            if is_normal:
                ok_count += 1
                color = (0, 255, 0) # Verde
                status = "OK"
            else:
                ng_count += 1
                color = (0, 0, 255) # Vermelho
                status = "NG"
                
                # Salvar imagem (Salvamos a versão CLAHE para perceber o que o modelo viu)
                # Create detailed timestamp structure
                now = datetime.now()
                year_dir = now.strftime("%Y")
                month_dir = now.strftime("%m")
                save_dir = os.path.join("data", "defects", year_dir, month_dir)
                os.makedirs(save_dir, exist_ok=True)
                
                # Re-generate CLAHE for visualization save (slightly inefficient but safe)
                img_clahe = apply_clahe(aligned_can) if aligner else apply_clahe(can_img)

                timestamp = now.strftime("%Y%m%d_%H%M%S")
                defect_filename = os.path.join(save_dir, f"NOK_{timestamp}_can{can_id}_score{score:.2f}.png")
                cv2.imwrite(defect_filename, img_clahe)

            # Desenhar na folha
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_sheet, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_sheet, f"{status} {score:.1f}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        num_cans = len(cans)
        timings['total_cans'] = (time.time() - t_cans_start) * 1000
        timings['total_pipeline'] = (time.time() - t_trigger) * 1000
        
        # Mostrar
        cv2.imshow("Inspection Result", cv2.resize(annotated_sheet, (1024, 768)))
        print(f"\n✓ Done: {ok_count} OK, {ng_count} NG")
        
        # PRINT TIMINGS REPORT
        print("\n" + "="*50)
        print(f" PERFORMANCE REPORT ({num_cans} cans)")
        print("="*50)
        print(f" TOTAL TIME:         {timings['total_pipeline']:.1f} ms  ({timings['total_pipeline']/1000:.2f} s)")
        print("-" * 50)
        print(f" 1. Detect Corners:  {timings['detect_corners']:.1f} ms")
        print(f" 2. Rectify Sheet:   {timings['rectify']:.1f} ms")
        print(f" 3. Crop Cans:       {timings['crop']:.1f} ms")
        print("-" * 50)
        print(f" 4. Processing Loop: {timings['total_cans']:.1f} ms")
        if num_cans > 0:
            print(f"    - Avg Per Can:   {timings['total_cans']/num_cans:.1f} ms")
            print(f"    - Avg Align:     {acc_align/num_cans:.1f} ms")
            print(f"    - Avg Infer:     {acc_infer/num_cans:.1f} ms")
            print(f"        * CLAHE:     {acc_clahe/num_cans:.1f} ms")
            print(f"        * Resize:    {acc_resize/num_cans:.1f} ms")
            print(f"        * Norm:      {acc_norm/num_cans:.1f} ms")
            print(f"        * OpenVINO:  {acc_ov/num_cans:.1f} ms")
        print("="*50 + "\n")

        cv2.waitKey(0)
        
    except Exception as e:
        print(f" ✗ Error: {e}")
        import traceback
        traceback.print_exc()
