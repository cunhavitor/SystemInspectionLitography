import cv2
import numpy as np
import os
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
    try:
        # ... (previous code truncated for brevity)
        
        # Step 4: Inspect Each Can
        print(f"  [4/5] Running PatchCore on {len(cans)} cans...")
        
        annotated_sheet = rectified.copy()
        ok_count = 0
        ng_count = 0
        
        for i, item in enumerate(cans[:10]): 
            can_id = item['id']
            can_img = item['image'] # Imagem cortada (BGR Raw)
            bbox = item['bbox']
            
            # 1. ALINHAMENTO
            if aligner:
                aligned_can = aligner.align(can_img)
            else:
                aligned_can = can_img
            
            # 3. INFERÊNCIA
            # Passamos a imagem RAW ALINHADA. O inferencer faz CLAHE + Resize.
            score, is_normal, heatmap = inferencer.predict(aligned_can)
            
            # DEBUG: Ver se o score baixou para valores normais (0 a 5)
            print(f"    DEBUG: Can #{can_id} Score: {score:.4f}")
            
            if is_normal:
                ok_count += 1
                color = (0, 255, 0) # Verde
                status = "OK"
            else:
                ng_count += 1
                color = (0, 0, 255) # Vermelho
                status = "NG"
                
                # Salvar imagem (Salvamos a versão CLAHE para perceber o que o modelo viu)
                defect_filename = f"defects/NOK_{timestamp}_can{can_id}_score{score:.2f}.png"
                cv2.imwrite(defect_filename, img_clahe)

            # Desenhar na folha
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_sheet, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_sheet, f"{status} {score:.1f}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Mostrar
        cv2.imshow("Inspection Result", cv2.resize(annotated_sheet, (1024, 768)))
        print(f"\n✓ Done: {ok_count} OK, {ng_count} NG")
        cv2.waitKey(0)
        
    except Exception as e:
        print(f" ✗ Error: {e}")
        import traceback
        traceback.print_exc()
