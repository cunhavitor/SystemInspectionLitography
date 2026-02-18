import cv2
import glob
import os
import numpy as np
import math

def main():
    # Define paths
    project_root = "/home/cunhav/projects/InspectionVisionCamera"
    image_folder = os.path.join(project_root, "data/raw_sheet_crops")
    ref_path = os.path.join(project_root, "models/can_reference/aligned_can_reference448_bpo-rr125.png")
    mask_path = os.path.join(project_root, "data/can_mask_448x448.png")
    
    # Load reference image
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"ERROR: Could not load reference image: {ref_path}")
        return
    
    # Load can mask
    can_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if can_mask is None:
        print(f"ERROR: Could not load can mask: {mask_path}")
        return
    
    print(f"Reference loaded: {ref_img.shape}")
    print(f"Mask loaded: {can_mask.shape}")
    
    # Garantir que a máscara tem o mesmo tamanho que a referência
    ref_h, ref_w = ref_img.shape[:2]
    if can_mask.shape[0] != ref_h or can_mask.shape[1] != ref_w:
        print(f"Resizing mask from {can_mask.shape} to ({ref_h}, {ref_w})")
        can_mask = cv2.resize(can_mask, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
    
    # Garantir que a máscara é binária (0 ou 255)
    _, can_mask = cv2.threshold(can_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Pre-process reference
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # Criar versão mascarada da referência (para ECC fino)
    ref_masked = cv2.bitwise_and(ref_img, ref_img, mask=can_mask)
    ref_masked_gray = cv2.cvtColor(ref_masked, cv2.COLOR_BGR2GRAY)
    
    # Find images
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not image_files:
        print(f"No PNG images found in {image_folder}")
        return
    print(f"Found {len(image_files)} images.")

    # --- SIFT: Pré-calcular referência (uma só vez) ---
    sift = cv2.SIFT_create(nfeatures=2000)
    kp_ref, des_ref = sift.detectAndCompute(ref_gray, can_mask)
    print(f"Reference SIFT: {len(kp_ref)} keypoints (masked)")
    
    # FLANN matcher (reutilizável)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # --- INICIALIZAÇÃO DA JANELA E TRACKBARS ---
    window_name = "Can Alignment Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def nothing(x): pass
    
    # Trackbars
    cv2.createTrackbar("Canny Low", window_name, 80, 255, nothing)
    cv2.createTrackbar("Canny High", window_name, 151, 255, nothing)
    cv2.createTrackbar("Blur Size", window_name, 3, 15, nothing)
    # View: 0=Original, 1=Coarse Aligned, 2=Masked, 3=Fine Aligned, 4=Overlay, 5=Edges
    cv2.createTrackbar("View", window_name, 0, 5, nothing)

    img_idx = 0

    while True:
        img_path = image_files[img_idx]
        original_img = cv2.imread(img_path)
        
        if original_img is None:
            img_idx = (img_idx + 1) % len(image_files)
            continue

        h_img, w_img = original_img.shape[:2]
        center_x, center_y = w_img // 2, h_img // 2

        # --- Ler Trackbars ---
        canny_low = cv2.getTrackbarPos("Canny Low", window_name)
        canny_high = cv2.getTrackbarPos("Canny High", window_name)
        blur_size = cv2.getTrackbarPos("Blur Size", window_name)
        view_mode = cv2.getTrackbarPos("View", window_name)
        
        if blur_size < 1: blur_size = 1
        if blur_size % 2 == 0: blur_size += 1

        info_lines = []

        # ============================================================
        # STAGE 1: COARSE CENTERING (SIFT Feature Matching + AffinePartial2D)
        # ============================================================
        
        input_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        kp_in, des_in = sift.detectAndCompute(input_gray, can_mask)
        
        coarse_aligned = original_img.copy()
        homography_ok = False
        
        if des_ref is not None and des_in is not None and len(des_ref) > 0 and len(des_in) > 0:
            matches = flann.knnMatch(des_ref, des_in, k=2)
            
            # Lowe's ratio test (0.75 para mais matches)
            good_matches = []
            ratios = []
            for m, n in matches:
                ratio = m.distance / n.distance if n.distance > 0 else 1.0
                ratios.append(ratio)
                if ratio < 0.75:
                    good_matches.append(m)
            
            avg_ratio = np.mean(ratios) if ratios else 1.0
            
            info_lines.append(f"SIFT: {len(kp_ref)} ref, {len(kp_in)} in, {len(good_matches)} good")
            
            print(f"\n--- Image {img_idx+1}: {os.path.basename(img_path)} ---")
            print(f"  SIFT: ref={len(kp_ref)} kp, input={len(kp_in)} kp")
            print(f"  Matches: {len(matches)} total, {len(good_matches)} good (avg ratio={avg_ratio:.3f})")
            
            if len(good_matches) > 10:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_in[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Affine Parcial: só scale uniforme + rotação + translação (4 DOF)
                # Muito mais robusto que Homografia (8 DOF) com poucos matches
                A, inlier_mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                
                if A is not None:
                    inliers = inlier_mask.ravel().sum()
                    inlier_ratio = inliers / len(good_matches)
                    
                    # Decompor affine parcial: A = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
                    a_scale = math.sqrt(A[0,0]**2 + A[1,0]**2)
                    a_tx, a_ty = A[0,2], A[1,2]
                    a_angle = math.degrees(math.atan2(A[1,0], A[0,0]))
                    
                    print(f"  Affine: {inliers}/{len(good_matches)} inliers ({inlier_ratio:.0%})")
                    print(f"  A decomp: scale={a_scale:.3f}, tx={a_tx:.1f}, ty={a_ty:.1f}, angle={a_angle:.1f}°")
                    print(f"  A matrix: {A}")
                    
                    info_lines.append(f"A: {inliers} inl, s={a_scale:.3f}, a={a_angle:.1f}°")
                    
                    # Validação mais apertada (esperamos scale entre 0.85 e 1.25)
                    if a_scale < 0.8 or a_scale > 1.3:
                        print(f"  WARNING: Scale {a_scale:.3f} out of range [0.8, 1.3]! Skipping.")
                        info_lines.append("A: REJECTED (bad scale)")
                    elif inlier_ratio < 0.3:
                        print(f"  WARNING: Low inlier ratio {inlier_ratio:.0%}! Skipping.")
                        info_lines.append("A: REJECTED (low inliers)")
                    else:
                        coarse_aligned = cv2.warpAffine(original_img, A, (w_img, h_img))
                        homography_ok = True
                else:
                    info_lines.append("Homography: FAILED (null)")
                    print("  Homography: FAILED (null matrix)")
            else:
                info_lines.append(f"SIFT: Not enough matches ({len(good_matches)})")
                print(f"  WARNING: Not enough good matches: {len(good_matches)} (need >10)")
                # Mostrar as distâncias dos matches para entender porquê
                if good_matches:
                    dists = [m.distance for m in good_matches]
                    print(f"  Match distances: min={min(dists):.1f}, max={max(dists):.1f}, avg={np.mean(dists):.1f}")
        else:
            info_lines.append("SIFT: No descriptors found")
            print(f"  No descriptors found")
        
        # ============================================================
        # STAGE 2: APPLY MASK (à imagem grosseiramente centrada + escalada)
        # ============================================================
        masked_input = cv2.bitwise_and(coarse_aligned, coarse_aligned, mask=can_mask)
        
        info_lines.append(f"Stage2: Mask Applied")
        
        # ============================================================
        # STAGE 3: FINE ALIGNMENT (ECC na imagem mascarada)
        # ============================================================
        # Agora corremos ECC entre a imagem mascarada e a referência mascarada.
        # Como já estão grosseiramente alinhadas, o ECC converge facilmente.
        
        masked_input_gray = cv2.cvtColor(masked_input, cv2.COLOR_BGR2GRAY)
        
        # Edges para ECC (melhor que raw pixels para alinhar contornos)
        input_edges = cv2.Canny(cv2.GaussianBlur(masked_input_gray, (blur_size, blur_size), 0), 
                                canny_low, canny_high)
        ref_edges = cv2.Canny(cv2.GaussianBlur(ref_masked_gray, (blur_size, blur_size), 0),
                              canny_low, canny_high)
        
        input_edges_f = input_edges.astype(np.float32) / 255.0
        ref_edges_f = ref_edges.astype(np.float32) / 255.0
        
        # MOTION_EUCLIDEAN: corrige translação + rotação + escala residual
        warp_mode = cv2.MOTION_EUCLIDEAN
        fine_warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
        
        try:
            cc, fine_warp = cv2.findTransformECC(
                ref_edges_f, input_edges_f, fine_warp, warp_mode, criteria
            )
            
            fine_dx = fine_warp[0, 2]
            fine_dy = fine_warp[1, 2]
            fine_angle = math.degrees(math.atan2(fine_warp[1, 0], fine_warp[0, 0]))
            
            info_lines.append(f"ECC: dx={fine_dx:.1f} dy={fine_dy:.1f} a={fine_angle:.2f}° cc={cc:.4f}")
            print(f"  ECC fine: dx={fine_dx:.2f}, dy={fine_dy:.2f}, angle={fine_angle:.2f}°, cc={cc:.4f}")
            
            # Aplicar ajuste fino ao coarse_aligned (já transformado pelo SIFT)
            fine_aligned = cv2.warpAffine(coarse_aligned, fine_warp, (w_img, h_img),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
            # Resultado final: alinhado + mascarado
            final_masked = cv2.bitwise_and(fine_aligned, fine_aligned, mask=can_mask)
            
            ecc_ok = True
            
        except cv2.error as e:
            info_lines.append(f"ECC FAILED: {str(e)[:50]}")
            fine_aligned = coarse_aligned.copy()
            final_masked = masked_input.copy()
            ecc_ok = False

        # ============================================================
        # DISPLAY
        # ============================================================
        view_labels = {
            0: "Original", 
            1: "Coarse Aligned", 
            2: "Masked (after coarse)", 
            3: "Fine Aligned + Mask",
            4: "Overlay (Fine vs Ref)",
            5: "Edges (Input vs Ref)"
        }
        
        if view_mode == 0:
            display_img = original_img.copy()
        elif view_mode == 1:
            display_img = coarse_aligned.copy()
        elif view_mode == 2:
            display_img = masked_input.copy()
        elif view_mode == 3:
            display_img = final_masked.copy()
        elif view_mode == 4:
            # Overlay: resultado final vs referência
            display_img = cv2.addWeighted(fine_aligned, 0.5, ref_img, 0.5, 0)
        elif view_mode == 5:
            # Mostrar edges lado a lado (input | ref)
            edges_bgr_in = cv2.cvtColor(input_edges, cv2.COLOR_GRAY2BGR)
            edges_bgr_ref = cv2.cvtColor(ref_edges, cv2.COLOR_GRAY2BGR)
            # Colorir: input=verde, ref=vermelho
            edges_bgr_in[:,:,0] = 0; edges_bgr_in[:,:,2] = 0  # Só verde
            edges_bgr_ref[:,:,0] = 0; edges_bgr_ref[:,:,1] = 0  # Só vermelho
            display_img = cv2.addWeighted(edges_bgr_in, 1.0, edges_bgr_ref, 1.0, 0)
        else:
            display_img = original_img.copy()

        # --- MIRA NO CENTRO ---
        cv2.drawMarker(display_img, (center_x, center_y), (0, 255, 255), 
                       cv2.MARKER_CROSS, 30, 2)

        # --- TEXTO INFO ---
        def put_text(img, text, pos, color=(255,255,255)):
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        img_info = f"Image: {os.path.basename(img_path)} ({img_idx+1}/{len(image_files)})"
        put_text(display_img, img_info, (10, 20))
        
        y_offset = 42
        for line in info_lines:
            if "Stage1" in line: color = (0, 200, 255)
            elif "Stage2" in line: color = (255, 200, 0)
            elif "Stage3" in line: color = (0, 255, 0)
            elif "TOTAL" in line: color = (0, 255, 255)
            elif "FAILED" in line: color = (0, 0, 255)
            else: color = (200, 200, 200)
            put_text(display_img, line, (10, y_offset), color=color)
            y_offset += 20

        mode_text = f"View: {view_labels.get(view_mode, '?')}"
        put_text(display_img, mode_text, (10, y_offset), color=(255, 200, 0))

        h, w = display_img.shape[:2]
        help_text = "'n' Next | 'p' Prev | 'q' Quit"
        put_text(display_img, help_text, (10, h - 10), color=(200, 200, 200))

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('n'):
            img_idx = (img_idx + 1) % len(image_files)
        elif key == ord('p'):
            img_idx = (img_idx - 1) % len(image_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()