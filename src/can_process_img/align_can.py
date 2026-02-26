import cv2
import numpy as np
import os
import math
from datetime import datetime


class CanAligner:
    """Alinha crops de latas à imagem de referência usando SIFT + ECC.
    
    Pipeline:
        1. SIFT Feature Matching → estimateAffinePartial2D (scale + rotate + translate)
        2. Aplicar máscara fixa (can_mask_448x448.png)
        3. ECC EUCLIDEAN fine-tuning (ajuste sub-pixel)
    """

    # Parâmetros fixos (validados interativamente)
    CANNY_LOW = 40
    CANNY_HIGH = 180
    BLUR_SIZE = 7
    SIFT_FEATURES = 2000
    RATIO_TEST = 0.75
    SCALE_MIN = 0.8
    SCALE_MAX = 1.2
    MIN_GOOD_MATCHES = 20
    ECC_ITERATIONS = 200
    ECC_EPSILON = 1e-7
    # Reject ECC warp when cc < this – a bad ECC solution is worse than SIFT alone.
    # Observed: good cans cc ≈ 0.69-0.73 | bad alignment cc ≈ 0.44-0.62
    MIN_ECC_CC = 0.72

    def __init__(self, reference_image_path, target_size=(448, 448)):
        self.target_size = target_size

        # 1. Carregar imagem de referência
        self.ref_img = cv2.imread(reference_image_path)
        if self.ref_img is None:
            raise ValueError(f"Não foi possível carregar a referência: {reference_image_path}")

        self.ref_h, self.ref_w = self.ref_img.shape[:2]
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)

        # 2. Carregar máscara fixa
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        mask_path = os.path.join(project_root, "data", "can_mask_448x448.png")
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise ValueError(f"Não foi possível carregar a máscara: {mask_path}")

        # Garantir tamanho e binarização da máscara
        if self.mask.shape[0] != self.ref_h or self.mask.shape[1] != self.ref_w:
            self.mask = cv2.resize(self.mask, (self.ref_w, self.ref_h), interpolation=cv2.INTER_NEAREST)
        _, self.mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)

        # 3. Pré-calcular SIFT da referência (masked, uma só vez)
        self.sift = cv2.SIFT_create(nfeatures=self.SIFT_FEATURES, contrastThreshold=0.03)
        self.kp_ref, self.des_ref = self.sift.detectAndCompute(self.ref_gray, self.mask)
        # print(f"[Align] Reference SIFT: {len(self.kp_ref)} keypoints (masked)")

        # 4. FLANN matcher (reutilizável)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=30)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 5. Pré-calcular referência para ECC
        # Compute CLAHE on the image to create a rich continuous gradient instead of binary edges
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ref_clahe = clahe.apply(self.ref_gray)
        ref_blurred = cv2.GaussianBlur(ref_clahe, (self.BLUR_SIZE, self.BLUR_SIZE), 0)
        
        self.ecc_mask = self.mask
        
        # Mask the background
        ref_masked = cv2.bitwise_and(ref_blurred, ref_blurred, mask=self.ecc_mask)
        self.ref_edges_f = ref_masked.astype(np.float32) / 255.0

        # print(f"[Align] CanAligner initialized: ref={self.ref_w}x{self.ref_h}, mask loaded")

    def align(self, can_crop):
        """Alinha um crop de lata à referência.
        
        Args:
            can_crop: Imagem BGR do crop da lata
            
        Returns:
            Tuple (aligned_image, is_aligned)
            - aligned_image: Imagem BGR alinhada e mascarada (448x448)
            - is_aligned: True se o alinhamento SIFT foi bem sucedido, False caso contrário
        """
        self.last_align_info = {'sift': None, 'ecc': None, 'ecc_rejected': False}

        if can_crop is None:
            return None, False

        # Resize input → 448x448
        input_resized = cv2.resize(can_crop, (self.ref_w, self.ref_h))

        # =============================================
        # STAGE 1: SIFT + AffinePartial2D (Coarse)
        # =============================================
        coarse_aligned, sift_confidence = self._sift_align(input_resized)
        is_aligned = sift_confidence > 0.0

        # =============================================
        # STAGE 2: Aplicar máscara fixa
        # =============================================
        masked = cv2.bitwise_and(coarse_aligned, coarse_aligned, mask=self.mask)

        # =============================================
        # STAGE 3: ECC EUCLIDEAN Fine-tuning (skip se SIFT já é excelente)
        # =============================================
        if sift_confidence < 0.95:
            fine_aligned = self._ecc_fine_align(coarse_aligned)
        else:
            fine_aligned = coarse_aligned

        # Resultado final: alinhado + mascarado
        result = cv2.bitwise_and(fine_aligned, fine_aligned, mask=self.mask)

        # Resize final se necessário
        if result.shape[:2] != self.target_size:
            result = cv2.resize(result, self.target_size)

        return result, is_aligned

    def _sift_align(self, input_img):
        """Stage 1: Alinhamento grosseiro via SIFT feature matching."""
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # CRITICAL FIX: Unmask the input image so SIFT can find features even if the can is shifted 
        kp_in, des_in = self.sift.detectAndCompute(input_gray, None)

        if self.des_ref is None or des_in is None or len(self.des_ref) == 0 or len(des_in) == 0:
            # print("[Align] SIFT: No descriptors, skipping")
            return input_img.copy(), 0.0

        # FLANN matching
        matches = self.flann.knnMatch(self.des_ref, des_in, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.RATIO_TEST * n.distance:
                good_matches.append(m)

        if len(good_matches) <= self.MIN_GOOD_MATCHES:
            # print(f"[Align] SIFT: {len(good_matches)} matches (<{self.MIN_GOOD_MATCHES}), skipping")
            return input_img.copy(), 0.0

        # Calcular Affine Parcial (4 DOF: scale uniforme + rotação + translação)
        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_in[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        A, inlier_mask = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if A is None:
            # print("[Align] SIFT: Affine failed")
            return input_img.copy(), 0.0

        inliers = inlier_mask.ravel().sum()
        inlier_ratio = inliers / len(good_matches)
        
        a_scale = math.sqrt(A[0, 0] ** 2 + A[1, 0] ** 2)
        a_angle = math.degrees(math.atan2(A[1, 0], A[0, 0]))

        self.last_align_info['sift'] = {
            'matches': len(good_matches),
            'inliers': inliers,
            'inlier_ratio': inlier_ratio,
            'scale': a_scale,
            'angle': a_angle
        }
        # print(f"[Align] SIFT: {len(good_matches)} matches, {inliers} inliers ({inlier_ratio:.0%}), "
        #       f"scale={a_scale:.3f}, angle={a_angle:.1f}°")

        # Validar transformação
        if a_scale < self.SCALE_MIN or a_scale > self.SCALE_MAX:
            # print(f"[Align] SIFT: scale={a_scale:.3f} out of range, skipping")
            return input_img.copy(), 0.0

        if abs(a_angle) > 15.0:
            # print(f"[Align] SIFT: angle={a_angle:.1f}° out of bounds, skipping")
            return input_img.copy(), 0.0

        if inlier_ratio < 0.3:
            # print(f"[Align] SIFT: inliers={inlier_ratio:.0%}, skipping")
            return input_img.copy(), 0.0

        # Aplicar transformação
        # Confiança = inlier_ratio (>0.7 = excelente, skip ECC)
        # Use BORDER_REPLICATE so we don't introduce hard black edges that trigger the anomaly model
        return cv2.warpAffine(
            input_img, A, (self.ref_w, self.ref_h), 
            borderMode=cv2.BORDER_REPLICATE
        ), inlier_ratio

    def _ecc_fine_align(self, coarse_aligned):
        """Stage 3: Ajuste fino via ECC EUCLIDEAN."""
        input_gray = cv2.cvtColor(coarse_aligned, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to create a rich gradient for ECC to climb
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        input_clahe = clahe.apply(input_gray)
        input_blurred = cv2.GaussianBlur(input_clahe, (self.BLUR_SIZE, self.BLUR_SIZE), 0)
        
        # Mask the background out
        input_masked = cv2.bitwise_and(input_blurred, input_blurred, mask=self.ecc_mask)
        
        input_edges_f = input_masked.astype(np.float32) / 255.0

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.ECC_ITERATIONS, self.ECC_EPSILON)

        try:
            # Change to MOTION_AFFINE so ECC doesn't fight SIFT's hallucinated scale
            cc, warp_matrix = cv2.findTransformECC(
                self.ref_edges_f, input_edges_f, warp_matrix,
                cv2.MOTION_AFFINE, criteria # cv2.MOTION_EUCLIDEAN, criteria
            )

            fine_dx = warp_matrix[0, 2]
            fine_dy = warp_matrix[1, 2]
            fine_angle = math.degrees(math.atan2(warp_matrix[1, 0], warp_matrix[0, 0]))

            self.last_align_info['ecc'] = {
                'dx': fine_dx,
                'dy': fine_dy,
                'angle': fine_angle,
                'cc': cc
            }
            # print(f"[Align] ECC: dx={fine_dx:.2f}, dy={fine_dy:.2f}, "
            #       f"angle={fine_angle:.2f}°, cc={cc:.4f}")

            # Quality gate: reject ECC warp if convergence was poor or drift was too large
            if cc < 0.50 or abs(fine_dx) > 5.0 or abs(fine_dy) > 5.0:
                self.last_align_info['ecc_rejected'] = True
                # print(f"[Align] ECC: cc={cc:.4f}, dx={fine_dx:.1f}, dy={fine_dy:.1f} – rejecting ECC, keeping SIFT result")
                return coarse_aligned

            return cv2.warpAffine(
                coarse_aligned, warp_matrix, (self.ref_w, self.ref_h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE
            )

        except cv2.error as e:
            # print(f"[Align] ECC failed: {str(e)[:60]}")
            return coarse_aligned
