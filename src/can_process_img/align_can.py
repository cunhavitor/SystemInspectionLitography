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
    CANNY_LOW = 25
    CANNY_HIGH = 121
    BLUR_SIZE = 1
    SIFT_FEATURES = 1000
    RATIO_TEST = 0.75
    SCALE_MIN = 0.8
    SCALE_MAX = 1.3
    MIN_GOOD_MATCHES = 10
    ECC_ITERATIONS = 100
    ECC_EPSILON = 1e-7

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
        self.sift = cv2.SIFT_create(nfeatures=self.SIFT_FEATURES)
        self.kp_ref, self.des_ref = self.sift.detectAndCompute(self.ref_gray, self.mask)
        print(f"[Align] Reference SIFT: {len(self.kp_ref)} keypoints (masked)")

        # 4. FLANN matcher (reutilizável)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=30)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 5. Pré-calcular referência mascarada para ECC
        ref_masked = cv2.bitwise_and(self.ref_img, self.ref_img, mask=self.mask)
        ref_masked_gray = cv2.cvtColor(ref_masked, cv2.COLOR_BGR2GRAY)
        ref_edges = cv2.Canny(ref_masked_gray, self.CANNY_LOW, self.CANNY_HIGH)
        self.ref_edges_f = ref_edges.astype(np.float32) / 255.0

        print(f"[Align] CanAligner initialized: ref={self.ref_w}x{self.ref_h}, mask loaded")

    def align(self, can_crop):
        """Alinha um crop de lata à referência.
        
        Args:
            can_crop: Imagem BGR do crop da lata
            
        Returns:
            Imagem BGR alinhada e mascarada (448x448)
        """
        if can_crop is None:
            return None

        # Resize input → 448x448
        input_resized = cv2.resize(can_crop, (self.ref_w, self.ref_h))

        # =============================================
        # STAGE 1: SIFT + AffinePartial2D (Coarse)
        # =============================================
        coarse_aligned, sift_confidence = self._sift_align(input_resized)

        # =============================================
        # STAGE 2: Aplicar máscara fixa
        # =============================================
        masked = cv2.bitwise_and(coarse_aligned, coarse_aligned, mask=self.mask)

        # =============================================
        # STAGE 3: ECC EUCLIDEAN Fine-tuning (skip se SIFT já é excelente)
        # =============================================
        if sift_confidence < 0.7:
            fine_aligned = self._ecc_fine_align(coarse_aligned, masked)
        else:
            fine_aligned = coarse_aligned

        # Resultado final: alinhado + mascarado
        result = cv2.bitwise_and(fine_aligned, fine_aligned, mask=self.mask)

        # Resize final se necessário
        if result.shape[:2] != self.target_size:
            result = cv2.resize(result, self.target_size)

        return result

    def _sift_align(self, input_img):
        """Stage 1: Alinhamento grosseiro via SIFT feature matching."""
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        kp_in, des_in = self.sift.detectAndCompute(input_gray, self.mask)

        if self.des_ref is None or des_in is None or len(self.des_ref) == 0 or len(des_in) == 0:
            print("[Align] SIFT: No descriptors, skipping")
            return input_img.copy(), 0.0

        # FLANN matching
        matches = self.flann.knnMatch(self.des_ref, des_in, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.RATIO_TEST * n.distance:
                good_matches.append(m)

        if len(good_matches) <= self.MIN_GOOD_MATCHES:
            print(f"[Align] SIFT: {len(good_matches)} matches (<{self.MIN_GOOD_MATCHES}), skipping")
            return input_img.copy(), 0.0

        # Calcular Affine Parcial (4 DOF: scale uniforme + rotação + translação)
        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_in[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        A, inlier_mask = cv2.estimateAffinePartial2D(
            dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if A is None:
            print("[Align] SIFT: Affine failed")
            return input_img.copy(), 0.0

        inliers = inlier_mask.ravel().sum()
        inlier_ratio = inliers / len(good_matches)
        a_scale = math.sqrt(A[0, 0] ** 2 + A[1, 0] ** 2)
        a_angle = math.degrees(math.atan2(A[1, 0], A[0, 0]))

        print(f"[Align] SIFT: {len(good_matches)} matches, {inliers} inliers ({inlier_ratio:.0%}), "
              f"scale={a_scale:.3f}, angle={a_angle:.1f}°")

        # Validar transformação
        if a_scale < self.SCALE_MIN or a_scale > self.SCALE_MAX:
            print(f"[Align] SIFT: scale={a_scale:.3f} out of range, skipping")
            return input_img.copy(), 0.0

        if inlier_ratio < 0.3:
            print(f"[Align] SIFT: inliers={inlier_ratio:.0%}, skipping")
            return input_img.copy(), 0.0

        # Aplicar transformação
        # Confiança = inlier_ratio (>0.7 = excelente, skip ECC)
        return cv2.warpAffine(input_img, A, (self.ref_w, self.ref_h)), inlier_ratio

    def _ecc_fine_align(self, coarse_aligned, masked_input):
        """Stage 3: Ajuste fino via ECC EUCLIDEAN."""
        masked_gray = cv2.cvtColor(masked_input, cv2.COLOR_BGR2GRAY)

        # Canny edges
        input_edges = cv2.Canny(masked_gray, self.CANNY_LOW, self.CANNY_HIGH)
        input_edges_f = input_edges.astype(np.float32) / 255.0

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.ECC_ITERATIONS, self.ECC_EPSILON)

        try:
            cc, warp_matrix = cv2.findTransformECC(
                self.ref_edges_f, input_edges_f, warp_matrix,
                cv2.MOTION_EUCLIDEAN, criteria
            )

            fine_dx = warp_matrix[0, 2]
            fine_dy = warp_matrix[1, 2]
            fine_angle = math.degrees(math.atan2(warp_matrix[1, 0], warp_matrix[0, 0]))

            print(f"[Align] ECC: dx={fine_dx:.2f}, dy={fine_dy:.2f}, "
                  f"angle={fine_angle:.2f}°, cc={cc:.4f}")

            return cv2.warpAffine(
                coarse_aligned, warp_matrix, (self.ref_w, self.ref_h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )

        except cv2.error as e:
            print(f"[Align] ECC failed: {str(e)[:60]}")
            return coarse_aligned
