"""
Reference-Based Inspector Module (Golden Image Comparison)
============================================================
Aligns a test image to a known-good "Golden Image" using ORB feature
matching + Homography (or Phase Correlation fallback), then computes
per-pixel SSIM to detect any structural deviation.

Algorithm:
1. Load golden (reference) and test images, resize to processing width.
2. ORB keypoint matching → RANSAC Homography for sub-pixel alignment.
   If ORB fails (< 10 matches), fall back to Phase Correlation.
3. Compute SSIM map between aligned reference and test image.
4. Threshold SSIM map (low-SSIM regions = defects) using Sauvola local
   thresholding for robustness against uneven lighting.
5. Connected-component extraction → defect list.

Detects: Any deviation — stains, holes, missing threads, crooked seams,
         puckers — without needing to tune Gabor/FFT parameters.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_sauvola


class ReferenceInspector:
    """Detects defects by comparing a test image against a golden reference."""

    PROCESS_WIDTH = 800
    MIN_ORB_MATCHES = 10
    SSIM_WIN_SIZE = 11

    def __init__(self):
        self.defects: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────
    # Pre-processing
    # ──────────────────────────────────────────
    def _decode_and_resize(
        self, img_buffer: BinaryIO
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Decode image, resize to PROCESS_WIDTH, return (original, gray, scale)."""
        if hasattr(img_buffer, "seek"):
            img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image file")

        h, w = img.shape[:2]
        scale = self.PROCESS_WIDTH / w
        target_h = int(h * scale)
        img_small = cv2.resize(img, (self.PROCESS_WIDTH, target_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        # CLAHE illumination correction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_gray, scale

    # ──────────────────────────────────────────
    # Alignment: ORB + Homography
    # ──────────────────────────────────────────
    def _align_orb(
        self, ref_gray: np.ndarray, test_gray: np.ndarray
    ) -> Optional[np.ndarray]:
        """Align test image to reference using ORB features + Homography.

        Returns the warped test_gray, or None if alignment failed.
        """
        orb = cv2.ORB_create(nfeatures=2000)
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        kp_test, des_test = orb.detectAndCompute(test_gray, None)

        if des_ref is None or des_test is None:
            return None
        if len(kp_ref) < self.MIN_ORB_MATCHES or len(kp_test) < self.MIN_ORB_MATCHES:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des_test, des_ref, k=2)

        # Lowe's ratio test
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.MIN_ORB_MATCHES:
            return None

        src_pts = np.float32(
            [kp_test[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_ref[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        h, w = ref_gray.shape
        aligned = cv2.warpPerspective(test_gray, H, (w, h))
        return aligned

    # ──────────────────────────────────────────
    # Alignment fallback: Phase Correlation
    # ──────────────────────────────────────────
    def _align_phase_correlation(
        self, ref_gray: np.ndarray, test_gray: np.ndarray
    ) -> np.ndarray:
        """Align using Phase Correlation (translation only).

        Faster and more robust than ORB for fixed-camera setups
        where only small shifts occur.
        """
        h, w = ref_gray.shape
        # Resize test to match ref dimensions exactly
        test_resized = cv2.resize(test_gray, (w, h))

        ref_f = np.float32(ref_gray)
        test_f = np.float32(test_resized)

        shift, _ = cv2.phaseCorrelate(ref_f, test_f)

        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        aligned = cv2.warpAffine(test_resized, M, (w, h))
        return aligned

    # ──────────────────────────────────────────
    # SSIM comparison
    # ──────────────────────────────────────────
    def _compute_ssim_map(
        self, ref_gray: np.ndarray, test_aligned: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute full SSIM and per-pixel SSIM map.

        Returns (global_ssim_score, ssim_map_uint8).
        The map is inverted so high values = low similarity = defects.
        """
        h_ref, w_ref = ref_gray.shape
        h_test, w_test = test_aligned.shape

        # Ensure same dimensions
        min_h = min(h_ref, h_test)
        min_w = min(w_ref, w_test)
        ref_crop = ref_gray[:min_h, :min_w]
        test_crop = test_aligned[:min_h, :min_w]

        # Ensure odd win_size ≤ smallest dimension
        win = self.SSIM_WIN_SIZE
        min_dim = min(min_h, min_w)
        if win > min_dim:
            win = min_dim if min_dim % 2 == 1 else min_dim - 1
        if win < 3:
            win = 3

        score, ssim_map = ssim(
            ref_crop, test_crop,
            win_size=win,
            full=True,
            data_range=255,
        )

        # Invert: defect regions have LOW ssim → HIGH in inverted map
        defect_map = (1.0 - ssim_map)
        defect_map = np.clip(defect_map * 255, 0, 255).astype(np.uint8)

        return score, defect_map

    # ──────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────
    def detect_defects(
        self,
        test_buffer: BinaryIO,
        ref_buffer: BinaryIO,
        sensitivity: float = 2.0,
        min_area: int = 300,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Run the full Reference-Based inspection pipeline.

        Args:
            test_buffer:  The fabric image to inspect.
            ref_buffer:   The known-good "Golden Image".
            sensitivity:  Not used for global Z-score — kept for API compat.
                          Sauvola window_size scales inversely with this.
            min_area:     Minimum defect area in original pixels.

        Returns:
            (original_bgr, result_annotated_bgr, ssim_heatmap_bgr, defect_list)
        """
        # 1. Decode both images
        orig_test, test_gray, scale = self._decode_and_resize(test_buffer)
        _, ref_gray, _ = self._decode_and_resize(ref_buffer)

        # Ensure same dimensions
        h_ref, w_ref = ref_gray.shape
        h_test, w_test = test_gray.shape
        target_h = min(h_ref, h_test)
        target_w = min(w_ref, w_test)
        ref_gray = cv2.resize(ref_gray, (target_w, target_h))
        test_gray_resized = cv2.resize(test_gray, (target_w, target_h))

        # 2. Align test to reference
        aligned = self._align_orb(ref_gray, test_gray_resized)
        alignment_method = "ORB+Homography"
        if aligned is None:
            aligned = self._align_phase_correlation(ref_gray, test_gray_resized)
            alignment_method = "Phase Correlation"

        # 3. SSIM comparison
        global_ssim, defect_map = self._compute_ssim_map(ref_gray, aligned)

        # 4. Sauvola local thresholding on the defect map
        #    (adaptive to local illumination differences)
        sauvola_window = max(25, int(101 / max(sensitivity, 0.5)))
        sauvola_window = sauvola_window if sauvola_window % 2 == 1 else sauvola_window + 1
        sauvola_thresh = threshold_sauvola(defect_map, window_size=sauvola_window, k=0.2)
        binary_map = (defect_map > sauvola_thresh).astype(np.uint8) * 255

        # Additional global floor: ignore regions with very small SSIM deviation
        floor_thresh = max(30, int(50 / max(sensitivity, 0.5)))
        binary_map[defect_map < floor_thresh] = 0

        # 5. Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel, iterations=1)

        # 6. Connected-component extraction
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_map, connectivity=8
        )

        self.defects = []
        result = orig_test.copy()
        proc_h, proc_w = defect_map.shape[:2]

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            real_area = int(area / (scale ** 2))
            if real_area < min_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Map to original coordinates and clamp to bounds
            orig_h_img, orig_w_img = orig_test.shape[:2]
            ox = max(0, int(x / scale))
            oy = max(0, int(y / scale))
            ow = max(1, min(int(w / scale), orig_w_img - ox))
            oh = max(1, min(int(h / scale), orig_h_img - oy))
            # Pad so the box wraps the SSIM-detected region comfortably
            _PAD = 10
            ox = max(0, ox - _PAD)
            oy = max(0, oy - _PAD)
            ow = min(orig_w_img - ox, ow + 2 * _PAD)
            oh = min(orig_h_img - oy, oh + 2 * _PAD)

            # Shape metrics
            aspect_ratio = w / max(h, 1)
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            solidity = 0.0
            if contours:
                cnt = contours[0]
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = cv2.contourArea(cnt) / hull_area

            # SSIM-based confidence: mean defect_map value in the region
            region_vals = defect_map[labels == i]
            mean_deviation = float(np.mean(region_vals)) if len(region_vals) > 0 else 0
            confidence = min(99, max(10, int(mean_deviation / 2.55 * 100 / 100)))
            # Scale: 0-255 defect_map → 0-100 confidence
            confidence = min(99, max(10, int(mean_deviation * 100 / 255)))

            # ── Oil Stain vs Hole discrimination ──
            # Grayscale features: darkness ratio, texture preservation, boundary sharpness
            ref_roi = aligned[y:y+h, x:x+w] if aligned is not None else None
            inner_mean_r = float(np.mean(ref_roi)) if ref_roi is not None and ref_roi.size > 0 else 0.0
            rh, rw = aligned.shape[:2] if aligned is not None else (1, 1)
            ex_r, ey_r = max(10, w // 2), max(10, h // 2)
            ny1_r, nx1_r = max(0, y - ey_r), max(0, x - ex_r)
            ny2_r, nx2_r = min(rh, y + h + ey_r), min(rw, x + w + ex_r)
            neigh_r = aligned[ny1_r:ny2_r, nx1_r:nx2_r] if aligned is not None else None
            neigh_mean_r = max(1.0, float(np.mean(neigh_r))) if neigh_r is not None and neigh_r.size > 0 else 1.0
            darkness_ratio_r = inner_mean_r / neigh_mean_r

            inner_var_r = float(np.var(ref_roi.astype(np.float32))) if ref_roi is not None and ref_roi.size > 0 else 0.0
            neigh_var_r = max(1.0, float(np.var(neigh_r.astype(np.float32)))) if neigh_r is not None and neigh_r.size > 0 else 1.0
            texture_ratio_r = inner_var_r / neigh_var_r

            boundary_gradient = 0.0
            global_gradient = 1.0
            if contours:
                cnt = contours[0]
                rim_mask = np.zeros(defect_map.shape, dtype=np.uint8)
                cv2.drawContours(rim_mask, [cnt], -1, 255, thickness=3)
                sx = cv2.Sobel(aligned, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(aligned, cv2.CV_64F, 0, 1, ksize=3)
                gmag = np.sqrt(sx**2 + sy**2)
                rim_px = gmag[rim_mask > 0]
                boundary_gradient = float(np.mean(rim_px)) if rim_px.size > 0 else 0.0
                global_gradient = max(1.0, float(np.mean(gmag)))
            has_torn_edges = boundary_gradient > global_gradient * 1.3

            # Classification: darkness + texture + boundary + shape
            if (solidity > 0.6 and 0.4 < aspect_ratio < 2.5
                    and darkness_ratio_r > 0.50 and texture_ratio_r > 0.35
                    and not has_torn_edges):
                d_type = "Oil Stain"
                color = (0, 140, 255)
            elif aspect_ratio > 3.0 or aspect_ratio < 0.33:
                d_type = "Missing Thread"
                color = (0, 0, 255)
            elif (darkness_ratio_r < 0.50 and texture_ratio_r < 0.35) or (solidity < 0.5 and real_area > 1000):
                d_type = "Hole"
                color = (255, 0, 0)
            elif has_torn_edges and real_area > 1000:
                d_type = "Hole"
                color = (255, 0, 0)
            elif solidity < 0.5:
                d_type = "Snag"
                color = (0, 200, 200)
            else:
                d_type = "Slub"
                color = (0, 165, 255)

            cv2.rectangle(result, (ox, oy), (ox + ow, oy + oh), color, 3)
            cv2.putText(
                result, f"{d_type} ({confidence}%)", (ox, max(oy - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

            self.defects.append({
                "ID": len(self.defects) + 1,
                "Type": d_type,
                "Area (px)": real_area,
                "Solidity": f"{solidity:.2f}",
                "Confidence": f"{confidence}%",
                "SSIM Deviation": f"{mean_deviation:.1f}",
                "Location": f"({ox}, {oy})",
                "bbox_x": ox, "bbox_y": oy, "bbox_w": ow, "bbox_h": oh,
            })

        # Heatmap visualization
        ssim_heatmap = cv2.applyColorMap(defect_map, cv2.COLORMAP_INFERNO)

        return orig_test, result, ssim_heatmap, self.defects

    def get_alignment_info(self) -> Dict[str, Any]:
        """Return info about the last alignment for diagnostics."""
        return {"defect_count": len(self.defects)}


# Module instance
reference_inspector = ReferenceInspector()
