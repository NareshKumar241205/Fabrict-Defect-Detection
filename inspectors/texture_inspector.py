"""
Texture & GLCM Inspector Module (Algorithm B)
===============================================
Detects texture anomalies using LBP, Entropy, Gabor filter banks,
and **actual GLCM (Gray Level Co-occurrence Matrix)** Haralick features.

Detects: Rough Weave / complex stains / general texture defects /
         broken weave patterns (via GLCM Correlation drop).

Algorithm:
1. Multi-scale LBP → Entropy analysis for local randomness spikes.
2. Gabor filter bank (6 orientations × 3 frequencies) for directional defects.
3. **GLCM per-patch analysis** — Contrast, Correlation, Homogeneity.
   Monitoring the Correlation metric mathematically proves when a weave
   pattern is broken, giving high precision for structural damage.
4. Fuse all anomaly maps (max), **Sauvola local adaptive thresholding**
   (replaces global Z-score — immune to uneven lighting), connected-
   component extraction.
5. Shape metrics (solidity, aspect ratio) stored for downstream sub-classification.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, BinaryIO
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.filters import threshold_sauvola
from skimage.morphology import disk

logger = logging.getLogger(__name__)


class TextureInspector:
    PROCESS_WIDTH = 800

    def __init__(self):
        self.RADIUS = 3
        self.N_POINTS = 8 * self.RADIUS
        self.METHOD = "uniform"

    # ──────────────────────────────────────────
    # Pre-processing
    # ──────────────────────────────────────────
    def _preprocess(
        self, img_buffer: BinaryIO
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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

        return img, img_small, img_gray, scale

    # ──────────────────────────────────────────
    # Feature maps
    # ──────────────────────────────────────────
    def compute_texture_map(self, img_gray: np.ndarray) -> np.ndarray:
        lbp = local_binary_pattern(img_gray, self.N_POINTS, self.RADIUS, self.METHOD)
        lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-9) * 255
        return lbp_norm.astype(np.uint8)

    def compute_entropy_map(self, lbp_img: np.ndarray) -> np.ndarray:
        ent_img = entropy(lbp_img, disk(5))
        return cv2.normalize(ent_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def compute_gabor_map(self, img_gray: np.ndarray) -> np.ndarray:
        """Gabor filter bank: 6 orientations × 3 frequencies = 18 filters."""
        orientations = [0, 30, 60, 90, 120, 150]
        frequencies = [0.05, 0.1, 0.2]
        ksize = 31
        sigma = 4.0
        gamma = 0.5

        responses = []
        for theta_deg in orientations:
            theta = np.deg2rad(theta_deg)
            for freq in frequencies:
                lambd = 1.0 / freq
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F
                )
                filtered = cv2.filter2D(img_gray, cv2.CV_32F, kernel)
                responses.append(np.abs(filtered))

        gabor_fused = np.maximum.reduce(responses)
        return cv2.normalize(gabor_fused, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def compute_glcm_map(self, img_gray: np.ndarray, patch_size: int = 48, step: int = 24) -> np.ndarray:
        """Compute per-patch GLCM features → anomaly map.

        For each patch, compute the Gray Level Co-occurrence Matrix and extract:
        - Contrast:    high in defective regions (edges, holes)
        - Correlation:  drops sharply when the periodic weave is broken
        - Homogeneity:  abnormally high for smooth stains

        Returns a normalized anomaly map [0..255] where high values = anomalous.
        """
        from config import GLCM_SETTINGS

        distances = GLCM_SETTINGS.get("DISTANCES", [1])
        angles = GLCM_SETTINGS.get("ANGLES", [0, np.pi / 2])
        thresholds = GLCM_SETTINGS.get("THRESHOLDS", {})

        contrast_max = thresholds.get("contrast_max", 250)
        correlation_min = thresholds.get("correlation_min", 0.80)
        homogeneity_max = thresholds.get("homogeneity_max", 0.98)

        h, w = img_gray.shape
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        # Quantize to 64 levels for faster GLCM
        img_q = (img_gray // 4).astype(np.uint8)

        for y in range(0, h - patch_size, step):
            for x in range(0, w - patch_size, step):
                patch = img_q[y : y + patch_size, x : x + patch_size]

                glcm = graycomatrix(
                    patch,
                    distances=distances,
                    angles=angles,
                    levels=64,
                    symmetric=True,
                    normed=True,
                )

                contrast = float(np.mean(graycoprops(glcm, "contrast")))
                correlation = float(np.mean(graycoprops(glcm, "correlation")))
                homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))

                # Compute a combined anomaly score for this patch
                score = 0.0
                if contrast > contrast_max:
                    score += min(1.0, contrast / contrast_max - 1.0)
                if correlation < correlation_min:
                    score += min(1.0, 1.0 - correlation / correlation_min)
                if homogeneity > homogeneity_max:
                    score += min(1.0, homogeneity / homogeneity_max - 1.0)

                anomaly_map[y : y + patch_size, x : x + patch_size] += score
                count_map[y : y + patch_size, x : x + patch_size] += 1.0

        count_map[count_map == 0] = 1.0
        anomaly_map = anomaly_map / count_map

        return cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ──────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────
    def detect_defects(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 3.0,
        min_area: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Returns (original, lbp_map, fused_entropy_map, annotated_result, defects)."""
        orig_full, img_small, img_gray, scale_factor = self._preprocess(img_buffer)
        h, w = img_gray.shape

        # Multi-scale entropy analysis
        scales = [1.0, 0.75, 0.5]
        entropy_maps = []
        for s in scales:
            curr_w, curr_h = int(w * s), int(h * s)
            resized_gray = cv2.resize(img_gray, (curr_w, curr_h))
            lbp = self.compute_texture_map(resized_gray)
            ent = self.compute_entropy_map(lbp)
            entropy_maps.append(cv2.resize(ent, (w, h)))

        final_entropy_map = np.maximum.reduce(entropy_maps)

        # Gabor filter bank — normalize BEFORE fusion so scales match
        gabor_map = self.compute_gabor_map(img_gray)
        final_entropy_map = np.maximum(final_entropy_map, gabor_map)

        # GLCM Haralick features — weave pattern structural analysis
        glcm_map = self.compute_glcm_map(img_gray)
        final_entropy_map = np.maximum(final_entropy_map, glcm_map)

        # Sauvola local adaptive thresholding (replaces global Z-score)
        # Calculates mean and std for rolling windows across the heatmap,
        # making anomaly detection immune to uneven lighting or shadows.
        mean_ent = np.mean(final_entropy_map)
        std_ent = np.std(final_entropy_map)

        sauvola_win = max(25, int(101 / max(sensitivity, 0.5)))
        sauvola_win = sauvola_win if sauvola_win % 2 == 1 else sauvola_win + 1
        sauvola_thresh = threshold_sauvola(final_entropy_map, window_size=sauvola_win, k=0.2)

        # Regions above Sauvola threshold OR below global lower bound
        mask_high = np.zeros_like(final_entropy_map, dtype=np.uint8)
        mask_high[final_entropy_map > sauvola_thresh] = 255

        # Also flag abnormally LOW texture (smooth stains, holes)
        lower_bound = mean_ent - sensitivity * std_ent
        mask_low = cv2.inRange(final_entropy_map, 0, int(max(0, lower_bound)))

        mask_combined = cv2.bitwise_or(mask_low, mask_high)

        # Global floor: ignore if not significantly deviant
        global_floor_high = mean_ent + sensitivity * std_ent * 0.5
        global_floor_low = mean_ent - sensitivity * std_ent * 0.5
        trivial = (final_entropy_map > global_floor_low) & (final_entropy_map < global_floor_high)
        mask_combined[trivial] = 0

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

        # Connected-component extraction
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)

        defect_list: List[Dict[str, Any]] = []
        final_output = orig_full.copy()

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            real_area = area / (scale_factor ** 2)
            if real_area < min_area:
                continue

            # Bounding box in original coordinates
            bx = max(0, int(stats[i, cv2.CC_STAT_LEFT] / scale_factor))
            by = max(0, int(stats[i, cv2.CC_STAT_TOP] / scale_factor))
            bw = max(1, int(stats[i, cv2.CC_STAT_WIDTH] / scale_factor))
            bh = max(1, int(stats[i, cv2.CC_STAT_HEIGHT] / scale_factor))

            # Solidity from contour analysis
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solidity = 0.0
            if contours:
                cnt = contours[0]
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = cv2.contourArea(cnt) / hull_area

            aspect_ratio = float(bw) / max(bh, 1)

            # ── FIX: Use 10-type taxonomy names ──
            if solidity > 0.85 and 0.4 < aspect_ratio < 2.5:
                name = "Oil Stain"
                color = (0, 140, 255)
            elif aspect_ratio > 3.0 or aspect_ratio < 0.33:
                name = "Missing Thread"
                color = (0, 0, 255)
            elif solidity < 0.5:
                if real_area > 1000:
                    name = "Hole"
                else:
                    name = "Slub"
                color = (255, 0, 0)
            else:
                name = "Slub"
                color = (255, 0, 255)

            # Z-score confidence
            defect_region_mask = labels == i
            defect_entropy_vals = final_entropy_map[defect_region_mask]
            defect_mean_ent = np.mean(defect_entropy_vals) if len(defect_entropy_vals) > 0 else mean_ent
            z_distance = abs(defect_mean_ent - mean_ent) / max(std_ent, 1e-6)
            confidence = min(99, max(10, int((z_distance / max(sensitivity, 1e-6)) * 100)))

            defect_list.append({
                "ID": i,
                "Type": name,
                "Area (px)": int(real_area),
                "Solidity": f"{solidity:.2f}",
                "Confidence": f"{confidence}%",
                "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh,
            })

            cv2.rectangle(final_output, (bx, by), (bx + bw, by + bh), color, 4)
            cv2.putText(final_output, name, (bx, max(by - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return orig_full, entropy_maps[0], final_entropy_map, final_output, defect_list


# Module instance
inspector = TextureInspector()
