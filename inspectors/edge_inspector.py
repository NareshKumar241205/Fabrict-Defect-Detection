"""
Edge / Structural Inspector Module (Algorithm C)
=================================================
Detects structural/geometric defects like holes, tears, and snags using
classical edge detection, line-spacing regularity, and **Frangi vesselness**
analysis for thread-level inspection.

Algorithm:
1. CLAHE illumination correction.
2. Canny edge detection (auto-threshold via Otsu).
3. Hough Line Transform → dominant weave-line spacing analysis.
4. Per-patch Laplacian variance → detects ruptures and structural changes.
5. **Frangi filter** (Hessian eigenvalue analysis) → isolates individual
   threads / continuous line structures to flag loose snags or missing threads.
6. Combined anomaly mask → **Sauvola local thresholding** (replaces global
   Z-score for immunity to lighting gradients).
7. Connected-component extraction.

Detects: Holes, Tears, Snags, Missing Threads.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO
from skimage.filters import frangi, threshold_sauvola


class EdgeInspector:
    """Detects structural defects using edge analysis and Laplacian variance."""

    PROCESS_WIDTH = 800
    PATCH_SIZE = 48
    PATCH_STEP = 24  # 50 % overlap

    def __init__(self):
        self.defects: List[Dict[str, Any]] = []

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

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, scale

    # ──────────────────────────────────────────
    # Laplacian variance map
    # ──────────────────────────────────────────
    def _compute_laplacian_variance_map(self, img_gray: np.ndarray) -> np.ndarray:
        """Per-patch Laplacian variance — sharp edges / ruptures spike here."""
        h, w = img_gray.shape
        var_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - self.PATCH_SIZE, self.PATCH_STEP):
            for x in range(0, w - self.PATCH_SIZE, self.PATCH_STEP):
                patch = img_gray[y : y + self.PATCH_SIZE, x : x + self.PATCH_SIZE]
                var = cv2.Laplacian(patch, cv2.CV_64F).var()
                var_map[y : y + self.PATCH_SIZE, x : x + self.PATCH_SIZE] += var
                count_map[y : y + self.PATCH_SIZE, x : x + self.PATCH_SIZE] += 1

        count_map[count_map == 0] = 1
        return var_map / count_map

    # ──────────────────────────────────────────
    # Hough line regularity
    # ──────────────────────────────────────────
    def _analyze_line_regularity(self, img_gray: np.ndarray) -> np.ndarray:
        """Flag regions where weave-line spacing is irregular."""
        h, w = img_gray.shape
        anomaly_mask = np.zeros((h, w), dtype=np.uint8)

        otsu_thresh, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(img_gray, int(otsu_thresh * 0.5), int(otsu_thresh))

        # FIX: less strict Hough params to capture shorter defect-related lines
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=60,
            minLineLength=w // 6, maxLineGap=25,
        )

        if lines is None or len(lines) < 3:
            return anomaly_mask

        h_lines: List[int] = []
        v_lines: List[int] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 30:
                h_lines.append((y1 + y2) // 2)
            elif angle > 60:
                v_lines.append((x1 + x2) // 2)

        for positions, axis in [(sorted(h_lines), "h"), (sorted(v_lines), "v")]:
            if len(positions) < 3:
                continue
            spacings = np.diff(positions).astype(float)
            if len(spacings) < 2:
                continue
            mean_sp = np.mean(spacings)
            std_sp = np.std(spacings)
            if std_sp < 1:
                continue

            for i, sp in enumerate(spacings):
                z = abs(sp - mean_sp) / max(std_sp, 1e-6)
                if z > 2.0:
                    pos = positions[i]
                    gap = int(sp)
                    if axis == "h":
                        anomaly_mask[max(0, pos) : min(h, pos + gap), :] = 255
                    else:
                        anomaly_mask[:, max(0, pos) : min(w, pos + gap)] = 255

        return anomaly_mask

    # ──────────────────────────────────────────
    # Frangi vesselness filter (thread detection)
    # ──────────────────────────────────────────
    def _compute_frangi_anomaly(
        self, img_gray: np.ndarray, sensitivity: float
    ) -> np.ndarray:
        """Use the Frangi (vesselness) filter to detect thread-like structures.

        The Frangi filter analyses eigenvalues of the Hessian matrix to
        enhance continuous, tube-like / line-like structures.  It was
        originally developed for blood-vessel segmentation in medical
        imaging but is mathematically ideal for isolating individual
        threads in a fabric weave.

        We compute the vesselness response, invert it (so *missing* or
        *disrupted* threads become bright), and return an anomaly mask.
        """
        img_f = img_gray.astype(np.float64) / 255.0

        # Frangi across multiple scales to capture different thread widths
        sigmas = range(1, 5)
        vesselness = frangi(
            img_f,
            sigmas=sigmas,
            alpha=0.5,
            beta=0.5,
            gamma=15,
            black_ridges=False,
        )

        # Normalize to [0, 255]
        vesselness_norm = cv2.normalize(
            vesselness, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # In a healthy weave, vesselness is uniformly high along threads.
        # Regions where vesselness *drops* indicate missing/broken threads.
        # Invert so anomalies (low vesselness) become bright.
        inverted = 255 - vesselness_norm

        # Sauvola local thresholding to find locally anomalous regions
        sauvola_win = max(25, int(101 / max(sensitivity, 0.5)))
        sauvola_win = sauvola_win if sauvola_win % 2 == 1 else sauvola_win + 1
        sauvola_thresh = threshold_sauvola(inverted, window_size=sauvola_win, k=0.15)
        frangi_mask = np.zeros_like(inverted, dtype=np.uint8)
        frangi_mask[inverted > sauvola_thresh] = 255

        # Remove noise: only keep regions where vesselness truly broke down
        mean_v = np.mean(inverted)
        std_v = np.std(inverted)
        # Raise the floor multiplier: 0.6 instead of 0.4 to require stronger anomaly signal
        global_floor = mean_v + sensitivity * std_v * 0.6
        frangi_mask[inverted < global_floor] = 0

        return frangi_mask

    # ──────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────
    def detect_defects(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.0,
        min_area: int = 800,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Returns (original, edge_viz, anomaly_heatmap, annotated, defects)."""
        from config import UNIFIED_SETTINGS

        use_frangi = UNIFIED_SETTINGS.get("USE_FRANGI", False)

        original, img_small, img_gray, scale = self._preprocess(img_buffer)
        h, w = img_gray.shape

        # 1. Raw Laplacian variance map (float)
        var_map = self._compute_laplacian_variance_map(img_gray)
        mean_var = np.mean(var_map)
        std_var = np.std(var_map)

        # ── Global z-score thresholding on the Laplacian variance ──
        var_norm_f = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        var_z = (var_map - mean_var) / max(std_var, 1e-6)
        laplacian_mask = np.zeros((h, w), dtype=np.uint8)
        laplacian_mask[var_z > sensitivity] = 255

        # Global floor so flat regions are not falsely flagged
        if std_var > 1e-6:
            z_map = np.abs(var_map - mean_var) / std_var
        else:
            z_map = np.zeros_like(var_map)
        laplacian_mask[z_map < sensitivity * 0.5] = 0

        # 2. Line regularity analysis
        line_mask = self._analyze_line_regularity(img_gray)

        # 3. Frangi vesselness anomaly (thread-level structural defects) — optional
        if use_frangi:
            frangi_mask = self._compute_frangi_anomaly(img_gray, sensitivity)
        else:
            frangi_mask = np.zeros((h, w), dtype=np.uint8)

        # 4. Combine Laplacian and line masks (primary for stains/holes)
        combined_mask = cv2.bitwise_or(laplacian_mask, line_mask)
        # Optionally OR with Frangi if enabled
        if use_frangi:
            combined_mask = cv2.bitwise_or(combined_mask, frangi_mask)

        # 5. Morphological cleanup: open first to remove isolated speckles,
        #    then close to reconnect legitimate fragmented detections.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 6. Extract defects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)

        defect_list: List[Dict[str, Any]] = []
        result = original.copy()

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            real_area = area / (scale ** 2)
            if real_area < min_area:
                continue

            bx = max(0, int(stats[i, cv2.CC_STAT_LEFT] / scale))
            by = max(0, int(stats[i, cv2.CC_STAT_TOP] / scale))
            bw = max(1, int(stats[i, cv2.CC_STAT_WIDTH] / scale))
            bh = max(1, int(stats[i, cv2.CC_STAT_HEIGHT] / scale))

            aspect_ratio = float(bw) / max(bh, 1)

            # ── FIX: Use 10-type taxonomy names ──
            if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                name = "Tear"
                color = (0, 0, 255)
            elif real_area < 800:
                name = "Snag"
                color = (0, 200, 200)
            elif real_area >= 1000:
                name = "Hole"
                color = (255, 0, 0)
            else:
                name = "Hole"
                color = (255, 128, 0)

            # Confidence
            region_mask = labels == i
            if region_mask.shape == var_map.shape:
                region_var = var_map[region_mask]
            else:
                region_var = np.array([])

            if len(region_var) > 0:
                region_mean = np.mean(region_var)
                z_dist = abs(region_mean - mean_var) / max(std_var, 1e-6)
                confidence = min(99, max(10, int((z_dist / max(sensitivity, 1e-6)) * 100)))
            else:
                confidence = 50

            defect_list.append({
                "ID": i,
                "Type": name,
                "Area (px)": int(real_area),
                "Confidence": f"{confidence}%",
                "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh,
            })

            cv2.rectangle(result, (bx, by), (bx + bw, by + bh), color, 3)
            cv2.putText(result, name, (bx, max(by - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Visualization outputs
        edge_viz = cv2.Canny(img_gray, 50, 150)
        var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        anomaly_heatmap = cv2.applyColorMap(var_norm, cv2.COLORMAP_MAGMA)

        return original, edge_viz, anomaly_heatmap, result, defect_list


# Module instance
edge_inspector = EdgeInspector()
