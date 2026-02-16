"""
Edge / Structural Inspector Module
===================================
Detects structural/geometric defects like wrinkles, folds, and broken weave
lines using classical edge detection and line analysis.

Algorithm:
1. Canny edge detection with automatic thresholding (Otsu's method).
2. Hough Line Transform to detect dominant weave lines.
3. Line spacing regularity analysis — irregular spacing = structural defect.
4. Laplacian variance per patch for wrinkle/fold detection (local blur/sharpness).
5. Combined anomaly mask from both methods.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO


class EdgeInspector:
    """Detects structural defects using edge analysis and Laplacian variance."""

    RESIZE_WIDTH = 800
    PATCH_SIZE = 48
    PATCH_STEP = 24  # 50% overlap

    def __init__(self):
        self.defects = []

    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Standardize input resolution and color space."""
        if hasattr(img_buffer, 'seek'):
            img_buffer.seek(0)

        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image file")

        h, w = img.shape[:2]
        scale = self.RESIZE_WIDTH / w
        target_h = int(h * scale)

        img_small = cv2.resize(img, (self.RESIZE_WIDTH, target_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        # CLAHE for illumination correction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, scale

    def _compute_laplacian_variance_map(self, img_gray: np.ndarray) -> np.ndarray:
        """Compute per-patch Laplacian variance to detect wrinkles and folds.
        
        Wrinkles cause local blur or unusual sharpness changes.
        Patches with variance far from the image mean are flagged.
        """
        h, w = img_gray.shape
        var_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - self.PATCH_SIZE, self.PATCH_STEP):
            for x in range(0, w - self.PATCH_SIZE, self.PATCH_STEP):
                patch = img_gray[y:y + self.PATCH_SIZE, x:x + self.PATCH_SIZE]
                lap = cv2.Laplacian(patch, cv2.CV_64F)
                var = lap.var()
                var_map[y:y + self.PATCH_SIZE, x:x + self.PATCH_SIZE] += var
                count_map[y:y + self.PATCH_SIZE, x:x + self.PATCH_SIZE] += 1

        # Average overlapping patches
        count_map[count_map == 0] = 1
        var_map = var_map / count_map

        return var_map

    def _analyze_line_regularity(self, img_gray: np.ndarray) -> np.ndarray:
        """Use Hough Lines to detect dominant weave lines and flag irregular spacing.
        
        Returns a binary anomaly mask where irregular line gaps are marked.
        """
        h, w = img_gray.shape
        anomaly_mask = np.zeros((h, w), dtype=np.uint8)

        # Auto-threshold Canny using Otsu's method
        otsu_thresh, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = int(otsu_thresh * 0.5)
        high_thresh = int(otsu_thresh)
        edges = cv2.Canny(img_gray, low_thresh, high_thresh)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=80,
            minLineLength=w // 4, maxLineGap=20
        )

        if lines is None or len(lines) < 3:
            return anomaly_mask

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 30:
                h_lines.append((y1 + y2) // 2)  # y-center
            elif angle > 60:
                v_lines.append((x1 + x2) // 2)  # x-center

        # Analyze spacing regularity for whichever group has more lines
        for positions, axis in [(sorted(h_lines), 'h'), (sorted(v_lines), 'v')]:
            if len(positions) < 3:
                continue
            spacings = np.diff(positions)
            if len(spacings) < 2:
                continue
            mean_sp = np.mean(spacings)
            std_sp = np.std(spacings)
            if std_sp < 1:
                continue

            for i, sp in enumerate(spacings):
                z = abs(sp - mean_sp) / max(std_sp, 1e-6)
                if z > 2.0:  # irregular gap
                    pos = positions[i]
                    if axis == 'h':
                        gap = int(sp)
                        anomaly_mask[max(0, pos):min(h, pos + gap), :] = 255
                    else:
                        gap = int(sp)
                        anomaly_mask[:, max(0, pos):min(w, pos + gap)] = 255

        return anomaly_mask

    def detect_defects(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.0,
        min_area: int = 800
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Main pipeline.
        
        Returns: (original, edge_map, anomaly_heatmap, annotated_result, defect_list)
        """
        original, img_small, img_gray, scale = self._preprocess(img_buffer)
        h, w = img_gray.shape

        # 1. Laplacian variance map
        var_map = self._compute_laplacian_variance_map(img_gray)
        mean_var = np.mean(var_map)
        std_var = np.std(var_map)

        # Z-score thresholding on Laplacian variance
        # Both very low (blurry/folded) and very high (sharp edges/tears) are anomalies
        lower_bound = mean_var - (sensitivity * std_var)
        upper_bound = mean_var + (sensitivity * std_var)

        var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask_low = cv2.inRange(var_norm, 0, int(max(0, lower_bound / max(mean_var, 1) * 128)))
        mask_high = cv2.inRange(var_norm, int(min(255, upper_bound / max(mean_var, 1) * 128)), 255)
        laplacian_mask = cv2.bitwise_or(mask_low, mask_high)

        # 2. Line regularity analysis
        line_mask = self._analyze_line_regularity(img_gray)

        # 3. Combine both masks
        combined_mask = cv2.bitwise_or(laplacian_mask, line_mask)

        # 4. Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 5. Extract defects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )

        defect_list = []
        result = original.copy()

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            real_area = area / (scale ** 2)
            if real_area < min_area:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT] / scale)
            y = int(stats[i, cv2.CC_STAT_TOP] / scale)
            bw = int(stats[i, cv2.CC_STAT_WIDTH] / scale)
            bh = int(stats[i, cv2.CC_STAT_HEIGHT] / scale)
            x, y = max(0, x), max(0, y)

            aspect_ratio = float(bw) / max(bh, 1)

            # Classify defect type
            if aspect_ratio > 4.0:
                name = "Wrinkle / Fold (Horiz)"
                color = (0, 200, 200)  # Yellow
            elif aspect_ratio < 0.25:
                name = "Wrinkle / Fold (Vert)"
                color = (0, 200, 200)
            elif real_area > 5000:
                name = "Structural Break"
                color = (0, 0, 255)  # Red
            else:
                name = "Weave Irregularity"
                color = (255, 128, 0)  # Blue-ish

            # Confidence from Z-score
            region_mask = (labels == i)
            region_var = var_map[region_mask[:h, :w]] if region_mask.shape == var_map.shape else []
            if len(region_var) > 0:
                region_mean = np.mean(region_var)
                z_dist = abs(region_mean - mean_var) / max(std_var, 1e-6)
                confidence = min(99, int((z_dist / max(sensitivity, 1e-6)) * 100))
            else:
                confidence = 50

            defect_list.append({
                "ID": i,
                "Type": name,
                "Area (px)": int(real_area),
                "Confidence": f"{confidence}%",
                "bbox_x": x, "bbox_y": y, "bbox_w": bw, "bbox_h": bh
            })

            cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 3)
            cv2.putText(result, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Visualization maps
        edge_viz = cv2.Canny(img_gray, 50, 150)
        anomaly_heatmap = cv2.applyColorMap(var_norm, cv2.COLORMAP_MAGMA)

        return original, edge_viz, anomaly_heatmap, result, defect_list


# Module instance
edge_inspector = EdgeInspector()
