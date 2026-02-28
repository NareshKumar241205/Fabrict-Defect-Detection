import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, BinaryIO, Optional
from config import SEAM_SETTINGS

logger = logging.getLogger(__name__)


class SeamInspector:
    """
    Stitch Quality Inspector (Group II)
    ====================================
    Detects 5 seam defect types using specialized algorithms:
    
    - Skip Stitch / Broken Stitch / Run-off Stitch → Projection Profiling
    - Crooked Stitch → Linear Regression on stitch line
    - Pucker → Laplacian Variance near seam region
    """
    
    def __init__(self):
        # Load defaults from config
        self.stitch_thresh = SEAM_SETTINGS.get("STITCH_COLOR_THRESH", 180)
        self.gap_tolerance = SEAM_SETTINGS.get("GAP_TOLERANCE", 10)
        self.crooked_r2_thresh = SEAM_SETTINGS.get("CROOKED_R2_THRESH", 0.85)
        self.pucker_var_sigma = SEAM_SETTINGS.get("PUCKER_VAR_SIGMA", 2.0)
        self.runoff_edge_margin = SEAM_SETTINGS.get("RUNOFF_EDGE_MARGIN", 0.10)
        self.broken_gap_min = 20  # Will be overridden by unified processor if needed

    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if hasattr(img_buffer, "seek"):
            img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
             raise ValueError("Could not decode image file")

        h, w = img.shape[:2]
        target_w = 800
        scale = target_w / w
        img_small = cv2.resize(img, (target_w, int(h * scale)))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Illumination Correction (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)
        
        return img, img_small, img_gray, scale

    def _deskew(self, img_gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """Auto-rotate to align seam horizontally. Returns deskewed image and angle."""
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=20)
        
        rot_img = img_gray
        angle_used = 0.0
        if lines is not None:
            for x1, y1, x2, y2 in lines[0]:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45 and abs(angle) > 0.5:
                    center = (img_gray.shape[1] // 2, img_gray.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rot_img = cv2.warpAffine(img_gray, M,
                                             (img_gray.shape[1], img_gray.shape[0]))
                    angle_used = angle
                    break
        
        return rot_img, angle_used

    def _extract_stitch_mask(self, rot_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the stitch line mask and its projection profile."""
        _, thread_mask = cv2.threshold(rot_img, self.stitch_thresh, 255,
                                       cv2.THRESH_BINARY)
        proj = np.sum(thread_mask, axis=0) / 255
        return thread_mask, proj

    # ──────────────────────────────────────────────
    # DETECTION ENGINE 1: Projection Profiling
    # Detects: Skip Stitch, Broken Stitch, Run-off Stitch
    # ──────────────────────────────────────────────
    def _detect_projection_defects(self, proj: np.ndarray, rot_img: np.ndarray
                                    ) -> List[Dict[str, Any]]:
        """Analyze projection profile for gaps (skip/broken) and edge run-off."""
        h, w = rot_img.shape[:2]
        defects = []
        
        # --- Skip / Broken detection via gap scanning ---
        gap_counter = 0
        in_gap = False
        gap_start = 0
        
        for i, val in enumerate(proj):
            if val < 2:  # Gap in stitch
                if not in_gap:
                    in_gap = True
                    gap_start = i
                gap_counter += 1
            else:
                if in_gap:
                    if gap_counter > self.gap_tolerance:
                        # Avoid edges
                        if gap_start > 10 and i < (len(proj) - 10):
                            # Classify: large gap = broken, small gap = skip
                            if gap_counter > self.broken_gap_min:
                                d_type = "Broken Stitch"
                            else:
                                d_type = "Skip Stitch"
                            
                            defects.append({
                                "x": gap_start, "y": 10,
                                "w": gap_counter, "h": h - 20,
                                "type": d_type,
                                "score": gap_counter
                            })
                    in_gap = False
                    gap_counter = 0
        
        # --- Run-off detection: stitch density drops at image edges ---
        edge_margin = int(w * self.runoff_edge_margin)
        if edge_margin > 5:
            # Check left edge
            left_density = np.mean(proj[:edge_margin])
            center_density = np.mean(proj[edge_margin:-edge_margin]) if w > 2 * edge_margin else 0
            
            if center_density > 15 and left_density > center_density * 0.5:
                # Stitch runs INTO the left edge — run-off
                defects.append({
                    "x": 0, "y": 10,
                    "w": edge_margin, "h": h - 20,
                    "type": "Run-off Stitch",
                    "score": int(left_density)
                })
            
            # Check right edge
            right_density = np.mean(proj[-edge_margin:])
            if center_density > 15 and right_density > center_density * 0.5:
                defects.append({
                    "x": w - edge_margin, "y": 10,
                    "w": edge_margin, "h": h - 20,
                    "type": "Run-off Stitch",
                    "score": int(right_density)
                })
        
        return defects

    # ──────────────────────────────────────────────
    # DETECTION ENGINE 2: Linear Regression
    # Detects: Crooked Stitch
    # ──────────────────────────────────────────────
    def _detect_crooked(self, thread_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fit a linear regression to the stitch line centroids.
        If R² is below threshold, the seam is crooked.
        """
        h, w = thread_mask.shape[:2]
        defects = []
        
        # Find centroid of stitch pixels per column
        centroids_y = []
        centroids_x = []
        
        step = max(1, w // 100)  # Sample ~100 points across the width
        valid_points = 0
        for x in range(0, w, step):
            col = thread_mask[:, x]
            stitch_pixels = np.where(col > 0)[0]
            if len(stitch_pixels) > 5: # Need a solid block of pixels, not just random weave noise
                centroids_y.append(np.mean(stitch_pixels))
                centroids_x.append(float(x))
                valid_points += 1
        
        # A real seam should span a significant portion of the image
        if valid_points < 30:
            return defects  # Not enough continuous stitch data; likely just weave noise
        
        centroids_x = np.array(centroids_x)
        centroids_y = np.array(centroids_y)
        
        # Linear regression: y = mx + b
        n = len(centroids_x)
        sum_x = np.sum(centroids_x)
        sum_y = np.sum(centroids_y)
        sum_xy = np.sum(centroids_x * centroids_y)
        sum_x2 = np.sum(centroids_x ** 2)
        
        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return defects
        
        m = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / n
        
        # R² (coefficient of determination)
        y_pred = m * centroids_x + b
        ss_res = np.sum((centroids_y - y_pred) ** 2)
        ss_tot = np.sum((centroids_y - np.mean(centroids_y)) ** 2)
        
        r_squared = 1.0 - (ss_res / max(ss_tot, 1e-10))
        
        if r_squared < self.crooked_r2_thresh:
            # Max deviation from the fitted line
            max_dev = np.max(np.abs(centroids_y - y_pred))
            
            defects.append({
                "x": 0, "y": max(0, int(np.min(centroids_y) - 20)),
                "w": w, "h": min(h, int(np.max(centroids_y) - np.min(centroids_y) + 40)),
                "type": "Crooked Stitch",
                "score": int((1.0 - r_squared) * 100),
                "r_squared": round(r_squared, 4),
                "max_deviation_px": round(float(max_dev), 1)
            })
        
        return defects

    # ──────────────────────────────────────────────
    # DETECTION ENGINE 3: Laplacian Variance
    # Detects: Pucker
    # ──────────────────────────────────────────────
    def _detect_pucker(self, img_gray: np.ndarray, thread_mask: np.ndarray
                       ) -> List[Dict[str, Any]]:
        """
        Compute Laplacian variance in the neighbourhood of the seam.
        High variance = high-frequency wrinkling = pucker.
        """
        h, w = img_gray.shape[:2]
        defects = []
        
        # Find seam vertical region from thread mask
        row_sums = np.sum(thread_mask, axis=1)
        seam_rows = np.where(row_sums > w * 0.2)[0]  # Rows must have at least 20% stitch coverage
        
        if len(seam_rows) < 5:
            return defects  # No distinct seam found
        
        seam_top = max(0, int(np.min(seam_rows)) - 30)
        seam_bottom = min(h, int(np.max(seam_rows)) + 30)
        
        # Extract seam neighborhood region
        seam_region = img_gray[seam_top:seam_bottom, :]
        
        if seam_region.shape[0] < 10 or seam_region.shape[1] < 10:
            return defects
        
        # Compute patch-wise Laplacian variance
        patch_size = 32
        step = 16
        variances = []
        patch_positions = []
        
        for x in range(0, seam_region.shape[1] - patch_size, step):
            patch = seam_region[:, x:x + patch_size]
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            var = lap.var()
            variances.append(var)
            patch_positions.append(x)
        
        if len(variances) < 5:
            return defects
        
        variances = np.array(variances)
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        
        if std_var < 1e-6:
            return defects
        
        # Find patches with abnormally high variance (wrinkling)
        threshold = mean_var + self.pucker_var_sigma * std_var
        
        pucker_start = None
        for i, (var, x_pos) in enumerate(zip(variances, patch_positions)):
            if var > threshold:
                if pucker_start is None:
                    pucker_start = x_pos
            else:
                if pucker_start is not None:
                    pucker_w = x_pos - pucker_start
                    if pucker_w > patch_size:  # Must be wider than a single patch
                        z_score = (np.max(variances[patch_positions.index(pucker_start) 
                                   if pucker_start in patch_positions else 0:i]) - mean_var) / max(std_var, 1e-6)
                        defects.append({
                            "x": pucker_start, "y": seam_top,
                            "w": pucker_w, "h": seam_bottom - seam_top,
                            "type": "Pucker",
                            "score": min(99, int(z_score * 20)),
                        })
                    pucker_start = None
        
        # Handle pucker at end of image
        if pucker_start is not None:
            pucker_w = patch_positions[-1] - pucker_start
            if pucker_w > patch_size:
                defects.append({
                    "x": pucker_start, "y": seam_top,
                    "w": pucker_w, "h": seam_bottom - seam_top,
                    "type": "Pucker",
                    "score": 60,
                })
        
        return defects

    # ──────────────────────────────────────────────
    # MAIN PIPELINE
    # ──────────────────────────────────────────────
    def detect_defects(self, img_buffer: BinaryIO, 
                       settings: Optional[Dict[str, Any]] = None
                       ) -> Tuple[np.ndarray, None, None, np.ndarray, List[Dict[str, Any]]]:
        """
        Full seam inspection pipeline.
        Runs all three engines: Projection, Regression, Laplacian.
        """
        if settings is None:
            settings = {}

        # Allow runtime overrides
        self.stitch_thresh = settings.get("STITCH_COLOR_THRESH", self.stitch_thresh)
        self.gap_tolerance = settings.get("GAP_TOLERANCE", self.gap_tolerance)

        img_orig, img_small, img_gray, scale = self._preprocess(img_buffer)
        
        # 1. Deskew
        rot_img, _ = self._deskew(img_gray)
        
        # 2. Extract stitch mask and projection
        thread_mask, proj = self._extract_stitch_mask(rot_img)
        
        # GLOBAL BAILOUT: If there is barely any stitch detected in the whole image,
        # this is likely just background noise on a plain fabric. Bail out immediately.
        stitch_density = np.count_nonzero(thread_mask) / (thread_mask.shape[0] * thread_mask.shape[1])
        if stitch_density < 0.01:  # Less than 1% of the image is thread
            return img_orig, None, None, img_small.copy(), []
        
        # 3. Run all three detection engines
        all_raw_defects = []
        
        # Engine 1: Projection Profiling (Skip, Broken, Run-off)
        proj_defects = self._detect_projection_defects(proj, rot_img)
        all_raw_defects.extend(proj_defects)
        
        # Engine 2: Linear Regression (Crooked)
        crooked_defects = self._detect_crooked(thread_mask)
        all_raw_defects.extend(crooked_defects)
        
        # Engine 3: Laplacian Variance (Pucker)
        pucker_defects = self._detect_pucker(img_gray, thread_mask)
        all_raw_defects.extend(pucker_defects)

        # Draw results on output image
        output_img = img_small.copy()
        
        # Color map per defect type
        type_colors = {
            "Skip Stitch": (0, 0, 255),       # Red
            "Broken Stitch": (0, 0, 200),      # Dark Red
            "Run-off Stitch": (0, 165, 255),   # Orange
            "Crooked Stitch": (255, 0, 255),   # Magenta
            "Pucker": (255, 255, 0),           # Cyan
        }
        
        defect_log = []
        for d in all_raw_defects:
            color = type_colors.get(d['type'], (0, 0, 255))
            
            cv2.rectangle(output_img, (d['x'], d['y']),
                         (d['x'] + d['w'], d['y'] + d['h']), color, 2)
            cv2.putText(output_img, d['type'], (d['x'], d['y'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Confidence from score
            confidence = min(99, max(10, d.get('score', 50)))
            
            defect_log.append({
                "ID": len(defect_log) + 1,
                "Type": d['type'],
                "Group": "Stitch Quality",
                "Area (px)": int(d['w'] * d['h'] * (scale ** -2) if scale else d['w'] * d['h']),
                "Confidence": f"{confidence}%",
                "bbox_x": int(d['x'] / scale) if scale else d['x'],
                "bbox_y": int(d['y'] / scale) if scale else d['y'],
                "bbox_w": int(d['w'] / scale) if scale else d['w'],
                "bbox_h": int(d['h'] / scale) if scale else d['h'],
            })
            
            # Add extra metadata if present
            if 'r_squared' in d:
                defect_log[-1]["R²"] = d['r_squared']
            if 'max_deviation_px' in d:
                defect_log[-1]["Max Deviation (px)"] = d['max_deviation_px']

        return img_orig, None, None, output_img, defect_log


seam_inspector = SeamInspector()
