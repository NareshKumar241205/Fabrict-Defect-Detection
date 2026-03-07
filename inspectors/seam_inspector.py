"""
Seam Inspector Module (Algorithms D / E / F) — IMPROVED
========================================================
Detects stitch & seam quality defects in the assembly (sewing) process.

Improvements:
- Projection smoothing (sliding window) before gap scan → fewer false Skip Stitches
- Run-off: requires center_density > CENTER_DENSITY_MIN guard
- Crooked: lowers min centroid count to 15 to detect partial seams
- All thresholds updated to match tuned config.py values

Engines:
  D — Projection Profiling    → Skip Stitch, Broken Stitch, Run-off Stitch
  E — Linear Regression       → Crooked Stitch
  F — Laplacian Variance      → Seam Pucker
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, BinaryIO, Optional
from config import SEAM_SETTINGS

logger = logging.getLogger(__name__)


class SeamInspector:
    """Stitch Quality Inspector (Group II)."""

    def __init__(self):
        self.stitch_thresh  = SEAM_SETTINGS.get("STITCH_COLOR_THRESH", 180)
        self.gap_tolerance  = SEAM_SETTINGS.get("GAP_TOLERANCE", 10)
        self.crooked_r2_thresh   = SEAM_SETTINGS.get("CROOKED_R2_THRESH", 0.85)
        self.pucker_var_sigma    = SEAM_SETTINGS.get("PUCKER_VAR_SIGMA", 1.5)       # tuned
        self.runoff_edge_margin  = SEAM_SETTINGS.get("RUNOFF_EDGE_MARGIN", 0.08)    # tuned
        self.broken_gap_min      = SEAM_SETTINGS.get("BROKEN_GAP_MIN", 15)          # tuned
        self.min_centroid_count  = SEAM_SETTINGS.get("MIN_CENTROID_COUNT", 15)      # tuned
        self.center_density_min  = SEAM_SETTINGS.get("CENTER_DENSITY_MIN", 8)       # new guard
        self.runoff_density_ratio = SEAM_SETTINGS.get("RUNOFF_DENSITY_RATIO", 0.35) # tuned

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
        target_w = 800
        scale = target_w / w
        img_small = cv2.resize(img, (target_w, int(h * scale)))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, scale

    # ──────────────────────────────────────────
    # Deskew — FIX: use MEDIAN angle of all near-horizontal lines
    # ──────────────────────────────────────────
    def _deskew(self, img_gray: np.ndarray) -> Tuple[np.ndarray, float]:
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20
        )

        if lines is None or len(lines) == 0:
            return img_gray, 0.0

        # Collect angles of all near-horizontal lines
        angles: List[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45 and abs(angle) > 0.5:
                angles.append(angle)

        if not angles:
            return img_gray, 0.0

        # Use median angle for robustness against outlier lines
        median_angle = float(np.median(angles))
        center = (img_gray.shape[1] // 2, img_gray.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rot_img = cv2.warpAffine(
            img_gray, M, (img_gray.shape[1], img_gray.shape[0])
        )
        return rot_img, median_angle

    # ──────────────────────────────────────────
    # Adaptive stitch mask — FIX: handle both bright and dark threads
    # ──────────────────────────────────────────
    def _extract_stitch_mask(
        self, rot_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Threshold the thread using whichever polarity (bright/dark) yields
        more stitch-like pixels. Bright threads on dark fabric use a high
        threshold; dark threads on light fabric use an inverted low threshold.
        """
        # Bright thread mask (original behaviour)
        _, mask_bright = cv2.threshold(
            rot_img, self.stitch_thresh, 255, cv2.THRESH_BINARY
        )
        bright_count = np.count_nonzero(mask_bright)

        # Dark thread mask (inverted)
        dark_thresh = 255 - self.stitch_thresh
        _, mask_dark = cv2.threshold(
            rot_img, dark_thresh, 255, cv2.THRESH_BINARY_INV
        )
        dark_count = np.count_nonzero(mask_dark)

        # Pick whichever gives a reasonable stitch signal (but not the whole image)
        img_pixels = rot_img.shape[0] * rot_img.shape[1]
        b_ratio = bright_count / max(img_pixels, 1)
        d_ratio = dark_count / max(img_pixels, 1)

        # A real stitch covers roughly 1-40 % of the image
        def _valid(r: float) -> bool:
            return 0.01 < r < 0.40

        if _valid(b_ratio) and (not _valid(d_ratio) or b_ratio < d_ratio):
            thread_mask = mask_bright
        elif _valid(d_ratio):
            thread_mask = mask_dark
        elif bright_count > 0:
            thread_mask = mask_bright
        else:
            thread_mask = mask_dark

        proj = np.sum(thread_mask, axis=0) / 255.0
        return thread_mask, proj

    # ──────────────────────────────────────────
    # Engine D: Projection Profiling
    # ──────────────────────────────────────────
    def _detect_projection_defects(
        self, proj: np.ndarray, rot_img: np.ndarray
    ) -> List[Dict[str, Any]]:
        h, w = rot_img.shape[:2]
        defects: List[Dict[str, Any]] = []

        # Smooth projection with sliding window (window=5) to reduce spike noise
        # before gap scanning → fewer false Skip Stitch detections
        smooth_proj = np.convolve(proj, np.ones(5) / 5.0, mode="same")

        # Gap scanning → Skip / Broken
        gap_counter = 0
        in_gap = False
        gap_start = 0

        for i, val in enumerate(smooth_proj):
            if val < 2:
                if not in_gap:
                    in_gap = True
                    gap_start = i
                gap_counter += 1
            else:
                if in_gap:
                    if gap_counter > self.gap_tolerance and gap_start > 10 and i < len(smooth_proj) - 10:
                        d_type = "Broken Stitch" if gap_counter > self.broken_gap_min else "Skip Stitch"
                        defects.append({
                            "x": gap_start, "y": 10,
                            "w": gap_counter, "h": h - 20,
                            "type": d_type,
                            "score": gap_counter,
                        })
                    in_gap = False
                    gap_counter = 0

        # Run-off detection (tuned: requires center_density > center_density_min)
        edge_margin = int(w * self.runoff_edge_margin)
        if edge_margin > 5 and w > 2 * edge_margin:
            center_density = np.mean(smooth_proj[edge_margin: -edge_margin])

            # Only trigger run-off if the seam is actually present in the center
            if center_density > self.center_density_min:
                left_density = np.mean(smooth_proj[:edge_margin])
                if 0 < left_density < center_density * self.runoff_density_ratio:
                    defects.append({
                        "x": 0, "y": 10,
                        "w": edge_margin, "h": h - 20,
                        "type": "Run-off Stitch",
                        "score": int((1.0 - (left_density / center_density)) * 100),
                    })

                right_density = np.mean(smooth_proj[-edge_margin:])
                if 0 < right_density < center_density * self.runoff_density_ratio:
                    defects.append({
                        "x": w - edge_margin, "y": 10,
                        "w": edge_margin, "h": h - 20,
                        "type": "Run-off Stitch",
                        "score": int((1.0 - (right_density / center_density)) * 100),
                    })
        return defects

    # ──────────────────────────────────────────
    # Engine E: Linear Regression (Crooked)
    # ──────────────────────────────────────────
    def _detect_crooked(self, thread_mask: np.ndarray) -> List[Dict[str, Any]]:
        h, w = thread_mask.shape[:2]
        defects: List[Dict[str, Any]] = []

        centroids_y: List[float] = []
        centroids_x: List[float] = []

        step = max(1, w // 100)
        for x in range(0, w, step):
            col = thread_mask[:, x]
            stitch_pixels = np.where(col > 0)[0]
            if len(stitch_pixels) > 5:
                centroids_y.append(float(np.mean(stitch_pixels)))
                centroids_x.append(float(x))

        if len(centroids_x) < self.min_centroid_count:
            return defects

        cx = np.array(centroids_x)
        cy = np.array(centroids_y)

        # ── GUARD 1: Y-spread check ──────────────────────────────────────────
        # A real seam runs in a NARROW horizontal band — stitch centroids cluster
        # close together in Y.  A knit/fabric pattern has pixels spread across
        # the full image height → large Y-spread.
        # If Y span > 35% of image height → fabric pattern, NOT a seam.
        y_spread = float(np.max(cy) - np.min(cy))
        if y_spread > h * 0.35:
            return defects

        # ── GUARD 2: X-span check ────────────────────────────────────────────
        # A real seam crosses at least 40% of the image width.
        x_span = float(np.max(cx) - np.min(cx))
        if x_span < w * 0.40:
            return defects

        n = len(cx)
        sum_x = np.sum(cx)
        sum_y = np.sum(cy)
        sum_xy = np.sum(cx * cy)
        sum_x2 = np.sum(cx ** 2)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return defects

        m = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / n

        y_pred = m * cx + b
        
        # --- NEW LOGIC: Use Deviation instead of R-squared ---
        # Calculate how far off the worst pixel is, and the average error
        max_dev = float(np.max(np.abs(cy - y_pred)))
        mse = np.sum((cy - y_pred) ** 2) / n
        
        # Flag as crooked if the stitch deviates by more than 8 pixels 
        # from a straight line, or if the average error is high.
        if max_dev > 8.0 or mse > 5.0:
            x_start = int(np.min(cx))
            x_end = int(np.max(cx))
            
            # Map deviation to a 10-99% confidence score
            score = min(99, int((max_dev / 8.0) * 40)) 
            
            defects.append({
                "x": x_start,
                "y": max(0, int(np.min(cy) - 20)),
                "w": max(1, x_end - x_start),
                "h": min(h, int(np.max(cy) - np.min(cy) + 40)),
                "type": "Crooked Stitch",
                "score": score,
                "max_deviation_px": round(max_dev, 1),
                "mse": round(mse, 2)
            })

        return defects
    # ──────────────────────────────────────────
    # Engine F: Laplacian Variance (Pucker)
    # ──────────────────────────────────────────
    def _detect_pucker(
        self, img_gray: np.ndarray, thread_mask: np.ndarray
    ) -> List[Dict[str, Any]]:
        h, w = img_gray.shape[:2]
        defects: List[Dict[str, Any]] = []

        row_sums = np.sum(thread_mask, axis=1)
        seam_rows = np.where(row_sums > w * 0.2)[0]
        if len(seam_rows) < 5:
            return defects

        seam_top = max(0, int(np.min(seam_rows)) - 30)
        seam_bottom = min(h, int(np.max(seam_rows)) + 30)

        seam_region = img_gray[seam_top:seam_bottom, :]
        if seam_region.shape[0] < 10 or seam_region.shape[1] < 10:
            return defects

        patch_size = 32
        step = 16
        variances: List[float] = []
        patch_positions: List[int] = []

        for x in range(0, seam_region.shape[1] - patch_size, step):
            patch = seam_region[:, x : x + patch_size]
            variances.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
            patch_positions.append(x)

        if len(variances) < 5:
            return defects

        var_arr = np.array(variances)
        mean_var = np.mean(var_arr)
        std_var = np.std(var_arr)
        if std_var < 1e-6:
            return defects

        threshold = mean_var + self.pucker_var_sigma * std_var

        # ── FIX: use index-based tracking instead of position lookup ──
        pucker_start_idx: Optional[int] = None
        for idx, (var, x_pos) in enumerate(zip(variances, patch_positions)):
            if var > threshold:
                if pucker_start_idx is None:
                    pucker_start_idx = idx
            else:
                if pucker_start_idx is not None:
                    _start_x = patch_positions[pucker_start_idx]
                    pucker_w = x_pos - _start_x
                    if pucker_w > patch_size:
                        span = var_arr[pucker_start_idx : idx]
                        z_score = (np.max(span) - mean_var) / max(std_var, 1e-6)
                        defects.append({
                            "x": _start_x, "y": seam_top,
                            "w": pucker_w, "h": seam_bottom - seam_top,
                            "type": "Pucker",
                            "score": min(99, int(z_score * 20)),
                        })
                    pucker_start_idx = None

        # Handle pucker at end of image
        if pucker_start_idx is not None:
            _start_x = patch_positions[pucker_start_idx]
            pucker_w = patch_positions[-1] - _start_x
            if pucker_w > patch_size:
                defects.append({
                    "x": _start_x, "y": seam_top,
                    "w": pucker_w, "h": seam_bottom - seam_top,
                    "type": "Pucker",
                    "score": 60,
                })

        return defects

    # ──────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────
    def detect_defects(
        self,
        img_buffer: BinaryIO,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, None, None, np.ndarray, List[Dict[str, Any]]]:
        if settings is None:
            settings = {}

        self.stitch_thresh = settings.get("STITCH_COLOR_THRESH", self.stitch_thresh)
        self.gap_tolerance = settings.get("GAP_TOLERANCE", self.gap_tolerance)

        img_orig, img_small, img_gray, scale = self._preprocess(img_buffer)

        # 1. Deskew
        rot_img, _ = self._deskew(img_gray)

        # 2. Stitch mask + projection
        thread_mask, proj = self._extract_stitch_mask(rot_img)

        # Bail out if there's barely any signal OR if the signal is massive background noise
        stitch_density = np.count_nonzero(thread_mask) / max(
            thread_mask.shape[0] * thread_mask.shape[1], 1
        )
        if stitch_density < 0.015 or stitch_density > 0.35:
            return img_orig, None, None, img_small.copy(), []

        # 3. Run all three engines
        all_raw: List[Dict[str, Any]] = []
        all_raw.extend(self._detect_projection_defects(proj, rot_img))
        all_raw.extend(self._detect_crooked(thread_mask))
        all_raw.extend(self._detect_pucker(img_gray, thread_mask))

        # Draw
        output_img = img_small.copy()
        type_colors = {
            "Skip Stitch": (0, 0, 255),
            "Broken Stitch": (0, 0, 200),
            "Run-off Stitch": (0, 165, 255),
            "Crooked Stitch": (255, 0, 255),
            "Pucker": (255, 255, 0),
        }

        defect_log: List[Dict[str, Any]] = []
        for d in all_raw:
            color = type_colors.get(d["type"], (0, 0, 255))
            cv2.rectangle(
                output_img, (d["x"], d["y"]),
                (d["x"] + d["w"], d["y"] + d["h"]), color, 2,
            )
            cv2.putText(
                output_img, d["type"], (d["x"], max(d["y"] - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

            confidence = min(99, max(10, d.get("score", 50)))
            entry: Dict[str, Any] = {
                "ID": len(defect_log) + 1,
                "Type": d["type"],
                "Group": "Stitch Quality",
                "Area (px)": int(d["w"] * d["h"] / (scale ** 2)),
                "Confidence": f"{confidence}%",
                "bbox_x": int(d["x"] / scale),
                "bbox_y": int(d["y"] / scale),
                "bbox_w": int(d["w"] / scale),
                "bbox_h": int(d["h"] / scale),
            }
            if "r_squared" in d:
                entry["R²"] = d["r_squared"]
            if "max_deviation_px" in d:
                entry["Max Deviation (px)"] = d["max_deviation_px"]
            defect_log.append(entry)

        return img_orig, None, None, output_img, defect_log


seam_inspector = SeamInspector()
