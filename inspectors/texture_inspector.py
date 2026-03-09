"""
Template Self-Similarity Inspector Module (Replaces Texture Inspector)
======================================================================
Uses local pattern analysis instead of global entropy.
The inspector automatically extracts a "clean" patch of fabric
(the lowest variance region) and uses Template Matching (NCC) to
slide it across the image. Areas where the cross-correlation
drops drastically are flagged as structural or cluster anomalies.

Detects: Slub, Tear
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, BinaryIO
from skimage.filters import threshold_sauvola
from config import UNIFIED_SETTINGS
from inspectors.defect_score import compute_severity

logger = logging.getLogger(__name__)

class TextureInspector:
    """Detects texture anomalies using dynamic Self-Similarity (Template Matching).
    (Kept class name TextureInspector for pipeline compatibility).
    """

    PROCESS_WIDTH = 800
    PATCH_SIZE = 64  # Size of the template patch

    def __init__(self):
        self.defects: List[Dict[str, Any]] = []

    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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

    def _compute_texture_deviation(self, img_gray: np.ndarray) -> np.ndarray:
        """True Normalized Cross-Correlation (NCC) based deviation map.

        1. Auto-extracts a clean 64×64 reference patch (region with the
           lowest local variance — assumed defect-free fabric).
        2. Slides the patch across the entire image using
           cv2.matchTemplate(TM_CCOEFF_NORMED).
        3. Returns a uint8 deviation map where HIGH values = LOW correlation
           (structural anomalies).
        """
        patch_size = self.PATCH_SIZE  # 64
        h, w = img_gray.shape[:2]

        if h < patch_size * 2 or w < patch_size * 2:
            return np.zeros((h, w), dtype=np.uint8)

        # ── Step 1: find the cleanest patch (lowest local variance) ──
        # Guard: only consider patches whose mean is close to the image
        # global mean — this prevents selecting a dark defect hole
        # (which has low variance but anomalous intensity) as "clean".
        step = patch_size // 2
        best_var = float('inf')
        best_patch = None

        global_mean = float(np.mean(img_gray))
        global_std = float(np.std(img_gray))
        mean_lo = global_mean - 1.5 * global_std
        mean_hi = global_mean + 1.5 * global_std

        for y in range(0, h - patch_size, step):
            for x in range(0, w - patch_size, step):
                patch = img_gray[y:y + patch_size, x:x + patch_size]
                patch_mean = float(np.mean(patch))
                # Reject patches with anomalous intensity (defect regions)
                if patch_mean < mean_lo or patch_mean > mean_hi:
                    continue
                local_var = float(np.var(patch.astype(np.float32)))
                # Reject near-constant patches (blank/background) with var < 1
                if 1.0 < local_var < best_var:
                    best_var = local_var
                    best_patch = patch.copy()

        if best_patch is None:
            return np.zeros((h, w), dtype=np.uint8)

        # ── Step 2: NCC sliding-window ──
        ncc_map = cv2.matchTemplate(
            img_gray, best_patch, cv2.TM_CCOEFF_NORMED
        )  # float32 in [-1, 1], shape = (h-ps+1, w-ps+1)

        # ── Step 3: correlation → deviation (invert and scale to 0-255) ──
        deviation = 1.0 - ncc_map          # high where correlation is low
        deviation = np.clip(deviation, 0, 2)
        deviation = (deviation / 2.0 * 255).astype(np.uint8)

        # Pad back to original image dimensions (matchTemplate shrinks)
        pad_y = patch_size // 2
        pad_x = patch_size // 2
        dh, dw = deviation.shape[:2]
        deviation_full = np.zeros((h, w), dtype=np.uint8)
        deviation_full[pad_y:pad_y + dh, pad_x:pad_x + dw] = deviation

        return deviation_full

    def process(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.5,
        debug_viz: bool = False
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        
        img, img_small, img_gray, scale = self._preprocess(img_buffer)
        
        # Compute texture deviation map
        dist_map = self._compute_texture_deviation(img_gray)
        
        # Hybrid Thresholding: global noise-floor + Sauvola local adaptation
        # The global floor kills the noisy baseline that Sauvola alone passes.
        # Sauvola handles local lighting variation that a global threshold misses.
        mean_dist = float(np.mean(dist_map))
        std_dist = float(np.std(dist_map))
        mult = max(1.0, 4.0 - (sensitivity * 0.5))
        global_floor = mean_dist + mult * std_dist
        global_floor = max(25, min(global_floor, 150))

        window_size = max(15, int(151 - sensitivity * 28)) | 1  # ensure odd
        sauvola_k = max(0.05, 0.5 - sensitivity * 0.08)
        sauvola_thresh = threshold_sauvola(
            dist_map, window_size=window_size, k=sauvola_k
        )
        # Effective threshold = stricter of global floor and local Sauvola
        effective_thresh = np.maximum(global_floor, sauvola_thresh)
        binary = ((dist_map > effective_thresh) * 255).astype(np.uint8)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.defects = []
        proc_h, proc_w = img_gray.shape[:2]
        img_total_area = proc_h * proc_w
        
        # Minimum area must be reasonably large to be a structural slub/tear
        min_area = UNIFIED_SETTINGS.get("HOLE_AREA_MIN", 800)
        max_area = img_total_area * UNIFIED_SETTINGS["MAX_BOX_AREA_RATIO"]

        result = img.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            # Geometry
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            aspect_ratio = w / max(h, 1)
            hull = cv2.convexHull(contour)
            hull_area = max(cv2.contourArea(hull), 1)
            solidity = area / hull_area

            # Scale back to original image
            ox, oy = max(0, int(x / scale)), max(0, int(y / scale))
            ow, oh = max(1, int(w / scale)), max(1, int(h / scale))
            real_area = max(1, int(area / (scale**2)))

            # Intensity check to distinguish Hole (dark) vs Slub (bright/thick)
            roi_gray = img_gray[y:y+h, x:x+w]
            defect_gray_mean = np.mean(roi_gray) if roi_gray.size > 0 else 127
            global_gray_mean = np.mean(img_gray)
            
            # Classification
            if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                d_type = "Tear"
                color = (0, 0, 255)
            elif defect_gray_mean < global_gray_mean * 0.80:
                d_type = "Hole"
                color = (255, 0, 0)
            elif solidity > 0.4:
                d_type = "Slub"
                color = (0, 165, 255)
            else:
                d_type = "Slub"
                color = (255, 0, 255)

            # ── Multi-metric severity scoring ──
            # Build a temporary defect dict for the scorer (processing coords)
            tmp_defect = {
                "Type": d_type,
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "Area (px)": area,
            }
            hsv_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
            score_result = compute_severity(
                tmp_defect, img_gray,
                hsv=hsv_small,
                deviation_map=dist_map,
                contour=contour,
            )
            severity = score_result["Severity"]
            score_details = score_result["Score_Details"]

            self.defects.append({
                "ID": 0,
                "Type": d_type,
                "Area (px)": real_area,
                "Solidity": f"{solidity:.2f}",
                "Confidence": f"{severity}%",
                "Severity": severity,
                "Score_Details": score_details,
                "bbox_x": ox, "bbox_y": oy, "bbox_w": ow, "bbox_h": oh,
                "Category": "Structural" if d_type in ("Hole", "Tear", "Missing Thread") else "Surface",
                "Group": "Fabric Structure",
                "Engine": "Adaptive Texture Analysis"
            })

            cv2.rectangle(result, (ox, oy), (ox + ow, oy + oh), color, 4)

        return self.defects, dist_map

# Expose instance
texture_inspector = TextureInspector()
