"""
Adaptive Background Subtractor (Replaces Edge Inspector)
========================================================
Detects Holes and Oil Stains.
Instead of relying on noisy Canny/Laplacian edge detectors,
this inspector calculates a heavily blurred "lighting background"
and subtracts it from the image. This flattens out shadows, folds,
and lighting gradients, leaving only true dark anomalies.

Detects: Hole, Oil Stain
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO
from skimage.filters import threshold_sauvola
from config import UNIFIED_SETTINGS
from inspectors.defect_score import compute_severity

class EdgeInspector:
    """Detects Holes and Oil Stains using Adaptive Background Subtraction.
    (Kept class name EdgeInspector for pipeline compatibility).
    """

    PROCESS_WIDTH = 800

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
        
        # Keep HSV for Oil Stain classification
        img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, img_hsv, scale

    def _compute_background_subtraction(self, img_gray: np.ndarray) -> np.ndarray:
        """Estimates lighting background and isolates dark anomalies.
        Uses a band-pass equivalent (Heavy Blur minus Light Blur) to ignore 
        microscopic dark knit holes while preserving massive true anomalies.
        """
        # 1. Light blur to destroy high-frequency knit texture and tiny dark spots
        # Kernel larger than individual knit holes (~10-15px) so they are erased.
        img_blur = cv2.GaussianBlur(img_gray, (25, 25), 0)
        
        # 2. Heavy blur to estimate the macro lighting gradient
        bg = cv2.GaussianBlur(img_gray, (151, 151), 0)
        
        # 3. Subtract: Background (lighting) minus Light Blur (features)
        # Deep large holes stay dark in img_blur, making them bright in diff.
        # Tiny knit holes are destroyed in img_blur, so they disappear from diff.
        diff = cv2.subtract(bg, img_blur)
        
        return diff

    def detect_defects(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.5,
        debug_viz: bool = False
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        
        img, img_small, img_gray, img_hsv, scale = self._preprocess(img_buffer)
        
        # 1. Subtraction Map
        diff_map = self._compute_background_subtraction(img_gray)
        
        # Hybrid Thresholding: global noise-floor + Sauvola local adaptation
        # The global floor kills the noisy baseline that Sauvola alone passes.
        # Sauvola handles local lighting variation that a global threshold misses.
        mean_diff = float(np.mean(diff_map))
        std_diff = float(np.std(diff_map))
        # Floor multiplier: just enough to cut texture noise, not the defect.
        # (Original 7.0-sens was for standalone use; now we only need a noise floor.)
        floor_mult = max(2.0, 4.5 - sensitivity * 0.6)
        global_floor = mean_diff + floor_mult * std_diff
        global_floor = max(20, min(global_floor, 120))

        window_size = max(15, int(151 - sensitivity * 28)) | 1  # ensure odd
        sauvola_k = max(0.05, 0.5 - sensitivity * 0.08)
        sauvola_thresh = threshold_sauvola(
            diff_map, window_size=window_size, k=sauvola_k
        )
        # Effective threshold = stricter of global floor and local Sauvola
        effective_thresh = np.maximum(global_floor, sauvola_thresh)
        binary = ((diff_map > effective_thresh) * 255).astype(np.uint8)
        
        # 3. Morphological Cleanup
        # Close small gaps in holes, then open to remove speckle noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.defects = []
        proc_h, proc_w = img_gray.shape[:2]
        img_total_area = proc_h * proc_w
        
        min_hole = UNIFIED_SETTINGS.get("HOLE_AREA_MIN", 800)
        max_area = img_total_area * UNIFIED_SETTINGS["MAX_BOX_AREA_RATIO"]

        result = img.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_hole or area > max_area:
                continue

            # Geometry
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            aspect_ratio = w / max(h, 1)
            hull = cv2.convexHull(contour)
            hull_area = max(cv2.contourArea(hull), 1)
            solidity = area / hull_area

            # Discard extremely thin/jagged noise (unless it's a massive tear handled by SSIM)
            if solidity < 0.2:
                continue

            # Check HSV Saturation for Oil Stain
            roi_sat = img_hsv[y:y+h, x:x+w, 1]
            global_sat_mean = np.mean(img_hsv[:,:,1])
            local_sat_mean = np.mean(roi_sat) if roi_sat.size > 0 else global_sat_mean
            
            # An Oil Stain is a soft structural defect with high solidity and a color/saturation shift
            is_stain = (solidity > 0.6) and (local_sat_mean > global_sat_mean * 1.2)
            
            # Scale back to original image
            ox, oy = max(0, int(x / scale)), max(0, int(y / scale))
            ow, oh = max(1, int(w / scale)), max(1, int(h / scale))
            real_area = max(1, int(area / (scale**2)))

            # Classification
            if is_stain:
                d_type = "Oil Stain"
                color = (0, 140, 255)
            elif aspect_ratio > 3.0 or aspect_ratio < 0.33:
                d_type = "Tear"
                color = (0, 0, 255)
            else:
                d_type = "Hole"
                color = (255, 0, 0)

            # ── Multi-metric severity scoring ──
            tmp_defect = {
                "Type": d_type,
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
                "Area (px)": area,
            }
            score_result = compute_severity(
                tmp_defect, img_gray,
                hsv=img_hsv,
                deviation_map=diff_map,
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
                "Category": "Surface" if d_type == "Oil Stain" else "Structural",
                "Group": "Fabric Structure",
                "Engine": "Adaptive Subtraction"
            })

            cv2.rectangle(result, (ox, oy), (ox + ow, oy + oh), color, 3)

        return self.defects, diff_map

# Expose instance
edge_inspector = EdgeInspector()
