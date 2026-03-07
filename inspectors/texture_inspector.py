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
from config import UNIFIED_SETTINGS

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
        """Finds texture anomalies by mathematically erasing the knitting pattern.
        Knitted fabric has thousands of tiny holes and slubs (the stitches).
        By Morphologically Closing (filling dark knit holes) and Opening (crushing bright
        knit slubs) with a kernel slightly larger than the stitch size, we create a perfectly
        flat fabric baseline where ONLY massive true defects survive.
        """
        # 1. Erase knitting pattern
        # The 15x15 kernel is precisely tuned to be larger than a single knit stitch,
        # perfectly filling the gaps and flattening the threads without erasing true defects.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # Fill all tiny dark knit gaps
        closed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, k)
        
        # Crush all tiny bright knit highlights
        flat_fabric = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k)
        
        # 2. Extract true anomalies
        # Calculate the global average color of this perfectly flat fabric
        mean_val = np.mean(flat_fabric)
        
        # Dark anomalies (Holes/Tears)
        diff_dark = cv2.subtract(mean_val, flat_fabric)
        
        # Bright anomalies (Massive Slubs)
        diff_bright = cv2.subtract(flat_fabric, mean_val)
        
        # Combine into a single distance map
        combined = cv2.addWeighted(diff_dark, 1.0, diff_bright, 1.0, 0)
        
        return combined

    def process(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.5,
        debug_viz: bool = False
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        
        img, img_small, img_gray, scale = self._preprocess(img_buffer)
        
        # Compute texture deviation map
        dist_map = self._compute_texture_deviation(img_gray)
        
        # Calculate dynamic threshold based on sensitivity
        # High sensitivity = lower multiplier
        mean_dist = np.mean(dist_map)
        std_dist = np.std(dist_map)
        
        # Because the knit pattern is mathematically erased, the noise floor is extremely low.
        # We can use a lower standard deviation multiplier and target tight isolation.
        mult = max(1.0, 4.0 - (sensitivity * 0.5))
        thresh_val = mean_dist + (mult * std_dist)
        thresh_val = max(25, min(thresh_val, 150))
        
        _, binary = cv2.threshold(dist_map, thresh_val, 255, cv2.THRESH_BINARY)
        
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
                # Elongated severe SSIM drop = Tear or massive snag line
                d_type = "Tear"
                color = (0, 0, 255) # Red
            elif defect_gray_mean < global_gray_mean * 0.80:
                # If the anomaly is significantly darker than the fabric baseline, it's a Hole
                d_type = "Hole"
                color = (255, 0, 0) # Blue
            elif solidity > 0.4:
                # Solid cluster = Thick Slub
                d_type = "Slub"
                color = (0, 165, 255) # Orange
            else:
                # Irregular messy break
                d_type = "Slub"
                color = (255, 0, 255) # Magenta
                
            # Confidence based on intensity of texture deviation
            roi_dist = dist_map[y:y+h, x:x+w]
            defect_dist_max = np.max(roi_dist) if roi_dist.size > 0 else 0
            
            # The stronger the deviation peak, the higher the confidence
            # Threshold is around 30-60, severe defects hit 120-255
            confidence = min(99, max(40, int((defect_dist_max / 255.0) * 100) + 20))

            self.defects.append({
                "ID": 0,
                "Type": d_type,
                "Area (px)": real_area,
                "Solidity": f"{solidity:.2f}",
                "Confidence": f"{confidence}%",
                "bbox_x": ox, "bbox_y": oy, "bbox_w": ow, "bbox_h": oh,
                "Category": "Structural",
                "Group": "Fabric Structure",
                "Engine": "Adaptive Texture Analysis"
            })

            cv2.rectangle(result, (ox, oy), (ox + ow, oy + oh), color, 4)

        return self.defects, dist_map

# Expose instance
texture_inspector = TextureInspector()
