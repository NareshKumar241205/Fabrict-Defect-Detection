"""
Gabor Filter Inspector Module (Replaces Spectral Inspector)
===========================================================
Uses directional Gabor filter banks to detect Missing Threads
and Snags. These defects perfectly disrupt the regular weave
pattern in specific directional frequencies (0°, 45°, 90°, 135°).

Detects: Missing Thread, Snag
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO
from config import UNIFIED_SETTINGS

class SpectralInspector:
    """Detects directional line defects using Gabor filter banks.
    (Kept class name SpectralInspector for pipeline compatibility).
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
        
        # CLAHE for contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, scale

    def _build_filters(self, ksize=31, sigma=4.0, theta_list=[0, np.pi/4, np.pi/2, 3*np.pi/4], lam=10.0, gamma=0.5):
        """Prebuilds a bank of Gabor filters to mimic human visual texture perception."""
        filters = []
        for theta in theta_list:
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lam, gamma, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()  # normalize
            filters.append(kern)
        return filters

    def _process_gabor(self, img_gray: np.ndarray, filters: List[np.ndarray]) -> np.ndarray:
        """Applies Gabor filter bank and returns the maximum directional response."""
        accum = np.zeros_like(img_gray, dtype=np.float32)
        for kern in filters:
            fimg = cv2.filter2D(img_gray, cv2.CV_32F, kern)
            np.maximum(accum, fimg, accum)
        
        # Normalize to 8-bit
        cv2.normalize(accum, accum, 0, 255, cv2.NORM_MINMAX)
        return accum.astype(np.uint8)

    def process(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.5,
        debug_viz: bool = False
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        
        img, img_small, img_gray, scale = self._preprocess(img_buffer)
        
        # Apply Gabor filter bank
        filters = self._build_filters(sigma=3.0, lam=8.0)  # Tune based on fabric weave size
        gabor_response = self._process_gabor(img_gray, filters)
        
        # Adaptive Threshold (Sauvola-like local thresholding)
        # We look for areas that spike heavily in the Gabor response (breaking weave)
        blur = cv2.GaussianBlur(gabor_response, (31, 31), 0)
        diff = cv2.absdiff(gabor_response, blur)
        
        # Threshold: deviation must be significant
        thresh_val = int(30 - (sensitivity * 2))  # Adjust sensitivity
        thresh_val = max(15, min(thresh_val, 60))
        _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological cleanup to connect broken threads
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.defects = []
        proc_h, proc_w = img_gray.shape[:2]
        img_total_area = proc_h * proc_w
        
        # Dynamic areas based on config
        min_snag = 150  # Snags can be tiny spikes
        max_area = img_total_area * UNIFIED_SETTINGS["MAX_BOX_AREA_RATIO"]

        result = img.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_snag or area > max_area:
                continue

            # Geometry
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            aspect_ratio = w / max(h, 1)
            hull = cv2.convexHull(contour)
            hull_area = max(cv2.contourArea(hull), 1)
            solidity = area / hull_area

            # Discard noisy/jagged web-like detections that aren't solid lines or snags
            if solidity < 0.15:
                continue

            # Scale back to original image
            ox, oy = max(0, int(x / scale)), max(0, int(y / scale))
            ow, oh = max(1, int(w / scale)), max(1, int(h / scale))
            real_area = max(1, int(area / (scale**2)))

            # Classification
            # Missing threads are highly elongated lines
            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                d_type = "Missing Thread"
                color = (0, 0, 255)  # Red BGR
            # Snags are smaller, intense local protrusions
            elif real_area < UNIFIED_SETTINGS.get("SNAG_AREA_MAX", 600):
                d_type = "Snag"
                color = (255, 105, 180)  # Hot Pink
            else:
                # If it doesn't fit the directional anomaly profile, skip it to let 
                # Template/SSIM inspector handle it
                continue

            # Calculate confidence based on Gabor spike intensity vs background
            roi_gabor = gabor_response[y:y+h, x:x+w]
            spike_val = np.mean(roi_gabor) if roi_gabor.size > 0 else 0
            bg_val = np.mean(blur[y:y+h, x:x+w]) if roi_gabor.size > 0 else 1
            ratio = (spike_val / max(bg_val, 1))
            
            # Bound confidence 40-99%
            confidence = min(99, max(40, int((ratio * 20))))

            self.defects.append({
                "ID": 0,
                "Type": d_type,
                "Area (px)": real_area,
                "Solidity": f"{solidity:.2f}",
                "Confidence": f"{confidence}%",
                "bbox_x": ox, "bbox_y": oy, "bbox_w": ow, "bbox_h": oh,
                "Category": "Structural",
                "Group": "Fabric Structure",
                "Engine": "Gabor Frequency Map"
            })

            cv2.rectangle(result, (ox, oy), (ox + ow, oy + oh), color, 3)
            
        return self.defects, gabor_response

# Expose instance
spectral_inspector = SpectralInspector()
