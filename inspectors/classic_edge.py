# inspectors/classic_edge.py
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO
from skimage.filters import threshold_sauvola
from config import CLASSIC_SETTINGS

class ClassicEdgeInspector:
    """PIPELINE 1: Detects Holes, Tears, and Oil Stains using Adaptive Background Subtraction."""

    def __init__(self):
        self.defects: List[Dict[str, Any]] = []

    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        if hasattr(img_buffer, "seek"):
            img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        h, w = img.shape[:2]
        scale = CLASSIC_SETTINGS["IMAGE_RESIZE_WIDTH"] / w
        target_h = int(h * scale)

        img_small = cv2.resize(img, (CLASSIC_SETTINGS["IMAGE_RESIZE_WIDTH"], target_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, img_hsv, scale

    def _compute_background_subtraction(self, img_gray: np.ndarray) -> np.ndarray:
        img_blur = cv2.GaussianBlur(img_gray, (25, 25), 0)
        bg = cv2.GaussianBlur(img_gray, (151, 151), 0)
        return cv2.subtract(bg, img_blur)

    def detect_defects(self, img_buffer: BinaryIO, sensitivity: float = 2.5) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        img, img_small, img_gray, img_hsv, scale = self._preprocess(img_buffer)
        diff_map = self._compute_background_subtraction(img_gray)
        
        mean_diff = float(np.mean(diff_map))
        std_diff = float(np.std(diff_map))
        floor_mult = max(2.0, 4.5 - sensitivity * 0.6)
        global_floor = max(20, min(mean_diff + floor_mult * std_diff, 120))

        window_size = max(15, int(151 - sensitivity * 28)) | 1
        sauvola_k = max(0.05, 0.5 - sensitivity * 0.08)
        sauvola_thresh = threshold_sauvola(diff_map, window_size=window_size, k=sauvola_k)
        effective_thresh = np.maximum(global_floor, sauvola_thresh)
        binary = ((diff_map > effective_thresh) * 255).astype(np.uint8)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.defects = []
        proc_h, proc_w = img_gray.shape[:2]
        img_total_area = proc_h * proc_w
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < CLASSIC_SETTINGS["HOLE_AREA_MIN"] or area > (img_total_area * CLASSIC_SETTINGS["MAX_BOX_AREA_RATIO"]):
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            
            aspect_ratio = w / max(h, 1)
            hull = cv2.convexHull(contour)
            hull_area = max(cv2.contourArea(hull), 1)
            solidity = area / hull_area

            if solidity < 0.2:
                continue

            roi_gray = img_gray[y:y+h, x:x+w]
            inner_mean = float(np.mean(roi_gray)) if roi_gray.size > 0 else 0.0
            
            expand_x, expand_y = max(10, w // 2), max(10, h // 2)
            ny1, nx1 = max(0, y - expand_y), max(0, x - expand_x)
            ny2, nx2 = min(proc_h, y + h + expand_y), min(proc_w, x + w + expand_x)
            neigh = img_gray[ny1:ny2, nx1:nx2]
            neigh_mean = max(1.0, float(np.mean(neigh)))
            darkness_ratio = inner_mean / neigh_mean 

            inner_var = float(np.var(roi_gray.astype(np.float32))) if roi_gray.size > 0 else 0.0
            neigh_var = max(1.0, float(np.var(neigh.astype(np.float32))))
            texture_ratio = inner_var / neigh_var  

            contour_mask = np.zeros(img_gray.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=5)
            sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
            boundary_pixels = grad_mag[contour_mask > 0]
            boundary_gradient = float(np.mean(boundary_pixels)) if boundary_pixels.size > 0 else 0.0
            global_gradient = max(1.0, float(np.mean(grad_mag)))
            has_torn_edges = boundary_gradient > global_gradient * 1.3

            roi_sat = img_hsv[y:y+h, x:x+w, 1]
            global_sat_mean = float(np.mean(img_hsv[:, :, 1]))
            local_sat_mean = float(np.mean(roi_sat)) if roi_sat.size > 0 else global_sat_mean
            has_sat_shift = (global_sat_mean > 2.0) and (local_sat_mean > global_sat_mean * 1.2)

            is_stain = (solidity > 0.6 and darkness_ratio > 0.50 and not has_torn_edges and (has_sat_shift or texture_ratio > 0.35))

            if is_stain:
                d_type = "Oil Stain"
            elif aspect_ratio > CLASSIC_SETTINGS["TEAR_ASPECT_RATIO_MIN"] or aspect_ratio < (1.0 / CLASSIC_SETTINGS["TEAR_ASPECT_RATIO_MIN"]):
                d_type = "Tear"
            else:
                d_type = "Hole"

            orig_h, orig_w = img.shape[:2]
            ox = max(0, int(x / scale))
            oy = max(0, int(y / scale))
            ow = max(1, min(int(w / scale), orig_w - ox))
            oh = max(1, min(int(h / scale), orig_h - oy))
            real_area = max(1, int(area / (scale**2)))

            self.defects.append({
                "Type": d_type,
                "Area (px)": real_area,
                "Solidity": f"{solidity:.2f}",
                "Confidence": f"{int(min(99, 50 + (solidity * 40)))}%",
                "bbox_x": ox, "bbox_y": oy, "bbox_w": ow, "bbox_h": oh,
                "Pipeline": "Classical",
                "Category": "Surface" if d_type == "Oil Stain" else "Structural"
            })

        return self.defects, diff_map