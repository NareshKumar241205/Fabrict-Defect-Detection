import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, BinaryIO, Optional
from config import SEAM_SETTINGS

logger = logging.getLogger(__name__)

class SeamInspector:
    def __init__(self):
        # Load defaults from config
        self.stitch_thresh = SEAM_SETTINGS.get("STITCH_COLOR_THRESH", 180)
        self.gap_tolerance = SEAM_SETTINGS.get("GAP_TOLERANCE", 10)

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
        
        # --- UPGRADE: Illumination Correction (CLAHE) ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_gray = clahe.apply(img_gray)
        
        return img, img_small, img_gray, scale

    def detect_defects(self, img_buffer: BinaryIO, settings: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, None, None, np.ndarray, List[Dict[str, Any]]]:
        if settings is None:
            settings = {}

        # Allow runtime overrides
        stitch_thresh = settings.get("STITCH_COLOR_THRESH", self.stitch_thresh)
        gap_tolerance = settings.get("GAP_TOLERANCE", self.gap_tolerance)

        img_orig, img_small, img_gray, scale = self._preprocess(img_buffer)
        
        # 1. Auto-Rotate (Deskew)
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
        
        rot_img = img_gray
        if lines is not None:
            for x1, y1, x2, y2 in lines[0]:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45 and abs(angle) > 0.5:
                    center = (img_gray.shape[1]//2, img_gray.shape[0]//2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rot_img = cv2.warpAffine(img_gray, M, (img_gray.shape[1], img_gray.shape[0]))
        
        # 2. Extract Stitch Line (Dynamic Threshold)
        _, thread_mask = cv2.threshold(rot_img, stitch_thresh, 255, cv2.THRESH_BINARY)
        
        # 3. Projection Profile
        proj = np.sum(thread_mask, axis=0) / 255 
        
        defects = []
        gap_counter = 0
        in_gap = False
        gap_start = 0

        # Scan projection
        for i, val in enumerate(proj):
            if val < 2: # Gap
                if not in_gap:
                    in_gap = True
                    gap_start = i
                gap_counter += 1
            else: # Thread found
                if in_gap:
                    if gap_counter > gap_tolerance:
                        # Avoid edges
                        if gap_start > 10 and i < (len(proj) - 10):
                            score = gap_counter
                            defects.append({
                                "x": gap_start, "y": 10, 
                                "w": gap_counter, "h": rot_img.shape[0]-20,
                                "type": "Skip Stitch",
                                "score": score
                            })
                    in_gap = False
                    gap_counter = 0

        # Draw results on output image
        output_img = img_small.copy()
        
        defect_log = []
        for d in defects:
            cv2.rectangle(output_img, (d['x'], d['y']), 
                         (d['x'] + d['w'], d['y'] + d['h']), (0, 0, 255), 2)
            cv2.putText(output_img, d['type'], (d['x'], d['y']-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            # Estimate area in original scale
            # Note: Seam detection works on 1D projection, so area is approximate
            # Confidence: how much the gap exceeds the tolerance threshold
            gap_confidence = min(99, int((d['score'] / max(gap_tolerance, 1)) * 50))
            defect_log.append({
                 "ID": len(defect_log) + 1,
                 "Type": d['type'],
                 "Area (px)": int(d['w'] * d['h'] * (scale**-2) if scale else d['w'] * d['h']),
                 "Confidence": f"{gap_confidence}%",
                 "bbox_x": int(d['x'] / scale) if scale else d['x'],
                 "bbox_y": int(d['y'] / scale) if scale else d['y'],
                 "bbox_w": int(d['w'] / scale) if scale else d['w'],
                 "bbox_h": int(d['h'] / scale) if scale else d['h']
            })

        return img_orig, None, None, output_img, defect_log

seam_inspector = SeamInspector()
