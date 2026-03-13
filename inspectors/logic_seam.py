# inspectors/logic_seam.py
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO, Optional
from config import LOGIC_SEAM_SETTINGS, SEAM_SETTINGS

class LogicSeamInspector:
    """PIPELINE 2: Detects Stitch Quality issues (Skip/Miss/Crooked Stitch)."""

    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if hasattr(img_buffer, "seek"):
            img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        target_w = 800
        scale = target_w / w
        img_small = cv2.resize(img, (target_w, int(h * scale)))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, scale

    def _deskew(self, img_gray: np.ndarray) -> Tuple[np.ndarray, float]:
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)

        if lines is None or len(lines) == 0:
            return img_gray, 0.0

        angles: List[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45 and abs(angle) > 0.5:
                angles.append(angle)

        if not angles:
            return img_gray, 0.0

        median_angle = float(np.median(angles))
        center = (img_gray.shape[1] // 2, img_gray.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rot_img = cv2.warpAffine(img_gray, M, (img_gray.shape[1], img_gray.shape[0]))
        return rot_img, median_angle

    def _extract_stitch_mask(self, rot_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, mask_bright = cv2.threshold(rot_img, LOGIC_SEAM_SETTINGS["STITCH_COLOR_THRESH"], 255, cv2.THRESH_BINARY)
        bright_count = np.count_nonzero(mask_bright)

        dark_thresh = 255 - LOGIC_SEAM_SETTINGS["STITCH_COLOR_THRESH"]
        _, mask_dark = cv2.threshold(rot_img, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        dark_count = np.count_nonzero(mask_dark)

        img_pixels = rot_img.shape[0] * rot_img.shape[1]
        b_ratio = bright_count / max(img_pixels, 1)
        d_ratio = dark_count / max(img_pixels, 1)

        def _valid(r: float) -> bool: return 0.01 < r < 0.40

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

    def _detect_projection_defects(self, proj: np.ndarray, rot_img: np.ndarray) -> List[Dict[str, Any]]:
        h, w = rot_img.shape[:2]
        defects: List[Dict[str, Any]] = []

        gap_counter = 0
        in_gap = False
        gap_start = 0

        for i, val in enumerate(proj):
            if val < 2:
                if not in_gap:
                    in_gap = True
                    gap_start = i
                gap_counter += 1
            else:
                if in_gap:
                    if gap_counter > LOGIC_SEAM_SETTINGS["GAP_TOLERANCE"] and gap_start > 10 and i < len(proj) - 10:
                        d_type = "Broken Stitch" if gap_counter > LOGIC_SEAM_SETTINGS["BROKEN_GAP_MIN"] else "Skip Stitch"
                        defects.append({"x": gap_start, "y": 10, "w": gap_counter, "h": h - 20, "type": d_type, "score": min(99, 40 + gap_counter)})
                    in_gap = False
                    gap_counter = 0

        edge_margin = int(w * LOGIC_SEAM_SETTINGS["RUNOFF_EDGE_MARGIN"])
        if edge_margin > 5 and w > 2 * edge_margin:
            center_density = np.mean(proj[edge_margin : -edge_margin])
            left_density = np.mean(proj[:edge_margin])
            if center_density > 5 and 0 < left_density < center_density * 0.35:
                defects.append({"x": 0, "y": 10, "w": edge_margin, "h": h - 20, "type": "Run-off Stitch", "score": int((1.0 - (left_density / center_density)) * 100)})

            right_density = np.mean(proj[-edge_margin:])
            if center_density > 5 and 0 < right_density < center_density * 0.35:
                defects.append({"x": w - edge_margin, "y": 10, "w": edge_margin, "h": h - 20, "type": "Run-off Stitch", "score": int((1.0 - (right_density / center_density)) * 100)})
        return defects

    def _detect_crooked(self, thread_mask: np.ndarray) -> List[Dict[str, Any]]:
        h, w = thread_mask.shape[:2]
        defects: List[Dict[str, Any]] = []

        centroids_y, centroids_x = [], []
        step = max(1, w // 100)
        for x in range(0, w, step):
            col = thread_mask[:, x]
            stitch_pixels = np.where(col > 0)[0]
            if len(stitch_pixels) > 5:
                centroids_y.append(float(np.mean(stitch_pixels)))
                centroids_x.append(float(x))

        if len(centroids_x) < (w * 0.15):
            return defects

        cx, cy = np.array(centroids_x), np.array(centroids_y)
        n = len(cx)
        sum_x, sum_y, sum_xy, sum_x2 = np.sum(cx), np.sum(cy), np.sum(cx * cy), np.sum(cx ** 2)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10: return defects

        m = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - m * sum_x) / n

        y_pred = m * cx + b
        max_dev = float(np.max(np.abs(cy - y_pred)))
        mse = np.sum((cy - y_pred) ** 2) / n
        
        if max_dev > 8.0 or mse > 5.0:
            x_start, x_end = int(np.min(cx)), int(np.max(cx))
            score = min(99, int((max_dev / 8.0) * 40)) 
            defects.append({
                "x": x_start, "y": max(0, int(np.min(cy) - 20)),
                "w": max(1, x_end - x_start), "h": min(h, int(np.max(cy) - np.min(cy) + 40)),
                "type": "Crooked Stitch", "score": score
            })

        return defects

    def _detect_puckering(self, proj: np.ndarray, rot_img: np.ndarray) -> List[Dict[str, Any]]:
        """Detect puckering using periodicity analysis and entropy filtering."""
        h, w = rot_img.shape[:2]
        defects: List[Dict[str, Any]] = []

        # Periodicity Analysis using 1D FFT
        proj_norm = proj - np.mean(proj)
        fft = np.fft.fft(proj_norm)
        freqs = np.fft.fftfreq(len(proj))
        
        # Focus on positive frequencies, corresponding to spacing of 15-25 pixels
        pos_freqs = freqs[:len(freqs)//2]
        magnitudes = np.abs(fft[:len(fft)//2])
        
        # Find spacing in pixels (inverse of frequency)
        spacings = 1.0 / pos_freqs[1:]  # Skip DC component
        valid_spacings = spacings[(spacings >= 15) & (spacings <= 25)]
        
        if len(valid_spacings) > 0:
            # Check if there's a significant peak at pucker frequency
            peak_idx = np.argmax(magnitudes[1:]) + 1  # +1 because we skipped DC
            peak_spacing = spacings[peak_idx - 1] if peak_idx < len(spacings) else 0
            
            if 15 <= peak_spacing <= 25 and magnitudes[peak_idx] > np.mean(magnitudes) * 2:
                # Found periodic pattern, likely puckering
                # Now check local entropy in regions of high projection variation
                var_threshold = np.mean(proj) + SEAM_SETTINGS["PUCKER_VAR_SIGMA"] * np.std(proj)
                high_var_regions = np.where(proj > var_threshold)[0]
                
                if len(high_var_regions) > 0:
                    # Group consecutive high variance regions
                    groups = []
                    current_group = [high_var_regions[0]]
                    
                    for i in range(1, len(high_var_regions)):
                        if high_var_regions[i] - high_var_regions[i-1] <= 5:  # Close proximity
                            current_group.append(high_var_regions[i])
                        else:
                            groups.append(current_group)
                            current_group = [high_var_regions[i]]
                    groups.append(current_group)
                    
                    # Filter groups by minimum length (at least 3 peaks for periodicity)
                    for group in groups:
                        if len(group) >= 3:
                            x_start, x_end = group[0], group[-1]
                            y_start, y_end = 10, h - 10
                            
                            # Entropy check: compare ROI entropy to surrounding fabric
                            roi = rot_img[y_start:y_end, x_start:x_end]
                            if roi.size > 0:
                                roi_entropy = self._calculate_entropy(roi)
                                
                                # Get surrounding fabric entropy (left and right of ROI)
                                left_roi = rot_img[y_start:y_end, max(0, x_start-50):x_start]
                                right_roi = rot_img[y_start:y_end, x_end:min(w, x_end+50)]
                                
                                surround_entropy = 0
                                count = 0
                                if left_roi.size > 0:
                                    surround_entropy += self._calculate_entropy(left_roi)
                                    count += 1
                                if right_roi.size > 0:
                                    surround_entropy += self._calculate_entropy(right_roi)
                                    count += 1
                                
                                if count > 0:
                                    surround_entropy /= count
                                    
                                    # If entropy is similar but intensity varies, it's puckering
                                    entropy_ratio = roi_entropy / max(surround_entropy, 1e-8)
                                    if 0.8 <= entropy_ratio <= 1.2:  # Similar entropy
                                        score = min(99, int(len(group) * 10))
                                        defects.append({
                                            "x": x_start, "y": y_start,
                                            "w": max(1, x_end - x_start), "h": y_end - y_start,
                                            "type": "Seam Puckering", "score": score
                                        })

        return defects

    def _calculate_entropy(self, img: np.ndarray) -> float:
        """Calculate Shannon entropy of an image."""
        if img.size == 0:
            return 0.0
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def detect_defects(self, img_buffer: BinaryIO) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        img_orig, img_small, img_gray, scale = self._preprocess(img_buffer)
        rot_img, _ = self._deskew(img_gray)
        thread_mask, proj = self._extract_stitch_mask(rot_img)

        stitch_density = np.count_nonzero(thread_mask) / max(thread_mask.shape[0] * thread_mask.shape[1], 1)
        if stitch_density < 0.015 or stitch_density > 0.35:
            return [], rot_img

        all_raw = self._detect_projection_defects(proj, rot_img) + self._detect_crooked(thread_mask) + self._detect_puckering(proj, rot_img)

        defect_log: List[Dict[str, Any]] = []
        for d in all_raw:
            confidence = min(99, max(10, d.get("score", 50)))
            defect_log.append({
                "Type": d["type"],
                "Area (px)": int(d["w"] * d["h"] / (scale ** 2)),
                "Quality Score": f"{confidence}%",
                "bbox_x": int(d["x"] / scale),
                "bbox_y": int(d["y"] / scale),
                "bbox_w": int(d["w"] / scale),
                "bbox_h": int(d["h"] / scale),
                "Pipeline": "Logic",
                "Category": "Structural"
            })

        viz_maps = {
            "logic_seam_map": rot_img,
            "grayscale_clahe": img_gray,
            "projection_profile": proj,
            "binary_thread_mask": thread_mask,
            "deskewed_image": rot_img,
        }

        return defect_log, viz_maps