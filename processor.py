import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.signal import find_peaks
import config as cfg

class QualityEngine:
    def __init__(self):
        pass

    def _preprocess(self, img_buffer):
        """Standardize image buffer to CV2 format."""
        if hasattr(img_buffer, 'seek'): img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Resize logic
        h, w = img.shape[:2]
        scale = cfg.IMAGE_RESIZE_WIDTH / w
        new_h = int(h * scale)
        img_small = cv2.resize(img, (cfg.IMAGE_RESIZE_WIDTH, new_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        return img_small, img_gray, scale

    # ======================================================
    # ALGORITHM 1: Difference of Gaussians (For DENIM/TEXTURE)
    # ======================================================
    def analyze_textured(self, img_gray, scale):
        # 1. Bandpass Filter (DoG)
        # Cancels out the weave pattern, leaves the defects
        g1 = cv2.GaussianBlur(img_gray, (0, 0), cfg.TEXTURED_SETTINGS["SIGMA_FINE"])
        g2 = cv2.GaussianBlur(img_gray, (0, 0), cfg.TEXTURED_SETTINGS["SIGMA_COARSE"])
        dog = cv2.absdiff(g1, g2)
        
        # 2. Thresholding
        _, mask = cv2.threshold(dog, cfg.TEXTURED_SETTINGS["THRESHOLD"], 255, cv2.THRESH_BINARY)
        
        # 3. Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        return self._find_blobs(mask, scale, cfg.TEXTURED_SETTINGS["MIN_AREA"], "Structural Defect"), dog

    # ======================================================
    # ALGORITHM 2: LBP + Entropy (For SMOOTH FABRICS)
    # ======================================================
    def analyze_smooth(self, img_gray, scale):
        # 1. Texture Map
        radius = cfg.SMOOTH_SETTINGS["LBP_RADIUS"]
        lbp = local_binary_pattern(img_gray, 8 * radius, radius, 'uniform')
        lbp_uint8 = (lbp / np.max(lbp) * 255).astype(np.uint8)
        
        # 2. Entropy (Chaos) Map
        ent = entropy(lbp_uint8, disk(5))
        ent_norm = cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. Statistical Anomaly Detection
        mean, std = np.mean(ent_norm), np.std(ent_norm)
        thresh_val = mean - (cfg.SMOOTH_SETTINGS["SENSITIVITY"] * std)
        
        _, mask = cv2.threshold(ent_norm, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return self._find_blobs(mask, scale, cfg.SMOOTH_SETTINGS["MIN_AREA"], "Texture Defect"), ent_norm

    # ======================================================
    # ALGORITHM 3: Signal Analysis (For SEAMS)
    # ======================================================
    def analyze_seam(self, img_gray, scale):
        h, w = img_gray.shape
        defects = []
        
        # 1. Find the Seam Line (Horizontal projection)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        proj_y = np.sum(np.abs(sobel_y), axis=1)
        seam_y = np.argmax(proj_y)
        
        # 2. Extract 1D Signal (Brightness profile along the seam)
        margin = cfg.SEAM_SETTINGS["ROI_HEIGHT"] // 2
        y1, y2 = max(0, seam_y - margin), min(h, seam_y + margin)
        roi = img_gray[y1:y2, :]
        
        # Vertical summation of ROI creates a 1D signal of stitches
        signal = np.sum(roi, axis=0)
        # Invert so stitches are peaks
        signal = np.max(signal) - signal 
        
        # 3. Peak Detection (Find individual stitches)
        peaks, _ = find_peaks(signal, prominence=cfg.SEAM_SETTINGS["PEAK_HEIGHT"], distance=10)
        
        # 4. Rhythm Check
        if len(peaks) > 5:
            intervals = np.diff(peaks)
            median_gap = np.median(intervals)
            
            for i, gap in enumerate(intervals):
                # If a gap is too wide, it's a skip stitch
                if gap > (median_gap * cfg.SEAM_SETTINGS["GAP_TOLERANCE"]):
                    x1 = peaks[i]
                    x2 = peaks[i+1]
                    defects.append({
                        "Type": "Skip Stitch",
                        "Loc": (x1, seam_y, x2-x1, 40), # x, y, w, h
                        "Area": int(gap)
                    })
        
        return defects, None

    # ======================================================
    # UTILITIES
    # ======================================================
    def _find_blobs(self, mask, scale, min_area, label):
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        defects = []
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area: continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            defects.append({
                "Type": label,
                "Loc": (x, y, w, h),
                "Area": int(area / scale**2)
            })
        return defects

    def run(self, img_file, mode):
        img_small, img_gray, scale = self._preprocess(img_file)
        result_img = img_small.copy()
        debug_img = None
        defects = []

        # ROUTING LOGIC
        if mode == "Smooth Fabric":
            defects, debug_img = self.analyze_smooth(img_gray, scale)
        elif mode == "Textured / Denim":
            defects, debug_img = self.analyze_textured(img_gray, scale)
        elif mode == "Seam Assembly":
            defects, debug_img = self.analyze_seam(img_gray, scale)

        # VISUALIZATION
        for d in defects:
            x, y, w, h = d["Loc"]
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(result_img, d["Type"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return result_img, defects, debug_img

# Singleton
engine = QualityEngine()