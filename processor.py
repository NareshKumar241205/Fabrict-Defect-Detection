import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk

class TextureInspector:
    def __init__(self):
        # Industry Standard: LBP with Radius 3 is optimal for textile weave
        self.RADIUS = 3
        self.N_POINTS = 8 * self.RADIUS
        # 'uniform' method makes it Rotation Invariant
        self.METHOD = 'uniform'
        
    def _preprocess(self, img_buffer):
        """Standardizes input resolution and color space."""
        if hasattr(img_buffer, 'seek'): img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Optimization: Resize large 4K images to 800px width for real-time speed
        h, w = img.shape[:2]
        target_w = 800
        scale = target_w / w
        target_h = int(h * scale)
        
        img_small = cv2.resize(img, (target_w, target_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        return img, img_small, img_gray, scale

    def compute_texture_map(self, img_gray):
        """Generates the LBP Texture Feature Map."""
        lbp = local_binary_pattern(img_gray, self.N_POINTS, self.RADIUS, self.METHOD)
        lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255
        return lbp_norm.astype(np.uint8)

    def compute_entropy_map(self, lbp_img):
        """Calculates Local Entropy (Randomness)."""
        ent_img = entropy(lbp_img, disk(5))
        ent_norm = cv2.normalize(ent_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return ent_norm

    def detect_defects(self, img_buffer, sensitivity=3.0, min_area=200):
        """Main Pipeline: Returns Original, LBP, Entropy, Final Result, and Data Log."""
        # 1. Ingest
        orig_full, img_small, img_gray, scale_factor = self._preprocess(img_buffer)
        
        # 2. Feature Extraction
        lbp_map = self.compute_texture_map(img_gray)
        entropy_map = self.compute_entropy_map(lbp_map)
        
        # 3. Statistical Anomaly Detection (Z-Score)
        mean_ent = np.mean(entropy_map)
        std_ent = np.std(entropy_map)
        
        lower_bound = mean_ent - (sensitivity * std_ent)
        upper_bound = mean_ent + (sensitivity * std_ent)
        
        # Find Outliers
        mask_low = cv2.inRange(entropy_map, 0, lower_bound)
        mask_high = cv2.inRange(entropy_map, upper_bound, 255)
        mask_combined = cv2.bitwise_or(mask_low, mask_high)
        
        # 4. Cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 5. Classification
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
        
        defect_list = []
        final_output = orig_full.copy()
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            real_area = area / (scale_factor ** 2)
            
            if real_area < min_area: continue
            
            # Extract ROI (Scaled)
            x = int(stats[i, cv2.CC_STAT_LEFT] / scale_factor)
            y = int(stats[i, cv2.CC_STAT_TOP] / scale_factor)
            w = int(stats[i, cv2.CC_STAT_WIDTH] / scale_factor)
            h = int(stats[i, cv2.CC_STAT_HEIGHT] / scale_factor)
            
            # Logic
            aspect_ratio = float(w) / h
            roi = img_gray[stats[i, cv2.CC_STAT_TOP]:stats[i, cv2.CC_STAT_TOP]+stats[i, cv2.CC_STAT_HEIGHT], 
                           stats[i, cv2.CC_STAT_LEFT]:stats[i, cv2.CC_STAT_LEFT]+stats[i, cv2.CC_STAT_WIDTH]]
            roi_mean = np.mean(roi) if roi.size > 0 else 0
            bg_mean = np.mean(img_gray)
            
            if roi_mean < (bg_mean - 40):
                name = "Stain / Oil"
                color = (0, 140, 255) # Orange
            elif aspect_ratio > 3.0:
                name = "Cut / Tear"
                color = (0, 0, 255) # Red
            else:
                name = "Texture Defect"
                color = (255, 0, 255) # Magenta

            defect_list.append({
                "ID": i, "Type": name, "Area (px)": int(real_area),
                "Confidence": f"{min(99, int(abs(mean_ent - 128)))}%"
            })
            
            # Drawing
            cv2.rectangle(final_output, (x, y), (x+w, y+h), color, 4)
            label = f"{name}"
            cv2.putText(final_output, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return orig_full, lbp_map, entropy_map, final_output, defect_list

# Initialize Instance
inspector = TextureInspector()