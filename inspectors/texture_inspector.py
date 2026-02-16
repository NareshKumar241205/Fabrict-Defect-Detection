import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, BinaryIO
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy
from skimage.morphology import disk

logger = logging.getLogger(__name__)

class TextureInspector:
    def __init__(self):
        # Industry Standard: LBP with Radius 3 is optimal for textile weave
        self.RADIUS = 3
        self.N_POINTS = 8 * self.RADIUS
        # 'uniform' method makes it Rotation Invariant
        self.METHOD = 'uniform'
        
    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Standardizes input resolution and color space."""
        if hasattr(img_buffer, 'seek'): img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None:
            raise ValueError("Could not decode image file")
        
        # Optimization: Resize large 4K images to 800px width for real-time speed
        h, w = img.shape[:2]
        target_w = 800
        scale = target_w / w
        target_h = int(h * scale)
        
        img_small = cv2.resize(img, (target_w, target_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # --- UPGRADE: Illumination Correction (CLAHE) ---
        # Enhances local contrast, making defects visible even in shadows
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_gray = clahe.apply(img_gray)
        
        return img, img_small, img_gray, scale

    def compute_texture_map(self, img_gray: np.ndarray) -> np.ndarray:
        """Generates the LBP Texture Feature Map."""
        lbp = local_binary_pattern(img_gray, self.N_POINTS, self.RADIUS, self.METHOD)
        lbp_norm = (lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255
        return lbp_norm.astype(np.uint8)

    def compute_entropy_map(self, lbp_img: np.ndarray) -> np.ndarray:
        """Calculates Local Entropy (Randomness)."""
        ent_img = entropy(lbp_img, disk(5))
        ent_norm = cv2.normalize(ent_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return ent_norm

    def compute_gabor_map(self, img_gray):
        """Gabor filter bank for directional defect detection.
        
        Uses 6 orientations × 3 frequencies = 18 filters.
        Fabric weave has strong directional patterns; a defect that disrupts
        one orientation lights up in that filter's response.
        """
        orientations = [0, 30, 60, 90, 120, 150]  # degrees
        frequencies = [0.05, 0.1, 0.2]              # cycles/pixel
        ksize = 31
        sigma = 4.0
        gamma = 0.5  # spatial aspect ratio
        
        responses = []
        for theta_deg in orientations:
            theta = np.deg2rad(theta_deg)
            for freq in frequencies:
                lambd = 1.0 / freq  # wavelength
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F
                )
                filtered = cv2.filter2D(img_gray, cv2.CV_32F, kernel)
                responses.append(np.abs(filtered))
        
        # Fuse: take max response across all 18 filters
        gabor_fused = np.maximum.reduce(responses)
        gabor_norm = cv2.normalize(gabor_fused, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gabor_norm

    def detect_defects(self, img_buffer, sensitivity=3.0, min_area=200):
        """Main Pipeline: Returns Original, LBP, Entropy, Final Result, and Data Log."""
        # 1. Ingest
        orig_full, img_small, img_gray, scale_factor = self._preprocess(img_buffer)
        h, w = img_gray.shape

        # --- UPGRADE: Multi-Scale Analysis (Image Pyramid) ---
        # Detects defects of varying sizes (e.g. tiny pinholes vs large tears)
        scales = [1.0, 0.75, 0.5] 
        entropy_maps = []

        for s in scales:
            # Resize
            curr_w, curr_h = int(w * s), int(h * s)
            resized_gray = cv2.resize(img_gray, (curr_w, curr_h))
            
            # Feature Extraction at this scale
            lbp = self.compute_texture_map(resized_gray)
            ent = self.compute_entropy_map(lbp)
            
            # Normalize back to original size for fusion
            ent_restored = cv2.resize(ent, (w, h))
            entropy_maps.append(ent_restored)

        # Fusion: Take the MAXIMUM anomaly score across all scales
        # This ensures that if a defect is visible at ANY scale, it is detected.
        final_entropy_map = np.maximum.reduce(entropy_maps)
        
        # --- UPGRADE: Gabor filter bank for directional defects ---
        gabor_map = self.compute_gabor_map(img_gray)
        # Fuse Gabor with Entropy: union of both anomaly detectors
        final_entropy_map = np.maximum(final_entropy_map, gabor_map)
        
        # 3. Statistical Anomaly Detection (Z-Score) on Fused Map
        mean_ent = np.mean(final_entropy_map)
        std_ent = np.std(final_entropy_map)
        
        lower_bound = mean_ent - (sensitivity * std_ent)
        upper_bound = mean_ent + (sensitivity * std_ent)
        
        # Find Outliers
        mask_low = cv2.inRange(final_entropy_map, 0, lower_bound)
        mask_high = cv2.inRange(final_entropy_map, upper_bound, 255)
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
            
            # Extract ROI
            x = int(stats[i, cv2.CC_STAT_LEFT] / scale_factor)
            y = int(stats[i, cv2.CC_STAT_TOP] / scale_factor)
            w = int(stats[i, cv2.CC_STAT_WIDTH] / scale_factor)
            h = int(stats[i, cv2.CC_STAT_HEIGHT] / scale_factor)

            # Ensure coordinates are within bounds
            x, y = max(0, x), max(0, y)
            
            # --- UPGRADE: Advanced Classification ---
            # Use Contour properties (Solidity) for better naming
            # Get specific contour for this component
            component_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            solidity = 0
            if contours:
                cnt = contours[0]
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = cv2.contourArea(cnt) / float(hull_area)
            
            aspect_ratio = float(w) / h
            
            # Smart Logic
            if solidity > 0.9:
                name = "Oil / Water Stain" # Very round/solid
                color = (0, 140, 255) # Orange
            elif aspect_ratio > 3.0:
                name = "Cut / Tear (Horiz)"
                color = (0, 0, 255) # Red
            elif aspect_ratio < 0.33:
                name = "Cut / Tear (Vert)"
                color = (0, 0, 255) # Red
            elif solidity < 0.5:
                # Distinguish based on area for "Ragged Hole" vs "Complex Texture"
                if real_area > 1000:
                   name = "Ragged Hole" # Irregular shape large
                else:
                   name = "Rough Weave"
                color = (255, 0, 0) # Blue
            else:
                name = "Texture Defect"
                color = (255, 0, 255) # Magenta

            # --- CONFIDENCE: Z-score distance from threshold ---
            # Measure how anomalous each defect's entropy is relative to the image
            defect_region_mask = (labels == i)
            defect_entropy_vals = final_entropy_map[defect_region_mask]
            defect_mean_ent = np.mean(defect_entropy_vals) if len(defect_entropy_vals) > 0 else mean_ent
            z_distance = abs(defect_mean_ent - mean_ent) / max(std_ent, 1e-6)
            confidence = min(99, int((z_distance / max(sensitivity, 1e-6)) * 100))

            defect_list.append({
                "ID": i, 
                "Type": name, 
                "Area (px)": int(real_area),
                "Solidity": f"{solidity:.2f}",
                "Confidence": f"{confidence}%",
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h
            })
            
            # Drawing
            cv2.rectangle(final_output, (x, y), (x+w, y+h), color, 4)
            label = f"{name}"
            cv2.putText(final_output, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        # Return fused map as "LBP" placeholder for viz, or just the main map
        return orig_full, entropy_maps[0], final_entropy_map, final_output, defect_list

# Initialize Instance
inspector = TextureInspector()
