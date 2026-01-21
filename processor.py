# inspectors.py
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import config as cfg

class BaseInspector:
    """Shared utilities for image processing."""
    def preprocess(self, img_buffer):
        """Converts input stream to optimized grayscale for analysis."""
        if hasattr(img_buffer, 'seek'): img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Optimization: Resize strictly to 500px width
        h, w = img.shape[:2]
        scale = cfg.IMAGE_RESIZE_WIDTH / w
        new_h = int(h * scale)
        
        img_small = cv2.resize(img, (cfg.IMAGE_RESIZE_WIDTH, new_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        return img_small, img_gray

class SurfaceInspector(BaseInspector):
    """Module A: Detects textural defects (Slubs, Holes, Missing Threads)."""
    
    def scan(self, img_gray, output_img):
        h, w = img_gray.shape
        defects = []
        patch_size = cfg.GLCM["PATCH_SIZE"]
        stride = cfg.GLCM["STRIDE"]

        # Sliding Window Logic
        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                # 1. Extract Patch
                patch = img_gray[y:y+patch_size, x:x+patch_size]
                
                # 2. Compute GLCM Matrix
                glcm = graycomatrix(patch, cfg.GLCM["DISTANCES"], cfg.GLCM["ANGLES"], 
                                    cfg.GLCM["LEVELS"], symmetric=True, normed=True)
                
                # 3. Extract Features
                contrast = graycoprops(glcm, 'contrast').mean()
                correlation = graycoprops(glcm, 'correlation').mean()
                
                # 4. Defect Classification
                if contrast > cfg.GLCM["THRESH_CONTRAST"]:
                    self._mark_defect(output_img, x, y, patch_size, cfg.COLORS["RED"])
                    defects.append({"Type": "Slub / Hole", "Location": f"Grid ({x},{y})", "Severity": "Critical"})
                
                elif correlation < cfg.GLCM["THRESH_CORRELATION"]:
                    self._mark_defect(output_img, x, y, patch_size, cfg.COLORS["ORANGE"])
                    defects.append({"Type": "Missing Thread", "Location": f"Grid ({x},{y})", "Severity": "Major"})

        return output_img, defects

    def _mark_defect(self, img, x, y, size, color):
        cv2.rectangle(img, (x, y), (x+size, y+size), color, 2)

class AssemblyInspector(BaseInspector):
    """Module B: Detects geometric defects (Skip Stitch, Puckering)."""
    
    def scan(self, img_gray, output_img):
        h, w = img_gray.shape
        defects = []
        
        # 1. Find Seam Line (Strongest Horizontal Edge)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        seam_y = np.argmax(np.sum(np.abs(sobel_y), axis=1))
        
        # Visualize Seam
        cv2.line(output_img, (0, seam_y), (w, seam_y), cfg.COLORS["CYAN"], 1)
        
        # 2. Extract ROI
        margin = cfg.SEAM["ROI_MARGIN"]
        y1, y2 = max(0, seam_y - margin), min(h, seam_y + margin)
        roi = img_gray[y1:y2, :]
        
        # 3. Algorithm: Projection Profiling (Skip Stitch)
        _, bin_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj = np.sum(bin_roi, axis=0)
        if proj.max() > 0: proj = proj / proj.max() # Normalize
        
        current_gap = 0
        gap_limit = 10 # Approx pixel width of a normal stitch gap
        
        for i in range(len(proj)):
            if proj[i] < 0.2: # Gap detected
                current_gap += 1
            else:
                # If gap is 2.5x larger than normal -> Skip Stitch
                if current_gap > (gap_limit * cfg.SEAM["GAP_TOLERANCE"]):
                    center_x = i - current_gap//2
                    self._mark_skip(output_img, center_x, seam_y)
                    defects.append({"Type": "Skip Stitch", "Location": f"X:{center_x}", "Severity": "Critical"})
                current_gap = 0
                
        # 4. Algorithm: Laplacian Variance (Puckering)
        roughness = cv2.Laplacian(roi, cv2.CV_64F).var()
        if roughness > cfg.SEAM["PUCKER_VAR"]:
            cv2.putText(output_img, "PUCKER DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cfg.COLORS["ORANGE"], 2)
            defects.append({"Type": "Seam Pucker", "Location": "Seam ROI", "Severity": "Moderate"})
            
        return output_img, defects

    def _mark_skip(self, img, x, y):
        cv2.circle(img, (x, y), 15, cfg.COLORS["RED"], 2)
        cv2.putText(img, "SKIP", (x-10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg.COLORS["RED"], 2)

class QualityEngine:
    """Facade: Single entry point for the Application."""
    def __init__(self):
        self.surface = SurfaceInspector()
        self.assembly = AssemblyInspector()
        
    def run_inspection(self, img_file, mode):
        # Route request to the correct engine
        if mode == "Fabric Surface":
            img, gray = self.surface.preprocess(img_file)
            return self.surface.scan(gray, img)
        else:
            img, gray = self.assembly.preprocess(img_file)
            return self.assembly.scan(gray, img)

# Initialize Singleton
engine = QualityEngine()