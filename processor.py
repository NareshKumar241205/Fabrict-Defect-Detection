import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from config import DEFAULT_GLCM, DEFAULT_SEAM, DEFAULT_SYSTEM

class FabricInspector:
    def __init__(self):
        # Load defaults initially
        self.patch_size = DEFAULT_SYSTEM["PATCH_SIZE"]
        self.step = DEFAULT_SYSTEM["STEP_SIZE"]

    def _preprocess(self, img_buffer, bg_thresh):
        if hasattr(img_buffer, "seek"):
            img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        h, w = img.shape[:2]
        target_w = DEFAULT_SYSTEM["IMAGE_RESIZE_WIDTH"]
        scale = target_w / w
        img_small = cv2.resize(img, (target_w, int(h * scale)))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        

        _, mask = cv2.threshold(img_gray, bg_thresh, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return img, img_small, img_gray, mask

    def _merge_boxes(self, boxes):
        """ Combines overlapping/adjacent boxes into single large detections """
        if not boxes:
            return []
            
        rects = []
        for b in boxes:
            rects.append([b['x'], b['y'], b['w'], b['h']])
            
        rects, weights = cv2.groupRectangles(rects, groupThreshold=1, eps=0.2)
        
        merged_defects = []
        for (x, y, w, h) in rects:
            merged_defects.append({
                "x": x, "y": y, "w": w, "h": h,
                "type": "Defect Area", 
                "score": 1.0
            })
        return merged_defects

    # =========================================
    # MODULE A: GLCM (Texture & Structure)
    # =========================================
    def analyze_texture_glcm(self, img_gray, mask, settings):
        h, w = img_gray.shape
        raw_defects = []
        
        # Unpack settings for speed
        corr_thresh = settings.get("CORRELATION_MIN", DEFAULT_GLCM["CORRELATION_MIN"])
        cont_thresh = settings.get("CONTRAST_MAX", DEFAULT_GLCM["CONTRAST_MAX"])
        homo_thresh = settings.get("HOMOGENEITY_MAX", DEFAULT_GLCM["HOMOGENEITY_MAX"])
        
        for y in range(0, h - self.patch_size, self.step):
            for x in range(0, w - self.patch_size, self.step):
                
                # Check Background
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                if cv2.countNonZero(mask_patch) < (self.patch_size * self.patch_size * 0.5):
                    continue 

                patch = img_gray[y:y+self.patch_size, x:x+self.patch_size]
                
                # GLCM Calculation
                glcm = graycomatrix(patch, distances=DEFAULT_GLCM['DISTANCES'], 
                                    angles=DEFAULT_GLCM['ANGLES'], levels=256, 
                                    symmetric=True, normed=True)
                
                contrast = graycoprops(glcm, 'contrast').mean()
                correlation = graycoprops(glcm, 'correlation').mean()
                homogeneity = graycoprops(glcm, 'homogeneity').mean()
                
                # Logic Mapping
                if correlation < corr_thresh:
                    raw_defects.append({
                        "x": x, "y": y, "w": self.patch_size, "h": self.patch_size,
                        "type": "Structure Break"
                    })
                elif contrast > cont_thresh:
                    raw_defects.append({
                        "x": x, "y": y, "w": self.patch_size, "h": self.patch_size,
                        "type": "Slub / Hole"
                    })
                elif homogeneity > homo_thresh:
                     raw_defects.append({
                        "x": x, "y": y, "w": self.patch_size, "h": self.patch_size,
                        "type": "Stain / Oil"
                    })

        return self._merge_boxes(raw_defects)

    # =========================================
    # MODULE B: SEAM INSPECTOR (Geometry)
    # =========================================
    def analyze_seam_geometry(self, img_gray, mask, settings):
        # Unpack settings
        stitch_thresh = settings.get("STITCH_COLOR_THRESH", DEFAULT_SEAM["STITCH_COLOR_THRESH"])
        gap_tolerance = settings.get("GAP_TOLERANCE", DEFAULT_SEAM["GAP_TOLERANCE"])

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

        for i, val in enumerate(proj):
            if val < 2: # Gap
                if not in_gap:
                    in_gap = True
                    gap_start = i
                gap_counter += 1
            else: # Thread
                if in_gap:
                    if gap_counter > gap_tolerance:
                        if gap_start > 10 and i < (len(proj) - 10):
                            defects.append({
                                "x": gap_start, "y": 10, 
                                "w": gap_counter, "h": rot_img.shape[0]-20,
                                "type": "Skip Stitch",
                                "score": gap_counter
                            })
                    in_gap = False
                    gap_counter = 0

        return defects

    # =========================================
    # MAIN PIPELINE
    # =========================================
    def process(self, img_buffer, mode="fabric", settings=None):
        if settings is None:
            settings = {}

        bg_thresh = settings.get("BACKGROUND_THRESH", DEFAULT_SYSTEM["BACKGROUND_THRESH"])
        
        orig, small, gray, mask = self._preprocess(img_buffer, bg_thresh)
        output_img = small.copy()
        
        found_defects = []
        
        if mode == "fabric":
            found_defects = self.analyze_texture_glcm(gray, mask, settings)
        elif mode == "seam":
            found_defects = self.analyze_seam_geometry(gray, mask, settings)
            
        defect_log = []
        for d in found_defects:
            cv2.rectangle(output_img, (d['x'], d['y']), 
                         (d['x'] + d['w'], d['y'] + d['h']), (0, 0, 255), 2)
            cv2.putText(output_img, d['type'], (d['x'], d['y']-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            
            defect_log.append({
                "Type": d['type'],
                "Area (px)": d['w'] * d['h']
            })

        return output_img, defect_log

inspector = FabricInspector()
