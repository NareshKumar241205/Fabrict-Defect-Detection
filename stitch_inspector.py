"""
Stitch Inspector Module
=======================
Analyzes seam/stitch quality using two complementary algorithms:
1. Projection Profiling - Detects skip stitches (gaps in seam)
2. Laplacian Variance - Detects puckers (wrinkled fabric near seam)

Usage:
    from stitch_inspector import stitch_inspector
    result = stitch_inspector.check_seam_quality(image_buffer)
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO


class StitchInspector:
    """Analyzes seam/stitch quality in fabric images."""
    
    # Configuration
    SEAM_WIDTH = 100            # Expected seam width in pixels (after resize)
    LAPLACIAN_THRESHOLD = 200   # Above = Pucker detected
    GAP_THRESHOLD = 0.3         # Projection valley depth ratio
    
    def __init__(self):
        """Initialize the Stitch Inspector."""
        self.defects: List[Dict[str, Any]] = []
    
    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Standardizes input: decode, resize, convert to grayscale.
        
        Returns:
            Tuple of (original_image, grayscale, scale_factor)
        """
        if hasattr(img_buffer, 'seek'):
            img_buffer.seek(0)
        
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Resize for performance (target 720p width)
        h, w = img.shape[:2]
        target_w = 720
        scale = target_w / w
        target_h = int(h * scale)
        
        img_resized = cv2.resize(img, (target_w, target_h))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        return img, img_gray, scale
    
    def _detect_seam_region(self, img_gray: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Auto-detect the seam region using edge detection and Hough lines.
        Falls back to center strip if no clear seam is found.
        
        Returns:
            (x, y, width, height) of detected seam region
        """
        h, w = img_gray.shape
        
        # Edge detection
        edges = cv2.Canny(img_gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=h//3, maxLineGap=20)
        
        if lines is not None and len(lines) > 0:
            # Find the most vertical line (seam is usually vertical or horizontal)
            best_x = w // 2
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly vertical
                if abs(x2 - x1) < 50:  # Near-vertical
                    best_x = (x1 + x2) // 2
                    break
            
            # Return region around detected seam
            seam_x = max(0, best_x - self.SEAM_WIDTH // 2)
            return seam_x, 0, self.SEAM_WIDTH, h
        
        # Fallback: center strip
        center_x = w // 2 - self.SEAM_WIDTH // 2
        return center_x, 0, self.SEAM_WIDTH, h
    
    def _projection_profile_analysis(self, seam_roi: np.ndarray) -> Tuple[bool, np.ndarray, List[int]]:
        """
        Detect skip stitches using vertical projection profiling.
        
        A skip stitch appears as a significant gap (valley) in the projection profile.
        
        Returns:
            Tuple of (has_skip_stitch, projection_profile, gap_positions)
        """
        # Binarize using Otsu's method
        _, binary = cv2.threshold(seam_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Compute vertical projection (sum pixels per row)
        projection = np.sum(binary, axis=1).astype(np.float32)
        
        # Normalize
        if projection.max() > 0:
            projection = projection / projection.max()
        
        # Detect gaps (valleys below threshold)
        mean_proj = np.mean(projection)
        gap_positions = []
        
        for i, val in enumerate(projection):
            if val < mean_proj * self.GAP_THRESHOLD:
                gap_positions.append(i)
        
        # Cluster nearby gaps
        clustered_gaps = []
        if gap_positions:
            cluster_start = gap_positions[0]
            for i in range(1, len(gap_positions)):
                if gap_positions[i] - gap_positions[i-1] > 10:
                    clustered_gaps.append((cluster_start, gap_positions[i-1]))
                    cluster_start = gap_positions[i]
            clustered_gaps.append((cluster_start, gap_positions[-1]))
        
        has_skip = len(clustered_gaps) > 0
        return has_skip, projection, clustered_gaps
    
    def _laplacian_variance_analysis(self, seam_roi: np.ndarray) -> Tuple[bool, float]:
        """
        Detect puckers using Laplacian variance.
        
        High variance = wrinkled/puckered surface
        Low variance = flat/smooth surface
        
        Returns:
            Tuple of (is_puckered, variance_value)
        """
        # Apply Laplacian filter
        laplacian = cv2.Laplacian(seam_roi, cv2.CV_64F)
        
        # Calculate variance
        variance = laplacian.var()
        
        is_puckered = variance > self.LAPLACIAN_THRESHOLD
        return is_puckered, variance
    
    def check_seam_quality(
        self, 
        img_buffer: BinaryIO,
        seam_roi: Tuple[int, int, int, int] = None,
        laplacian_threshold: float = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Main pipeline: Analyze seam quality for skip stitches and puckers.
        
        Args:
            img_buffer: Image file buffer
            seam_roi: Optional manual ROI (x, y, w, h). Auto-detected if None.
            laplacian_threshold: Override default pucker threshold
            
        Returns:
            Tuple of (original_image, annotated_result, defect_list)
        """
        if laplacian_threshold is not None:
            self.LAPLACIAN_THRESHOLD = laplacian_threshold
        
        # Preprocess
        original, img_gray, scale = self._preprocess(img_buffer)
        result = original.copy()
        h, w = img_gray.shape
        
        self.defects = []
        
        # Detect or use provided seam region
        if seam_roi is None:
            sx, sy, sw, sh = self._detect_seam_region(img_gray)
        else:
            sx, sy, sw, sh = seam_roi
        
        # Extract seam ROI
        seam_gray = img_gray[sy:sy+sh, sx:sx+sw]
        
        # Analysis 1: Projection Profiling for Skip Stitch
        has_skip, projection, gap_clusters = self._projection_profile_analysis(seam_gray)
        
        if has_skip:
            for gap_start, gap_end in gap_clusters:
                # Scale back to original coordinates
                y1_orig = int((sy + gap_start) / scale)
                y2_orig = int((sy + gap_end) / scale)
                x1_orig = int(sx / scale)
                x2_orig = int((sx + sw) / scale)
                
                # Draw red dashed line effect (solid for simplicity)
                cv2.line(result, (x1_orig, y1_orig), (x2_orig, y1_orig), (0, 0, 255), 3)
                cv2.line(result, (x1_orig, y2_orig), (x2_orig, y2_orig), (0, 0, 255), 3)
                cv2.putText(result, "Skip Stitch", (x1_orig, y1_orig - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                self.defects.append({
                    "ID": len(self.defects) + 1,
                    "Type": "Skip Stitch",
                    "Location": f"Row {y1_orig}-{y2_orig}",
                    "Severity": "High"
                })
        
        # Analysis 2: Laplacian Variance for Pucker
        is_puckered, variance = self._laplacian_variance_analysis(seam_gray)
        
        if is_puckered:
            # Highlight entire seam region
            x1_orig = int(sx / scale)
            y1_orig = int(sy / scale)
            x2_orig = int((sx + sw) / scale)
            y2_orig = int((sy + sh) / scale)
            
            # Draw orange box for pucker
            cv2.rectangle(result, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 165, 255), 3)
            cv2.putText(result, f"Pucker (var={variance:.0f})", (x1_orig, y1_orig - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            self.defects.append({
                "ID": len(self.defects) + 1,
                "Type": "Pucker",
                "Location": "Seam Region",
                "Variance": f"{variance:.1f}",
                "Severity": "Medium" if variance < 300 else "High"
            })
        
        # Draw seam detection guide (green)
        sx_orig = int(sx / scale)
        sy_orig = int(sy / scale)
        sw_orig = int(sw / scale)
        sh_orig = int(sh / scale)
        cv2.rectangle(result, (sx_orig, sy_orig), (sx_orig + sw_orig, sy_orig + sh_orig), 
                      (0, 255, 0), 1)
        
        return original, result, self.defects


# Module-level instance for easy import
stitch_inspector = StitchInspector()
