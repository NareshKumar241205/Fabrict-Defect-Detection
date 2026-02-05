"""
GLCM Inspector Module
=====================
Detects structural weave defects (slubs, missing warp threads) using
Gray-Level Co-occurrence Matrix (GLCM) texture features.

Algorithm:
1. Quantize grayscale image to 32 levels (reduces GLCM size from 256x256 to 32x32)
2. Apply sliding window (64x64 px, stride 32)
3. Compute GLCM and extract: contrast, correlation, homogeneity
4. Flag anomalies based on statistical thresholds
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import Tuple, List, Dict, Any, BinaryIO


class GLCMInspector:
    """Detects structural fabric defects using GLCM texture analysis."""
    
    # Configuration (Tuned to avoid false positives on regular weave)
    WINDOW_SIZE = 64        # Larger window for stable analysis
    STRIDE = 64             # No overlap to reduce detections
    GRAY_LEVELS = 16        # Fewer levels = faster
    
    # Thresholds (STRICT - only flag obvious defects)
    CONTRAST_HIGH = 300     # Very high = only major defects
    CORRELATION_LOW = 0.4   # Very low = only severe irregularity
    HOMOGENEITY_LOW = 0.2   # Very low = only severe texture breaks
    
    def __init__(self):
        """Initialize the GLCM Inspector."""
        self.defects: List[Dict[str, Any]] = []
    
    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Standardizes input: decode, resize, convert to grayscale, quantize.
        
        Returns:
            Tuple of (original_image, quantized_gray, scale_factor)
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
        
        # Quantize to reduce GLCM computation
        img_quantized = (img_gray // (256 // self.GRAY_LEVELS)).astype(np.uint8)
        
        return img, img_quantized, scale
    
    def _compute_glcm_features(self, window: np.ndarray) -> Dict[str, float]:
        """
        Compute GLCM features for a single window.
        
        Args:
            window: 2D grayscale image patch (quantized)
            
        Returns:
            Dictionary with contrast, correlation, homogeneity values
        """
        # GLCM at distance=1, only 2 angles for speed (0 and 90 degrees)
        glcm = graycomatrix(
            window, 
            distances=[1], 
            angles=[0, np.pi/2],  # Reduced from 4 angles
            levels=self.GRAY_LEVELS,
            symmetric=True,
            normed=True
        )
        
        # Average across all angles for rotation invariance
        contrast = graycoprops(glcm, 'contrast').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        
        return {
            'contrast': contrast,
            'correlation': correlation,
            'homogeneity': homogeneity
        }
    
    def _classify_defect(self, features: Dict[str, float]) -> Tuple[str, Tuple[int, int, int]]:
        """
        Classify defect type based on GLCM features.
        
        Returns:
            Tuple of (defect_name, BGR_color) or (None, None) if normal
        """
        if features['contrast'] > self.CONTRAST_HIGH and features['homogeneity'] < self.HOMOGENEITY_LOW:
            return "Slub / Knot", (0, 165, 255)  # Orange
        elif features['correlation'] < self.CORRELATION_LOW:
            return "Missing Thread", (0, 0, 255)  # Red
        elif features['homogeneity'] < self.HOMOGENEITY_LOW:
            return "Weave Irregularity", (255, 0, 255)  # Magenta
        return None, None
    
    def detect_structural_defects(
        self, 
        img_buffer: BinaryIO,
        contrast_threshold: float = None,
        correlation_threshold: float = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Main detection pipeline using sliding window GLCM analysis.
        
        Args:
            img_buffer: Image file buffer
            contrast_threshold: Override default contrast threshold
            correlation_threshold: Override default correlation threshold
            
        Returns:
            Tuple of (original_image, annotated_result, defect_list)
        """
        # Allow threshold overrides
        if contrast_threshold is not None:
            self.CONTRAST_HIGH = contrast_threshold
        if correlation_threshold is not None:
            self.CORRELATION_LOW = correlation_threshold
        
        # Preprocess
        original, img_quantized, scale = self._preprocess(img_buffer)
        result = original.copy()
        h, w = img_quantized.shape
        
        self.defects = []
        defect_id = 0
        
        # Create feature maps for visualization
        contrast_map = np.zeros((h, w), dtype=np.float32)
        
        # Sliding window analysis
        for y in range(0, h - self.WINDOW_SIZE, self.STRIDE):
            for x in range(0, w - self.WINDOW_SIZE, self.STRIDE):
                # Extract window
                window = img_quantized[y:y+self.WINDOW_SIZE, x:x+self.WINDOW_SIZE]
                
                # Compute features
                features = self._compute_glcm_features(window)
                
                # Store for heatmap
                contrast_map[y:y+self.WINDOW_SIZE, x:x+self.WINDOW_SIZE] = features['contrast']
                
                # Classify
                defect_type, color = self._classify_defect(features)
                
                if defect_type:
                    defect_id += 1
                    
                    # Scale coordinates back to original image
                    x_orig = int(x / scale)
                    y_orig = int(y / scale)
                    w_orig = int(self.WINDOW_SIZE / scale)
                    h_orig = int(self.WINDOW_SIZE / scale)
                    
                    # Draw on result
                    cv2.rectangle(result, (x_orig, y_orig), 
                                  (x_orig + w_orig, y_orig + h_orig), color, 3)
                    cv2.putText(result, defect_type, (x_orig, y_orig - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    self.defects.append({
                        "ID": defect_id,
                        "Type": defect_type,
                        "Location": f"({x_orig}, {y_orig})",
                        "Contrast": f"{features['contrast']:.1f}",
                        "Correlation": f"{features['correlation']:.2f}",
                        "Homogeneity": f"{features['homogeneity']:.2f}"
                    })
        
        # Normalize contrast map for visualization
        contrast_map_norm = cv2.normalize(contrast_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        contrast_heatmap = cv2.applyColorMap(contrast_map_norm, cv2.COLORMAP_JET)
        
        return original, result, contrast_heatmap, self.defects


# Module-level instance for easy import
glcm_inspector = GLCMInspector()
