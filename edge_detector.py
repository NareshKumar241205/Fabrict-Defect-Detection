"""
Edge & Line Detector Module
===========================
Detects linear defects (tears, missing threads, cracks) using:
1. Canny Edge Detection + Hough Line Transform
2. Gradient Magnitude Analysis (Sobel)
3. Gabor Filters for directional defects

These methods are specifically designed for THIN LINEAR DEFECTS
that LBP and GLCM cannot detect effectively.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO


class EdgeLineDetector:
    """Detects linear fabric defects using edge and gradient analysis."""
    
    # Configuration (STRICT to avoid detecting regular weave patterns)
    CANNY_LOW = 100         # Higher = less sensitive
    CANNY_HIGH = 200
    HOUGH_THRESHOLD = 100   # Higher = only strong lines
    MIN_LINE_LENGTH = 150   # Longer = only significant defects
    MAX_LINE_GAP = 10
    GRADIENT_THRESHOLD = 150  # Higher = only major intensity changes
    
    def __init__(self):
        """Initialize the Edge/Line Detector."""
        self.defects: List[Dict[str, Any]] = []
    
    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, float]:
        """Load and prepare image."""
        if hasattr(img_buffer, 'seek'):
            img_buffer.seek(0)
        
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Resize for performance
        h, w = img.shape[:2]
        target_w = 720
        scale = target_w / w
        target_h = int(h * scale)
        
        img_resized = cv2.resize(img, (target_w, target_h))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur to reduce noise
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        
        return img, img_blur, scale
    
    def _detect_edges_and_lines(self, img_gray: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect edges using Canny and find lines using Hough Transform.
        
        Returns:
            Tuple of (edge_map, list_of_lines)
        """
        edges = cv2.Canny(img_gray, self.CANNY_LOW, self.CANNY_HIGH)
        
        # Morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges_closed, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.HOUGH_THRESHOLD,
            minLineLength=self.MIN_LINE_LENGTH,
            maxLineGap=self.MAX_LINE_GAP
        )
        
        return edges_closed, lines if lines is not None else []
    
    def _compute_gradient_anomalies(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Find areas with high gradient magnitude (sharp intensity changes).
        
        Returns:
            Binary mask of gradient anomalies
        """
        # Sobel gradients
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Threshold to get anomalies
        _, anomaly_mask = cv2.threshold(magnitude_norm, self.GRADIENT_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
        
        return anomaly_mask
    
    def _apply_gabor_filter(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Apply Gabor filters at multiple orientations to detect directional defects.
        
        Returns:
            Combined Gabor response highlighting directional anomalies
        """
        gabor_responses = []
        
        # Gabor parameters
        ksize = 31
        sigma = 4.0
        lambd = 10.0
        gamma = 0.5
        psi = 0
        
        # Apply at different orientations (0, 45, 90, 135 degrees)
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F
            )
            filtered = cv2.filter2D(img_gray, cv2.CV_64F, kernel)
            gabor_responses.append(np.abs(filtered))
        
        # Combine responses (take maximum across orientations)
        combined = np.max(np.array(gabor_responses), axis=0)
        combined_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return combined_norm
    
    def _classify_line_defect(self, line: np.ndarray) -> Tuple[str, Tuple[int, int, int]]:
        """
        Classify a detected line as a specific defect type based on orientation.
        """
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Classify based on orientation
        if angle < 30 or angle > 150:
            return "Horizontal Tear", (0, 0, 255)  # Red - likely weft issue
        elif 60 < angle < 120:
            return "Missing Warp Thread", (255, 0, 0)  # Blue - vertical = warp issue
        else:
            return "Diagonal Crack", (0, 165, 255)  # Orange
    
    def detect_linear_defects(
        self, 
        img_buffer: BinaryIO,
        min_line_length: int = None,
        gradient_threshold: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Main detection pipeline for linear defects.
        
        Returns:
            Tuple of (original, annotated_result, edge_map, defect_list)
        """
        if min_line_length is not None:
            self.MIN_LINE_LENGTH = min_line_length
        if gradient_threshold is not None:
            self.GRADIENT_THRESHOLD = gradient_threshold
        
        # Preprocess
        original, img_gray, scale = self._preprocess(img_buffer)
        result = original.copy()
        
        self.defects = []
        defect_id = 0
        
        # Method 1: Edge + Hough Lines
        edges, lines = self._detect_edges_and_lines(img_gray)
        
        for line in lines:
            defect_type, color = self._classify_line_defect(line)
            x1, y1, x2, y2 = line[0]
            
            # Scale back to original
            x1_orig, y1_orig = int(x1 / scale), int(y1 / scale)
            x2_orig, y2_orig = int(x2 / scale), int(y2 / scale)
            
            # Calculate line length
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Draw on result
            cv2.line(result, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 3)
            cv2.putText(result, defect_type, (x1_orig, y1_orig - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            defect_id += 1
            self.defects.append({
                "ID": defect_id,
                "Type": defect_type,
                "Location": f"({x1_orig},{y1_orig}) to ({x2_orig},{y2_orig})",
                "Length": f"{int(length / scale)} px"
            })
        
        # Method 2: Gradient anomalies (for more subtle defects)
        gradient_mask = self._compute_gradient_anomalies(img_gray)
        
        # Find contours in gradient mask
        contours, _ = cv2.findContours(gradient_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Only larger anomalies (was 100)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Only flag very elongated regions (likely real linear defects)
                if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                    # Scale back
                    x_orig, y_orig = int(x / scale), int(y / scale)
                    w_orig, h_orig = int(w / scale), int(h / scale)
                    
                    defect_id += 1
                    defect_type = "Linear Anomaly"
                    color = (255, 255, 0)  # Cyan
                    
                    cv2.rectangle(result, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 2)
                    
                    self.defects.append({
                        "ID": defect_id,
                        "Type": defect_type,
                        "Location": f"({x_orig}, {y_orig})",
                        "Length": f"{max(w_orig, h_orig)} px"
                    })
        
        # Colorize edge map for visualization
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return original, result, edges_rgb, self.defects


# Module-level instance
edge_detector = EdgeLineDetector()
