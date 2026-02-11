"""
Spectral Residual Inspector Module (Industry Level)
===================================================
Uses Fourier Transform (FFT) to detect anomalies in repetitive textures.

Algorithm (Spectral Residual Approach):
1. Transform image to Frequency Domain (FFT).
2. Compute Log Amplitude Spectrum.
3. Compute Spectral Residual (Log Spectrum - Average Spectrum).
4. Inverse FFT to get Saliency Map.
5. Threshold Saliency Map to find defects.

This method works because the "repetitive background" (weave) has a strong, 
predictable spectral signature. Subtracting the average spectrum removes this 
background, leaving only the "surprise" (defect) in the Saliency Map.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, BinaryIO


class SpectralInspector:
    """Detects defects using Spectral Residual (FFT) Saliency."""
    
    # Configuration
    RESIZE_WIDTH = 256  # Downsample for FFT speed & broad selection
    SMOOTHING_KERNEL = 5
    
    def __init__(self):
        self.defects = []

    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, float]:
        """Standardize input."""
        if hasattr(img_buffer, 'seek'):
            img_buffer.seek(0)
        
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image file")
        
        h, w = img.shape[:2]
        
        # Calculate scale factor based on 720p display target (not FFT target)
        display_scale = 720 / w
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, img_gray, display_scale

    def _compute_saliency_map(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Compute the Spectral Residual Saliency Map.
        """
        # 1. Resize for processing (Spectral Residual works best on smaller scales)
        #    This captures the "global" anomaly rather than pixel noise.
        img_small = cv2.resize(img_gray, (self.RESIZE_WIDTH, self.RESIZE_WIDTH))
        
        # 2. FFT
        f = np.fft.fft2(img_small)
        f_shift = np.fft.fftshift(f)
        
        # 3. Log Amplitude Spectrum
        amplitude = np.abs(f_shift)
        log_amplitude = np.log(amplitude + 1e-9)
        
        # 4. Spectral Residual
        #    SR = LogAmp - Average(LogAmp)
        #    We approximate Average(LogAmp) using box filter smoothing
        spectrum_smooth = cv2.blur(log_amplitude, (3, 3))
        spectral_residual = log_amplitude - spectrum_smooth
        
        # 5. Inverse FFT to Spatial Domain
        #    Reconstruct image using mainly the "residual" (surprise) phases
        phase = np.angle(f_shift)
        f_ishift = np.fft.ifftshift(np.exp(spectral_residual + 1j * phase))
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # 6. Smooth the Saliency Map
        saliency_map = cv2.GaussianBlur(img_back, (self.SMOOTHING_KERNEL, self.SMOOTHING_KERNEL), 0)
        
        # Normalize to 0-255
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Resize back to input size
        return cv2.resize(saliency_map, (img_gray.shape[1], img_gray.shape[0]))

    def detect_defects(
        self, 
        img_buffer: BinaryIO, 
        sensitivity: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Main pipeline.
        
        Args:
            sensitivity: Z-score threshold multiplier (standard deviations).
                         Lower = more sensitive.
        """
        # Preprocess
        original, img_gray, scale = self._preprocess(img_buffer)
        result = original.copy()
        
        # 1. Compute Saliency
        saliency_map = self._compute_saliency_map(img_gray)
        
        # 2. Dynamic Thresholding (Auto-Calibration)
        #    We assume the saliency map is mostly dark (0).
        #    Defects are statistical outliers.
        mean_sal = np.mean(saliency_map)
        std_sal = np.std(saliency_map)
        
        # Threshold: Mean + (Sigma * StdDev)
        # Industry standard usually around 2-3 sigma
        thresh_val = mean_sal + (sensitivity * std_sal)
        _, binary_map = cv2.threshold(saliency_map, thresh_val, 255, cv2.THRESH_BINARY)
        
        # 2b. Morphological Closing to Connect Fragments
        #     One big issue with FFT is that it can "break" a line into dots.
        #     We use a strong closing operation to merge them back together.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel_close)
        
        # 3. Defect Extraction
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.defects = []
        defect_id = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter noise (relative to image size)
            if area > (img_gray.shape[0] * img_gray.shape[1] * 0.0005): # ~0.05% of image
                x, y, w, h = cv2.boundingRect(contour)
                
                defect_id += 1
                
                # Classify roughly by shape
                aspect_ratio = w / float(h)
                
                # Logic: If it's very long, it's a structural line
                if aspect_ratio > 3.0: 
                    d_type = "Horizontal Tear/Thread"
                    color = (0, 0, 255) # Red
                elif aspect_ratio < 0.33:
                    d_type = "Vertical Tear/Thread"
                    color = (0, 0, 255) # Red
                else:
                    d_type = "Texture Anomaly"
                    color = (0, 165, 255) # Orange
                
                # Draw
                cv2.rectangle(result, (x, y), (x+w, y+h), color, 3)
                cv2.putText(result, f"{d_type} ({int(area)})", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                self.defects.append({
                    "ID": defect_id,
                    "Type": d_type,
                    "Location": f"({x}, {y})",
                    "Saliency": f"{int(np.mean(saliency_map[y:y+h, x:x+w]))}"
                })
        
        # Colorize saliency for visualization
        saliency_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_INFERNO)
        
        return original, result, saliency_heatmap, self.defects

# Module instance
spectral_inspector = SpectralInspector()
