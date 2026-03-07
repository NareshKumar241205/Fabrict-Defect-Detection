"""
Spectral Residual Inspector Module (Algorithm A)
=================================================
Uses Fourier Transform (FFT) **and** Discrete Wavelet Transform (DWT)
to detect anomalies in repetitive textures.

Algorithm (Hybrid FFT + DWT Approach):
1. Resize to processing resolution & apply CLAHE illumination correction.
2. FFT path: Spectral Residual Saliency (original pipeline).
3. DWT path: Multi-level wavelet decomposition (PyWavelets) →
   suppress periodic weave in detail coefficients → Inverse DWT
   reconstructs a *defect-only* image with pixel-perfect boundaries.
4. Fuse FFT saliency and DWT defect map (element-wise max).
5. Sauvola local thresholding (replaces global Z-score) for
   robustness against uneven lighting or shadows.

Detects: Missing Threads, Slubs, Oil Stains.
"""

import cv2
import numpy as np
import pywt
from typing import Tuple, List, Dict, Any, BinaryIO
from skimage.filters import threshold_sauvola


class SpectralInspector:
    """Detects defects using Spectral Residual (FFT) Saliency."""

    # Processing resolution (all inspectors use a consistent 800 px width)
    PROCESS_WIDTH = 800
    SMOOTHING_KERNEL = 5

    def __init__(self):
        self.defects: List[Dict[str, Any]] = []

    # ──────────────────────────────────────────
    # Pre-processing
    # ──────────────────────────────────────────
    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Standardize input: resize to PROCESS_WIDTH, apply CLAHE.

        Returns (original_bgr, img_small_bgr, img_gray, scale).
        *scale* = PROCESS_WIDTH / original_width  so that
        original_coord = process_coord / scale.
        """
        if hasattr(img_buffer, "seek"):
            img_buffer.seek(0)

        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image file")

        h, w = img.shape[:2]
        scale = self.PROCESS_WIDTH / w
        target_h = int(h * scale)

        img_small = cv2.resize(img, (self.PROCESS_WIDTH, target_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

        # CLAHE illumination correction (matches texture & edge inspectors)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, scale

    # ──────────────────────────────────────────
    # Core FFT saliency
    # ──────────────────────────────────────────
    def _compute_saliency_map(self, img_gray: np.ndarray, fft_size: int = 256) -> np.ndarray:
        """Compute the Spectral Residual Saliency Map at a given FFT scale."""
        img_small = cv2.resize(img_gray, (fft_size, fft_size))

        f = np.fft.fft2(img_small)
        f_shift = np.fft.fftshift(f)

        amplitude = np.abs(f_shift)
        log_amplitude = np.log(amplitude + 1e-9)

        spectrum_smooth = cv2.blur(log_amplitude, (3, 3))
        spectral_residual = log_amplitude - spectrum_smooth

        phase = np.angle(f_shift)
        f_ishift = np.fft.ifftshift(np.exp(spectral_residual + 1j * phase))
        img_back = np.abs(np.fft.ifft2(f_ishift))

        saliency = cv2.GaussianBlur(
            img_back, (self.SMOOTHING_KERNEL, self.SMOOTHING_KERNEL), 0
        )
        saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize back to the processing resolution (img_gray size)
        return cv2.resize(saliency, (img_gray.shape[1], img_gray.shape[0]))

    # ──────────────────────────────────────────
    # DWT defect map (Wavelet)
    # ──────────────────────────────────────────
    def _compute_dwt_defect_map(
        self, img_gray: np.ndarray, wavelet: str = "db4", level: int = 3
    ) -> np.ndarray:
        """Decompose with DWT, suppress periodic weave, reconstruct defect-only image.

        The repeating weave pattern concentrates energy in the approximation
        coefficients and in regular, low-energy detail bands.  By soft-thresholding
        the approximation to zero and keeping only the *anomalous* detail
        coefficients, the inverse DWT produces an image where only defects
        (slubs, missing threads, stains) remain, with pixel-perfect boundaries.
        """
        img_f = img_gray.astype(np.float64)

        # Multi-level 2-D DWT
        coeffs = pywt.wavedec2(img_f, wavelet, level=level)

        # Zero out the approximation (low-frequency weave background)
        coeffs[0] = np.zeros_like(coeffs[0])

        # For each detail level, soft-threshold to suppress periodic structure
        # but preserve anomalies (defects with unusually high energy)
        for i in range(1, len(coeffs)):
            details = list(coeffs[i])  # (cH, cV, cD)
            for j in range(len(details)):
                d = details[j]
                # Universal threshold (VisuShrink)
                sigma = np.median(np.abs(d)) / 0.6745
                thresh = sigma * np.sqrt(2 * np.log(max(d.size, 2)))
                # Keep only coefficients ABOVE the threshold (anomalies)
                details[j] = pywt.threshold(d, thresh, mode="hard")
            coeffs[i] = tuple(details)

        # Inverse DWT → defect-only reconstruction
        reconstructed = pywt.waverec2(coeffs, wavelet)
        reconstructed = np.abs(reconstructed)

        # Resize to match img_gray (DWT may pad slightly)
        h, w = img_gray.shape
        reconstructed = cv2.resize(reconstructed, (w, h))

        return cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ──────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────
    def detect_defects(
        self,
        img_buffer: BinaryIO,
        sensitivity: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Run the full Spectral Residual pipeline.

        Args:
            sensitivity: Z-score multiplier (lower = more sensitive).

        Returns:
            (original_bgr, result_bgr, saliency_heatmap, defect_list)
        """
        from config import UNIFIED_SETTINGS

        use_sauvola = UNIFIED_SETTINGS.get("USE_SAUVOLA_SPECTRAL", False)
        use_dwt = UNIFIED_SETTINGS.get("USE_DWT", True)
        sauvola_window = UNIFIED_SETTINGS.get("SAUVOLA_WINDOW", 151)

        original, img_small, img_gray, scale = self._preprocess(img_buffer)
        result = original.copy()

        # 1. Multi-Resolution FFT Saliency (Image Pyramid)
        fft_scales = [512, 256, 128]
        saliency_maps = [self._compute_saliency_map(img_gray, s) for s in fft_scales]
        saliency_fft = np.maximum.reduce(saliency_maps)

        # 1b. DWT defect map (wavelet) — optional
        if use_dwt:
            dwt_map = self._compute_dwt_defect_map(img_gray, wavelet="db4", level=3)
            saliency_map = np.maximum(saliency_fft, dwt_map)
        else:
            saliency_map = saliency_fft

        # 2. Global z-score thresholding for FFT saliency
        mean_sal = np.mean(saliency_map)
        std_sal = np.std(saliency_map)
        thresh_val = mean_sal + sensitivity * std_sal
        _, binary_map = cv2.threshold(saliency_map, thresh_val, 255, cv2.THRESH_BINARY)

        # 2b. Morphological ops: CLOSE FIRST to fuse the fragmented thread, 
        # then OPEN to remove the background noise.
        
        # Use a large 25x25 kernel to bridge large vertical/horizontal gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Use a small kernel to clean up speckles after the thread is fused
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # 3. Defect extraction
        contours, _ = cv2.findContours(
            binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.defects = []
        defect_id = 0
        proc_h, proc_w = img_gray.shape[:2]
        img_total_area = proc_h * proc_w
        min_area = max(300, int(img_total_area * 0.004))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            defect_id += 1

            # ── FIX: coordinates are in processing space → map to original ──
            ox = int(x / scale)
            oy = int(y / scale)
            ow = max(1, int(w / scale))
            oh = max(1, int(h / scale))
            real_area = max(1, int(area / (scale ** 2)))

            # Shape metrics (computed in processing space for stability)
            aspect_ratio = w / max(h, 1)

            # Solidity (convex hull ratio): smooth blobs → high solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = (cv2.contourArea(contour) / max(hull_area, 1)) if hull_area > 0 else 0

            # ── Classification using the 10-type taxonomy ──
            if solidity > 0.85 and 0.4 < aspect_ratio < 2.5:
                d_type = "Oil Stain"
                color = (0, 140, 255)  # Orange
            elif aspect_ratio > 3.0 or aspect_ratio < 0.33:
                d_type = "Missing Thread"
                color = (0, 0, 255)    # Red
            else:
                d_type = "Slub"
                color = (0, 165, 255)  # Orange-yellow

            # ── FIX: Z-score confidence (consistent with other inspectors) ──
            roi_sal = saliency_map[y : y + h, x : x + w]
            roi_mean = np.mean(roi_sal) if roi_sal.size > 0 else mean_sal
            z_distance = abs(roi_mean - mean_sal) / max(std_sal, 1e-6)
            confidence = min(99, max(10, int((z_distance / max(sensitivity, 1e-6)) * 100)))

            # Draw on original-scale result image
            cv2.rectangle(result, (ox, oy), (ox + ow, oy + oh), color, 3)
            cv2.putText(
                result, f"{d_type} ({confidence}%)", (ox, max(oy - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

            self.defects.append({
                "ID": defect_id,
                "Type": d_type,
                "Area (px)": real_area,
                "Solidity": f"{solidity:.2f}",
                "Confidence": f"{confidence}%",
                "Location": f"({ox}, {oy})",
                "Saliency": f"{int(roi_mean)}",
                "bbox_x": ox, "bbox_y": oy, "bbox_w": ow, "bbox_h": oh,
            })

        saliency_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_INFERNO)
        return original, result, saliency_heatmap, self.defects


# Module instance
spectral_inspector = SpectralInspector()
