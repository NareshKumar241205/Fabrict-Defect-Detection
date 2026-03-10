# inspectors/logic_spectral.py
import cv2
import numpy as np
import pywt
from typing import Tuple, List, Dict, Any, BinaryIO
from config import LOGIC_SPECTRAL_SETTINGS

class LogicSpectralInspector:
    """PIPELINE 2: Detects Slub and Skip/Miss Stitch using FFT+DWT Saliency."""

    def __init__(self):
        self.defects: List[Dict[str, Any]] = []

    def _preprocess(self, img_buffer: BinaryIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if hasattr(img_buffer, "seek"):
            img_buffer.seek(0)
        file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        h, w = img.shape[:2]
        scale = LOGIC_SPECTRAL_SETTINGS["IMAGE_RESIZE_WIDTH"] / w
        target_h = int(h * scale)

        img_small = cv2.resize(img, (LOGIC_SPECTRAL_SETTINGS["IMAGE_RESIZE_WIDTH"], target_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        return img, img_small, img_gray, scale

    def _compute_saliency_map(self, img_gray: np.ndarray, fft_size: int = 256) -> np.ndarray:
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

        saliency = cv2.GaussianBlur(img_back, (LOGIC_SPECTRAL_SETTINGS["SMOOTHING_KERNEL"], LOGIC_SPECTRAL_SETTINGS["SMOOTHING_KERNEL"]), 0)
        saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.resize(saliency, (img_gray.shape[1], img_gray.shape[0]))

    def _compute_dwt_defect_map(self, img_gray: np.ndarray) -> np.ndarray:
        img_f = img_gray.astype(np.float64)
        coeffs = pywt.wavedec2(img_f, "db4", level=3)
        coeffs[0] = np.zeros_like(coeffs[0])

        for i in range(1, len(coeffs)):
            details = list(coeffs[i])
            for j in range(len(details)):
                d = details[j]
                sigma = np.median(np.abs(d)) / 0.6745
                thresh = sigma * np.sqrt(2 * np.log(max(d.size, 2)))
                details[j] = pywt.threshold(d, thresh, mode="hard")
            coeffs[i] = tuple(details)

        reconstructed = np.abs(pywt.waverec2(coeffs, "db4"))
        h, w = img_gray.shape
        reconstructed = cv2.resize(reconstructed, (w, h))
        return cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def detect_defects(self, img_buffer: BinaryIO, sensitivity: float = 2.0) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        original, img_small, img_gray, scale = self._preprocess(img_buffer)
        
        fft_scales = [512, 256, 128]
        saliency_maps = [self._compute_saliency_map(img_gray, s) for s in fft_scales]
        saliency_fft = np.maximum.reduce(saliency_maps)

        if LOGIC_SPECTRAL_SETTINGS["USE_DWT"]:
            dwt_map = self._compute_dwt_defect_map(img_gray)
            saliency_map = np.maximum(saliency_fft, dwt_map)
        else:
            saliency_map = saliency_fft

        mean_sal = np.mean(saliency_map)
        std_sal = np.std(saliency_map)
        thresh_val = mean_sal + sensitivity * std_sal
        _, binary_map = cv2.threshold(saliency_map, thresh_val, 255, cv2.THRESH_BINARY)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.defects = []
        proc_h, proc_w = img_gray.shape[:2]
        min_area = max(300, int(proc_h * proc_w * 0.004))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            ox = int(x / scale)
            oy = int(y / scale)
            ow = max(1, int(w / scale))
            oh = max(1, int(h / scale))
            real_area = max(1, int(area / (scale ** 2)))

            aspect_ratio = w / max(h, 1)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = (cv2.contourArea(contour) / max(hull_area, 1)) if hull_area > 0 else 0

            # Strip Oil Stain logic. Only classify as Slub or Skip/Miss Stitch.
            if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                d_type = "Skip/Miss Stitch"
            else:
                d_type = "Slub"

            roi_sal = saliency_map[y : y + h, x : x + w]
            roi_mean = np.mean(roi_sal) if roi_sal.size > 0 else mean_sal
            z_distance = abs(roi_mean - mean_sal) / max(std_sal, 1e-6)
            confidence = min(99, max(10, int((z_distance / max(sensitivity, 1e-6)) * 100)))

            self.defects.append({
                "Type": d_type,
                "Area (px)": real_area,
                "Solidity": f"{solidity:.2f}",
                "Quality Score": f"{confidence}%",
                "bbox_x": ox, "bbox_y": oy, "bbox_w": ow, "bbox_h": oh,
                "Pipeline": "Logic",
                "Category": "Structural" if d_type == "Skip/Miss Stitch" else "Surface"
            })

        return self.defects, saliency_map