"""
Multi-Metric Defect Severity Scoring
=====================================
Replaces the single arbitrary "confidence %" with 10 measurable
classical CV metrics.  Each defect type uses a different weighted
combination so that the final Severity Score (0–100) reflects
genuine physical evidence rather than a pseudo-classification
probability.

Every defect dict receives two new keys:
    "Severity"       – int 0-100 (the final combined score)
    "Score_Details"   – dict of individual metric names → 0-100 values

Metric catalogue (10 metrics):
──────────────────────────────────────────────────────────────────
 #  Name                    What it measures
 1  max_pixel_dev_35        Max deviation from local mean in a 35×35 window
 2  local_contrast          |inner_mean − neigh_mean| / neigh_mean
 3  edge_gradient           Mean Sobel gradient magnitude inside ROI
 4  area_ratio              Defect area / image area  (larger = severe)
 5  compactness             4π·area / perimeter²  (circularity)
 6  ncc_peak_dev            Peak NCC deviation inside ROI (texture engine)
 7  variance_ratio          σ²_inside / σ²_neighbourhood
 8  darkness_depth          (neigh_mean − inner_mean) / neigh_mean for dark
 9  saturation_shift        |sat_inside − sat_global| / sat_global (color)
10  boundary_strength       Mean Canny edge density on the contour rim
──────────────────────────────────────────────────────────────────

Weights per defect type are defined in SCORE_WEIGHTS below.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional


# ─── Per-type weight profiles ────────────────────────────────────────────────
# Each value is a dict  metric_name → weight (0-1).
# Missing metrics get weight 0.  Weights are normalised internally.
SCORE_WEIGHTS: Dict[str, Dict[str, float]] = {

    # ── Texture engine defects ──
    "Slub": {
        "ncc_peak_dev":       0.25,
        "max_pixel_dev_35":   0.15,
        "variance_ratio":     0.15,
        "edge_gradient":      0.10,
        "area_ratio":         0.10,
        "compactness":        0.10,
        "local_contrast":     0.05,
        "boundary_strength":  0.05,
        "darkness_depth":     0.00,
        "saturation_shift":   0.05,
    },
    "Tear": {
        "ncc_peak_dev":       0.20,
        "max_pixel_dev_35":   0.15,
        "edge_gradient":      0.20,
        "area_ratio":         0.15,
        "compactness":        0.05,   # tears are elongated → low compactness OK
        "variance_ratio":     0.10,
        "local_contrast":     0.05,
        "boundary_strength":  0.05,
        "darkness_depth":     0.05,
        "saturation_shift":   0.00,
    },

    # ── Edge / background-subtraction defects ──
    "Hole": {
        "darkness_depth":     0.25,
        "max_pixel_dev_35":   0.15,
        "local_contrast":     0.15,
        "edge_gradient":      0.10,
        "area_ratio":         0.10,
        "compactness":        0.10,
        "boundary_strength":  0.05,
        "variance_ratio":     0.05,
        "ncc_peak_dev":       0.00,
        "saturation_shift":   0.05,
    },
    "Oil Stain": {
        "saturation_shift":   0.30,
        "local_contrast":     0.15,
        "max_pixel_dev_35":   0.10,
        "area_ratio":         0.10,
        "compactness":        0.10,
        "edge_gradient":      0.05,
        "boundary_strength":  0.05,
        "variance_ratio":     0.05,
        "darkness_depth":     0.05,
        "ncc_peak_dev":       0.05,
    },

    # ── Gabor / spectral defects ──
    "Missing Thread": {
        "edge_gradient":      0.20,
        "max_pixel_dev_35":   0.15,
        "area_ratio":         0.15,
        "local_contrast":     0.10,
        "variance_ratio":     0.10,
        "boundary_strength":  0.10,
        "compactness":        0.05,   # elongated by nature
        "darkness_depth":     0.05,
        "ncc_peak_dev":       0.05,
        "saturation_shift":   0.05,
    },
    "Snag": {
        "max_pixel_dev_35":   0.20,
        "edge_gradient":      0.20,
        "variance_ratio":     0.15,
        "local_contrast":     0.10,
        "area_ratio":         0.10,
        "compactness":        0.10,
        "boundary_strength":  0.05,
        "darkness_depth":     0.05,
        "ncc_peak_dev":       0.00,
        "saturation_shift":   0.05,
    },
}

# Seam defect types use a separate fast-path (no ROI image metrics).
SEAM_TYPES = {
    "Skip Stitch", "Broken Stitch", "Run-off Stitch",
    "Crooked Stitch", "Pucker",
}

# Default fallback weights (equal)
_DEFAULT_WEIGHT = 0.10


# ─── Individual metric computers ────────────────────────────────────────────

def _safe_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
    """Extract ROI clamped to image bounds."""
    ih, iw = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _neighbourhood(img: np.ndarray, x: int, y: int, w: int, h: int,
                    expand: float = 0.5) -> Optional[np.ndarray]:
    """Local neighbourhood = bbox expanded by *expand* fraction on each side."""
    ih, iw = img.shape[:2]
    ex = max(10, int(w * expand))
    ey = max(10, int(h * expand))
    nx1, ny1 = max(0, x - ex), max(0, y - ey)
    nx2, ny2 = min(iw, x + w + ex), min(ih, y + h + ey)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return img[ny1:ny2, nx1:nx2]


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


# ── Metric 1: Max Pixel Deviation (35×35 local-mean filter) ──────────────────
def _metric_max_pixel_dev_35(gray: np.ndarray, x: int, y: int,
                              w: int, h: int) -> float:
    roi = _safe_roi(gray, x, y, w, h)
    if roi is None or roi.size == 0:
        return 0.0
    local_mean = cv2.blur(roi, (35, 35))
    dev = np.abs(roi.astype(np.float32) - local_mean.astype(np.float32))
    max_dev = float(np.max(dev))
    # Scale: 0 → 0, 80+ → 100
    return _clamp(max_dev / 80.0 * 100.0)


# ── Metric 2: Local Contrast Ratio ───────────────────────────────────────────
def _metric_local_contrast(gray: np.ndarray, x: int, y: int,
                            w: int, h: int) -> float:
    roi = _safe_roi(gray, x, y, w, h)
    neigh = _neighbourhood(gray, x, y, w, h)
    if roi is None or neigh is None:
        return 0.0
    inner_mean = float(np.mean(roi))
    neigh_mean = max(1.0, float(np.mean(neigh)))
    ratio = abs(inner_mean - neigh_mean) / neigh_mean
    # Scale: 0 → 0, 0.5+ → 100
    return _clamp(ratio / 0.5 * 100.0)


# ── Metric 3: Edge Gradient Magnitude (Sobel) ────────────────────────────────
def _metric_edge_gradient(gray: np.ndarray, x: int, y: int,
                           w: int, h: int) -> float:
    roi = _safe_roi(gray, x, y, w, h)
    if roi is None or roi.size == 0:
        return 0.0
    sx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mean_mag = float(np.mean(mag))
    # Scale: 0 → 0, 150+ → 100
    return _clamp(mean_mag / 150.0 * 100.0)


# ── Metric 4: Area Ratio ─────────────────────────────────────────────────────
def _metric_area_ratio(img_area: int, defect_area: int) -> float:
    if img_area <= 0:
        return 0.0
    ratio = defect_area / img_area
    # Scale: 0 → 0, 0.05 (5% of image) → 100
    return _clamp(ratio / 0.05 * 100.0)


# ── Metric 5: Compactness (circularity) ──────────────────────────────────────
def _metric_compactness(area: float, perimeter: float) -> float:
    if perimeter <= 0:
        return 0.0
    c = (4.0 * np.pi * area) / (perimeter ** 2)
    # c ∈ [0,1] where 1 = perfect circle.  More compact = more "real" defect.
    return _clamp(c * 100.0)


# ── Metric 6: NCC Peak Deviation ─────────────────────────────────────────────
def _metric_ncc_peak_dev(deviation_map: Optional[np.ndarray],
                          x: int, y: int, w: int, h: int) -> float:
    if deviation_map is None:
        return 0.0
    roi = _safe_roi(deviation_map, x, y, w, h)
    if roi is None or roi.size == 0:
        return 0.0
    peak = float(np.max(roi))
    # deviation_map is uint8 0-255.  Scale: 80 → 0, 200+ → 100
    normalised = (peak - 80.0) / 120.0
    return _clamp(normalised * 100.0)


# ── Metric 7: Variance Ratio (inside / neighbourhood) ────────────────────────
def _metric_variance_ratio(gray: np.ndarray, x: int, y: int,
                            w: int, h: int) -> float:
    roi = _safe_roi(gray, x, y, w, h)
    neigh = _neighbourhood(gray, x, y, w, h)
    if roi is None or neigh is None:
        return 0.0
    var_in = float(np.var(roi.astype(np.float32)))
    var_out = max(1.0, float(np.var(neigh.astype(np.float32))))
    # If inside is MORE varied than outside → texture anomaly
    # If inside is LESS varied (e.g. dark hole) → also anomalous
    ratio = abs(var_in - var_out) / var_out
    # Scale: 0 → 0, 2.0+ → 100
    return _clamp(ratio / 2.0 * 100.0)


# ── Metric 8: Darkness Depth ─────────────────────────────────────────────────
def _metric_darkness_depth(gray: np.ndarray, x: int, y: int,
                            w: int, h: int) -> float:
    roi = _safe_roi(gray, x, y, w, h)
    neigh = _neighbourhood(gray, x, y, w, h)
    if roi is None or neigh is None:
        return 0.0
    inner_mean = float(np.mean(roi))
    neigh_mean = max(1.0, float(np.mean(neigh)))
    # Only scores when inner is DARKER than neighbourhood
    depth = max(0.0, (neigh_mean - inner_mean) / neigh_mean)
    # Scale: 0 → 0, 0.4+ → 100
    return _clamp(depth / 0.4 * 100.0)


# ── Metric 9: Saturation Shift ───────────────────────────────────────────────
def _metric_saturation_shift(hsv: Optional[np.ndarray],
                              x: int, y: int, w: int, h: int) -> float:
    if hsv is None:
        return 0.0
    roi = _safe_roi(hsv[:, :, 1], x, y, w, h)
    if roi is None or roi.size == 0:
        return 0.0
    local_sat = float(np.mean(roi))
    global_sat = max(1.0, float(np.mean(hsv[:, :, 1])))
    shift = abs(local_sat - global_sat) / global_sat
    # Scale: 0 → 0, 0.5+ → 100
    return _clamp(shift / 0.5 * 100.0)


# ── Metric 10: Boundary Strength (Canny edge density on contour rim) ─────────
def _metric_boundary_strength(gray: np.ndarray, x: int, y: int,
                               w: int, h: int) -> float:
    roi = _safe_roi(gray, x, y, w, h)
    if roi is None or roi.size == 0:
        return 0.0
    edges = cv2.Canny(roi, 50, 150)
    # Fraction of rim pixels that have a strong edge
    # Create a 3-pixel wide border mask
    mask = np.zeros_like(edges, dtype=np.uint8)
    border = 3
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255
    rim_edges = cv2.bitwise_and(edges, mask)
    rim_total = max(1, np.count_nonzero(mask))
    density = np.count_nonzero(rim_edges) / rim_total
    # Scale: 0 → 0, 0.3+ → 100
    return _clamp(density / 0.3 * 100.0)


# ─── Main scoring function ──────────────────────────────────────────────────

def compute_severity(
    defect: Dict[str, Any],
    gray: np.ndarray,
    hsv: Optional[np.ndarray] = None,
    deviation_map: Optional[np.ndarray] = None,
    contour: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute multi-metric severity for a single defect.

    Parameters
    ----------
    defect : dict
        Must contain  bbox_x, bbox_y, bbox_w, bbox_h, Type, Area (px).
    gray : ndarray
        Grayscale image at processing resolution (same coords as bbox).
    hsv : ndarray or None
        HSV image at processing resolution (for saturation metric).
    deviation_map : ndarray or None
        Engine-specific map (NCC deviation, bg-subtraction diff, Gabor).
    contour : ndarray or None
        OpenCV contour for compactness.  If None, estimated from bbox.

    Returns
    -------
    dict with keys "Severity" (int 0-100) and "Score_Details" (dict).
    """
    d_type = defect.get("Type", "Unknown")
    bx = defect.get("bbox_x", 0)
    by = defect.get("bbox_y", 0)
    bw = defect.get("bbox_w", 1)
    bh = defect.get("bbox_h", 1)
    d_area = defect.get("Area (px)", bw * bh)
    img_h, img_w = gray.shape[:2]
    img_area = img_h * img_w

    # Compute perimeter and area from contour if available
    if contour is not None:
        peri = float(cv2.arcLength(contour, True))
        c_area = float(cv2.contourArea(contour))
    else:
        peri = float(2 * (bw + bh))
        c_area = float(d_area)

    # ── Compute all 10 metrics ──
    scores: Dict[str, float] = {}

    scores["max_pixel_dev_35"]  = _metric_max_pixel_dev_35(gray, bx, by, bw, bh)
    scores["local_contrast"]    = _metric_local_contrast(gray, bx, by, bw, bh)
    scores["edge_gradient"]     = _metric_edge_gradient(gray, bx, by, bw, bh)
    scores["area_ratio"]        = _metric_area_ratio(img_area, d_area)
    scores["compactness"]       = _metric_compactness(c_area, peri)
    scores["ncc_peak_dev"]      = _metric_ncc_peak_dev(deviation_map, bx, by, bw, bh)
    scores["variance_ratio"]    = _metric_variance_ratio(gray, bx, by, bw, bh)
    scores["darkness_depth"]    = _metric_darkness_depth(gray, bx, by, bw, bh)
    scores["saturation_shift"]  = _metric_saturation_shift(hsv, bx, by, bw, bh)
    scores["boundary_strength"] = _metric_boundary_strength(gray, bx, by, bw, bh)

    # Round for readability
    scores = {k: round(v, 1) for k, v in scores.items()}

    # ── Weighted combination ──
    weights = SCORE_WEIGHTS.get(d_type, {})
    total_w = 0.0
    weighted_sum = 0.0
    for metric_name, metric_val in scores.items():
        w = weights.get(metric_name, _DEFAULT_WEIGHT)
        weighted_sum += w * metric_val
        total_w += w

    severity = int(round(weighted_sum / max(total_w, 1e-6)))
    severity = max(0, min(100, severity))

    return {
        "Severity": severity,
        "Score_Details": scores,
    }


def compute_seam_severity(
    defect_type: str,
    raw_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Scoring for seam/stitch defects that don't use ROI image metrics.

    Parameters
    ----------
    defect_type : str
        One of the SEAM_TYPES.
    raw_metrics : dict
        Engine-provided measurements.  Keys vary by type:
            Skip Stitch   → {"gap_width": px}
            Broken Stitch → {"gap_width": px}
            Run-off Stitch→ {"density_ratio": 0-1}
            Crooked Stitch→ {"max_deviation_px": px, "mse": float}
            Pucker        → {"z_score": float, "region_width": px}

    Returns
    -------
    dict with "Severity" and "Score_Details".
    """
    scores: Dict[str, float] = {}

    if defect_type in ("Skip Stitch", "Broken Stitch"):
        gw = raw_metrics.get("gap_width", 0)
        # Wider gap = more severe.  Scale: 10px → 0, 80+ → 100
        scores["gap_width_score"] = _clamp((gw - 10) / 70.0 * 100.0)
        # Broken > Skip by nature (threshold already in classifier)
        scores["type_bonus"] = 60.0 if defect_type == "Broken Stitch" else 30.0
        severity = int(round(scores["gap_width_score"] * 0.7
                              + scores["type_bonus"] * 0.3))

    elif defect_type == "Run-off Stitch":
        dr = raw_metrics.get("density_ratio", 0)
        # Lower density at edges → worse.  Scale: 0.35 → 0, 0.0 → 100
        scores["edge_density_loss"] = _clamp((0.35 - dr) / 0.35 * 100.0)
        severity = int(round(scores["edge_density_loss"]))

    elif defect_type == "Crooked Stitch":
        md = raw_metrics.get("max_deviation_px", 0)
        mse = raw_metrics.get("mse", 0)
        scores["max_deviation_score"] = _clamp(md / 20.0 * 100.0)
        scores["mse_score"] = _clamp(mse / 15.0 * 100.0)
        severity = int(round(scores["max_deviation_score"] * 0.6
                              + scores["mse_score"] * 0.4))

    elif defect_type == "Pucker":
        zs = raw_metrics.get("z_score", 0)
        rw = raw_metrics.get("region_width", 0)
        scores["laplacian_z_score"] = _clamp(zs / 4.0 * 100.0)
        scores["region_width_score"] = _clamp(rw / 100.0 * 100.0)
        severity = int(round(scores["laplacian_z_score"] * 0.7
                              + scores["region_width_score"] * 0.3))
    else:
        severity = 50

    severity = max(0, min(100, severity))
    scores = {k: round(v, 1) for k, v in scores.items()}

    return {
        "Severity": severity,
        "Score_Details": scores,
    }
