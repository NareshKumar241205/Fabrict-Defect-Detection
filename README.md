# Fabric Defect Detector

**Automated Optical Inspection (AOI) System for Fabric Defect Detection** — *100% Classical Computer Vision, Zero ML/DL*

![Project Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Classical%20CV-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)

---

## Overview

**FabricQA** is a reference-free inspection tool that detects manufacturing defects in textile fabrics using **only classical computer vision** — no neural networks, no pre-trained models, no training step. It uses Gabor filter banks, Sauvola adaptive thresholding, NCC template matching, Canny + Hough line analysis, projection profiling, and multi-metric severity scoring to detect **11 defect types** across two categories.

### Defect Taxonomy

| Category | Group | Defect Type | Detection Engine |
|----------|-------|-------------|-----------------|
| **Structural** | Fabric Structure | Hole | Edge Inspector (Sauvola + contour geometry) |
| **Structural** | Fabric Structure | Tear | Edge Inspector (aspect-ratio elongation) |
| **Structural** | Fabric Structure | Missing Thread | Spectral Inspector (Gabor filter bank) |
| **Structural** | Stitch Quality | Skip Stitch | Seam Inspector (projection profiling) |
| **Structural** | Stitch Quality | Broken Stitch | Seam Inspector (projection profiling) |
| **Surface** | Fabric Structure | Slub | Spectral Inspector (Gabor filter bank) |
| **Surface** | Fabric Structure | Snag | Edge Inspector (small-area contour) |
| **Surface** | Fabric Structure | Oil Stain | Edge Inspector (high-solidity contour) |
| **Surface** | Stitch Quality | Run-off Stitch | Seam Inspector (projection profiling) |
| **Surface** | Stitch Quality | Crooked Stitch | Seam Inspector (linear regression R²) |
| **Surface** | Stitch Quality | Pucker | Seam Inspector (Laplacian variance) |

---

## Architecture

```
FabricQA/
├── app.py                          # Streamlit dashboard (UI + routing)
├── config.py                       # Global settings & defect taxonomy
├── requirements.txt                # Pinned dependencies
├── runtime.txt                     # Python version for deployment
├── inspectors/
│   ├── __init__.py                 # Package exports
│   ├── unified_processor.py        # Main pipeline: pre-classify → route → NMS → score → validate
│   ├── defect_score.py             # 10-metric severity scoring (fabric + seam)
│   ├── spectral_inspector.py       # Engine A: Gabor filter bank → saliency map
│   ├── texture_inspector.py        # Engine B: NCC template matching + Sauvola
│   ├── edge_inspector.py           # Engine C: Sobel/Canny + contour geometry
│   ├── seam_inspector.py           # Engines D/E/F: projection, regression, Laplacian
│   └── reference_inspector.py      # ORB alignment + SSIM golden-image compare
└── Dataset/                        # COCO-annotated images (train/valid/test splits)
```

---

## Pipeline

```
Input Image
    │
    ├─► Shadow Removal (optional)
    │
    ├─► Deskew (rotation correction via Hough angle — pre-classifier only)
    │
    ├─► Pre-Classify
    │       ├─ Canny + HoughLinesP → seam line detection (horizontal/vertical)
    │       └─ Fallback: dual-axis intensity projection profiling
    │
    ├─► Route
    │       ├─ Group I: Fabric Structure (always runs)
    │       │     ├─ Engine A: Spectral (Gabor) → Missing Thread, Slub, Snag
    │       │     ├─ Engine B: Texture (NCC + Sauvola) → Hole, Tear, Slub
    │       │     └─ Engine C: Edge (Sobel/Canny + contours) → Hole, Tear, Oil Stain
    │       │
    │       └─ Group II: Stitch Quality (only if seam detected)
    │             ├─ Engine D: Projection Profiling → Skip/Broken/Run-off Stitch
    │             ├─ Engine E: Linear Regression (R²) → Crooked Stitch
    │             └─ Engine F: Laplacian Variance → Pucker
    │
    ├─► Non-Maximum Suppression (IoU = 0.35)
    │
    ├─► Multi-Metric Severity Scoring (10 classical CV metrics, 0–100)
    │
    ├─► Severity Gate (≥ 30)
    │
    └─► Geometry Validator (Sobel gradient + local contrast)
            └─► Final Defect List
```

---

## Algorithms

### Pre-Classification (Routing)
- **Canny edge detection + Hough Line Transform** (`HoughLinesP`) — detects dominant long lines to identify seam presence and orientation (horizontal vs vertical)
- **Intensity projection profiling** — horizontal (`axis=1`) and vertical (`axis=0`) mean-intensity projections as fallback seam detector
- Vertical seams: image is rotated 90° CCW before seam engines, bounding boxes transformed back afterward

### Group I: Fabric Structure

| Engine | Algorithm | Detects |
|--------|-----------|---------|
| **A — Spectral** | Multi-orientation **Gabor filter bank** (4 angles × configurable σ/λ) → saliency map → connected-component extraction | Missing Thread, Snag, Slub |
| **B — Texture** | **NCC (Normalized Cross-Correlation)** template matching + **Sauvola adaptive thresholding** (scikit-image) → anomaly binarization | Hole, Tear, Missing Thread, Slub |
| **C — Edge** | **Sobel gradients** + **Canny edge detection** → contour analysis with geometry rules (aspect ratio, solidity, area thresholds) | Hole, Tear, Oil Stain |

### Group II: Stitch Quality

| Engine | Algorithm | Detects |
|--------|-----------|---------|
| **D — Projection** | Row-wise intensity projection along the stitch line → gap/density-drop detection | Skip Stitch, Broken Stitch, Run-off Stitch |
| **E — Regression** | Stitch centroid extraction → **least-squares linear regression** → **R²** deviation from straight line | Crooked Stitch |
| **F — Laplacian** | **Laplacian operator** on local patches → texture variance → σ-threshold flagging | Pucker |

### Post-Processing
- **Non-Maximum Suppression (NMS)** at IoU = 0.35 — merges duplicate detections across engines
- **10-metric severity scoring** — contrast ratio, edge gradient, max pixel deviation, saturation shift, solidity, compactness, boundary sharpness, texture homogeneity, size ratio, isolation score — weighted per defect type
- **Severity gate** (≥ 30) — drops low-confidence proposals
- **Photometric geometry validator** — Sobel gradient + local contrast check (texture-engine and stitch-quality defects are exempt)

---

## Key Features

- **11 Defect Types** across Structural and Surface categories
- **Dual-orientation seam detection** — handles both horizontal and vertical seam lines
- **6 Inspection Modes** — Full, Texture, Spectral, Seam, Edge/Structural, Reference Compare
- **Batch Processing** — upload multiple images, get a combined CSV + PDF report
- **Inspection History** — track quality trends over time with charts
- **Before/After Comparison Slider** — visual diff overlay
- **PDF Report Generation** — downloadable inspection reports with annotated images
- **Live Camera** — real-time webcam inspection
- **Zero training data required** — all thresholds are statistically adaptive

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NareshKumar241205/Fabrict-Defect-Detection.git
   cd Fabrict-Defect-Detection
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
streamlit run app.py
```

---

## Inspection Modes

| Mode | Description |
|------|-------------|
| 🔬 **Full Inspection** | Runs all engines (Spectral + Texture + Edge + Seam) via UnifiedProcessor |
| 🧵 **Texture Analysis** | NCC template matching + Sauvola adaptive thresholding |
| 📡 **Spectral (Gabor)** | Multi-orientation Gabor filter bank saliency |
| 🪡 **Seam / Stitch** | Projection profiling + regression + Laplacian (both orientations) |
| 📐 **Edge / Structure** | Canny + Sobel + contour geometry analysis |
| 🖼️ **Reference Compare** | ORB alignment + SSIM golden-image comparison |
| 📦 **Batch Inspection** | Process multiple images with combined CSV/PDF report |
| 📊 **History** | View past inspection results and defect trends |

## Configuration

All tunable parameters are in [`config.py`](config.py):

| Setting | Default | Purpose |
|---------|---------|---------|
| `IMAGE_RESIZE_WIDTH` | 800 | Processing resolution |
| `SEAM_DETECTION_THRESH` | 0.15 | Projection gradient threshold for seam gate |
| `HOLE_AREA_MIN` | 1500 | Minimum contour area (px²) for Hole classification |
| `FP_MIN_CONTRAST` | 0.05 | Geometry validator: min local contrast ratio |
| `FP_MIN_GRADIENT` | 8.0 | Geometry validator: min Sobel gradient magnitude |
| `CROOKED_R2_THRESH` | 0.85 | R² threshold below which stitch is "crooked" |
| `PUCKER_VAR_SIGMA` | 1.5 | Laplacian variance σ-multiplier for pucker detection |

## License

This project is open source and available under the [MIT License](LICENSE).
