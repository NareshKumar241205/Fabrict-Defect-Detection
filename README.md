# Fabric Defect Detector

**Automated Optical Inspection (AOI) System for Fabric Defect Detection** *Built with Classical Computer Vision & Python*

![Project Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)

## Overview
**FabricQA** is a robust, reference-free inspection tool designed to detect manufacturing defects in textile fabrics. Unlike modern AI approaches that require thousands of training images, this system uses **Classical Computer Vision** techniques (LBP, Gabor Filters, FFT Spectral Analysis, Morphological Processing, and Statistical Profiling) to detect defects mathematically.

It can identify and classify:
* **Holes** (Punctures, Ragged Holes)
* **Oil/Water Stains** (Discoloration)
* **Cuts & Tears** (Structural damage — horizontal and vertical)
* **Snags / Knots** (Texture anomalies)
* **Skip Stitches** (Seam defects)
* **Wrinkles / Folds** (Structural deformation)
* **Weave Irregularities** (Broken weave lines)

## Key Features
* **5 Inspection Modes:** Full, Texture, Spectral, Seam, Edge/Structural
* **Batch Processing:** Upload multiple images and get a combined CSV report
* **Inspection History:** Track quality trends over time with charts
* **Gabor Filter Bank:** 18 directional filters to catch diagonal/oriented defects
* **Multi-Resolution FFT:** Spectral analysis at multiple scales for micro & macro defects
* **Smart Confidence Scores:** Z-score based confidence metrics (not hardcoded)
* **Auto-Calibration:** Statistical thresholding adapts to each image
* **Live Camera:** Real-time webcam inspection
* **Automated Reporting:** Downloadable CSV defect reports

## Architecture

```
FabricQA/
├── app.py                      # Streamlit dashboard (UI + routing)
├── config.py                   # Global settings (GLCM, Seam, System)
├── requirements.txt            # Pinned dependencies
├── inspectors/
│   ├── __init__.py             # Package exports
│   ├── texture_inspector.py    # LBP + Entropy + Gabor filter bank
│   ├── spectral_inspector.py   # Multi-res FFT Spectral Residual
│   ├── seam_inspector.py       # Stitch gap analysis via projection
│   ├── edge_inspector.py       # Laplacian variance + Hough lines
│   └── reference_inspector.py  # ORB alignment + pixel diff
├── tests/
│   ├── test_texture.py
│   ├── test_spectral.py
│   ├── test_seam.py
│   ├── test_edge.py
│   └── test_reference.py
└── Dataset/                    # Sample images (train/valid/test)
```

## How It Works

| Inspector | Algorithm | Detects |
|-----------|-----------|---------|
| **Texture** | LBP + Local Entropy + Gabor bank (18 filters) + Multi-scale pyramid | Holes, stains, cuts, texture anomalies |
| **Spectral** | FFT Spectral Residual at 3 resolutions + Z-score thresholding | Repetitive pattern disruptions |
| **Seam** | CLAHE + Canny + HoughLines deskew + Projection profile | Skip stitches, thread gaps |
| **Edge** | Laplacian variance patches + Hough line spacing regularity | Wrinkles, folds, broken weave |
| **Reference** | ORB feature matching + Homography alignment + Pixel diff | Any deviation from golden sample |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NareshKumar241205/Fabrict-Defect-Detection.git
    cd Fabrict-Defect-Detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate    # Windows
    # source .venv/bin/activate  # Linux/Mac
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

### Running Tests

```bash
pytest tests/ -v
```

## Inspection Modes

| Mode | Description |
|------|-------------|
| 🔬 **Full Inspection** | Runs all inspectors (Texture + Spectral + Seam + Edge) |
| 🧵 **Texture Analysis** | LBP + Entropy + Gabor directional filters |
| 📡 **Spectral (FFT)** | Multi-resolution frequency saliency |
| 🪡 **Seam / Stitch** | Skip-stitch and thread gap detection |
| 📐 **Edge / Structure** | Wrinkle, fold, and weave irregularity detection |
|  **Batch Inspection** | Process multiple images with combined report |
| 📊 **History** | View past inspection results and defect trends |

## License

This project is open source and available under the [MIT License](LICENSE).
