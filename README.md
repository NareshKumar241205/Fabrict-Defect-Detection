# Fabric Inspector Pro

**Industry-Standard Automated Optical Inspection (AOI) System**  
*Powered by Spectral Residual (FFT) & Unsupervised Computer Vision.*

![Project Status](https://img.shields.io/badge/Status-Industrial%20Release-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)

## Overview
**Fabric Inspector Pro** is an automated quality control system for textile manufacturing. It replaces human visual inspection with **math-based AI**.

Unlike deep learning models that require thousands of training images, this system uses **Unsupervised Computer Vision** to "learn" the fabric texture instantly and identify anomalies. It uses **Spectral Residual Analysis (FFT)**—the same algorithm used in high-speed industrial AOI machines—to mathematically subtract the weave pattern and isolate defects.

### Defects Detected
| Defect Type | Method | Description |
|:---:|:---:|:---|
| **Structural Tears** | Spectral Residual (FFT) | Cuts, holes, or missing threads hidden in the weave. |
| **Slubs / Knots** | Spectral Residual (FFT) | Thick bunches of thread or weave irregularities. |
| **Skip Stitches** | Projection Profiling | Missing stitches in seam lines (Projection Analysis). |
| **Seam Puckering** | Laplacian Variance | Wrinkled seams indicating poor tension. |
| **Stains / Oil** | LBP Texture Analysis | Local Binary Patterns detect surface discoloration. |

## Key Features
* **Zero-Touch Automation:** No manual tuning. The system auto-detects `Surface`, `Seam`, and `Texture` defects in parallel.
* **Spectral Saliency Engine:** Uses **Fourier Transforms** to remove repetitive background textures, making it robust against complex weaves (Twill, Denim, Plain).
* **Multi-Layer Analysis:** 
    * **Layer 1:** Spectral (Geometry)
    * **Layer 2:** Seam (Stitch Quality)
    * **Layer 3:** Texture (Surface Finish)
* **Real-Time Performance:** Processes 720p feeds in <200ms using optimized NumPy/OpenCV pipelines.
* **Saliency Heatmaps:** Visualizes "what the machine sees" for full explainability.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NareshKumar241205/Fabrict-Defect-Detection.git
    cd Fabrict-Defect-Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the web interface:

```bash
streamlit run app.py
```

### How to Use
1.  **Upload Mode:** Drag & Drop a fabric image.
2.  **Camera Mode:** Use a webcam for live inspection.
3.  **Result:** The system outputs a **Pass/Fail** certification and a detailed Defect Manifest.

## The AI Under the Hood
This project proves you don't need Deep Learning to solve complex vision problems.
-   **Spectral Residual:** `Log(Amplitude(FFT)) - Avg(Log(Amplitude))` = Saliency.
-   **LBP (Local Binary Pattern):** Encodes texture micro-patterns into a histogram.
-   **Projection Profiling:** Sums pixel intensity along axes to find gaps.
