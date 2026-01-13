# 🧵 Fabric Inspector Pro

**Automated Optical Inspection (AOI) System for Fabric Defect Detection** *Built with Classical Computer Vision & Python — No Deep Learning Required.*

![Project Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)

## 📖 Overview
**Fabric Inspector Pro** is a robust, reference-free inspection tool designed to detect manufacturing defects in textile fabrics. Unlike modern AI approaches that require thousands of training images, this system uses **Classical Computer Vision** techniques (Gabor Filters, Morphological Processing, and Statistical Profiling) to detect defects mathematically.

It can identify and classify:
* **Holes** (Punctures)
* **Oil/Water Stains** (Discoloration)
* **Cuts & Tears** (Structural damage)
* **Snags / Knots** (Texture anomalies)

## ✨ Key Features
* **📸 Dual-Input Mode:** Analyze static images (Upload) or use a real-time **Live Webcam** feed for industrial mockups.
* **🧠 Smart Auto-Calibration:** One-click calibration analyzes a "Golden Sample" to automatically calculate the perfect sensitivity thresholds using statistical standard deviation.
* **🛡️ Negligible Defect Filtering:** adjustable noise tolerance to ignore dust or tiny lint (Pass/Fail logic).
* **🏭 Dual-Branch Detection:**
    * *Texture Branch:* Uses Gabor Filters to find structural breaks (Cuts/Wrinkles).
    * *Intensity Branch:* Uses Adaptive Thresholding to find chemical defects (Stains).
* **📊 Automated Reporting:** Generates a downloadable CSV report listing every defect found, its size, and type.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/fabric-inspector-pro.git](https://github.com/YOUR_USERNAME/fabric-inspector-pro.git)
    cd fabric-inspector-pro
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

Run the application using Streamlit:

```bash
streamlit run app.py
