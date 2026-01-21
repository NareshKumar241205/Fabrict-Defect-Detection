# config.py
import numpy as np

# --- SYSTEM CONFIG ---
IMAGE_RESIZE_WIDTH = 800  # High res for better defect detail
DEBUG_MODE = True         # Returns debug images (masks) to UI

# --- MODE A: SMOOTH FABRIC (LBP Logic) ---
# Best for: Cotton, Silk, Polyester (Low noise)
SMOOTH_SETTINGS = {
    "LBP_RADIUS": 3,
    "SENSITIVITY": 3.5,   # Sigma threshold (Higher = Less strict)
    "MIN_AREA": 150       # Minimum defect size in pixels
}

# --- MODE B: TEXTURED FABRIC (DoG Logic) ---
# Best for: Denim, Twill, Knits (High noise)
TEXTURED_SETTINGS = {
    "SIGMA_FINE": 1.0,    # Blur amount to remove grain
    "SIGMA_COARSE": 8.0,  # Blur amount to remove structure
    "THRESHOLD": 25,      # Intensity difference to trigger alert
    "MIN_AREA": 100
}

# --- MODE C: SEAM INSPECTION (Geometry) ---
SEAM_SETTINGS = {
    "ROI_HEIGHT": 100,    # Height of strip to analyze around seam
    "GAP_TOLERANCE": 1.8, # Anomaly threshold for stitch spacing
    "PEAK_HEIGHT": 20     # Minimum brightness of a thread to be seen
}