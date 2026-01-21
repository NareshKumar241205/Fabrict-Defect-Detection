# config.py
import numpy as np

# --- SYSTEM SETTINGS ---
IMAGE_RESIZE_WIDTH = 600
DEBUG_MODE = False

# --- LOGIC A: SURFACE (DIFFERENCE OF GAUSSIANS) ---
DOG = {
    "KERNEL_1": (3, 3),      # Fine details (Weave)
    "KERNEL_2": (21, 21),    # Coarse details (Structure)
    "THRESHOLD": 40,         # Sensitivity (Lower = More sensitive)
    "MIN_AREA": 150,         # Minimum defect size (px) to ignore noise
    "MAX_AREA": 5000         # Maximum defect size
}

# --- LOGIC B: SEAM (RHYTHMIC ANALYSIS) ---
SEAM = {
    "ROI_HEIGHT": 80,        # Height of the scan strip
    "PEAK_PROMINENCE": 20,   # How much a stitch must "pop" out from background
    "RHYTHM_TOLERANCE": 1.8, # If a gap is 1.8x the median gap -> ERROR
    "EDGE_DENSITY_TH": 0.25  # Pucker threshold
}

# --- UI COLORS (BGR) ---
COLORS = {
    "RED": (0, 0, 255),
    "ORANGE": (0, 140, 255),
    "CYAN": (255, 255, 0),
    "GREEN": (0, 255, 0),
    "MAGENTA": (255, 0, 255)
}