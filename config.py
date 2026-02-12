# config.py
import numpy as np

# --- SYSTEM ---
DEFAULT_SYSTEM = {
    "IMAGE_RESIZE_WIDTH": 800,
    "PATCH_SIZE": 48,         # Slightly smaller to catch edges, but we will merge them later
    "STEP_SIZE": 24,          # 50% Overlap
    "BACKGROUND_THRESH": 40   # Pixel intensity below this is considered "Background" and ignored
}

# --- MODULE A: GLCM (Fabric) ---
DEFAULT_GLCM = {
    "DISTANCES": [1],
    "ANGLES": [0, np.pi/2],
    "CONTRAST_MAX": 250,      # Anything above this is definitely a hole/slub
    "CORRELATION_MIN": 0.80,  # Strict: Only flag if correlation drops significantly
    "HOMOGENEITY_MAX": 0.98   # Very strict: Only flag if it's perfectly smooth (oil)
}

# --- MODULE B: SEAM (Stitch) ---
DEFAULT_SEAM = {
    "GAP_TOLERANCE": 10,       # Max pixels allowed between stitches
    "MIN_STITCH_LENGTH": 5,    # Noise filter
    "STITCH_COLOR_THRESH": 180 # Brightness of the thread (adjust if thread is dark)
}
