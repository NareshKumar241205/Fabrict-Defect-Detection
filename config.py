# config.py
import numpy as np

# ==========================================
# PIPELINE 1: CLASSICAL CV (Oil, Hole, Tear)
# ==========================================
CLASSIC_SETTINGS = {
    "IMAGE_RESIZE_WIDTH": 800,
    "HOLE_AREA_MIN": 800,
    "MAX_BOX_AREA_RATIO": 0.22,
    "TEAR_ASPECT_RATIO_MIN": 2.0,
}

# ==========================================
# PIPELINE 2: LOGIC CHANGE (Slub, Skip/Miss/Crooked Stitch)
# ==========================================
LOGIC_SPECTRAL_SETTINGS = {
    "IMAGE_RESIZE_WIDTH": 800,
    "SMOOTHING_KERNEL": 5,
    "USE_DWT": True,
}

LOGIC_SEAM_SETTINGS = {
    "STITCH_COLOR_THRESH": 180,
    "GAP_TOLERANCE": 10,
    "CROOKED_R2_THRESH": 0.85,
    "RUNOFF_EDGE_MARGIN": 0.10,
    "BROKEN_GAP_MIN": 20,
}

# ==========================================
# PIPELINE 3: SEAM INSPECTOR
# ==========================================
SEAM_SETTINGS = {
    "STITCH_COLOR_THRESH": 180,
    "GAP_TOLERANCE": 10,
    "CROOKED_R2_THRESH": 0.85,
    "PUCKER_VAR_SIGMA": 2.0,
    "RUNOFF_EDGE_MARGIN": 0.10,
    "BROKEN_GAP_MIN": 20,
}

# ==========================================
# PIPELINE 4: TEXTURE INSPECTOR
# ==========================================
TEXTURE_SETTINGS = {
    "USE_SAUVOLA_TEXTURE": True,
    "SAUVOLA_K": 0.2,
    "SAUVOLA_WINDOW": 151,
}

# ==========================================
# GLCM SETTINGS (for texture inspector)
# ==========================================
GLCM_SETTINGS = {
    "DISTANCES": [1],
    "ANGLES": [0, np.pi / 2],
    "THRESHOLDS": {
        "contrast_max": 250,
        "correlation_min": 0.80,
        "homogeneity_max": 0.98,
    },
}

# ==========================================
# UNIFIED SETTINGS (shared across inspectors)
# ==========================================
UNIFIED_SETTINGS = {
    "USE_SAUVOLA_TEXTURE": True,
    "SAUVOLA_WINDOW": 151,
}
