# config.py
import numpy as np

# --- SYSTEM ---
DEFAULT_SYSTEM = {
    "IMAGE_RESIZE_WIDTH": 800,
    "PATCH_SIZE": 48,
    "STEP_SIZE": 24,
    "BACKGROUND_THRESH": 40
}

# --- MODULE A: GABOR FILTER (Spectral) ---
GABOR_SETTINGS = {
    "SIGMA": 3.0,
    "LAMBDA": 8.0,
    "GAMMA": 0.5,
    "KSIZE": 31,
    "ANGLES": [0, np.pi/4, np.pi/2, 3*np.pi/4]
}

# --- MODULE A+: TEMPLATE MATCHING (Texture) ---
TEMPLATE_SETTINGS = {
    "PATCH_SIZE": 64,
    "MIN_SSIM_DROP": 0.15,
}

# --- MODULE A++: ADAPTIVE SUBTRACTION (Edge) ---
SUBTRACTION_SETTINGS = {
    "BLUR_KERNEL": (151, 151),
    "MIN_DEPTH": 15,
}

# --- MODULE A+++: REFERENCE COMPARE (Golden Image + SSIM) ---
REFERENCE_SETTINGS = {
    "MIN_ORB_MATCHES": 10,
    "SSIM_WIN_SIZE": 11,
    "SAUVOLA_WINDOW": 51,
    "SAUVOLA_K": 0.2,
}

# --- MODULE B: SEAM (Stitch) ---
SEAM_SETTINGS = {
    "GAP_TOLERANCE": 10,
    "MIN_STITCH_LENGTH": 5,
    "STITCH_COLOR_THRESH": 180,
    "CROOKED_R2_THRESH": 0.85,
    "PUCKER_VAR_SIGMA": 1.5,       # Tightened from 2.0 → catch puckers earlier
    "RUNOFF_EDGE_MARGIN": 0.08,    # Tightened from 0.10 → narrower edge check
    "BROKEN_GAP_MIN": 15,          # Lowered from 20 → catch broken stitches sooner
    "MIN_CENTROID_COUNT": 15,      # Lowered from 20 → detect partial seams
    "CENTER_DENSITY_MIN": 8,       # Min center density to trigger run-off check
    "RUNOFF_DENSITY_RATIO": 0.35,  # Edge must be < 35% of center to be a run-off
}

# --- DEFECT TAXONOMY ---
DEFECT_TYPES = {
    "group_i": {
        "name": "Fabric Structure",
        "types": {
            "Missing Thread": {"engine": "spectral", "description": "Absent warp/weft thread in weave"},
            "Slub":           {"engine": "spectral", "description": "Thick, uneven yarn lump"},
            "Oil Stain":      {"engine": "spectral", "description": "Contamination mark on fabric"},
            "Hole":           {"engine": "edge",     "description": "Complete perforation in fabric"},
            "Tear":           {"engine": "edge",     "description": "Ripped fabric along a line"},
            "Snag":           {"engine": "edge",     "description": "Pulled loop or thread on surface"},
        }
    },
    "group_ii": {
        "name": "Stitch Quality",
        "types": {
            "Skip Stitch":    {"engine": "seam_projection", "description": "Missing stitch in seam line"},
            "Broken Stitch":  {"engine": "seam_projection", "description": "Thread break within seam"},
            "Run-off Stitch": {"engine": "seam_projection", "description": "Stitch runs off fabric edge"},
            "Crooked Stitch": {"engine": "seam_regression", "description": "Seam deviates from straight line"},
            "Pucker":         {"engine": "seam_laplacian",  "description": "Wrinkling/gathering near seam"},
        }
    }
}

# --- UNIFIED PROCESSOR ---
UNIFIED_SETTINGS = {
    "DEFAULT_MODE": "full",
    "SEAM_DETECTION_THRESH": 0.15,
    "SUB_CLASSIFY": True,

    # Sub-classification thresholds (tuned)
    "OIL_STAIN_SOLIDITY_MIN": 0.80,   # Lowered from 0.85 → catch less-round stains
    "TEAR_ASPECT_RATIO_MIN": 3.0,
    "SNAG_AREA_MAX": 600,             # Lowered from 800 → snags are small
    "HOLE_AREA_MIN": 800,             # Lowered from 1000 → catch smaller holes
    "BROKEN_GAP_MIN": 15,

    "MAX_BOX_AREA_RATIO": 0.22,       # Max bbox area as fraction of image (reduced from 0.25)
    "MIN_SOLIDITY": 0.08,             # Discard extremely jagged noise contours
}
