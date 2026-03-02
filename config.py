# config.py
import numpy as np

# --- SYSTEM ---
DEFAULT_SYSTEM = {
    "IMAGE_RESIZE_WIDTH": 800,
    "PATCH_SIZE": 48,         # Slightly smaller to catch edges
    "STEP_SIZE": 24,          # 50% Overlap
    "BACKGROUND_THRESH": 40   # Pixel intensity below this is "Background"
}

# --- MODULE A: GLCM (Fabric) ---
GLCM_SETTINGS = {
    "DISTANCES": [1],
    "ANGLES": [0, np.pi/2],
    "THRESHOLDS": {
        "contrast_max": 250,      # Anything above this is definitely a hole/slub
        "correlation_min": 0.80,  # Strict: Only flag if correlation drops significantly
        "homogeneity_max": 0.98   # Very strict: Only flag if it's perfectly smooth (oil)
    }
}

# --- MODULE B: SEAM (Stitch) ---
SEAM_SETTINGS = {
    "GAP_TOLERANCE": 10,       # Max pixels allowed between stitches
    "MIN_STITCH_LENGTH": 5,    # Noise filter
    "STITCH_COLOR_THRESH": 180, # Brightness of the thread (adjust if thread is dark)
    "CROOKED_R2_THRESH": 0.85,  # Linear regression R² below this = crooked
    "PUCKER_VAR_SIGMA": 2.0,   # Laplacian variance Z-score for pucker detection
    "RUNOFF_EDGE_MARGIN": 0.10, # Fraction of image width to check for run-off
    "BROKEN_GAP_MIN": 20,      # Projection gap >= this = Broken Stitch (else Skip)
}

# --- DEFECT TAXONOMY ---
# All 10 defect types, grouped for the Unified Processor
DEFECT_TYPES = {
    "group_i": {
        "name": "Fabric Structure",
        "types": {
            # Spectral Residual (FFT) detects these pattern anomalies
            "Missing Thread": {"engine": "spectral", "description": "Absent warp/weft thread in weave"},
            "Slub":           {"engine": "spectral", "description": "Thick, uneven yarn lump"},
            "Oil Stain":      {"engine": "spectral", "description": "Contamination mark on fabric"},
            # Canny/Morphology detects these physical edge defects
            "Hole":           {"engine": "edge", "description": "Complete perforation in fabric"},
            "Tear":           {"engine": "edge", "description": "Ripped fabric along a line"},
            "Snag":           {"engine": "edge", "description": "Pulled loop or thread on surface"},
        }
    },
    "group_ii": {
        "name": "Stitch Quality",
        "types": {
            # Projection Profiling detects these stitch density issues
            "Skip Stitch":    {"engine": "seam_projection", "description": "Missing stitch in seam line"},
            "Broken Stitch":  {"engine": "seam_projection", "description": "Thread break within seam"},
            "Run-off Stitch": {"engine": "seam_projection", "description": "Stitch runs off fabric edge"},
            # Linear Regression checks straightness
            "Crooked Stitch": {"engine": "seam_regression", "description": "Seam deviates from straight line"},
            # Laplacian Variance detects wrinkling
            "Pucker":         {"engine": "seam_laplacian", "description": "Wrinkling/gathering near seam"},
        }
    }
}

# --- UNIFIED PROCESSOR ---
UNIFIED_SETTINGS = {
    "DEFAULT_MODE": "full",       # "full", "structure_only", "seam_only"
    "SEAM_DETECTION_THRESH": 0.15, # Projection gradient threshold for seam pre-classifier
    "SUB_CLASSIFY": True,         # Enable sub-classification refinement
    # Sub-classification thresholds
    "OIL_STAIN_SOLIDITY_MIN": 0.85,    # High solidity = oil stain (round/smooth)
    "TEAR_ASPECT_RATIO_MIN": 3.0,      # Elongated = tear
    "SNAG_AREA_MAX": 800,              # Small edge defect = snag
    "HOLE_AREA_MIN": 1000,             # Large edge defect = hole
    "BROKEN_GAP_MIN": 20,             # Large gap in projection = broken stitch
}
