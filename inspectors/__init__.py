"""
Inspectors Package
==================
All fabric defect detection engines grouped together.
"""

from .texture_inspector import inspector
from .spectral_inspector import spectral_inspector
from .stitch_inspector import stitch_inspector
from .edge_detector import edge_detector
from .glcm_inspector import glcm_inspector

__all__ = [
    "inspector",
    "spectral_inspector",
    "stitch_inspector",
    "edge_detector",
    "glcm_inspector",
]
