"""
Inspectors Package
==================
All fabric defect detection engines grouped together.
The UnifiedProcessor is the recommended entry point.
"""

from .texture_inspector import inspector as texture_inspector
from .spectral_inspector import spectral_inspector
from .seam_inspector import seam_inspector
from .edge_inspector import edge_inspector
from .unified_processor import unified_processor

__all__ = [
    "texture_inspector",
    "spectral_inspector",
    "seam_inspector",
    "edge_inspector",
    "unified_processor",
]