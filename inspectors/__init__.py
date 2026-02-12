"""
Inspectors Package
==================
All fabric defect detection engines grouped together.
"""

from .texture_inspector import inspector
from .spectral_inspector import spectral_inspector

__all__ = [
    "inspector",
    "spectral_inspector",
]