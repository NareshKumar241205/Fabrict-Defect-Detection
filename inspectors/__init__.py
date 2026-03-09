"""
Inspectors Package
==================
All fabric defect detection engines grouped together.
The UnifiedProcessor is the recommended entry point.
"""

from .unified_processor import unified_processor

__all__ = [
    "unified_processor",
]