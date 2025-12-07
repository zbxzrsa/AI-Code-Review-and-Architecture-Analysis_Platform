"""
CodeReviewAI_V3 - Legacy/Quarantine Code Review AI Module

This module provides quarantined code review capabilities for comparison.
Version: V3 (Legacy/Quarantine)
Status: Deprecated, read-only mode

Purpose:
- Baseline comparison for V1/V2 performance
- Re-evaluation of quarantined experiments
- Historical analysis
"""

__version__ = "3.0.0"
__status__ = "quarantine"
__deprecated__ = True

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .src.code_reviewer import CodeReviewer
    from .src.comparison_engine import ComparisonEngine

__all__ = [
    "CodeReviewer",
    "ComparisonEngine",
    "__version__",
    "__status__",
]
