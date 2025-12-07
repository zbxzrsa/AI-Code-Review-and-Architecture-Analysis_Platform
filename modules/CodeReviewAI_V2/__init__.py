"""
CodeReviewAI_V2 - Production Code Review AI Module

This module provides production-ready AI-powered code review capabilities.
Version: V2 (Production)
Status: Stable, production-ready

Features:
- All V1 features
- Enhanced hallucination detection
- Production-grade error handling
- Performance optimizations
- Extended language support
"""

__version__ = "2.0.0"
__status__ = "production"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .src.code_reviewer import CodeReviewer
    from .src.issue_detector import IssueDetector
    from .src.fix_suggester import FixSuggester
    from .src.hallucination_detector import HallucinationDetector

__all__ = [
    "CodeReviewer",
    "IssueDetector",
    "FixSuggester",
    "HallucinationDetector",
    "__version__",
    "__status__",
]
