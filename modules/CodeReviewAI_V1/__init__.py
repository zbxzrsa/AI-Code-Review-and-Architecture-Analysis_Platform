"""
CodeReviewAI_V1 - Experimental Code Review AI Module

This module provides AI-powered code review capabilities.
Version: V1 (Experimental)
Status: Under active development and testing

Features:
- Code analysis and review
- Issue detection
- Fix suggestions
- Quality scoring
"""

__version__ = "1.0.0"
__status__ = "experimental"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .src.code_reviewer import CodeReviewer
    from .src.issue_detector import IssueDetector
    from .src.fix_suggester import FixSuggester

__all__ = [
    "CodeReviewer",
    "IssueDetector",
    "FixSuggester",
    "__version__",
    "__status__",
]
