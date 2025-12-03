"""
V2 CR-AI Configuration

Production-grade configuration for code review AI operations.
"""

from .settings import settings
from .model_config import MODEL_CONFIG, CONSENSUS_CONFIG
from .review_config import REVIEW_DIMENSIONS, REVIEW_CONFIG

__all__ = [
    "settings",
    "MODEL_CONFIG",
    "CONSENSUS_CONFIG",
    "REVIEW_DIMENSIONS",
    "REVIEW_CONFIG",
]
