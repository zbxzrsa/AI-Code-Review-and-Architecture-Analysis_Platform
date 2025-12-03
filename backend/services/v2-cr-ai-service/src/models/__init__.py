"""
V2 CR-AI Data Models

Pydantic models for code review AI operations.
"""

from .review_models import (
    ReviewRequest,
    ReviewResponse,
    ReviewFinding,
    FindingSeverity,
    FindingCategory,
    CodeLocation,
    FixSuggestion,
)

from .consensus_models import (
    ConsensusResult,
    ModelVerification,
    ConsensusDecision,
)

__all__ = [
    # Review models
    "ReviewRequest",
    "ReviewResponse",
    "ReviewFinding",
    "FindingSeverity",
    "FindingCategory",
    "CodeLocation",
    "FixSuggestion",
    # Consensus models
    "ConsensusResult",
    "ModelVerification",
    "ConsensusDecision",
]
