"""
Hallucination Detection module for V1 Code Review AI.

Detects and mitigates hallucinations in code reviews:
- Consistency checking
- Fact verification
- Confidence scoring
"""

from .detector import (
    HallucinationDetector,
    HallucinationResult,
    ConsistencyChecker,
    FactChecker,
)

__all__ = [
    "HallucinationDetector",
    "HallucinationResult",
    "ConsistencyChecker",
    "FactChecker",
]
