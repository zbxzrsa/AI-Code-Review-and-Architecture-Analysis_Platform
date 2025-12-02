"""
Review module for V1 Code Review AI.

Contains:
- Code review engine with multi-strategy support
- Multi-dimensional analysis
- Finding aggregation
"""

from .engine import (
    ReviewEngine,
    ReviewResult,
    Finding,
)
from .strategies import (
    BaselineStrategy,
    ChainOfThoughtStrategy,
    FewShotStrategy,
    ContrastiveStrategy,
    EnsembleStrategy,
)
from .dimensions import (
    DimensionAnalyzer,
    CorrectnessAnalyzer,
    SecurityAnalyzer,
    PerformanceAnalyzer,
)

__all__ = [
    "ReviewEngine",
    "ReviewResult",
    "Finding",
    "BaselineStrategy",
    "ChainOfThoughtStrategy",
    "FewShotStrategy",
    "ContrastiveStrategy",
    "EnsembleStrategy",
    "DimensionAnalyzer",
    "CorrectnessAnalyzer",
    "SecurityAnalyzer",
    "PerformanceAnalyzer",
]
