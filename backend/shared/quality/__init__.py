"""
Code Quality Module

Provides code quality analysis and technical debt tracking.
"""
from .technical_debt_tracker import (
    DebtCategory,
    DebtItem,
    DebtPriority,
    DebtStatus,
    TechnicalDebtTracker,
    DuplicationDetector,
    ComplexityAnalyzer,
    DocumentationAnalyzer,
)
