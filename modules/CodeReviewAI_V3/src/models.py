"""
CodeReviewAI_V3 Data Models

Legacy models for quarantine and comparison.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from enum import Enum


class ReviewStatus(str, Enum):
    COMPLETED = "completed"
    QUARANTINED = "quarantined"
    RE_EVALUATING = "re_evaluating"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Dimension(str, Enum):
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"


@dataclass
class Finding:
    """Legacy finding model"""
    dimension: str
    issue: str
    line_numbers: List[int]
    severity: str
    confidence: float
    suggestion: str
    explanation: str
    cwe_id: Optional[str] = None
    rule_id: Optional[str] = None
    code_snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "issue": self.issue,
            "line_numbers": self.line_numbers,
            "severity": self.severity,
            "confidence": self.confidence,
            "suggestion": self.suggestion,
            "explanation": self.explanation,
            "cwe_id": self.cwe_id,
            "rule_id": self.rule_id,
            "code_snippet": self.code_snippet,
        }


@dataclass
class ReviewResult:
    """Legacy review result"""
    review_id: str
    code_hash: str
    status: ReviewStatus
    findings: List[Finding]
    overall_score: float
    dimension_scores: Dict[str, float]
    model_version: str
    processing_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # V3: Quarantine metadata
    quarantine_reason: Optional[str] = None
    quarantined_at: Optional[datetime] = None
    re_evaluation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "code_hash": self.code_hash,
            "status": self.status.value,
            "findings": [f.to_dict() for f in self.findings],
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "model_version": self.model_version,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "quarantine_reason": self.quarantine_reason,
            "quarantined_at": self.quarantined_at.isoformat() if self.quarantined_at else None,
            "re_evaluation_count": self.re_evaluation_count,
        }


@dataclass
class ComparisonResult:
    """Result of comparing versions"""
    comparison_id: str
    baseline_version: str
    compare_version: str

    # Metrics comparison
    score_delta: float
    findings_delta: int
    latency_delta_ms: float

    # Statistical analysis
    is_significant: bool
    confidence_interval: tuple
    p_value: float

    # Recommendation
    recommendation: str  # "promote", "keep", "quarantine"

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "baseline_version": self.baseline_version,
            "compare_version": self.compare_version,
            "score_delta": self.score_delta,
            "findings_delta": self.findings_delta,
            "latency_delta_ms": self.latency_delta_ms,
            "is_significant": self.is_significant,
            "confidence_interval": self.confidence_interval,
            "p_value": self.p_value,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }
