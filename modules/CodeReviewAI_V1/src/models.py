"""
CodeReviewAI_V1 Data Models

Data classes and enums for code review operations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from enum import Enum


class ReviewStatus(str, Enum):
    """Status of a review"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class Severity(str, Enum):
    """Severity levels for findings"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Dimension(str, Enum):
    """Review dimensions"""
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    ARCHITECTURE = "architecture"
    TESTING = "testing"


@dataclass
class ReviewConfig:
    """Configuration for code review"""
    dimensions: List[Dimension] = field(default_factory=lambda: list(Dimension))
    max_findings: int = 50
    min_confidence: float = 0.7
    include_suggestions: bool = True
    include_explanations: bool = True
    language: Optional[str] = None
    context_lines: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": [d.value for d in self.dimensions],
            "max_findings": self.max_findings,
            "min_confidence": self.min_confidence,
            "include_suggestions": self.include_suggestions,
            "include_explanations": self.include_explanations,
            "language": self.language,
            "context_lines": self.context_lines,
        }


@dataclass
class Finding:
    """A single finding from code review"""
    dimension: str
    issue: str
    line_numbers: List[int]
    severity: str
    confidence: float
    suggestion: str
    explanation: str

    # Optional metadata
    cwe_id: Optional[str] = None
    rule_id: Optional[str] = None
    code_snippet: Optional[str] = None
    fix_snippet: Optional[str] = None
    reasoning_steps: List[str] = field(default_factory=list)

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
            "fix_snippet": self.fix_snippet,
            "reasoning_steps": self.reasoning_steps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        return cls(
            dimension=data["dimension"],
            issue=data["issue"],
            line_numbers=data["line_numbers"],
            severity=data["severity"],
            confidence=data["confidence"],
            suggestion=data.get("suggestion", ""),
            explanation=data.get("explanation", ""),
            cwe_id=data.get("cwe_id"),
            rule_id=data.get("rule_id"),
            code_snippet=data.get("code_snippet"),
            fix_snippet=data.get("fix_snippet"),
            reasoning_steps=data.get("reasoning_steps", []),
        )


@dataclass
class ReviewResult:
    """Complete result from code review"""
    review_id: str
    code_hash: str
    status: ReviewStatus

    # Findings by dimension
    findings: List[Finding]

    # Scores
    overall_score: float  # 0-100
    dimension_scores: Dict[str, float]

    # Metadata
    model_version: str
    strategy_used: str
    processing_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Confidence
    avg_confidence: float = 0.0
    min_confidence: float = 0.0

    # Verification
    hallucination_check_passed: bool = True
    verified_findings_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "code_hash": self.code_hash,
            "status": self.status.value,
            "findings": [f.to_dict() for f in self.findings],
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "model_version": self.model_version,
            "strategy_used": self.strategy_used,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "hallucination_check_passed": self.hallucination_check_passed,
            "verified_findings_count": self.verified_findings_count,
        }

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        severity_counts = {}
        dimension_counts = {}

        for finding in self.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            dimension_counts[finding.dimension] = dimension_counts.get(finding.dimension, 0) + 1

        return {
            "total_findings": len(self.findings),
            "by_severity": severity_counts,
            "by_dimension": dimension_counts,
            "overall_score": self.overall_score,
            "avg_confidence": self.avg_confidence,
        }
