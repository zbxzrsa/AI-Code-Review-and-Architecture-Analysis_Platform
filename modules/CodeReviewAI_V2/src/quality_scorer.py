"""
CodeReviewAI_V2 - Production Quality Scorer

Enhanced scoring with:
- Configurable weights
- SLO-based grading
- Trend analysis
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import Finding, Dimension, Severity

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Enhanced quality score with SLO tracking"""
    overall: float
    dimensions: Dict[str, float]
    grade: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    findings_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    code_lines: int = 0
    issues_per_line: float = 0.0

    # V2: SLO tracking
    meets_slo: bool = True
    slo_violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall, 2),
            "dimensions": {k: round(v, 2) for k, v in self.dimensions.items()},
            "grade": self.grade,
            "timestamp": self.timestamp.isoformat(),
            "findings_count": self.findings_count,
            "severity_breakdown": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "code_lines": self.code_lines,
            "issues_per_line": round(self.issues_per_line, 4),
            "meets_slo": self.meets_slo,
            "slo_violations": self.slo_violations,
        }


@dataclass
class SLOConfig:
    """V2: SLO configuration"""
    max_critical_issues: int = 0
    max_high_issues: int = 3
    min_score: float = 70.0
    max_issues_per_line: float = 0.1


class QualityScorer:
    """
    Production quality scorer with SLO enforcement.
    """

    def __init__(
        self,
        slo_config: Optional[SLOConfig] = None,
    ):
        self.slo_config = slo_config or SLOConfig()
        self._history: List[QualityScore] = []

        self.severity_weights = {
            Severity.CRITICAL.value: 25.0,
            Severity.HIGH.value: 15.0,
            Severity.MEDIUM.value: 8.0,
            Severity.LOW.value: 3.0,
            Severity.INFO.value: 0.5,
        }

        self.dimension_weights = {
            Dimension.SECURITY.value: 1.5,
            Dimension.CORRECTNESS.value: 1.3,
            Dimension.PERFORMANCE.value: 1.0,
            Dimension.MAINTAINABILITY.value: 0.8,
            Dimension.ARCHITECTURE.value: 0.9,
            Dimension.TESTING.value: 0.7,
            Dimension.DOCUMENTATION.value: 0.6,
        }

        self.grade_thresholds = {"A": 90, "B": 80, "C": 70, "D": 60, "F": 0}

    async def score(
        self,
        code: str,
        findings: List[Finding],
    ) -> QualityScore:
        """Calculate score with SLO checking"""
        code_lines = len(code.split('\n'))

        dimension_scores = {d.value: 100.0 for d in Dimension}
        severity_counts = {s.value: 0 for s in Severity}

        total_penalty = 0.0

        for finding in findings:
            if finding.severity in severity_counts:
                severity_counts[finding.severity] += 1

            sev_weight = self.severity_weights.get(finding.severity, 5.0)
            dim_weight = self.dimension_weights.get(finding.dimension, 1.0)
            penalty = sev_weight * dim_weight * finding.confidence
            total_penalty += penalty

            if finding.dimension in dimension_scores:
                dimension_scores[finding.dimension] -= penalty

        dimension_scores = {k: max(0, v) for k, v in dimension_scores.items()}
        overall = max(0, min(100, 100.0 - total_penalty))
        grade = self._assign_grade(overall)

        # V2: SLO check
        slo_violations = []
        if severity_counts[Severity.CRITICAL.value] > self.slo_config.max_critical_issues:
            slo_violations.append("Critical issues exceed SLO")
        if severity_counts[Severity.HIGH.value] > self.slo_config.max_high_issues:
            slo_violations.append("High issues exceed SLO")
        if overall < self.slo_config.min_score:
            slo_violations.append("Score below SLO threshold")

        issues_per_line = len(findings) / max(1, code_lines)
        if issues_per_line > self.slo_config.max_issues_per_line:
            slo_violations.append("Issue density exceeds SLO")

        score = QualityScore(
            overall=overall,
            dimensions=dimension_scores,
            grade=grade,
            findings_count=len(findings),
            critical_count=severity_counts[Severity.CRITICAL.value],
            high_count=severity_counts[Severity.HIGH.value],
            medium_count=severity_counts[Severity.MEDIUM.value],
            low_count=severity_counts[Severity.LOW.value],
            code_lines=code_lines,
            issues_per_line=issues_per_line,
            meets_slo=len(slo_violations) == 0,
            slo_violations=slo_violations,
        )

        self._history.append(score)
        logger.info(f"Score: {overall:.1f} ({grade}), SLO: {'met' if score.meets_slo else 'violated'}")

        return score

    def _assign_grade(self, score: float) -> str:
        for grade, threshold in sorted(self.grade_thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return grade
        return "F"

    def get_trend(self, limit: int = 10) -> Dict[str, Any]:
        """Get quality trend"""
        if not self._history:
            return {"trend": "unknown", "data": []}

        recent = self._history[-limit:]

        if len(recent) < 2:
            trend = "stable"
        else:
            first = sum(s.overall for s in recent[:len(recent)//2]) / (len(recent)//2)
            second = sum(s.overall for s in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            delta = second - first
            trend = "improving" if delta > 5 else ("declining" if delta < -5 else "stable")

        return {
            "trend": trend,
            "current_score": recent[-1].overall,
            "current_grade": recent[-1].grade,
            "slo_compliance_rate": sum(1 for s in recent if s.meets_slo) / len(recent),
            "data": [{"score": s.overall, "grade": s.grade, "meets_slo": s.meets_slo} for s in recent],
        }
