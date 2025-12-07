"""
CodeReviewAI_V1 - Quality Scorer

Calculates code quality scores across multiple dimensions:
- Overall quality score
- Dimension-specific scores
- Trend analysis
- Benchmarking
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import Finding, Dimension, Severity

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score result"""
    overall: float  # 0-100
    dimensions: Dict[str, float]
    grade: str  # A, B, C, D, F
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Details
    findings_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Metadata
    code_lines: int = 0
    issues_per_line: float = 0.0

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
        }


@dataclass
class QualityConfig:
    """Configuration for quality scoring"""
    # Severity weights
    severity_weights: Dict[str, float] = field(default_factory=lambda: {
        Severity.CRITICAL.value: 25.0,
        Severity.HIGH.value: 15.0,
        Severity.MEDIUM.value: 8.0,
        Severity.LOW.value: 3.0,
        Severity.INFO.value: 0.5,
    })

    # Dimension weights
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        Dimension.SECURITY.value: 1.5,
        Dimension.CORRECTNESS.value: 1.3,
        Dimension.PERFORMANCE.value: 1.0,
        Dimension.MAINTAINABILITY.value: 0.8,
        Dimension.ARCHITECTURE.value: 0.9,
        Dimension.TESTING.value: 0.7,
    })

    # Grade thresholds
    grade_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "A": 90,
        "B": 80,
        "C": 70,
        "D": 60,
        "F": 0,
    })


class QualityScorer:
    """
    Calculates comprehensive quality scores.

    Supports:
    - Multi-dimensional scoring
    - Configurable weights
    - Grade assignment
    - Trend tracking
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize quality scorer"""
        self.config = config or QualityConfig()
        self._history: List[QualityScore] = []

    async def score(
        self,
        code: str,
        findings: List[Finding],
    ) -> QualityScore:
        """
        Calculate quality score for code.

        Args:
            code: Source code being scored
            findings: List of detected findings

        Returns:
            QualityScore with overall and dimension scores
        """
        code_lines = len(code.split('\n'))

        # Initialize dimension scores
        dimension_scores = {d.value: 100.0 for d in Dimension}

        # Count severities
        severity_counts = {
            Severity.CRITICAL.value: 0,
            Severity.HIGH.value: 0,
            Severity.MEDIUM.value: 0,
            Severity.LOW.value: 0,
            Severity.INFO.value: 0,
        }

        # Process findings
        total_penalty = 0.0

        for finding in findings:
            severity = finding.severity
            dimension = finding.dimension
            confidence = finding.confidence

            # Count severity
            if severity in severity_counts:
                severity_counts[severity] += 1

            # Calculate penalty
            severity_weight = self.config.severity_weights.get(severity, 5.0)
            dimension_weight = self.config.dimension_weights.get(dimension, 1.0)

            penalty = severity_weight * dimension_weight * confidence
            total_penalty += penalty

            # Apply to dimension score
            if dimension in dimension_scores:
                dimension_scores[dimension] -= penalty

        # Ensure non-negative dimension scores
        dimension_scores = {k: max(0, v) for k, v in dimension_scores.items()}

        # Calculate overall score
        overall = 100.0 - total_penalty
        overall = max(0, min(100, overall))

        # Calculate weighted average of dimensions
        weighted_sum = sum(
            dimension_scores[d] * self.config.dimension_weights.get(d, 1.0)
            for d in dimension_scores
        )
        total_weight = sum(self.config.dimension_weights.values())
        weighted_overall = weighted_sum / total_weight

        # Use the lower of the two methods
        overall = min(overall, weighted_overall)

        # Assign grade
        grade = self._assign_grade(overall)

        # Create score result
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
            issues_per_line=len(findings) / max(1, code_lines),
        )

        # Store in history
        self._history.append(score)

        logger.info(f"Quality score: {overall:.1f} ({grade}) - {len(findings)} findings")
        return score

    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score"""
        for grade, threshold in sorted(
            self.config.grade_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if score >= threshold:
                return grade
        return "F"

    async def compare_scores(
        self,
        score1: QualityScore,
        score2: QualityScore,
    ) -> Dict[str, Any]:
        """
        Compare two quality scores.

        Args:
            score1: First score (typically older)
            score2: Second score (typically newer)

        Returns:
            Comparison result with improvements/regressions
        """
        overall_delta = score2.overall - score1.overall

        dimension_deltas = {}
        for dim in score1.dimensions:
            if dim in score2.dimensions:
                dimension_deltas[dim] = score2.dimensions[dim] - score1.dimensions[dim]

        improvements = [d for d, delta in dimension_deltas.items() if delta > 0]
        regressions = [d for d, delta in dimension_deltas.items() if delta < 0]

        return {
            "overall_delta": overall_delta,
            "overall_improved": overall_delta > 0,
            "grade_change": f"{score1.grade} → {score2.grade}",
            "dimension_deltas": dimension_deltas,
            "improvements": improvements,
            "regressions": regressions,
            "findings_delta": score2.findings_count - score1.findings_count,
        }

    def get_trend(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get quality trend from history.

        Args:
            limit: Number of historical scores to include

        Returns:
            Trend analysis
        """
        if not self._history:
            return {"trend": "unknown", "data": []}

        recent = self._history[-limit:]

        # Calculate trend
        if len(recent) < 2:
            trend = "stable"
        else:
            first_half = sum(s.overall for s in recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(s.overall for s in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

            delta = second_half - first_half
            if delta > 5:
                trend = "improving"
            elif delta < -5:
                trend = "declining"
            else:
                trend = "stable"

        return {
            "trend": trend,
            "current_score": recent[-1].overall if recent else None,
            "current_grade": recent[-1].grade if recent else None,
            "history_count": len(self._history),
            "data": [
                {
                    "score": s.overall,
                    "grade": s.grade,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in recent
            ],
        }

    def get_benchmark(self, score: QualityScore) -> Dict[str, Any]:
        """
        Compare score against benchmarks.

        Args:
            score: Score to benchmark

        Returns:
            Benchmark comparison
        """
        # Industry benchmarks (configurable)
        benchmarks = {
            "excellent": 90,
            "good": 75,
            "acceptable": 60,
            "poor": 40,
        }

        category = "poor"
        for cat, threshold in sorted(benchmarks.items(), key=lambda x: x[1], reverse=True):
            if score.overall >= threshold:
                category = cat
                break

        return {
            "score": score.overall,
            "category": category,
            "benchmarks": benchmarks,
            "percentile": self._estimate_percentile(score.overall),
            "recommendations": self._get_recommendations(score),
        }

    def _estimate_percentile(self, score: float) -> int:
        """Estimate percentile ranking"""
        # Simplified percentile estimation
        if score >= 95:
            return 99
        elif score >= 90:
            return 95
        elif score >= 85:
            return 90
        elif score >= 80:
            return 80
        elif score >= 70:
            return 60
        elif score >= 60:
            return 40
        elif score >= 50:
            return 25
        else:
            return 10

    def _get_recommendations(self, score: QualityScore) -> List[str]:
        """Get improvement recommendations"""
        recommendations = []

        # Based on severity counts
        if score.critical_count > 0:
            recommendations.append(
                f"Address {score.critical_count} critical issues immediately"
            )

        if score.high_count > 2:
            recommendations.append(
                f"Prioritize fixing {score.high_count} high-severity issues"
            )

        # Based on dimension scores
        weak_dimensions = [
            dim for dim, val in score.dimensions.items()
            if val < 70
        ]

        for dim in weak_dimensions[:3]:  # Top 3 weak areas
            recommendations.append(f"Focus on improving {dim} (score: {score.dimensions[dim]:.1f})")

        # Based on issues per line
        if score.issues_per_line > 0.1:
            recommendations.append(
                "High issue density - consider code review and refactoring"
            )

        if not recommendations:
            recommendations.append("Code quality is good - maintain current standards")

        return recommendations
