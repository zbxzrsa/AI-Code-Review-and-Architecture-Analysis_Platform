"""
CodeReviewAI_V3 - Comparison Engine

Compares results across V1, V2, V3 for analysis and decisions.
"""

import uuid
import logging
import statistics
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

from .models import ReviewResult, ComparisonResult

logger = logging.getLogger(__name__)


class ComparisonEngine:
    """
    Compares review results across versions.

    Used for:
    - A/B testing analysis
    - Version promotion decisions
    - Re-evaluation assessment
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        min_improvement: float = 5.0,
    ):
        """
        Initialize comparison engine.

        Args:
            significance_threshold: P-value threshold for significance
            min_improvement: Minimum score improvement for promotion
        """
        self.significance_threshold = significance_threshold
        self.min_improvement = min_improvement
        self._comparison_history: List[ComparisonResult] = []

    async def compare(
        self,
        baseline_result: ReviewResult,
        compare_result: ReviewResult,
    ) -> ComparisonResult:
        """
        Compare two review results.

        Args:
            baseline_result: V3 baseline result
            compare_result: V1 or V2 result to compare

        Returns:
            ComparisonResult with analysis
        """
        comparison_id = str(uuid.uuid4())

        # Calculate deltas
        score_delta = compare_result.overall_score - baseline_result.overall_score
        findings_delta = len(compare_result.findings) - len(baseline_result.findings)
        latency_delta = compare_result.processing_time_ms - baseline_result.processing_time_ms

        # Statistical analysis (simplified)
        is_significant, confidence_interval, p_value = self._statistical_test(
            baseline_result.overall_score,
            compare_result.overall_score,
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            score_delta,
            findings_delta,
            latency_delta,
            is_significant,
        )

        result = ComparisonResult(
            comparison_id=comparison_id,
            baseline_version=baseline_result.model_version,
            compare_version=compare_result.model_version,
            score_delta=score_delta,
            findings_delta=findings_delta,
            latency_delta_ms=latency_delta,
            is_significant=is_significant,
            confidence_interval=confidence_interval,
            p_value=p_value,
            recommendation=recommendation,
        )

        self._comparison_history.append(result)

        logger.info(
            f"Comparison {comparison_id}: "
            f"Δscore={score_delta:+.1f}, Δfindings={findings_delta:+d}, "
            f"recommendation={recommendation}"
        )

        return result

    def _statistical_test(
        self,
        baseline_score: float,
        compare_score: float,
    ) -> tuple[bool, tuple, float]:
        """
        Simplified statistical test.

        In production, use proper statistical tests like t-test.
        """
        # Simplified: just check if difference is meaningful
        delta = abs(compare_score - baseline_score)

        # Mock p-value based on delta
        if delta >= 10:
            p_value = 0.01
        elif delta >= 5:
            p_value = 0.05
        else:
            p_value = 0.1

        is_significant = p_value <= self.significance_threshold

        # Confidence interval (simplified)
        margin = 5.0  # ±5 points
        ci_lower = compare_score - baseline_score - margin
        ci_upper = compare_score - baseline_score + margin

        return is_significant, (ci_lower, ci_upper), p_value

    def _generate_recommendation(
        self,
        score_delta: float,
        findings_delta: int,
        latency_delta: float,
        is_significant: bool,
    ) -> str:
        """Generate promotion/quarantine recommendation"""

        # Significant improvement
        if is_significant and score_delta >= self.min_improvement:
            if latency_delta <= 500:  # Not significantly slower
                return "promote"
            else:
                return "keep"  # Good but slow

        # Significant regression
        if is_significant and score_delta <= -self.min_improvement:
            return "quarantine"

        # Not significant - keep current
        return "keep"

    async def batch_compare(
        self,
        baseline_results: List[ReviewResult],
        compare_results: List[ReviewResult],
    ) -> Dict[str, Any]:
        """
        Compare batches of results.

        Args:
            baseline_results: List of V3 baseline results
            compare_results: List of results to compare

        Returns:
            Aggregated comparison statistics
        """
        if len(baseline_results) != len(compare_results):
            raise ValueError("Result lists must have same length")

        comparisons = []
        for base, compare in zip(baseline_results, compare_results):
            result = await self.compare(base, compare)
            comparisons.append(result)

        # Aggregate statistics
        score_deltas = [c.score_delta for c in comparisons]
        significant_count = sum(1 for c in comparisons if c.is_significant)

        recommendations = {}
        for c in comparisons:
            recommendations[c.recommendation] = recommendations.get(c.recommendation, 0) + 1

        return {
            "total_comparisons": len(comparisons),
            "avg_score_delta": statistics.mean(score_deltas) if score_deltas else 0,
            "std_score_delta": statistics.stdev(score_deltas) if len(score_deltas) > 1 else 0,
            "significant_count": significant_count,
            "significance_rate": significant_count / len(comparisons) if comparisons else 0,
            "recommendations": recommendations,
            "overall_recommendation": max(recommendations, key=recommendations.get) if recommendations else "keep",
        }

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get comparison history"""
        recent = self._comparison_history[-limit:]
        return [c.to_dict() for c in recent]

    def clear_history(self):
        """Clear comparison history"""
        self._comparison_history.clear()
