"""
Shadow Traffic Comparator

Compares V1 experimental output against V2 production baseline
to determine if V1 is ready for promotion.

This is the core evaluation component for the V1 → V2 transition:
- Collects paired V1/V2 outputs from shadow traffic
- Computes accuracy, latency, and cost deltas
- Runs statistical significance tests
- Provides promotion recommendation
"""

import asyncio
import logging
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import statistics
import math

logger = logging.getLogger(__name__)


@dataclass
class AnalysisOutput:
    """Output from a single analysis"""
    version: str
    version_id: str
    request_id: str
    code_hash: str
    language: str
    issues: List[Dict[str, Any]]
    latency_ms: float
    cost: float
    timestamp: datetime
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonPair:
    """A paired comparison between V1 and V2 outputs"""
    request_id: str
    code_hash: str
    v1_output: Optional[AnalysisOutput] = None
    v2_output: Optional[AnalysisOutput] = None
    
    @property
    def is_complete(self) -> bool:
        return self.v1_output is not None and self.v2_output is not None


@dataclass
class ComparisonMetrics:
    """Aggregated comparison metrics"""
    total_pairs: int = 0
    complete_pairs: int = 0
    
    # Issue detection comparison
    v1_total_issues: int = 0
    v2_total_issues: int = 0
    agreement_rate: float = 0.0
    
    # Accuracy metrics
    true_positive_agreement: int = 0
    false_positive_delta: int = 0
    false_negative_delta: int = 0
    accuracy_delta: float = 0.0
    
    # Latency metrics
    v1_avg_latency_ms: float = 0.0
    v2_avg_latency_ms: float = 0.0
    v1_p95_latency_ms: float = 0.0
    v2_p95_latency_ms: float = 0.0
    latency_improvement_pct: float = 0.0
    
    # Cost metrics
    v1_avg_cost: float = 0.0
    v2_avg_cost: float = 0.0
    cost_delta_pct: float = 0.0
    
    # Statistical tests
    latency_p_value: float = 1.0
    accuracy_p_value: float = 1.0
    is_statistically_significant: bool = False


@dataclass
class PromotionRecommendation:
    """Recommendation for V1 → V2 promotion"""
    recommend_promotion: bool
    confidence: float
    reasons: List[str]
    blockers: List[str]
    metrics: ComparisonMetrics
    next_evaluation_in_hours: Optional[int] = None


class ShadowComparator:
    """
    Compares shadow traffic outputs between V1 and V2.
    
    This is the decision engine for V1 → V2 promotion.
    """
    
    def __init__(
        self,
        min_pairs_for_evaluation: int = 1000,
        min_evaluation_hours: int = 24,
        accuracy_delta_threshold: float = 0.02,
        max_latency_increase_pct: float = 20.0,
        max_cost_increase_pct: float = 10.0,
        statistical_significance_p: float = 0.05,
    ):
        self.min_pairs = min_pairs_for_evaluation
        self.min_hours = min_evaluation_hours
        self.accuracy_threshold = accuracy_delta_threshold
        self.max_latency_increase = max_latency_increase_pct
        self.max_cost_increase = max_cost_increase_pct
        self.significance_p = statistical_significance_p
        
        # Storage for comparison pairs
        self.pending_pairs: Dict[str, ComparisonPair] = {}
        self.complete_pairs: List[ComparisonPair] = []
        
        # Evaluation state per version
        self.version_evaluations: Dict[str, datetime] = {}
        
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the comparator"""
        self._running = True
        # Store task reference to prevent garbage collection
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Shadow Comparator started")
    
    async def stop(self):
        """Stop the comparator"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                # Clean up task reference before re-raising
                self._cleanup_task = None
                # Re-raise to allow proper cancellation propagation
                raise
            finally:
                self._cleanup_task = None
    
    # ==================== Output Collection ====================
    
    def record_v1_output(self, output: AnalysisOutput):
        """Record V1 shadow traffic output"""
        pair_key = f"{output.code_hash}:{output.request_id}"
        
        if pair_key not in self.pending_pairs:
            self.pending_pairs[pair_key] = ComparisonPair(
                request_id=output.request_id,
                code_hash=output.code_hash
            )
        
        self.pending_pairs[pair_key].v1_output = output
        self._check_pair_complete(pair_key)
    
    def record_v2_output(self, output: AnalysisOutput):
        """Record V2 production output"""
        pair_key = f"{output.code_hash}:{output.request_id}"
        
        if pair_key not in self.pending_pairs:
            self.pending_pairs[pair_key] = ComparisonPair(
                request_id=output.request_id,
                code_hash=output.code_hash
            )
        
        self.pending_pairs[pair_key].v2_output = output
        self._check_pair_complete(pair_key)
    
    def _check_pair_complete(self, pair_key: str):
        """Check if pair is complete and move to complete list"""
        pair = self.pending_pairs.get(pair_key)
        if pair and pair.is_complete:
            self.complete_pairs.append(pair)
            del self.pending_pairs[pair_key]
            
            # Keep only last 10000 pairs
            if len(self.complete_pairs) > 10000:
                self.complete_pairs = self.complete_pairs[-10000:]
    
    # ==================== Comparison Logic ====================
    
    def compute_metrics(
        self,
        version_id: str,
        time_window_hours: int = 24
    ) -> ComparisonMetrics:
        """Compute comparison metrics for a version"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        
        # Filter pairs for this version and time window
        relevant_pairs = [
            p for p in self.complete_pairs
            if p.v1_output and p.v1_output.version_id == version_id
            and p.v1_output.timestamp >= cutoff
        ]
        
        if not relevant_pairs:
            return ComparisonMetrics()
        
        metrics = ComparisonMetrics(
            total_pairs=len(self.pending_pairs) + len(relevant_pairs),
            complete_pairs=len(relevant_pairs)
        )
        
        # Collect latencies and costs
        v1_latencies = []
        v2_latencies = []
        v1_costs = []
        v2_costs = []
        
        # Issue comparison
        agreements = 0
        
        for pair in relevant_pairs:
            v1 = pair.v1_output
            v2 = pair.v2_output
            
            # Latency
            v1_latencies.append(v1.latency_ms)
            v2_latencies.append(v2.latency_ms)
            
            # Cost
            v1_costs.append(v1.cost)
            v2_costs.append(v2.cost)
            
            # Issue comparison
            metrics.v1_total_issues += len(v1.issues)
            metrics.v2_total_issues += len(v2.issues)
            
            # Check agreement on critical issues
            v1_critical = {self._issue_key(i) for i in v1.issues if i.get("severity") in ["critical", "high"]}
            v2_critical = {self._issue_key(i) for i in v2.issues if i.get("severity") in ["critical", "high"]}
            
            if v1_critical == v2_critical:
                agreements += 1
            
            metrics.true_positive_agreement += len(v1_critical & v2_critical)
            metrics.false_positive_delta += len(v1_critical - v2_critical)
            metrics.false_negative_delta += len(v2_critical - v1_critical)
        
        # Compute averages
        if v1_latencies:
            metrics.v1_avg_latency_ms = statistics.mean(v1_latencies)
            metrics.v2_avg_latency_ms = statistics.mean(v2_latencies)
            metrics.v1_p95_latency_ms = self._percentile(v1_latencies, 95)
            metrics.v2_p95_latency_ms = self._percentile(v2_latencies, 95)
            
            # Latency improvement (negative = V1 is faster)
            if metrics.v2_avg_latency_ms > 0:
                metrics.latency_improvement_pct = (
                    (metrics.v2_avg_latency_ms - metrics.v1_avg_latency_ms) 
                    / metrics.v2_avg_latency_ms * 100
                )
        
        if v1_costs:
            metrics.v1_avg_cost = statistics.mean(v1_costs)
            metrics.v2_avg_cost = statistics.mean(v2_costs)
            
            if metrics.v2_avg_cost > 0:
                metrics.cost_delta_pct = (
                    (metrics.v1_avg_cost - metrics.v2_avg_cost)
                    / metrics.v2_avg_cost * 100
                )
        
        # Agreement rate
        if relevant_pairs:
            metrics.agreement_rate = agreements / len(relevant_pairs)
        
        # Accuracy delta (V1 finding more issues could be better or worse)
        # For now, use agreement + fewer false positives as proxy
        if metrics.v2_total_issues > 0:
            fp_rate_v1 = metrics.false_positive_delta / metrics.v2_total_issues
            fn_rate_v1 = metrics.false_negative_delta / metrics.v2_total_issues
            metrics.accuracy_delta = metrics.agreement_rate - fp_rate_v1 - fn_rate_v1
        
        # Statistical significance (simple t-test approximation)
        if len(v1_latencies) >= 30 and len(v2_latencies) >= 30:
            metrics.latency_p_value = self._simple_t_test(v1_latencies, v2_latencies)
            metrics.is_statistically_significant = metrics.latency_p_value < self.significance_p
        
        return metrics
    
    def _issue_key(self, issue: Dict[str, Any]) -> str:
        """Create unique key for an issue"""
        return f"{issue.get('type', '')}:{issue.get('pattern', '')}:{issue.get('line', '')}"
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    def _simple_t_test(self, a: List[float], b: List[float]) -> float:
        """Simple t-test p-value approximation"""
        if len(a) < 2 or len(b) < 2:
            return 1.0
        
        mean_a = statistics.mean(a)
        mean_b = statistics.mean(b)
        std_a = statistics.stdev(a)
        std_b = statistics.stdev(b)
        n_a = len(a)
        n_b = len(b)
        
        # Pooled standard error
        se = math.sqrt(std_a**2/n_a + std_b**2/n_b)
        
        if se == 0:
            return 1.0
        
        # t-statistic
        t = abs(mean_a - mean_b) / se
        
        # Approximate p-value (very rough)
        # For large samples, t > 2 roughly corresponds to p < 0.05
        if t > 3.5:
            return 0.001
        elif t > 2.5:
            return 0.01
        elif t > 2.0:
            return 0.05
        elif t > 1.5:
            return 0.15
        else:
            return 0.5
    
    # ==================== Promotion Decision ====================
    
    def _evaluate_accuracy(self, metrics, reasons: List[str], blockers: List[str]) -> float:
        """Evaluate accuracy metrics and return confidence delta."""
        if metrics.accuracy_delta >= self.accuracy_threshold:
            reasons.append(f"Accuracy improved by {metrics.accuracy_delta:.1%}")
            return 0.3
        if metrics.accuracy_delta >= 0:
            reasons.append(f"Accuracy maintained ({metrics.accuracy_delta:.1%})")
            return 0.2
        blockers.append(f"Accuracy regression: {metrics.accuracy_delta:.1%}")
        return 0.0
    
    def _evaluate_latency(self, metrics, reasons: List[str], blockers: List[str]) -> float:
        """Evaluate latency metrics and return confidence delta."""
        if metrics.latency_improvement_pct > 0:
            reasons.append(f"Latency improved by {metrics.latency_improvement_pct:.1f}%")
            return 0.2
        if metrics.latency_improvement_pct > -self.max_latency_increase:
            reasons.append(f"Latency acceptable ({metrics.latency_improvement_pct:.1f}%)")
            return 0.1
        blockers.append(f"Latency regression too high: {-metrics.latency_improvement_pct:.1f}%")
        return 0.0
    
    def _evaluate_cost(self, metrics, reasons: List[str], blockers: List[str]) -> float:
        """Evaluate cost metrics and return confidence delta."""
        if metrics.cost_delta_pct <= 0:
            reasons.append(f"Cost reduced by {-metrics.cost_delta_pct:.1f}%")
            return 0.2
        if metrics.cost_delta_pct <= self.max_cost_increase:
            reasons.append(f"Cost increase acceptable ({metrics.cost_delta_pct:.1f}%)")
            return 0.1
        blockers.append(f"Cost increase too high: {metrics.cost_delta_pct:.1f}%")
        return 0.0
    
    def _check_evaluation_duration(self, version_id: str, blockers: List[str]) -> None:
        """Check if minimum evaluation duration is met."""
        start_time = self.version_evaluations.get(version_id)
        if not start_time:
            return
        elapsed_hours = (datetime.now(timezone.utc) - start_time).total_seconds() / 3600
        if elapsed_hours < self.min_hours:
            blockers.append(f"Minimum evaluation time not met: {elapsed_hours:.1f}/{self.min_hours}h")
    
    def evaluate_promotion(
        self,
        version_id: str
    ) -> PromotionRecommendation:
        """
        Evaluate if V1 version should be promoted to V2.
        
        This is the main decision point for the V1 → V2 transition.
        """
        metrics = self.compute_metrics(version_id)
        reasons: List[str] = []
        blockers: List[str] = []
        
        # Check minimum data requirements
        if metrics.complete_pairs < self.min_pairs:
            blockers.append(f"Insufficient data: {metrics.complete_pairs}/{self.min_pairs} pairs")
            return PromotionRecommendation(
                recommend_promotion=False, confidence=0.0, reasons=reasons,
                blockers=blockers, metrics=metrics, next_evaluation_in_hours=6
            )
        
        # Run evaluations
        self._check_evaluation_duration(version_id, blockers)
        confidence = (
            self._evaluate_accuracy(metrics, reasons, blockers) +
            self._evaluate_latency(metrics, reasons, blockers) +
            self._evaluate_cost(metrics, reasons, blockers)
        )
        
        # Statistical significance
        if metrics.is_statistically_significant:
            reasons.append("Results statistically significant")
            confidence += 0.2
        elif metrics.complete_pairs >= self.min_pairs:
            blockers.append("Results not statistically significant")
        
        # Final decision
        recommend = len(blockers) == 0 and confidence >= 0.5
        
        return PromotionRecommendation(
            recommend_promotion=recommend,
            confidence=min(confidence, 1.0),
            reasons=reasons,
            blockers=blockers,
            metrics=metrics,
            next_evaluation_in_hours=6 if not recommend else None
        )
    
    def start_evaluation(self, version_id: str):
        """Start evaluation tracking for a version"""
        if version_id not in self.version_evaluations:
            self.version_evaluations[version_id] = datetime.now(timezone.utc)
            logger.info(f"Started evaluation tracking for {version_id}")
    
    # ==================== Maintenance ====================
    
    async def _cleanup_loop(self):
        """Cleanup old pending pairs"""
        while self._running:
            try:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
                
                # Remove old pending pairs
                to_remove = []
                for key, pair in self.pending_pairs.items():
                    if pair.v1_output and pair.v1_output.timestamp < cutoff:
                        to_remove.append(key)
                    elif pair.v2_output and pair.v2_output.timestamp < cutoff:
                        to_remove.append(key)
                
                for key in to_remove:
                    del self.pending_pairs[key]
                
                if to_remove:
                    logger.debug(f"Cleaned up {len(to_remove)} stale pending pairs")
                    
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            
            await asyncio.sleep(300)  # Every 5 minutes
    
    def get_status(self) -> Dict[str, Any]:
        """Get comparator status"""
        return {
            "pending_pairs": len(self.pending_pairs),
            "complete_pairs": len(self.complete_pairs),
            "versions_evaluating": list(self.version_evaluations.keys()),
            "config": {
                "min_pairs": self.min_pairs,
                "min_hours": self.min_hours,
                "accuracy_threshold": self.accuracy_threshold,
                "max_latency_increase_pct": self.max_latency_increase,
                "max_cost_increase_pct": self.max_cost_increase,
            }
        }

    def compute_detailed_metrics(
        self,
        version_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Compute detailed comparison metrics with breakdown.
        
        Args:
            version_id: Version ID to compute metrics for
            time_window_hours: Time window for filtering pairs
            
        Returns:
            Dictionary containing:
            - basic_metrics: Standard ComparisonMetrics
            - issue_breakdown: Issues categorized by severity and type
            - latency_distribution: Latency statistics (min, max, median, std)
            - hourly_trends: Hourly aggregated metrics
        """
        basic = self.compute_metrics(version_id, time_window_hours)
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        relevant_pairs = [
            p for p in self.complete_pairs
            if p.v1_output and p.v1_output.version_id == version_id
            and p.v1_output.timestamp >= cutoff
        ]
        
        if not relevant_pairs:
            return {
                "basic_metrics": basic.__dict__ if hasattr(basic, '__dict__') else {},
                "issue_breakdown": {},
                "latency_distribution": {},
                "hourly_trends": {}
            }
        
        # Issue breakdown by severity and type
        issue_by_severity: Dict[str, Dict[str, int]] = {}
        issue_by_type: Dict[str, Dict[str, int]] = {}
        
        for pair in relevant_pairs:
            for issue in (pair.v1_output.issues or []):
                sev = issue.get("severity", "unknown")
                typ = issue.get("type", "unknown")
                
                if sev not in issue_by_severity:
                    issue_by_severity[sev] = {"v1": 0, "v2": 0}
                issue_by_severity[sev]["v1"] += 1
                
                if typ not in issue_by_type:
                    issue_by_type[typ] = {"v1": 0, "v2": 0}
                issue_by_type[typ]["v1"] += 1
            
            for issue in (pair.v2_output.issues or []):
                sev = issue.get("severity", "unknown")
                typ = issue.get("type", "unknown")
                
                if sev not in issue_by_severity:
                    issue_by_severity[sev] = {"v1": 0, "v2": 0}
                issue_by_severity[sev]["v2"] += 1
                
                if typ not in issue_by_type:
                    issue_by_type[typ] = {"v1": 0, "v2": 0}
                issue_by_type[typ]["v2"] += 1
        
        # Latency distribution
        v1_latencies = [p.v1_output.latency_ms for p in relevant_pairs]
        v2_latencies = [p.v2_output.latency_ms for p in relevant_pairs]
        
        def calc_stats(data: List[float]) -> Dict[str, float]:
            if not data:
                return {"min": 0, "max": 0, "median": 0, "std": 0}
            return {
                "min": min(data),
                "max": max(data),
                "median": statistics.median(data),
                "std": statistics.stdev(data) if len(data) > 1 else 0,
            }
        
        latency_dist = {
            "v1": calc_stats(v1_latencies),
            "v2": calc_stats(v2_latencies),
        }
        
        # Hourly trends
        hourly: Dict[str, Dict[str, Any]] = {}
        for pair in relevant_pairs:
            hour_key = pair.v1_output.timestamp.strftime("%Y-%m-%d %H:00")
            if hour_key not in hourly:
                hourly[hour_key] = {
                    "count": 0,
                    "v1_latencies": [],
                    "v2_latencies": [],
                    "v1_issues": 0,
                    "v2_issues": 0,
                }
            hourly[hour_key]["count"] += 1
            hourly[hour_key]["v1_latencies"].append(pair.v1_output.latency_ms)
            hourly[hour_key]["v2_latencies"].append(pair.v2_output.latency_ms)
            hourly[hour_key]["v1_issues"] += len(pair.v1_output.issues or [])
            hourly[hour_key]["v2_issues"] += len(pair.v2_output.issues or [])
        
        # Compute hourly averages
        hourly_trends = {}
        for hour, data in hourly.items():
            hourly_trends[hour] = {
                "count": data["count"],
                "v1_avg_latency": statistics.mean(data["v1_latencies"]) if data["v1_latencies"] else 0,
                "v2_avg_latency": statistics.mean(data["v2_latencies"]) if data["v2_latencies"] else 0,
                "v1_avg_issues": data["v1_issues"] / data["count"] if data["count"] > 0 else 0,
                "v2_avg_issues": data["v2_issues"] / data["count"] if data["count"] > 0 else 0,
            }
        
        return {
            "basic_metrics": basic.__dict__ if hasattr(basic, '__dict__') else {},
            "issue_breakdown": {
                "by_severity": issue_by_severity,
                "by_type": issue_by_type,
            },
            "latency_distribution": latency_dist,
            "hourly_trends": hourly_trends,
        }

    def get_promotion_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of promotion recommendations.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of historical promotion recommendations
        """
        history = []
        for version_id, start_time in self.version_evaluations.items():
            recommendation = self.evaluate_promotion(version_id)
            history.append({
                "version_id": version_id,
                "evaluation_start": start_time.isoformat(),
                "recommendation": {
                    "recommend_promotion": recommendation.recommend_promotion,
                    "confidence": recommendation.confidence,
                    "reasons": recommendation.reasons,
                    "blockers": recommendation.blockers,
                },
                "metrics_summary": {
                    "complete_pairs": recommendation.metrics.complete_pairs,
                    "accuracy_delta": recommendation.metrics.accuracy_delta,
                    "latency_improvement_pct": recommendation.metrics.latency_improvement_pct,
                    "cost_delta_pct": recommendation.metrics.cost_delta_pct,
                }
            })
        return history[:limit]
