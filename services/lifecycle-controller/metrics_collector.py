"""
Metrics Collector for Lifecycle Controller

Collects and exposes Prometheus metrics for the self-evolution cycle.
Enables monitoring of cycle health, promotion rates, and recovery statistics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricValue:
    """A single metric value"""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    metric_type: MetricType


class CycleMetricsCollector:
    """
    Collects metrics for the self-evolution cycle.
    
    Exposes metrics in Prometheus format for monitoring.
    """
    
    def __init__(self):
        # Counters
        self._promotions_total = 0
        self._demotions_total = 0
        self._recoveries_total = 0
        self._recovery_attempts_total = 0
        self._evaluations_total = 0
        self._rollbacks_total = 0
        
        # Gauges (current state)
        self._versions_by_state: Dict[str, int] = {
            "experiment": 0,
            "shadow": 0,
            "gray_1_percent": 0,
            "gray_5_percent": 0,
            "gray_25_percent": 0,
            "gray_50_percent": 0,
            "stable": 0,
            "quarantine": 0,
            "re_evaluation": 0,
        }
        
        # Histograms (timing)
        self._promotion_durations: List[float] = []
        self._recovery_durations: List[float] = []
        self._evaluation_durations: List[float] = []
        
        # Success rates
        self._promotion_successes = 0
        self._promotion_failures = 0
        self._recovery_successes = 0
        self._recovery_failures = 0
        
        # Shadow comparison metrics
        self._shadow_pairs_complete = 0
        self._shadow_pairs_pending = 0
        self._shadow_accuracy_deltas: List[float] = []
        self._shadow_latency_improvements: List[float] = []
    
    # ==================== Counter Updates ====================
    
    def record_promotion(self, success: bool, duration_hours: float = 0):
        """Record a promotion attempt"""
        self._promotions_total += 1
        if success:
            self._promotion_successes += 1
        else:
            self._promotion_failures += 1
        
        if duration_hours > 0:
            self._promotion_durations.append(duration_hours)
    
    def record_demotion(self, reason: str):
        """Record a demotion to V3"""
        self._demotions_total += 1
        logger.info(f"Recorded demotion: {reason}")
    
    def record_recovery(self, success: bool, attempts: int, duration_hours: float = 0):
        """Record a recovery attempt"""
        self._recovery_attempts_total += attempts
        if success:
            self._recoveries_total += 1
            self._recovery_successes += 1
        else:
            self._recovery_failures += 1
        
        if duration_hours > 0:
            self._recovery_durations.append(duration_hours)
    
    def record_evaluation(self, duration_seconds: float):
        """Record an evaluation"""
        self._evaluations_total += 1
        self._evaluation_durations.append(duration_seconds)
    
    def record_rollback(self, from_phase: str, reason: str):
        """Record a rollback"""
        self._rollbacks_total += 1
        logger.info(f"Recorded rollback from {from_phase}: {reason}")
    
    # ==================== Gauge Updates ====================
    
    def update_version_counts(self, counts: Dict[str, int]):
        """Update current version counts by state"""
        self._versions_by_state.update(counts)
    
    def update_shadow_metrics(
        self,
        pairs_complete: int,
        pairs_pending: int,
        accuracy_delta: Optional[float] = None,
        latency_improvement: Optional[float] = None
    ):
        """Update shadow comparison metrics"""
        self._shadow_pairs_complete = pairs_complete
        self._shadow_pairs_pending = pairs_pending
        
        if accuracy_delta is not None:
            self._shadow_accuracy_deltas.append(accuracy_delta)
            # Keep last 100
            if len(self._shadow_accuracy_deltas) > 100:
                self._shadow_accuracy_deltas = self._shadow_accuracy_deltas[-100:]
        
        if latency_improvement is not None:
            self._shadow_latency_improvements.append(latency_improvement)
            if len(self._shadow_latency_improvements) > 100:
                self._shadow_latency_improvements = self._shadow_latency_improvements[-100:]
    
    # ==================== Metric Export ====================
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Counters
        lines.append("# HELP lifecycle_promotions_total Total number of promotion attempts")
        lines.append("# TYPE lifecycle_promotions_total counter")
        lines.append(f"lifecycle_promotions_total {self._promotions_total}")
        
        lines.append("# HELP lifecycle_demotions_total Total number of demotions to V3")
        lines.append("# TYPE lifecycle_demotions_total counter")
        lines.append(f"lifecycle_demotions_total {self._demotions_total}")
        
        lines.append("# HELP lifecycle_recoveries_total Total successful recoveries from V3")
        lines.append("# TYPE lifecycle_recoveries_total counter")
        lines.append(f"lifecycle_recoveries_total {self._recoveries_total}")
        
        lines.append("# HELP lifecycle_recovery_attempts_total Total recovery attempts")
        lines.append("# TYPE lifecycle_recovery_attempts_total counter")
        lines.append(f"lifecycle_recovery_attempts_total {self._recovery_attempts_total}")
        
        lines.append("# HELP lifecycle_evaluations_total Total evaluations performed")
        lines.append("# TYPE lifecycle_evaluations_total counter")
        lines.append(f"lifecycle_evaluations_total {self._evaluations_total}")
        
        lines.append("# HELP lifecycle_rollbacks_total Total rollbacks")
        lines.append("# TYPE lifecycle_rollbacks_total counter")
        lines.append(f"lifecycle_rollbacks_total {self._rollbacks_total}")
        
        # Gauges - version counts
        lines.append("# HELP lifecycle_versions_total Current versions by state")
        lines.append("# TYPE lifecycle_versions_total gauge")
        for state, count in self._versions_by_state.items():
            lines.append(f'lifecycle_versions_total{{state="{state}"}} {count}')
        
        # Gauges - shadow metrics
        lines.append("# HELP lifecycle_shadow_pairs_complete Complete shadow comparison pairs")
        lines.append("# TYPE lifecycle_shadow_pairs_complete gauge")
        lines.append(f"lifecycle_shadow_pairs_complete {self._shadow_pairs_complete}")
        
        lines.append("# HELP lifecycle_shadow_pairs_pending Pending shadow comparison pairs")
        lines.append("# TYPE lifecycle_shadow_pairs_pending gauge")
        lines.append(f"lifecycle_shadow_pairs_pending {self._shadow_pairs_pending}")
        
        # Success rates
        lines.append("# HELP lifecycle_promotion_success_rate Promotion success rate")
        lines.append("# TYPE lifecycle_promotion_success_rate gauge")
        total_promo = self._promotion_successes + self._promotion_failures
        rate = self._promotion_successes / total_promo if total_promo > 0 else 0
        lines.append(f"lifecycle_promotion_success_rate {rate}")
        
        lines.append("# HELP lifecycle_recovery_success_rate Recovery success rate")
        lines.append("# TYPE lifecycle_recovery_success_rate gauge")
        total_recov = self._recovery_successes + self._recovery_failures
        rate = self._recovery_successes / total_recov if total_recov > 0 else 0
        lines.append(f"lifecycle_recovery_success_rate {rate}")
        
        # Averages
        if self._shadow_accuracy_deltas:
            avg_delta = sum(self._shadow_accuracy_deltas) / len(self._shadow_accuracy_deltas)
            lines.append("# HELP lifecycle_shadow_accuracy_delta_avg Average accuracy delta")
            lines.append("# TYPE lifecycle_shadow_accuracy_delta_avg gauge")
            lines.append(f"lifecycle_shadow_accuracy_delta_avg {avg_delta}")
        
        if self._shadow_latency_improvements:
            avg_improvement = sum(self._shadow_latency_improvements) / len(self._shadow_latency_improvements)
            lines.append("# HELP lifecycle_shadow_latency_improvement_avg Average latency improvement %")
            lines.append("# TYPE lifecycle_shadow_latency_improvement_avg gauge")
            lines.append(f"lifecycle_shadow_latency_improvement_avg {avg_improvement}")
        
        return "\n".join(lines)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary"""
        return {
            "counters": {
                "promotions_total": self._promotions_total,
                "demotions_total": self._demotions_total,
                "recoveries_total": self._recoveries_total,
                "recovery_attempts_total": self._recovery_attempts_total,
                "evaluations_total": self._evaluations_total,
                "rollbacks_total": self._rollbacks_total,
            },
            "gauges": {
                "versions_by_state": self._versions_by_state,
                "shadow_pairs_complete": self._shadow_pairs_complete,
                "shadow_pairs_pending": self._shadow_pairs_pending,
            },
            "rates": {
                "promotion_success_rate": (
                    self._promotion_successes / (self._promotion_successes + self._promotion_failures)
                    if (self._promotion_successes + self._promotion_failures) > 0 else 0
                ),
                "recovery_success_rate": (
                    self._recovery_successes / (self._recovery_successes + self._recovery_failures)
                    if (self._recovery_successes + self._recovery_failures) > 0 else 0
                ),
            },
            "averages": {
                "promotion_duration_hours": (
                    sum(self._promotion_durations) / len(self._promotion_durations)
                    if self._promotion_durations else 0
                ),
                "recovery_duration_hours": (
                    sum(self._recovery_durations) / len(self._recovery_durations)
                    if self._recovery_durations else 0
                ),
                "shadow_accuracy_delta": (
                    sum(self._shadow_accuracy_deltas) / len(self._shadow_accuracy_deltas)
                    if self._shadow_accuracy_deltas else 0
                ),
            }
        }
    
    def reset_counters(self):
        """Reset all counters (for testing)"""
        self._promotions_total = 0
        self._demotions_total = 0
        self._recoveries_total = 0
        self._recovery_attempts_total = 0
        self._evaluations_total = 0
        self._rollbacks_total = 0
        self._promotion_successes = 0
        self._promotion_failures = 0
        self._recovery_successes = 0
        self._recovery_failures = 0
