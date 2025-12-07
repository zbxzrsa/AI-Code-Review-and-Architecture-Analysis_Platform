"""
Monitoring_V3 - Legacy/Quarantine Bridge

Read-only access to deprecated monitoring implementations.
"""

import warnings
from typing import Dict, Any, List
from datetime import datetime, timezone


warnings.warn(
    "Monitoring_V3 is deprecated. Use Monitoring_V2 for production.",
    DeprecationWarning,
    stacklevel=2,
)


class LegacyMetricsCollector:
    """V3 Legacy Metrics Collector - Read-Only."""

    __deprecated__ = True

    def __init__(self):
        self._historical_metrics: Dict[str, List[float]] = {}

    def load_historical_metrics(self, metric_name: str, values: List[float]):
        """Load historical metrics for comparison."""
        self._historical_metrics[metric_name] = values

    def compare_with_v2(self, v2_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare V2 metrics with V3 baseline."""
        comparison = {}

        for name, v2_value in v2_metrics.items():
            if name in self._historical_metrics:
                v3_values = self._historical_metrics[name]
                v3_avg = sum(v3_values) / len(v3_values) if v3_values else 0

                comparison[name] = {
                    "v3_baseline_avg": v3_avg,
                    "v2_current": v2_value,
                    "change_pct": ((v2_value - v3_avg) / v3_avg * 100) if v3_avg else 0,
                }

        return comparison

    def get_deprecation_info(self) -> Dict[str, Any]:
        """Get deprecation information."""
        return {
            "module": "Monitoring_V3",
            "status": "deprecated",
            "replacement": "Monitoring_V2",
            "reason": "V2 provides SLO tracking and distributed tracing",
        }


def get_legacy_metrics_collector() -> LegacyMetricsCollector:
    """Get legacy metrics collector."""
    warnings.warn("Using deprecated V3 module", DeprecationWarning)
    return LegacyMetricsCollector()


__all__ = ["LegacyMetricsCollector", "get_legacy_metrics_collector"]
