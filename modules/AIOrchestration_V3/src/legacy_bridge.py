"""
AIOrchestration_V3 - Legacy/Quarantine Bridge

Read-only access to deprecated orchestration implementations.
"""

import warnings
from typing import Dict, Any, List


warnings.warn(
    "AIOrchestration_V3 is deprecated. Use AIOrchestration_V2 for production.",
    DeprecationWarning,
    stacklevel=2,
)


class LegacyOrchestrator:
    """V3 Legacy Orchestrator - Read-Only."""

    __deprecated__ = True

    def __init__(self):
        self._historical_requests: List[Dict] = []

    def load_historical_requests(self, requests: List[Dict]):
        """Load historical request data."""
        self._historical_requests = requests

    def compare_performance(self, v2_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Compare V2 performance with V3 baseline."""
        if not self._historical_requests:
            return {"error": "No V3 data loaded"}

        v3_latencies = [r.get("latency_ms", 0) for r in self._historical_requests]
        v3_avg = sum(v3_latencies) / len(v3_latencies) if v3_latencies else 0

        return {
            "v3_avg_latency_ms": v3_avg,
            "v2_avg_latency_ms": v2_stats.get("avg_latency_ms", 0),
            "v2_features": {
                "circuit_breaker": True,
                "load_balancing": True,
                "slo_enforcement": True,
            },
        }

    def get_deprecation_info(self) -> Dict[str, Any]:
        return {
            "module": "AIOrchestration_V3",
            "status": "deprecated",
            "replacement": "AIOrchestration_V2",
            "reason": "V2 provides circuit breaker, load balancing, and SLO enforcement",
        }


def get_legacy_orchestrator() -> LegacyOrchestrator:
    warnings.warn("Using deprecated V3 module", DeprecationWarning)
    return LegacyOrchestrator()


__all__ = ["LegacyOrchestrator", "get_legacy_orchestrator"]
