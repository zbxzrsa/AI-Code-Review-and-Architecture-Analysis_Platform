"""
SelfHealing_V3 - Legacy/Quarantine Bridge

Read-only access to deprecated self-healing implementations.
Used for comparison baseline and rollback scenarios.
"""

import warnings
from typing import Dict, Any, Optional
from datetime import datetime, timezone


# Deprecation warning on import
warnings.warn(
    "SelfHealing_V3 is deprecated. Use SelfHealing_V2 for production.",
    DeprecationWarning,
    stacklevel=2,
)


class LegacyHealthMonitor:
    """
    V3 Legacy Health Monitor - Read-Only.

    Preserved for baseline comparison with V2.
    """

    __deprecated__ = True

    def __init__(self):
        self._snapshots: Dict[str, Any] = {}
        self._comparison_data: Dict[str, Any] = {}

    def load_snapshot(self, snapshot_id: str, data: Dict[str, Any]):
        """Load historical snapshot for comparison."""
        self._snapshots[snapshot_id] = {
            "data": data,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }

    def compare_with_v2(self, v2_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare V2 result with V3 baseline."""
        if not self._snapshots:
            return {"error": "No V3 snapshot loaded"}

        latest_snapshot = list(self._snapshots.values())[-1]

        return {
            "v3_baseline": latest_snapshot["data"],
            "v2_current": v2_result,
            "comparison_time": datetime.now(timezone.utc).isoformat(),
            "improvements": self._calculate_improvements(latest_snapshot["data"], v2_result),
        }

    def _calculate_improvements(self, v3: Dict, v2: Dict) -> Dict[str, Any]:
        """Calculate improvements from V3 to V2."""
        improvements = {}

        # Compare latency if available
        v3_latency = v3.get("latency_ms", 0)
        v2_latency = v2.get("latency_ms", 0)

        if v3_latency and v2_latency:
            improvements["latency_improvement_pct"] = (
                (v3_latency - v2_latency) / v3_latency * 100
                if v3_latency > 0 else 0
            )

        return improvements

    def get_deprecation_info(self) -> Dict[str, Any]:
        """Get deprecation information."""
        return {
            "module": "SelfHealing_V3",
            "status": "deprecated",
            "replacement": "SelfHealing_V2",
            "reason": "V2 provides better SLO tracking and predictive capabilities",
            "migration_guide": "See INTEGRATION_GUIDE.md",
        }


class LegacyRecoveryManager:
    """
    V3 Legacy Recovery Manager - Read-Only.
    """

    __deprecated__ = True

    def __init__(self):
        self._historical_recoveries: list = []

    def load_historical_data(self, recoveries: list):
        """Load historical recovery data."""
        self._historical_recoveries = recoveries

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics from historical recoveries."""
        if not self._historical_recoveries:
            return {"error": "No historical data loaded"}

        successful = sum(1 for r in self._historical_recoveries if r.get("success"))
        total = len(self._historical_recoveries)

        return {
            "total_recoveries": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "status": "historical_baseline",
        }


# Factory functions (read-only)
def get_legacy_health_monitor() -> LegacyHealthMonitor:
    """Get legacy health monitor for comparison."""
    warnings.warn("Using deprecated V3 module", DeprecationWarning)
    return LegacyHealthMonitor()


def get_legacy_recovery_manager() -> LegacyRecoveryManager:
    """Get legacy recovery manager for comparison."""
    warnings.warn("Using deprecated V3 module", DeprecationWarning)
    return LegacyRecoveryManager()


__all__ = [
    "LegacyHealthMonitor",
    "LegacyRecoveryManager",
    "get_legacy_health_monitor",
    "get_legacy_recovery_manager",
]
