"""
Caching_V3 - Legacy/Quarantine Bridge

Read-only access to deprecated caching implementations.
"""

import warnings
from typing import Dict, Any


warnings.warn(
    "Caching_V3 is deprecated. Use Caching_V2 for production.",
    DeprecationWarning,
    stacklevel=2,
)


class LegacyCacheManager:
    """V3 Legacy Cache Manager - Read-Only."""

    __deprecated__ = True

    def __init__(self):
        self._historical_stats: Dict[str, Any] = {}

    def load_historical_stats(self, stats: Dict[str, Any]):
        """Load historical cache statistics."""
        self._historical_stats = stats

    def compare_hit_rates(self, v2_hit_rate: float) -> Dict[str, Any]:
        """Compare V2 hit rate with V3 baseline."""
        v3_hit_rate = self._historical_stats.get("hit_rate", 0)

        return {
            "v3_hit_rate": v3_hit_rate,
            "v2_hit_rate": v2_hit_rate,
            "improvement_pct": ((v2_hit_rate - v3_hit_rate) / v3_hit_rate * 100) if v3_hit_rate else 0,
        }

    def get_deprecation_info(self) -> Dict[str, Any]:
        return {
            "module": "Caching_V3",
            "status": "deprecated",
            "replacement": "Caching_V2",
            "reason": "V2 provides SLO-aware TTL and cache warming",
        }


def get_legacy_cache_manager() -> LegacyCacheManager:
    warnings.warn("Using deprecated V3 module", DeprecationWarning)
    return LegacyCacheManager()


__all__ = ["LegacyCacheManager", "get_legacy_cache_manager"]
