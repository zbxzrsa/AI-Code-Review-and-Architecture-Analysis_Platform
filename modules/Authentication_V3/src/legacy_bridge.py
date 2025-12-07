"""
Authentication_V3 - Legacy/Quarantine Bridge

Read-only access to deprecated authentication implementations.
"""

import warnings
from typing import Dict, Any


warnings.warn(
    "Authentication_V3 is deprecated. Use Authentication_V2 for production.",
    DeprecationWarning,
    stacklevel=2,
)


class LegacyAuthManager:
    """V3 Legacy Auth Manager - Read-Only."""

    __deprecated__ = True

    def __init__(self):
        self._historical_data: Dict[str, Any] = {}

    def load_historical_data(self, data: Dict[str, Any]):
        """Load historical auth data."""
        self._historical_data = data

    def compare_security_metrics(self, v2_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare V2 security metrics with V3 baseline."""
        return {
            "v3_baseline": self._historical_data,
            "v2_current": v2_metrics,
            "improvements": {
                "mfa_enabled": v2_metrics.get("mfa_enabled", False),
                "oauth_providers": v2_metrics.get("oauth_provider_count", 0),
            },
        }

    def get_deprecation_info(self) -> Dict[str, Any]:
        return {
            "module": "Authentication_V3",
            "status": "deprecated",
            "replacement": "Authentication_V2",
            "reason": "V2 provides MFA, OAuth, and enhanced security",
        }


def get_legacy_auth_manager() -> LegacyAuthManager:
    warnings.warn("Using deprecated V3 module", DeprecationWarning)
    return LegacyAuthManager()


__all__ = ["LegacyAuthManager", "get_legacy_auth_manager"]
