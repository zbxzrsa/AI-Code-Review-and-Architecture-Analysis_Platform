"""
V3 Legacy Module Tests

Tests for quarantine/legacy modules used for comparison baseline.
"""

import pytest
import warnings
import sys
from pathlib import Path

# Add modules to path
_modules_path = Path(__file__).parent.parent
if str(_modules_path) not in sys.path:
    sys.path.insert(0, str(_modules_path))


class TestSelfHealingV3Legacy:
    """Test SelfHealing V3 legacy bridge."""

    def test_deprecation_warning(self):
        """Test that deprecation warning is issued."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from SelfHealing_V3.src.legacy_bridge import get_legacy_health_monitor

            monitor = get_legacy_health_monitor()
            assert any("deprecated" in str(warning.message).lower() for warning in w)

    def test_load_snapshot(self):
        """Test loading historical snapshot."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from SelfHealing_V3.src.legacy_bridge import LegacyHealthMonitor

            monitor = LegacyHealthMonitor()
            monitor.load_snapshot("snap-1", {"latency_ms": 500, "status": "healthy"})

            assert "snap-1" in monitor._snapshots

    def test_compare_with_v2(self):
        """Test comparison with V2 results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from SelfHealing_V3.src.legacy_bridge import LegacyHealthMonitor

            monitor = LegacyHealthMonitor()
            monitor.load_snapshot("baseline", {"latency_ms": 500})

            comparison = monitor.compare_with_v2({"latency_ms": 250})

            assert "v3_baseline" in comparison
            assert "v2_current" in comparison
            assert comparison["improvements"]["latency_improvement_pct"] == 50.0

    def test_deprecation_info(self):
        """Test deprecation info."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from SelfHealing_V3.src.legacy_bridge import LegacyHealthMonitor

            monitor = LegacyHealthMonitor()
            info = monitor.get_deprecation_info()

            assert info["status"] == "deprecated"
            assert info["replacement"] == "SelfHealing_V2"


class TestMonitoringV3Legacy:
    """Test Monitoring V3 legacy bridge."""

    def test_load_historical_metrics(self):
        """Test loading historical metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Monitoring_V3.src.legacy_bridge import LegacyMetricsCollector

            collector = LegacyMetricsCollector()
            collector.load_historical_metrics("latency", [100, 150, 200])

            assert "latency" in collector._historical_metrics

    def test_compare_with_v2(self):
        """Test comparison with V2 metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Monitoring_V3.src.legacy_bridge import LegacyMetricsCollector

            collector = LegacyMetricsCollector()
            collector.load_historical_metrics("latency", [100, 100, 100])

            comparison = collector.compare_with_v2({"latency": 50})

            assert comparison["latency"]["v3_baseline_avg"] == 100
            assert comparison["latency"]["v2_current"] == 50
            assert comparison["latency"]["change_pct"] == -50.0


class TestCachingV3Legacy:
    """Test Caching V3 legacy bridge."""

    def test_compare_hit_rates(self):
        """Test hit rate comparison."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Caching_V3.src.legacy_bridge import LegacyCacheManager

            manager = LegacyCacheManager()
            manager.load_historical_stats({"hit_rate": 0.7})

            comparison = manager.compare_hit_rates(0.85)

            assert comparison["v3_hit_rate"] == 0.7
            assert comparison["v2_hit_rate"] == 0.85
            assert comparison["improvement_pct"] > 0


class TestAuthenticationV3Legacy:
    """Test Authentication V3 legacy bridge."""

    def test_compare_security_metrics(self):
        """Test security metrics comparison."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Authentication_V3.src.legacy_bridge import LegacyAuthManager

            manager = LegacyAuthManager()
            manager.load_historical_data({"basic_auth": True})

            comparison = manager.compare_security_metrics({
                "mfa_enabled": True,
                "oauth_provider_count": 3,
            })

            assert comparison["improvements"]["mfa_enabled"] == True
            assert comparison["improvements"]["oauth_providers"] == 3


class TestAIOrchestrationV3Legacy:
    """Test AIOrchestration V3 legacy bridge."""

    def test_compare_performance(self):
        """Test performance comparison."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from AIOrchestration_V3.src.legacy_bridge import LegacyOrchestrator

            orchestrator = LegacyOrchestrator()
            orchestrator.load_historical_requests([
                {"latency_ms": 1000},
                {"latency_ms": 1200},
                {"latency_ms": 800},
            ])

            comparison = orchestrator.compare_performance({
                "avg_latency_ms": 500,
            })

            assert comparison["v3_avg_latency_ms"] == 1000
            assert comparison["v2_avg_latency_ms"] == 500
            assert comparison["v2_features"]["circuit_breaker"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
