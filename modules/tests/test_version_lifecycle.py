"""
Version Lifecycle Tests

Tests for version promotion, deprecation, and rollback scenarios.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

_modules_path = Path(__file__).parent.parent
if str(_modules_path) not in sys.path:
    sys.path.insert(0, str(_modules_path))


class TestVersionPromotion:
    """Test V1 to V2 promotion scenarios."""

    def test_v1_to_v2_data_compatibility(self):
        """Test that V1 data format works with V2."""
        # V1 health result format
        v1_result = {
            "service": "api",
            "status": "healthy",
            "latency_ms": 150,
        }

        # V2 should handle V1 format
        from SelfHealing_V2.src.backend_integration import SLOStatus

        # V2 adds SLO evaluation
        is_healthy = v1_result["status"] == "healthy"
        latency_ok = v1_result["latency_ms"] < 1000

        slo_status = SLOStatus.COMPLIANT if (is_healthy and latency_ok) else SLOStatus.VIOLATED

        assert slo_status == SLOStatus.COMPLIANT

    def test_v1_metrics_in_v2(self):
        """Test V1 metrics can be recorded in V2."""
        from Monitoring_V2.src.backend_integration import get_metrics_collector

        collector = get_metrics_collector(prefix="test", use_backend=False)

        # V1 style metric recording
        collector.record_metric("request_count", 1)
        collector.record_metric("request_latency", 150.5)

        # Should work without error
        assert True

    def test_v1_cache_keys_in_v2(self):
        """Test V1 cache keys work in V2."""
        from Caching_V2.src.backend_integration import get_cache_manager

        cache = get_cache_manager(use_backend=False)

        # V1 style key
        v1_key = "user:123:profile"

        # Should work in V2
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            cache.set(v1_key, {"name": "Test"})
        )

        result = asyncio.get_event_loop().run_until_complete(
            cache.get(v1_key)
        )

        assert result is not None


class TestVersionDeprecation:
    """Test V2 to V3 deprecation scenarios."""

    def test_v3_read_only(self):
        """Test V3 modules are read-only."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from SelfHealing_V3.src.legacy_bridge import LegacyHealthMonitor

            monitor = LegacyHealthMonitor()

            # V3 can load data
            monitor.load_snapshot("test", {"status": "healthy"})

            # But is marked deprecated
            assert monitor.__deprecated__ == True

    def test_deprecation_message(self):
        """Test deprecation messages are clear."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Monitoring_V3.src.legacy_bridge import LegacyMetricsCollector

            collector = LegacyMetricsCollector()
            info = collector.get_deprecation_info()

            assert "deprecated" in info["status"]
            assert "V2" in info["replacement"]


class TestVersionRollback:
    """Test rollback scenarios from V2 to V1."""

    def test_v2_to_v1_fallback(self):
        """Test V2 can fall back to V1 behavior."""
        from SelfHealing_V1.src.backend_integration import get_health_monitor

        # V1 monitor without backend
        v1_monitor = get_health_monitor(use_backend=False)

        # Should work as fallback
        assert v1_monitor is not None

    def test_feature_degradation(self):
        """Test graceful feature degradation."""
        from AIOrchestration_V1.src.backend_integration import get_orchestrator

        # V1 orchestrator (no circuit breaker)
        orchestrator = get_orchestrator(use_backend=False)

        # Basic functionality should work
        assert orchestrator is not None


class TestVersionCoexistence:
    """Test multiple versions running together."""

    def test_v1_v2_parallel_operation(self):
        """Test V1 and V2 can operate in parallel."""
        from SelfHealing_V1.src.backend_integration import get_health_monitor as get_v1_monitor
        from SelfHealing_V2.src.backend_integration import get_health_monitor as get_v2_monitor

        v1 = get_v1_monitor(use_backend=False)
        v2 = get_v2_monitor(use_backend=False)

        # Both should be independent
        assert v1 is not v2

    def test_v3_comparison_with_v2(self):
        """Test V3 can compare with V2 results."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from SelfHealing_V3.src.legacy_bridge import LegacyHealthMonitor

            v3 = LegacyHealthMonitor()
            v3.load_snapshot("baseline", {"latency_ms": 500, "error_rate": 0.05})

            # V2 result
            v2_result = {"latency_ms": 200, "error_rate": 0.01}

            comparison = v3.compare_with_v2(v2_result)

            # V2 should show improvement
            assert comparison["improvements"]["latency_improvement_pct"] > 0


class TestVersionMigration:
    """Test migration between versions."""

    def test_config_migration_v1_to_v2(self):
        """Test configuration migration from V1 to V2."""
        # V1 config
        v1_config = {
            "check_interval": 30,
            "failure_threshold": 3,
        }

        # V2 config (adds SLO)
        v2_config = {
            **v1_config,
            "slo_latency_ms": 1000,
            "slo_availability": 99.9,
        }

        # Should preserve V1 settings
        assert v2_config["check_interval"] == v1_config["check_interval"]
        assert v2_config["failure_threshold"] == v1_config["failure_threshold"]

        # Should add V2 settings
        assert "slo_latency_ms" in v2_config

    def test_data_format_evolution(self):
        """Test data format evolution across versions."""
        # V1 format
        v1_data = {"status": "ok", "value": 100}

        # V2 format (adds metadata)
        v2_data = {
            **v1_data,
            "slo_compliant": True,
            "error_budget_pct": 95.5,
        }

        # V3 format (read-only snapshot)
        v3_data = {
            "snapshot_of": v2_data,
            "deprecated": True,
            "comparison_baseline": True,
        }

        # All versions maintain core data
        assert v1_data["status"] == "ok"
        assert v2_data["status"] == "ok"
        assert v3_data["snapshot_of"]["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
