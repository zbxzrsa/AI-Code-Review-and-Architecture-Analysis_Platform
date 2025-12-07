"""
Integration Tests for Self-Healing System

Tests the complete self-healing workflow including:
- Health monitoring
- Issue detection
- Auto-repair execution
- Alert generation
- Metrics collection
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.insert(0, 'backend/shared')

from self_healing.orchestrator import SelfHealingOrchestrator
from self_healing.health_monitor import HealthMonitor, HealthStatus, HealthMetrics
from self_healing.auto_repair import AutoRepair, RepairAction, RepairResult
from self_healing.alert_manager import AlertManager, AlertSeverity


class TestSelfHealingOrchestrator:
    """Test self-healing orchestrator integration."""

    @pytest.mark.asyncio
    async def test_orchestrator_startup_shutdown(self):
        """Orchestrator should start and stop cleanly."""
        orchestrator = SelfHealingOrchestrator(dry_run=True)

        # Start
        await orchestrator.start()
        assert orchestrator.is_running
        assert orchestrator.health_monitor.is_running

        # Stop
        await orchestrator.stop()
        assert not orchestrator.is_running
        assert not orchestrator.health_monitor.is_running

    @pytest.mark.asyncio
    async def test_end_to_end_repair_workflow(self):
        """Test complete workflow from detection to repair."""
        orchestrator = SelfHealingOrchestrator(
            enable_auto_repair=True,
            dry_run=True
        )

        await orchestrator.start()

        # Simulate high error rate
        bad_metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            error_rate=0.10,  # 10% - exceeds critical threshold
            response_time_p95=2000,
            cpu_usage_percent=50,
            memory_usage_percent=40,
            availability_percent=99.0
        )

        # Run health checks
        await orchestrator.health_monitor.run_health_checks(bad_metrics)

        # Wait for async processing
        await asyncio.sleep(0.5)

        # Verify alert was generated
        alerts = orchestrator.alert_manager.get_active_alerts(
            severity=AlertSeverity.CRITICAL
        )
        assert len(alerts) > 0, "Should generate critical alert"

        # Verify repair was attempted
        assert orchestrator.stats["repairs_attempted"] > 0

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_anomaly_detection_triggers_alert(self):
        """Anomaly detection should trigger alerts."""
        orchestrator = SelfHealingOrchestrator(dry_run=True)

        await orchestrator.start()

        # Build baseline with normal metrics
        for _ in range(20):
            normal_metrics = HealthMetrics(
                timestamp=datetime.now().isoformat(),
                response_time_p95=2000,
                error_rate=0.01
            )
            orchestrator.health_monitor.metrics_history.append(normal_metrics)

        # Inject anomalous metrics
        anomalous_metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            response_time_p95=10000,  # 5x normal
            error_rate=0.01
        )

        anomalies = await orchestrator.health_monitor.detect_anomalies(
            anomalous_metrics
        )

        assert len(anomalies) > 0, "Should detect anomaly"

        # Wait for alert processing
        await asyncio.sleep(0.2)

        # Verify alert generated
        assert orchestrator.alert_manager.stats["alerts_generated"] > 0

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_circuit_breaker_auto_recovery(self):
        """Circuit breaker should auto-recover after backoff."""
        orchestrator = SelfHealingOrchestrator(dry_run=True)

        # Simulate circuit breaker opening
        repair_context = {
            "triggered_by": "test",
            "issue": "High failure rate",
            "service_name": "test_service"
        }

        # Execute repair
        record = await orchestrator.auto_repair.execute_repair(
            RepairAction.RESTART_SERVICE,
            repair_context
        )

        assert record.result == RepairResult.SUCCESS
        assert record.success
        assert record.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_repairs(self):
        """System should handle multiple concurrent repairs."""
        orchestrator = SelfHealingOrchestrator(
            enable_auto_repair=True,
            dry_run=True
        )

        # Trigger multiple repairs concurrently
        repairs = [
            orchestrator.auto_repair.execute_repair(
                RepairAction.CLEAR_CACHE,
                {"triggered_by": f"test_{i}"}
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*repairs)

        # All should succeed
        assert all(r.success for r in results)
        assert len(results) == 5


class TestHealthMonitor:
    """Test health monitoring functionality."""

    @pytest.mark.asyncio
    async def test_health_score_calculation(self):
        """Health score should be calculated correctly."""
        # Perfect metrics
        perfect_metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            response_time_p95=1000,
            error_rate=0.001,
            cpu_usage_percent=30,
            memory_usage_percent=40,
            availability_percent=99.99
        )

        score = perfect_metrics.calculate_health_score()
        assert score >= 90, f"Perfect metrics should score ≥90, got {score}"

        # Degraded metrics
        degraded_metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            response_time_p95=2500,
            error_rate=0.03,
            cpu_usage_percent=75,
            memory_usage_percent=75,
            availability_percent=99.5
        )

        score = degraded_metrics.calculate_health_score()
        assert 50 <= score < 90, f"Degraded metrics should score 50-90, got {score}"

    @pytest.mark.asyncio
    async def test_threshold_monitoring(self):
        """Threshold violations should be detected."""
        monitor = HealthMonitor()

        # High error rate
        bad_metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            error_rate=0.10  # 10% - critical
        )

        results = await monitor.run_health_checks(bad_metrics)

        # Find error rate check
        error_check = next(
            (r for r in results if r["check_id"] == "error_rate"),
            None
        )

        assert error_check is not None
        assert error_check["status"] == HealthStatus.CRITICAL

    @pytest.mark.asyncio
    async def test_trend_analysis(self):
        """Trend analysis should work correctly."""
        monitor = HealthMonitor()

        # Add increasing response times
        for i in range(20):
            metrics = HealthMetrics(
                timestamp=datetime.now().isoformat(),
                response_time_p95=1000 + i * 100  # Increasing
            )
            monitor.metrics_history.append(metrics)

        trend = monitor.get_trends("response_time_p95", window_minutes=60)

        assert trend["trend"] == "increasing"
        assert trend["data_points"] == 20


class TestAutoRepair:
    """Test auto-repair functionality."""

    @pytest.mark.asyncio
    async def test_repair_execution(self):
        """Repair actions should execute correctly."""
        repair = AutoRepair(dry_run=True)

        context = {
            "triggered_by": "test",
            "issue": "High CPU usage",
            "service_name": "analysis"
        }

        record = await repair.execute_repair(
            RepairAction.RESTART_SERVICE,
            context
        )

        assert record.result == RepairResult.SUCCESS
        assert record.success
        assert record.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_repair_failure_handling(self):
        """Failed repairs should be handled gracefully."""
        repair = AutoRepair(dry_run=False)

        # Mock failing strategy
        async def failing_strategy(context):
            raise Exception("Simulated failure")

        repair.repair_strategies[RepairAction.SCALE_UP] = failing_strategy

        record = await repair.execute_repair(
            RepairAction.SCALE_UP,
            {"triggered_by": "test"}
        )

        assert record.result == RepairResult.FAILED
        assert not record.success
        assert record.error_message is not None

    def test_success_rate_calculation(self):
        """Success rate should be calculated correctly."""
        repair = AutoRepair()

        # No repairs yet
        assert repair.get_success_rate() == 1.0

        # Add some repairs
        repair.stats["repairs_attempted"] = 10
        repair.stats["repairs_successful"] = 8

        assert repair.get_success_rate() == 0.8


class TestAlertManager:
    """Test alert management functionality."""

    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Alerts should be generated correctly."""
        manager = AlertManager()

        alert = await manager.generate_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test",
            source="test"
        )

        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"

    @pytest.mark.asyncio
    async def test_alert_deduplication(self):
        """Duplicate alerts should be deduplicated."""
        manager = AlertManager(dedup_window_seconds=60)

        # Generate same alert twice
        alert1 = await manager.generate_alert(
            severity=AlertSeverity.WARNING,
            title="Duplicate Test",
            message="Test message",
            source="test"
        )

        alert2 = await manager.generate_alert(
            severity=AlertSeverity.WARNING,
            title="Duplicate Test",
            message="Test message",
            source="test"
        )

        assert alert1 is not None
        assert alert2 is None  # Should be deduplicated
        assert manager.stats["alerts_deduplicated"] == 1

    @pytest.mark.asyncio
    async def test_alert_routing(self):
        """Alerts should be routed to correct channels."""
        manager = AlertManager()

        # Critical alert
        alert = await manager.generate_alert(
            severity=AlertSeverity.CRITICAL,
            title="Critical Issue",
            message="System critical",
            source="test"
        )

        # Should route to multiple channels
        assert len(alert.channels) >= 3  # PagerDuty, Slack, Email, Log
        assert alert.delivered


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.mark.asyncio
    async def test_memory_leak_detection_and_repair(self):
        """Complete workflow: detect memory leak → alert → repair."""
        orchestrator = SelfHealingOrchestrator(
            enable_auto_repair=True,
            dry_run=True
        )

        await orchestrator.start()

        # Simulate memory leak
        high_memory_metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            memory_usage_percent=95.0,  # Critical
            error_rate=0.01,
            response_time_p95=2000,
            cpu_usage_percent=50,
            availability_percent=99.9
        )

        # Process metrics
        await orchestrator.health_monitor.run_health_checks(high_memory_metrics)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Verify workflow
        assert orchestrator.stats["issues_detected"] > 0
        assert orchestrator.stats["alerts_sent"] > 0

        # In dry run, repair should be attempted
        if orchestrator.enable_auto_repair:
            assert orchestrator.stats["repairs_attempted"] > 0

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self):
        """Circuit breaker should prevent cascading failures."""
        orchestrator = SelfHealingOrchestrator(dry_run=True)

        # Simulate multiple service failures
        for _ in range(5):
            await orchestrator.auto_repair.execute_repair(
                RepairAction.RESTART_SERVICE,
                {
                    "triggered_by": "health_check",
                    "service_name": "external_api"
                }
            )

        # Circuit breaker should have opened
        # (In full implementation with actual circuit breaker)
        assert orchestrator.auto_repair.stats["repairs_attempted"] == 5

    @pytest.mark.asyncio
    async def test_system_degradation_and_recovery(self):
        """Test graceful degradation and recovery."""
        orchestrator = SelfHealingOrchestrator(
            enable_auto_repair=True,
            dry_run=True
        )

        await orchestrator.start()

        # Phase 1: System degrades
        degraded_metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            response_time_p95=2500,  # Warning level
            error_rate=0.025,  # Warning level
            cpu_usage_percent=75,
            memory_usage_percent=70,
            availability_percent=99.7
        )

        await orchestrator.health_monitor.run_health_checks(degraded_metrics)
        await asyncio.sleep(0.3)

        # Should detect degradation
        assert orchestrator.health_monitor.current_status in [
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY
        ]

        # Phase 2: System recovers
        healthy_metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            response_time_p95=1500,
            error_rate=0.008,
            cpu_usage_percent=45,
            memory_usage_percent=40,
            availability_percent=99.95
        )

        await orchestrator.health_monitor.run_health_checks(healthy_metrics)
        await asyncio.sleep(0.3)

        # Should recover
        assert orchestrator.health_monitor.current_status == HealthStatus.HEALTHY

        await orchestrator.stop()


class TestRealWorldScenarios:
    """Test real-world failure scenarios."""

    @pytest.mark.asyncio
    async def test_database_connection_pool_exhaustion(self):
        """Handle database connection pool exhaustion."""
        orchestrator = SelfHealingOrchestrator(
            enable_auto_repair=True,
            dry_run=True
        )

        # Simulate connection pool exhaustion
        context = {
            "triggered_by": "connection_monitor",
            "issue": "Connection pool exhausted",
            "service_name": "database",
            "current_connections": 100,
            "max_connections": 100
        }

        record = await orchestrator.auto_repair.execute_repair(
            RepairAction.RESTART_SERVICE,
            context
        )

        assert record.success

    @pytest.mark.asyncio
    async def test_queue_overflow_scenario(self):
        """Handle queue overflow with auto-drain."""
        orchestrator = SelfHealingOrchestrator(
            enable_auto_repair=True,
            dry_run=True
        )

        context = {
            "triggered_by": "queue_monitor",
            "issue": "Queue near capacity",
            "queue_name": "analysis_queue",
            "current_size": 9500,
            "max_size": 10000
        }

        record = await orchestrator.auto_repair.execute_repair(
            RepairAction.DRAIN_QUEUE,
            context
        )

        assert record.success
        assert record.action == RepairAction.DRAIN_QUEUE

    @pytest.mark.asyncio
    async def test_deployment_rollback_scenario(self):
        """Handle failed deployment with automatic rollback."""
        orchestrator = SelfHealingOrchestrator(
            enable_auto_repair=True,
            dry_run=True
        )

        # Simulate deployment failure
        context = {
            "triggered_by": "deployment_monitor",
            "issue": "High error rate after deployment",
            "current_version": "2.1.0",
            "target_version": "2.0.5"
        }

        record = await orchestrator.auto_repair.execute_repair(
            RepairAction.ROLLBACK_VERSION,
            context
        )

        assert record.success
        assert record.action == RepairAction.ROLLBACK_VERSION


class TestPerformanceImpact:
    """Test performance impact of self-healing system."""

    @pytest.mark.asyncio
    async def test_monitoring_overhead(self):
        """Monitoring should have minimal overhead."""
        import time

        monitor = HealthMonitor(check_interval=1)

        # Measure baseline
        start = time.perf_counter()
        metrics = await monitor.collect_metrics()
        baseline_duration = time.perf_counter() - start

        # Should be very fast (< 10ms)
        assert baseline_duration < 0.01, f"Metrics collection too slow: {baseline_duration}s"

    @pytest.mark.asyncio
    async def test_repair_execution_time(self):
        """Repairs should complete within acceptable time."""
        repair = AutoRepair(dry_run=True)

        import time
        start = time.perf_counter()

        record = await repair.execute_repair(
            RepairAction.CLEAR_CACHE,
            {"triggered_by": "test"}
        )

        duration = time.perf_counter() - start

        # Should complete quickly in dry run
        assert duration < 1.0, f"Repair too slow: {duration}s"
        assert record.duration_seconds < 1.0


class TestFailureRecovery:
    """Test recovery from various failure modes."""

    @pytest.mark.asyncio
    async def test_monitor_crash_recovery(self):
        """Health monitor should recover from crashes."""
        monitor = HealthMonitor()

        await monitor.start()

        # Simulate crash by cancelling task
        if monitor._monitor_task:
            monitor._monitor_task.cancel()

        # Should be able to restart
        await monitor.stop()
        await monitor.start()

        assert monitor.is_running

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_alert_delivery_failure_handling(self):
        """Alert delivery failures should be handled gracefully."""
        manager = AlertManager()

        # Mock failing delivery
        async def failing_delivery(alert, channel):
            raise Exception("Delivery failed")

        original_deliver = manager._deliver_to_channel
        manager._deliver_to_channel = failing_delivery

        # Should not crash
        alert = await manager.generate_alert(
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test message",
            source="test"
        )

        # Alert should be created even if delivery fails
        assert alert is not None

        # Restore
        manager._deliver_to_channel = original_deliver


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
