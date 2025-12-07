"""Tests for SelfHealing_V2"""

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.health_monitor import HealthMonitor, HealthStatus
from src.recovery_manager import RecoveryManager, Runbook, RecoveryStep, RecoveryAction, RecoveryStatus
from src.predictive_healer import PredictiveHealer, RiskLevel


class TestHealthMonitorV2:
    @pytest.fixture
    def monitor(self):
        return HealthMonitor()

    @pytest.mark.asyncio
    async def test_register_and_check(self, monitor):
        async def healthy_check():
            return True

        monitor.register_service("test-service", healthy_check)
        result = await monitor.check_service("test-service")

        assert result.service == "test-service"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNKNOWN]

    @pytest.mark.asyncio
    async def test_slo_latency_check(self, monitor):
        async def slow_check():
            await asyncio.sleep(0.05)
            return True

        monitor.register_service(
            "slow-service",
            slow_check,
            slo_latency_ms=10  # Will fail SLO
        )

        result = await monitor.check_service("slow-service")
        assert result.details.get("slo_met") == False

    @pytest.mark.asyncio
    async def test_unhealthy_threshold(self, monitor):
        async def failing_check():
            return False

        monitor.register_service(
            "failing-service",
            failing_check,
            unhealthy_threshold=2
        )

        await monitor.check_service("failing-service")
        await monitor.check_service("failing-service")

        status = monitor.get_status("failing-service")
        assert status == HealthStatus.UNHEALTHY

    def test_overall_status(self, monitor):
        monitor._status["service-1"] = HealthStatus.HEALTHY
        monitor._status["service-2"] = HealthStatus.HEALTHY

        assert monitor.get_overall_status() == HealthStatus.HEALTHY

        monitor._status["service-2"] = HealthStatus.DEGRADED
        assert monitor.get_overall_status() == HealthStatus.DEGRADED


class TestRecoveryManagerV2:
    @pytest.fixture
    def manager(self):
        return RecoveryManager()

    @pytest.mark.asyncio
    async def test_runbook_execution(self, manager):
        step_executed = []

        async def step1():
            step_executed.append(1)
            return True

        async def step2():
            step_executed.append(2)
            return True

        runbook = Runbook(
            name="test-runbook",
            description="Test recovery",
            steps=[
                RecoveryStep(RecoveryAction.CLEAR_CACHE, step1),
                RecoveryStep(RecoveryAction.RESTART, step2),
            ],
            cooldown_seconds=0,
        )

        manager.register_runbook(runbook)
        manager.assign_runbook("test-service", "test-runbook")

        execution = await manager.execute_recovery("test-service")

        assert execution.status == RecoveryStatus.SUCCESS
        assert step_executed == [1, 2]

    @pytest.mark.asyncio
    async def test_step_failure(self, manager):
        async def failing_step():
            return False

        runbook = Runbook(
            name="failing-runbook",
            description="Failing recovery",
            steps=[
                RecoveryStep(RecoveryAction.RESTART, failing_step, retry_count=1),
            ],
            cooldown_seconds=0,
        )

        manager.register_runbook(runbook)
        manager.assign_runbook("failing-service", "failing-runbook")

        execution = await manager.execute_recovery("failing-service")

        assert execution.status == RecoveryStatus.FAILED

    @pytest.mark.asyncio
    async def test_cooldown(self, manager):
        async def step():
            return True

        runbook = Runbook(
            name="cooldown-runbook",
            description="Cooldown test",
            steps=[RecoveryStep(RecoveryAction.RESTART, step)],
            cooldown_seconds=300,
        )

        manager.register_runbook(runbook)
        manager.assign_runbook("cooldown-service", "cooldown-runbook")

        # First execution
        await manager.execute_recovery("cooldown-service")

        # Second should be skipped (cooldown)
        execution = await manager.execute_recovery("cooldown-service")

        assert execution.status == RecoveryStatus.SKIPPED


class TestPredictiveHealer:
    @pytest.fixture
    def healer(self):
        return PredictiveHealer()

    @pytest.mark.asyncio
    async def test_record_metric(self, healer):
        for i in range(10):
            await healer.record_metric("test-service", "cpu_usage", 50 + i)

        summary = healer.get_metrics_summary("test-service")
        assert "cpu_usage" in summary
        assert summary["cpu_usage"]["samples"] == 10

    @pytest.mark.asyncio
    async def test_predict_healthy(self, healer):
        # Record healthy metrics
        for _ in range(20):
            await healer.record_metric("healthy-service", "cpu_usage", 30)

        prediction = await healer.predict_failures("healthy-service")

        assert prediction.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_predict_critical(self, healer):
        # Record critical metrics
        for _ in range(20):
            await healer.record_metric("critical-service", "cpu_usage", 95)

        prediction = await healer.predict_failures("critical-service")

        assert prediction.risk_level == RiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_proactive_actions(self, healer):
        for _ in range(20):
            await healer.record_metric("at-risk", "cpu_usage", 92)

        actions = await healer.get_proactive_actions()

        assert len(actions) > 0
        assert actions[0]["service"] == "at-risk"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
