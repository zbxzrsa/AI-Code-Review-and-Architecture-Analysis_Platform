"""Tests for SelfHealing_V1 Health Monitor"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.health_monitor import HealthMonitor, HealthStatus
from src.recovery_manager import RecoveryManager, RecoveryAction
from src.incident_detector import IncidentDetector


class TestHealthMonitor:
    @pytest.fixture
    def monitor(self):
        return HealthMonitor()

    def test_register_service(self, monitor):
        async def check():
            return True

        monitor.register_service("test-service", check)
        assert "test-service" in monitor._services

    @pytest.mark.asyncio
    async def test_check_healthy_service(self, monitor):
        async def check():
            return True

        monitor.register_service("test-service", check)
        result = await monitor.check_service("test-service")

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_unhealthy_service(self, monitor):
        async def check():
            return False

        monitor.register_service("test-service", check)

        # Multiple failures to reach unhealthy
        for _ in range(5):
            result = await monitor.check_service("test-service")

        assert result.status == HealthStatus.UNHEALTHY


class TestRecoveryManager:
    @pytest.fixture
    def recovery(self):
        return RecoveryManager()

    def test_register_handler(self, recovery):
        async def handler():
            return True

        recovery.register_recovery_handler("test", RecoveryAction.RESTART, handler)
        assert "test" in recovery._recovery_handlers


class TestIncidentDetector:
    @pytest.fixture
    def detector(self):
        return IncidentDetector()

    def test_get_active_incidents(self, detector):
        incidents = detector.get_active_incidents()
        assert isinstance(incidents, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
