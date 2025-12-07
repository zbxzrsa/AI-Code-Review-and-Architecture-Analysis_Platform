"""
SelfHealing_V1 - Backend Integration Bridge

Integrates with backend/shared/self_healing implementations.
This module provides a unified interface to the backend self-healing components.
"""

import sys
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add backend path for imports
_backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

# Import backend implementations
try:
    from shared.self_healing import (
        HealthMonitor as BackendHealthMonitor,
        HealthStatus as BackendHealthStatus,
        HealthMetrics as BackendHealthMetrics,
        AutoRepair as BackendAutoRepair,
        RepairAction as BackendRepairAction,
        RepairResult as BackendRepairResult,
        AlertManager as BackendAlertManager,
        Alert as BackendAlert,
        AlertSeverity as BackendAlertSeverity,
        MetricsCollector as BackendMetricsCollector,
        SelfHealingOrchestrator as BackendOrchestrator,
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    BackendHealthMonitor = None
    BackendAutoRepair = None
    BackendAlertManager = None
    BackendMetricsCollector = None
    BackendOrchestrator = None


class IntegratedHealthMonitor:
    """
    V1 Health Monitor with backend integration.

    Wraps backend HealthMonitor with V1-specific experimental features.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            self._backend = BackendHealthMonitor()
        else:
            # Fallback to local implementation
            from .health_monitor import HealthMonitor
            self._local = HealthMonitor()

    async def check_health(self, service: str) -> Dict[str, Any]:
        """Check service health using backend or local implementation."""
        if self.use_backend:
            return await self._backend.check_health(service)
        return await self._local.check_service(service)

    def get_status(self, service: str) -> str:
        """Get health status."""
        if self.use_backend:
            return self._backend.get_status(service).value
        return self._local.get_status(service).value


class IntegratedAutoRepair:
    """
    V1 Auto Repair with backend integration.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            self._backend = BackendAutoRepair()
        else:
            from .recovery_manager import RecoveryManager
            self._local = RecoveryManager()

    async def execute_repair(self, service: str, action: str) -> Dict[str, Any]:
        """Execute repair action."""
        if self.use_backend:
            return await self._backend.execute_repair(service, action)
        return await self._local.execute_recovery(service)


class IntegratedAlertManager:
    """
    V1 Alert Manager with backend integration.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            self._backend = BackendAlertManager()
        else:
            from .incident_detector import IncidentDetector
            self._local = IncidentDetector()

    async def send_alert(
        self,
        service: str,
        severity: str,
        message: str,
    ) -> bool:
        """Send alert."""
        if self.use_backend:
            return await self._backend.send_alert(service, severity, message)
        return self._local.detect_incident(service, severity)


class IntegratedMetricsCollector:
    """
    V1 Metrics Collector with backend integration.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            self._backend = BackendMetricsCollector()
        else:
            from modules.Monitoring_V1.src.metrics_collector import MetricsCollector
            self._local = MetricsCollector()

    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None):
        """Record a metric."""
        if self.use_backend:
            self._backend.record(name, value, tags or {})
        else:
            self._local.inc(name, value, tags or {})


# Factory functions
def get_health_monitor(use_backend: bool = True) -> IntegratedHealthMonitor:
    """Get integrated health monitor."""
    return IntegratedHealthMonitor(use_backend)


def get_auto_repair(use_backend: bool = True) -> IntegratedAutoRepair:
    """Get integrated auto repair."""
    return IntegratedAutoRepair(use_backend)


def get_alert_manager(use_backend: bool = True) -> IntegratedAlertManager:
    """Get integrated alert manager."""
    return IntegratedAlertManager(use_backend)


def get_metrics_collector(use_backend: bool = True) -> IntegratedMetricsCollector:
    """Get integrated metrics collector."""
    return IntegratedMetricsCollector(use_backend)


# Export backend availability
__all__ = [
    "BACKEND_AVAILABLE",
    "IntegratedHealthMonitor",
    "IntegratedAutoRepair",
    "IntegratedAlertManager",
    "IntegratedMetricsCollector",
    "get_health_monitor",
    "get_auto_repair",
    "get_alert_manager",
    "get_metrics_collector",
]
