"""
SelfHealing_V2 - Production Backend Integration

Enhanced integration with SLO tracking, predictive capabilities, and runbook automation.
"""

import sys
import asyncio
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Add backend path
_backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

# Import backend implementations
try:
    from shared.self_healing import (
        HealthMonitor as BackendHealthMonitor,
        AutoRepair as BackendAutoRepair,
        AlertManager as BackendAlertManager,
        MetricsCollector as BackendMetricsCollector,
        SelfHealingOrchestrator as BackendOrchestrator,
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


class SLOStatus(str, Enum):
    """SLO compliance status."""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    VIOLATED = "violated"


@dataclass
class HealthCheckResult:
    """Production health check result with SLO data."""
    service: str
    healthy: bool
    latency_ms: float
    slo_status: SLOStatus
    error_budget_remaining: float
    details: Dict[str, Any]


class ProductionHealthMonitor:
    """
    V2 Production Health Monitor with SLO integration.

    Features:
    - SLO-based health evaluation
    - Error budget tracking
    - Predictive health analysis
    """

    def __init__(
        self,
        slo_latency_ms: float = 1000,
        slo_availability: float = 99.9,
        use_backend: bool = True,
    ):
        self.slo_latency_ms = slo_latency_ms
        self.slo_availability = slo_availability
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            self._backend = BackendHealthMonitor()
        else:
            from .health_monitor import HealthMonitor
            self._local = HealthMonitor()

        # SLO tracking
        self._health_history: Dict[str, List[bool]] = {}
        self._latency_history: Dict[str, List[float]] = {}
        self._window_size = 100

    async def check_health(self, service: str) -> HealthCheckResult:
        """Check service health with SLO evaluation."""
        import time
        start = time.time()

        try:
            if self.use_backend:
                result = await self._backend.check_health(service)
                healthy = result.status == "healthy"
            else:
                result = await self._local.check_service(service)
                healthy = result.status.value == "healthy"

            latency = (time.time() - start) * 1000

        except Exception as e:
            healthy = False
            latency = (time.time() - start) * 1000

        # Track history
        self._record_check(service, healthy, latency)

        # Calculate SLO status
        slo_status = self._evaluate_slo(service, latency)
        error_budget = self._calculate_error_budget(service)

        return HealthCheckResult(
            service=service,
            healthy=healthy,
            latency_ms=latency,
            slo_status=slo_status,
            error_budget_remaining=error_budget,
            details={"backend": self.use_backend},
        )

    def _record_check(self, service: str, healthy: bool, latency: float):
        """Record health check for SLO tracking."""
        if service not in self._health_history:
            self._health_history[service] = []
            self._latency_history[service] = []

        self._health_history[service].append(healthy)
        self._latency_history[service].append(latency)

        # Maintain window
        if len(self._health_history[service]) > self._window_size:
            self._health_history[service] = self._health_history[service][-self._window_size:]
            self._latency_history[service] = self._latency_history[service][-self._window_size:]

    def _evaluate_slo(self, service: str, latency: float) -> SLOStatus:
        """Evaluate SLO compliance."""
        if service not in self._health_history:
            return SLOStatus.COMPLIANT

        history = self._health_history[service]
        availability = sum(history) / len(history) * 100 if history else 100

        # Latency SLO
        latency_violated = latency > self.slo_latency_ms

        # Availability SLO
        if availability < self.slo_availability - 1:
            return SLOStatus.VIOLATED
        elif availability < self.slo_availability or latency_violated:
            return SLOStatus.AT_RISK

        return SLOStatus.COMPLIANT

    def _calculate_error_budget(self, service: str) -> float:
        """Calculate remaining error budget percentage."""
        if service not in self._health_history:
            return 100.0

        history = self._health_history[service]
        if not history:
            return 100.0

        failures = sum(1 for h in history if not h)
        allowed_failures = len(history) * (100 - self.slo_availability) / 100

        if allowed_failures <= 0:
            return 100.0 if failures == 0 else 0.0

        return max(0, (1 - failures / allowed_failures) * 100)


class ProductionRecoveryManager:
    """
    V2 Production Recovery Manager with runbook automation.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            self._backend = BackendAutoRepair()
        else:
            from .recovery_manager import RecoveryManager
            self._local = RecoveryManager()

        self._runbooks: Dict[str, List[Dict]] = {}
        self._execution_history: List[Dict] = []

    def register_runbook(
        self,
        service: str,
        steps: List[Dict[str, Any]],
        cooldown_seconds: int = 300,
    ):
        """Register automated recovery runbook."""
        self._runbooks[service] = {
            "steps": steps,
            "cooldown": cooldown_seconds,
            "last_execution": None,
        }

    async def execute_recovery(
        self,
        service: str,
        reason: str = "health_check_failed",
    ) -> Dict[str, Any]:
        """Execute recovery with runbook."""
        import time
        from datetime import datetime, timezone

        runbook = self._runbooks.get(service)

        if runbook:
            # Check cooldown
            if runbook["last_execution"]:
                elapsed = time.time() - runbook["last_execution"]
                if elapsed < runbook["cooldown"]:
                    return {
                        "success": False,
                        "reason": "cooldown_active",
                        "remaining": runbook["cooldown"] - elapsed,
                    }

        # Execute recovery
        if self.use_backend:
            result = await self._backend.execute_repair(service, reason)
        else:
            result = await self._local.execute_recovery(service)

        # Record execution
        execution = {
            "service": service,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": result.status.value == "success" if hasattr(result, 'status') else True,
        }
        self._execution_history.append(execution)

        if runbook:
            runbook["last_execution"] = time.time()

        return execution


class ProductionPredictiveHealer:
    """
    V2 Predictive Healer with ML-based anomaly detection.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend

        from .predictive_healer import PredictiveHealer
        self._healer = PredictiveHealer()

    async def analyze_and_heal(self, service: str) -> Dict[str, Any]:
        """Analyze metrics and take proactive action."""
        prediction = await self._healer.predict_failures(service)

        if prediction.risk_level.value in ["high", "critical"]:
            actions = await self._healer.get_proactive_actions()
            return {
                "service": service,
                "risk": prediction.risk_level.value,
                "confidence": prediction.confidence,
                "actions_taken": actions,
            }

        return {
            "service": service,
            "risk": prediction.risk_level.value,
            "confidence": prediction.confidence,
            "actions_taken": [],
        }

    async def record_metric(self, service: str, metric: str, value: float):
        """Record metric for analysis."""
        await self._healer.record_metric(service, metric, value)


# Factory functions
def get_health_monitor(
    slo_latency_ms: float = 1000,
    slo_availability: float = 99.9,
    use_backend: bool = True,
) -> ProductionHealthMonitor:
    """Get production health monitor."""
    return ProductionHealthMonitor(slo_latency_ms, slo_availability, use_backend)


def get_recovery_manager(use_backend: bool = True) -> ProductionRecoveryManager:
    """Get production recovery manager."""
    return ProductionRecoveryManager(use_backend)


def get_predictive_healer(use_backend: bool = True) -> ProductionPredictiveHealer:
    """Get production predictive healer."""
    return ProductionPredictiveHealer(use_backend)


__all__ = [
    "BACKEND_AVAILABLE",
    "SLOStatus",
    "HealthCheckResult",
    "ProductionHealthMonitor",
    "ProductionRecoveryManager",
    "ProductionPredictiveHealer",
    "get_health_monitor",
    "get_recovery_manager",
    "get_predictive_healer",
]
