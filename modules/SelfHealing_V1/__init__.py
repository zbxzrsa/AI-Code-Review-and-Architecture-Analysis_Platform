"""
SelfHealing_V1 - Experimental Self-Healing Module

System self-healing and recovery capabilities.
Version: V1 (Experimental)
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Incident:
    """Detected incident."""
    incident_id: str
    component: str
    severity: IncidentSeverity
    description: str
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Monitors system component health."""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheck] = {}
        self._running = False

    def register_check(self, component: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[component] = check_func

    async def check_health(self, component: str) -> HealthCheck:
        """Run health check for a component."""
        check_func = self.health_checks.get(component)
        if not check_func:
            return HealthCheck(
                component=component,
                status=HealthStatus.UNKNOWN,
                message="No health check registered",
                timestamp=datetime.utcnow()
            )

        try:
            result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
            return HealthCheck(
                component=component,
                status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                message="Check passed" if result else "Check failed",
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            return HealthCheck(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                timestamp=datetime.utcnow()
            )

    async def check_all(self) -> Dict[str, HealthCheck]:
        """Check health of all registered components."""
        for component in self.health_checks:
            self.results[component] = await self.check_health(component)
        return self.results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in self.results.values()]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        return HealthStatus.DEGRADED


class RecoveryManager:
    """Manages system recovery operations."""

    def __init__(self):
        self.recovery_actions: Dict[str, Callable] = {}
        self.recovery_history: List[Dict[str, Any]] = []

    def register_recovery(self, component: str, recovery_func: Callable):
        """Register a recovery action for a component."""
        self.recovery_actions[component] = recovery_func

    async def attempt_recovery(self, component: str) -> bool:
        """Attempt to recover a component."""
        recovery_func = self.recovery_actions.get(component)
        if not recovery_func:
            return False

        try:
            result = await recovery_func() if asyncio.iscoroutinefunction(recovery_func) else recovery_func()
            self.recovery_history.append({
                "component": component,
                "timestamp": datetime.utcnow().isoformat(),
                "success": result
            })
            return result
        except Exception as e:
            self.recovery_history.append({
                "component": component,
                "timestamp": datetime.utcnow().isoformat(),
                "success": False,
                "error": str(e)
            })
            return False


class IncidentDetector:
    """Detects and tracks system incidents."""

    def __init__(self):
        self.incidents: Dict[str, Incident] = {}
        self.detection_rules: List[Callable] = []

    def add_detection_rule(self, rule: Callable):
        """Add an incident detection rule."""
        self.detection_rules.append(rule)

    def detect_incident(
        self,
        component: str,
        severity: IncidentSeverity,
        description: str
    ) -> Incident:
        """Create and track a new incident."""
        import secrets
        incident_id = f"INC-{secrets.token_hex(4).upper()}"

        incident = Incident(
            incident_id=incident_id,
            component=component,
            severity=severity,
            description=description,
            detected_at=datetime.utcnow()
        )
        self.incidents[incident_id] = incident
        return incident

    def resolve_incident(self, incident_id: str) -> bool:
        """Mark an incident as resolved."""
        incident = self.incidents.get(incident_id)
        if incident:
            incident.resolved_at = datetime.utcnow()
            return True
        return False

    def get_active_incidents(self) -> List[Incident]:
        """Get all unresolved incidents."""
        return [i for i in self.incidents.values() if i.resolved_at is None]


__version__ = "1.0.0"
__status__ = "experimental"
__all__ = [
    "HealthMonitor", "RecoveryManager", "IncidentDetector",
    "HealthStatus", "IncidentSeverity", "HealthCheck", "Incident"
]
