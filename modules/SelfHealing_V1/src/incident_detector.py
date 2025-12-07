"""
SelfHealing_V1 - Incident Detector

Detects and classifies system incidents.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .health_monitor import HealthStatus, SystemHealth

logger = logging.getLogger(__name__)


class IncidentSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IncidentType(str, Enum):
    SERVICE_DOWN = "service_down"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_SPIKE = "error_spike"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class Incident:
    """System incident"""
    incident_id: str
    type: IncidentType
    severity: IncidentSeverity
    affected_services: List[str]
    description: str
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None

    def is_active(self) -> bool:
        return self.resolved_at is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "type": self.type.value,
            "severity": self.severity.value,
            "affected_services": self.affected_services,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution": self.resolution,
            "is_active": self.is_active(),
        }


class IncidentDetector:
    """
    Incident detection system.

    Features:
    - Incident detection
    - Severity classification
    - Incident tracking
    """

    def __init__(
        self,
        latency_threshold_ms: float = 1000,
        error_rate_threshold: float = 0.05,
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold

        self._incidents: Dict[str, Incident] = {}
        self._incident_counter = 0

    async def analyze(self, health: SystemHealth) -> List[Incident]:
        """Analyze system health for incidents"""
        new_incidents = []

        for service_name, service in health.services.items():
            # Check for service down
            if service.status == HealthStatus.UNHEALTHY:
                incident = self._create_incident(
                    type=IncidentType.SERVICE_DOWN,
                    severity=IncidentSeverity.CRITICAL,
                    services=[service_name],
                    description=f"Service {service_name} is unhealthy: {service.error_message}",
                )
                new_incidents.append(incident)

            # Check for performance degradation
            elif service.response_time_ms and service.response_time_ms > self.latency_threshold_ms:
                incident = self._create_incident(
                    type=IncidentType.PERFORMANCE_DEGRADATION,
                    severity=IncidentSeverity.MEDIUM,
                    services=[service_name],
                    description=f"High latency on {service_name}: {service.response_time_ms:.0f}ms",
                )
                new_incidents.append(incident)

        return new_incidents

    def _create_incident(
        self,
        type: IncidentType,
        severity: IncidentSeverity,
        services: List[str],
        description: str,
    ) -> Incident:
        """Create new incident"""
        self._incident_counter += 1
        incident_id = f"INC-{self._incident_counter:06d}"

        incident = Incident(
            incident_id=incident_id,
            type=type,
            severity=severity,
            affected_services=services,
            description=description,
            detected_at=datetime.now(timezone.utc),
        )

        self._incidents[incident_id] = incident
        logger.warning(f"Incident detected: {incident_id} - {description}")

        return incident

    async def resolve_incident(
        self,
        incident_id: str,
        resolution: str,
    ) -> bool:
        """Mark incident as resolved"""
        incident = self._incidents.get(incident_id)

        if incident and incident.is_active():
            incident.resolved_at = datetime.now(timezone.utc)
            incident.resolution = resolution
            logger.info(f"Incident resolved: {incident_id}")
            return True

        return False

    def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents"""
        return [i for i in self._incidents.values() if i.is_active()]

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID"""
        return self._incidents.get(incident_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get incident statistics"""
        active = [i for i in self._incidents.values() if i.is_active()]
        resolved = [i for i in self._incidents.values() if not i.is_active()]

        by_severity = {}
        for i in active:
            by_severity[i.severity.value] = by_severity.get(i.severity.value, 0) + 1

        return {
            "total_incidents": len(self._incidents),
            "active_incidents": len(active),
            "resolved_incidents": len(resolved),
            "active_by_severity": by_severity,
        }
