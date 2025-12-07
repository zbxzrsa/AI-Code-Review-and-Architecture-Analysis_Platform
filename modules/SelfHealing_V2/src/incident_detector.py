"""
SelfHealing_V2 - Incident Detector

Production incident detection with pattern recognition and alert correlation.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable


class IncidentSeverity(str, Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    """Incident status."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class Incident:
    """Incident data structure."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    service: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    assignee: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "service": self.service,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "alerts": self.alerts,
            "tags": self.tags,
            "assignee": self.assignee,
        }


@dataclass
class DetectionRule:
    """Incident detection rule."""
    rule_id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: IncidentSeverity
    cooldown_seconds: int = 300
    last_triggered: Optional[datetime] = None


class IncidentDetector:
    """
    Production Incident Detector.

    Features:
    - Pattern-based detection
    - Alert correlation
    - Automatic severity classification
    - Deduplication
    """

    def __init__(
        self,
        correlation_window_minutes: int = 5,
        dedup_window_minutes: int = 30,
    ):
        self.correlation_window = timedelta(minutes=correlation_window_minutes)
        self.dedup_window = timedelta(minutes=dedup_window_minutes)

        self._incidents: Dict[str, Incident] = {}
        self._rules: Dict[str, DetectionRule] = {}
        self._alert_buffer: List[Dict[str, Any]] = []
        self._incident_counter = 0

    def register_rule(
        self,
        rule_id: str,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: IncidentSeverity = IncidentSeverity.MEDIUM,
        cooldown_seconds: int = 300,
    ):
        """Register detection rule."""
        self._rules[rule_id] = DetectionRule(
            rule_id=rule_id,
            name=name,
            condition=condition,
            severity=severity,
            cooldown_seconds=cooldown_seconds,
        )

    async def process_alert(self, alert: Dict[str, Any]) -> Optional[Incident]:
        """Process alert and detect incidents."""
        now = datetime.now(timezone.utc)

        # Add to buffer
        alert["received_at"] = now
        self._alert_buffer.append(alert)

        # Clean old alerts
        self._alert_buffer = [
            a for a in self._alert_buffer
            if (now - a["received_at"]) < self.correlation_window
        ]

        # Check rules
        for rule in self._rules.values():
            if self._should_trigger_rule(rule, now) and rule.condition(alert):
                incident = await self._create_or_correlate_incident(alert, rule)
                return incident

        return None

    def _should_trigger_rule(self, rule: DetectionRule, now: datetime) -> bool:
        """Check if rule cooldown has passed."""
        if rule.last_triggered is None:
            return True

        elapsed = (now - rule.last_triggered).total_seconds()
        return elapsed >= rule.cooldown_seconds

    async def _create_or_correlate_incident(
        self,
        alert: Dict[str, Any],
        rule: DetectionRule,
    ) -> Incident:
        """Create new incident or correlate with existing."""
        now = datetime.now(timezone.utc)
        service = alert.get("service", "unknown")

        # Check for existing incident to correlate
        for incident in self._incidents.values():
            if (
                incident.service == service
                and incident.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
                and (now - incident.created_at) < self.dedup_window
            ):
                # Correlate with existing
                incident.alerts.append(alert)
                incident.updated_at = now

                # Escalate severity if needed
                if rule.severity.value > incident.severity.value:
                    incident.severity = rule.severity

                return incident

        # Create new incident
        self._incident_counter += 1
        incident_id = f"INC-{self._incident_counter:06d}"

        incident = Incident(
            incident_id=incident_id,
            title=f"{rule.name}: {service}",
            description=alert.get("message", "Incident detected"),
            severity=rule.severity,
            status=IncidentStatus.OPEN,
            service=service,
            created_at=now,
            updated_at=now,
            alerts=[alert],
            tags=[rule.rule_id],
        )

        self._incidents[incident_id] = incident
        rule.last_triggered = now

        return incident

    async def acknowledge_incident(self, incident_id: str, assignee: str):
        """Acknowledge incident."""
        if incident_id in self._incidents:
            incident = self._incidents[incident_id]
            incident.status = IncidentStatus.ACKNOWLEDGED
            incident.assignee = assignee
            incident.updated_at = datetime.now(timezone.utc)

    async def resolve_incident(self, incident_id: str):
        """Resolve incident."""
        if incident_id in self._incidents:
            incident = self._incidents[incident_id]
            now = datetime.now(timezone.utc)
            incident.status = IncidentStatus.RESOLVED
            incident.resolved_at = now
            incident.updated_at = now

    def get_open_incidents(self) -> List[Incident]:
        """Get all open incidents."""
        return [
            i for i in self._incidents.values()
            if i.status in [IncidentStatus.OPEN, IncidentStatus.ACKNOWLEDGED, IncidentStatus.INVESTIGATING]
        ]

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        return self._incidents.get(incident_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get incident statistics."""
        incidents = list(self._incidents.values())

        by_severity = {}
        by_status = {}

        for incident in incidents:
            severity = incident.severity.value
            status = incident.status.value

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1

        # Calculate MTTR for resolved incidents
        resolved = [i for i in incidents if i.resolved_at]
        mttr = None
        if resolved:
            total_time = sum(
                (i.resolved_at - i.created_at).total_seconds()
                for i in resolved
            )
            mttr = total_time / len(resolved)

        return {
            "total_incidents": len(incidents),
            "open_incidents": len(self.get_open_incidents()),
            "by_severity": by_severity,
            "by_status": by_status,
            "mttr_seconds": mttr,
            "rules_registered": len(self._rules),
        }


# Default rules
def create_default_rules() -> List[DetectionRule]:
    """Create default detection rules."""
    return [
        DetectionRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            condition=lambda a: a.get("error_rate", 0) > 0.1,
            severity=IncidentSeverity.HIGH,
        ),
        DetectionRule(
            rule_id="service_down",
            name="Service Down",
            condition=lambda a: a.get("status") == "down",
            severity=IncidentSeverity.CRITICAL,
        ),
        DetectionRule(
            rule_id="high_latency",
            name="High Latency",
            condition=lambda a: a.get("latency_ms", 0) > 5000,
            severity=IncidentSeverity.MEDIUM,
        ),
    ]


__all__ = [
    "IncidentSeverity",
    "IncidentStatus",
    "Incident",
    "DetectionRule",
    "IncidentDetector",
    "create_default_rules",
]
