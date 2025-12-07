"""
Monitoring_V2 - Alert Manager

Production alert management with routing and escalation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Callable


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    name: str
    condition: str  # Expression
    severity: AlertSeverity
    for_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    labels: Dict[str, str]
    annotations: Dict[str, str]
    started_at: datetime
    ended_at: Optional[datetime] = None
    silenced_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "labels": self.labels,
            "annotations": self.annotations,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


@dataclass
class Silence:
    """Alert silence."""
    silence_id: str
    matchers: Dict[str, str]
    starts_at: datetime
    ends_at: datetime
    created_by: str
    comment: str


class AlertManager:
    """
    Production Alert Manager.

    Features:
    - Rule-based alerting
    - Alert grouping and deduplication
    - Silencing
    - Notification routing
    """

    def __init__(
        self,
        group_wait: timedelta = timedelta(seconds=30),
        group_interval: timedelta = timedelta(minutes=5),
        repeat_interval: timedelta = timedelta(hours=4),
    ):
        self.group_wait = group_wait
        self.group_interval = group_interval
        self.repeat_interval = repeat_interval

        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._silences: Dict[str, Silence] = {}
        self._notification_handlers: List[Callable] = []
        self._alert_counter = 0

    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self._rules[rule.rule_id] = rule

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler."""
        self._notification_handlers.append(handler)

    async def fire_alert(
        self,
        rule_id: str,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> Alert:
        """Fire an alert."""
        rule = self._rules.get(rule_id)
        if not rule:
            raise ValueError(f"Unknown rule: {rule_id}")

        now = datetime.now(timezone.utc)

        # Check for existing alert
        existing = self._find_existing_alert(rule_id, labels)
        if existing and existing.status == AlertStatus.FIRING:
            return existing

        # Create new alert
        self._alert_counter += 1
        alert_id = f"ALT-{self._alert_counter:06d}"

        alert = Alert(
            alert_id=alert_id,
            rule_id=rule_id,
            name=rule.name,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            labels={**rule.labels, **(labels or {})},
            annotations={**rule.annotations, **(annotations or {})},
            started_at=now,
        )

        # Check silences
        if self._is_silenced(alert):
            alert.status = AlertStatus.SILENCED
        else:
            # Send notifications
            await self._notify(alert)

        self._alerts[alert_id] = alert
        return alert

    async def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.ended_at = datetime.now(timezone.utc)
            await self._notify(alert)

    def _find_existing_alert(
        self,
        rule_id: str,
        labels: Optional[Dict[str, str]],
    ) -> Optional[Alert]:
        """Find existing alert matching rule and labels."""
        for alert in self._alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.FIRING:
                if labels is None or all(
                    alert.labels.get(k) == v for k, v in labels.items()
                ):
                    return alert
        return None

    def _is_silenced(self, alert: Alert) -> bool:
        """Check if alert matches any silence."""
        now = datetime.now(timezone.utc)

        for silence in self._silences.values():
            if silence.starts_at <= now <= silence.ends_at:
                if all(
                    alert.labels.get(k) == v
                    for k, v in silence.matchers.items()
                ):
                    alert.silenced_until = silence.ends_at
                    return True

        return False

    async def _notify(self, alert: Alert):
        """Send notifications for alert."""
        for handler in self._notification_handlers:
            try:
                if callable(handler):
                    import asyncio
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
            except Exception:
                pass  # Don't let handler errors break alerting

    def add_silence(
        self,
        matchers: Dict[str, str],
        duration: timedelta,
        created_by: str,
        comment: str = "",
    ) -> Silence:
        """Add a silence."""
        now = datetime.now(timezone.utc)
        silence_id = f"SIL-{len(self._silences) + 1:04d}"

        silence = Silence(
            silence_id=silence_id,
            matchers=matchers,
            starts_at=now,
            ends_at=now + duration,
            created_by=created_by,
            comment=comment,
        )

        self._silences[silence_id] = silence
        return silence

    def remove_silence(self, silence_id: str):
        """Remove a silence."""
        if silence_id in self._silences:
            del self._silences[silence_id]

    def get_firing_alerts(self) -> List[Alert]:
        """Get all firing alerts."""
        return [
            a for a in self._alerts.values()
            if a.status == AlertStatus.FIRING
        ]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity."""
        return [
            a for a in self._alerts.values()
            if a.severity == severity and a.status == AlertStatus.FIRING
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        alerts = list(self._alerts.values())
        firing = [a for a in alerts if a.status == AlertStatus.FIRING]

        by_severity = {}
        for alert in firing:
            sev = alert.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_alerts": len(alerts),
            "firing": len(firing),
            "resolved": len([a for a in alerts if a.status == AlertStatus.RESOLVED]),
            "silenced": len([a for a in alerts if a.status == AlertStatus.SILENCED]),
            "by_severity": by_severity,
            "rules_defined": len(self._rules),
            "active_silences": len([
                s for s in self._silences.values()
                if s.ends_at > datetime.now(timezone.utc)
            ]),
        }


__all__ = [
    "AlertSeverity",
    "AlertStatus",
    "AlertRule",
    "Alert",
    "Silence",
    "AlertManager",
]
