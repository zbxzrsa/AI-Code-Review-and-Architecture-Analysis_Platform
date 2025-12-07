"""
Monitoring_V1 - Alert Manager

Alert rules and notification management.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    for_duration: int = 0  # Seconds condition must be true
    enabled: bool = True


@dataclass
class Alert:
    """Active alert instance"""
    rule_name: str
    severity: AlertSeverity
    message: str
    status: AlertStatus
    labels: Dict[str, str]
    fired_at: datetime
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "status": self.status.value,
            "labels": self.labels,
            "fired_at": self.fired_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class AlertManager:
    """
    Alert rule management and evaluation.

    Features:
    - Rule definition
    - Threshold evaluation
    - Alert state management
    - Notification hooks
    """

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._notification_handlers: List[Callable[[Alert], None]] = []

        # For "for" duration tracking
        self._condition_start: Dict[str, datetime] = {}

    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self._rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str):
        """Remove alert rule"""
        self._rules.pop(name, None)
        self._condition_start.pop(name, None)

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler"""
        self._notification_handlers.append(handler)

    def evaluate(self):
        """Evaluate all alert rules"""
        now = datetime.now(timezone.utc)

        for name, rule in self._rules.items():
            if not rule.enabled:
                continue

            try:
                condition_met = rule.condition()
            except Exception as e:
                logger.error(f"Error evaluating rule {name}: {e}")
                continue

            if condition_met:
                # Track duration
                if name not in self._condition_start:
                    self._condition_start[name] = now

                duration = (now - self._condition_start[name]).total_seconds()

                if duration >= rule.for_duration:
                    # Fire alert if not already firing
                    if name not in self._active_alerts:
                        self._fire_alert(rule)
            else:
                # Condition no longer met
                self._condition_start.pop(name, None)

                # Resolve if firing
                if name in self._active_alerts:
                    self._resolve_alert(name)

    def _fire_alert(self, rule: AlertRule):
        """Fire an alert"""
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=rule.message,
            status=AlertStatus.FIRING,
            labels=rule.labels,
            fired_at=datetime.now(timezone.utc),
        )

        self._active_alerts[rule.name] = alert

        logger.warning(f"Alert fired: {rule.name} - {rule.message}")

        # Notify handlers
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")

    def _resolve_alert(self, name: str):
        """Resolve an alert"""
        if name in self._active_alerts:
            alert = self._active_alerts.pop(name)
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)

            self._alert_history.append(alert)

            logger.info(f"Alert resolved: {name}")

            # Notify handlers
            for handler in self._notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Notification handler error: {e}")

    def silence_alert(self, name: str, duration_minutes: int = 60):
        """Silence an alert temporarily"""
        if name in self._active_alerts:
            self._active_alerts[name].status = AlertStatus.SILENCED
            logger.info(f"Alert silenced: {name} for {duration_minutes} minutes")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self._active_alerts.values())

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity"""
        return [
            a for a in self._active_alerts.values()
            if a.severity == severity
        ]

    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history"""
        return [a.to_dict() for a in self._alert_history[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        by_severity = {}
        for alert in self._active_alerts.values():
            sev = alert.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "active_count": len(self._active_alerts),
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "by_severity": by_severity,
            "history_count": len(self._alert_history),
        }
