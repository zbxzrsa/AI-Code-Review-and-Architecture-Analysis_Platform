"""
Alert Management

Automatic alerting for system anomalies.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    LATENCY_HIGH = "latency_high"
    AVAILABILITY_LOW = "availability_low"
    ERROR_RATE_HIGH = "error_rate_high"
    MEMORY_HIGH = "memory_high"
    STORAGE_FULL = "storage_full"
    COLLECTION_FAILED = "collection_failed"
    QUALITY_LOW = "quality_low"
    TECHNOLOGY_DEPRECATED = "technology_deprecated"


@dataclass
class Alert:
    """
    System alert.
    
    Attributes:
        alert_id: Unique alert identifier
        alert_type: Type of alert
        severity: Alert severity
        message: Human-readable message
        details: Additional context
        created_at: When alert was created
        acknowledged: Whether alert has been acknowledged
        resolved: Whether alert has been resolved
    """
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    alert_type: AlertType
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: int = 300  # 5 minutes between same alerts


class AlertManager:
    """
    Manages system alerts and notifications.
    
    Features:
    - Rule-based alert triggering
    - Alert deduplication
    - Notification dispatch
    - Alert lifecycle management
    """
    
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._rules: List[AlertRule] = []
        self._last_alert_times: Dict[str, float] = {}
        self._notification_handlers: List[Callable[[Alert], None]] = []
        self._alert_counter = 0
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default alerting rules."""
        self.add_rule(AlertRule(
            name="high_latency",
            alert_type=AlertType.LATENCY_HIGH,
            condition=lambda m: m.get("avg_latency_ms", 0) > 500,
            severity=AlertSeverity.WARNING,
            message_template="Processing latency {avg_latency_ms:.0f}ms exceeds 500ms threshold",
        ))
        
        self.add_rule(AlertRule(
            name="low_availability",
            alert_type=AlertType.AVAILABILITY_LOW,
            condition=lambda m: m.get("availability", 100) < 99.9,
            severity=AlertSeverity.CRITICAL,
            message_template="System availability {availability:.2f}% below 99.9% SLA",
        ))
        
        self.add_rule(AlertRule(
            name="high_memory",
            alert_type=AlertType.MEMORY_HIGH,
            condition=lambda m: m.get("memory_percent", 0) > 70,
            severity=AlertSeverity.WARNING,
            message_template="Memory usage {memory_percent:.1f}% exceeds 70% threshold",
        ))
        
        self.add_rule(AlertRule(
            name="high_error_rate",
            alert_type=AlertType.ERROR_RATE_HIGH,
            condition=lambda m: m.get("error_rate", 0) > 0.05,
            severity=AlertSeverity.ERROR,
            message_template="Error rate {error_rate:.1%} exceeds 5% threshold",
        ))
        
        self.add_rule(AlertRule(
            name="low_quality",
            alert_type=AlertType.QUALITY_LOW,
            condition=lambda m: m.get("quality_pass_rate", 1) < 0.5,
            severity=AlertSeverity.WARNING,
            message_template="Quality pass rate {quality_pass_rate:.1%} below 50%",
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alerting rule."""
        self._rules.append(rule)
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler."""
        self._notification_handlers.append(handler)
    
    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """
        Evaluate all rules against current metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            List of triggered alerts
        """
        import time
        
        triggered = []
        now = time.time()
        
        for rule in self._rules:
            try:
                if rule.condition(metrics):
                    # Check cooldown
                    last_time = self._last_alert_times.get(rule.name, 0)
                    if now - last_time < rule.cooldown_seconds:
                        continue
                    
                    # Create alert
                    alert = self._create_alert(rule, metrics)
                    triggered.append(alert)
                    self._last_alert_times[rule.name] = now
                    
            except Exception as e:
                logger.error(f"Rule evaluation error for {rule.name}: {e}")
        
        return triggered
    
    def _create_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> Alert:
        """Create an alert from a rule."""
        self._alert_counter += 1
        alert_id = f"alert_{self._alert_counter:06d}"
        
        # Format message with metrics
        try:
            message = rule.message_template.format(**metrics)
        except KeyError:
            message = rule.message_template
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            message=message,
            details={"metrics": metrics, "rule": rule.name},
        )
        
        self._alerts[alert_id] = alert
        
        # Dispatch notifications
        self._dispatch_notifications(alert)
        
        logger.warning(f"Alert triggered: {alert.message}")
        return alert
    
    def _dispatch_notifications(self, alert: Alert):
        """Dispatch alert to notification handlers."""
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")
    
    def acknowledge(self, alert_id: str, user: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert to acknowledge
            user: User acknowledging the alert
            
        Returns:
            True if successful
        """
        alert = self._alerts.get(alert_id)
        if alert:
            alert.acknowledged = True
            alert.acknowledged_by = user
            return True
        return False
    
    def resolve(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert to resolve
            
        Returns:
            True if successful
        """
        alert = self._alerts.get(alert_id)
        if alert:
            alert.resolved = True
            alert.resolved_at = datetime.now(timezone.utc)
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        return [a for a in self._alerts.values() if not a.resolved]
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get a specific alert."""
        return self._alerts.get(alert_id)
    
    def clear_resolved(self, older_than_hours: int = 24):
        """Clear resolved alerts older than specified hours."""
        from datetime import timedelta
        
        cutoff = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        
        to_remove = [
            aid for aid, alert in self._alerts.items()
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff
        ]
        
        for aid in to_remove:
            del self._alerts[aid]
        
        return len(to_remove)
