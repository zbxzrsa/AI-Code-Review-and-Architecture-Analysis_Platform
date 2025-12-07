"""
Alert Manager - Alert Generation and Routing

Manages alert generation, routing, and notification delivery.

Features:
- Multi-channel alerting (Slack, Email, PagerDuty)
- Alert deduplication
- Alert escalation
- Alert history
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert message."""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str

    created_at: str
    fingerprint: str

    # Routing
    channels: List[AlertChannel] = field(default_factory=list)

    # Context
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Delivery
    delivered: bool = False
    delivery_attempts: int = 0
    delivered_at: Optional[str] = None


class AlertManager:
    """
    Manages alert lifecycle.

    Features:
    - Alert generation
    - Deduplication
    - Routing
    - Delivery
    - History tracking
    """

    def __init__(
        self,
        dedup_window_seconds: int = 300,
        max_history: int = 10000
    ):
        self.dedup_window = timedelta(seconds=dedup_window_seconds)
        self.max_history = max_history

        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # Routing rules
        self.routing_rules: Dict[AlertSeverity, List[AlertChannel]] = {
            AlertSeverity.INFO: [AlertChannel.LOG],
            AlertSeverity.WARNING: [AlertChannel.SLACK, AlertChannel.LOG],
            AlertSeverity.ERROR: [AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.LOG],
            AlertSeverity.CRITICAL: [
                AlertChannel.PAGERDUTY,
                AlertChannel.SLACK,
                AlertChannel.EMAIL,
                AlertChannel.LOG
            ]
        }

        # Statistics
        self.stats = {
            "alerts_generated": 0,
            "alerts_deduplicated": 0,
            "alerts_delivered": 0,
            "delivery_failures": 0
        }

    async def generate_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None
    ) -> Optional[Alert]:
        """Generate and route an alert."""
        # Create fingerprint for deduplication
        fingerprint = self._create_fingerprint(title, source, labels or {})

        # Check for duplicate
        if self._is_duplicate(fingerprint):
            self.stats["alerts_deduplicated"] += 1
            logger.debug(f"Alert deduplicated: {title}")
            return None

        # Create alert
        alert = Alert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            severity=severity,
            title=title,
            message=message,
            source=source,
            created_at=datetime.now().isoformat(),
            fingerprint=fingerprint,
            labels=labels or {},
            annotations=annotations or {}
        )

        # Determine channels
        alert.channels = self.routing_rules.get(severity, [AlertChannel.LOG])

        # Store alert
        self.active_alerts[fingerprint] = alert
        self.alert_history.append(alert)

        # Trim history
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        self.stats["alerts_generated"] += 1

        # Deliver alert
        await self._deliver_alert(alert)

        logger.info(
            f"Alert generated: [{severity.value}] {title} → "
            f"{[c.value for c in alert.channels]}"
        )

        return alert

    def _create_fingerprint(
        self,
        title: str,
        source: str,
        labels: Dict[str, str]
    ) -> str:
        """Create fingerprint for deduplication."""
        data = f"{title}:{source}:{sorted(labels.items())}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _is_duplicate(self, fingerprint: str) -> bool:
        """Check if alert is duplicate within dedup window."""
        if fingerprint not in self.active_alerts:
            return False

        alert = self.active_alerts[fingerprint]
        created_at = datetime.fromisoformat(alert.created_at)

        # Check if within dedup window
        if datetime.now() - created_at < self.dedup_window:
            return True
        else:
            # Remove from active alerts
            del self.active_alerts[fingerprint]
            return False

    async def _deliver_alert(self, alert: Alert) -> None:
        """Deliver alert to configured channels."""
        delivery_results = []

        for channel in alert.channels:
            try:
                success = await self._deliver_to_channel(alert, channel)
                delivery_results.append(success)
                alert.delivery_attempts += 1
            except Exception as e:
                logger.error(f"Alert delivery failed ({channel.value}): {e}")
                delivery_results.append(False)
                self.stats["delivery_failures"] += 1

        alert.delivered = any(delivery_results)
        if alert.delivered:
            alert.delivered_at = datetime.now().isoformat()
            self.stats["alerts_delivered"] += 1

    async def _deliver_to_channel(
        self,
        alert: Alert,
        channel: AlertChannel
    ) -> bool:
        """Deliver alert to specific channel."""
        if channel == AlertChannel.LOG:
            return self._log_alert(alert)
        elif channel == AlertChannel.SLACK:
            return await self._send_to_slack(alert)
        elif channel == AlertChannel.EMAIL:
            return await self._send_email(alert)
        elif channel == AlertChannel.PAGERDUTY:
            return await self._send_to_pagerduty(alert)
        else:
            logger.warning(f"Unknown channel: {channel}")
            return False

    def _log_alert(self, alert: Alert) -> bool:
        """Log alert to system logs."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)

        logger.log(
            log_level,
            f"[ALERT] {alert.title}: {alert.message}"
        )

        return True

    async def _send_to_slack(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        # Simulate Slack API call
        await asyncio.sleep(0.1)

        # In production:
        # async with aiohttp.ClientSession() as session:
        #     await session.post(slack_webhook_url, json={...})

        logger.info(f"Sent to Slack: {alert.title}")
        return True

    async def _send_email(self, alert: Alert) -> bool:
        """Send alert via email."""
        # Simulate email send
        await asyncio.sleep(0.1)

        # In production: Use SMTP or email service

        logger.info(f"Sent email: {alert.title}")
        return True

    async def _send_to_pagerduty(self, alert: Alert) -> bool:
        """Send alert to PagerDuty."""
        # Simulate PagerDuty API call
        await asyncio.sleep(0.1)

        # In production: PagerDuty Events API

        logger.info(f"Sent to PagerDuty: {alert.title}")
        return True

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get currently active alerts."""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        by_severity = defaultdict(int)
        for alert in self.alert_history[-1000:]:
            by_severity[alert.severity.value] += 1

        return {
            **self.stats,
            "active_alerts": len(self.active_alerts),
            "by_severity": dict(by_severity),
            "delivery_success_rate": (
                self.stats["alerts_delivered"] / self.stats["alerts_generated"]
                if self.stats["alerts_generated"] > 0 else 1.0
            )
        }
