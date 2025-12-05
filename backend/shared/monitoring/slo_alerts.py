"""
SLO Alerting Module

Implements:
- SLO violation detection
- Multi-level alerting (warning, critical)
- Error budget tracking
- PagerDuty/Slack integration
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    PAGE = "page"  # Immediate attention required


class AlertState(str, Enum):
    """Alert state."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class SLODefinition:
    """SLO definition."""
    name: str
    description: str
    target: float  # e.g., 0.999 for 99.9%
    window_hours: int = 720  # 30 days default
    warning_threshold: float = 0.9  # Alert when 90% of budget consumed
    critical_threshold: float = 0.75  # Critical when 75% of budget remains
    measurement: str = "availability"  # availability, latency, error_rate


@dataclass
class ErrorBudget:
    """Error budget tracking."""
    slo_name: str
    total_budget_minutes: float
    consumed_minutes: float
    remaining_minutes: float
    remaining_percentage: float
    burn_rate: float  # Budget consumption rate
    projected_exhaustion: Optional[datetime] = None
    
    @property
    def is_exhausted(self) -> bool:
        return self.remaining_minutes <= 0


@dataclass
class Alert:
    """Alert instance."""
    id: str
    slo_name: str
    severity: AlertSeverity
    state: AlertState
    title: str
    description: str
    current_value: float
    target_value: float
    started_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricWindow:
    """Sliding window for metrics."""
    
    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self._data: deque = deque()
    
    def add(self, value: float, timestamp: Optional[datetime] = None):
        """Add data point."""
        ts = timestamp or datetime.now(timezone.utc)
        self._data.append((ts, value))
        self._cleanup()
    
    def _cleanup(self):
        """Remove expired data points."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.window_minutes)
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()
    
    def average(self) -> float:
        """Calculate average over window."""
        self._cleanup()
        if not self._data:
            return 0.0
        return sum(v for _, v in self._data) / len(self._data)
    
    def percentile(self, p: float) -> float:
        """Calculate percentile over window."""
        self._cleanup()
        if not self._data:
            return 0.0
        
        values = sorted(v for _, v in self._data)
        idx = int(len(values) * p / 100)
        return values[min(idx, len(values) - 1)]
    
    def count(self) -> int:
        """Count data points in window."""
        self._cleanup()
        return len(self._data)


class SLOAlertManager:
    """
    SLO alert management.
    
    Features:
    - Multi-level alerting
    - Error budget tracking
    - Alert deduplication
    - Integration with notification channels
    """
    
    def __init__(
        self,
        pagerduty_key: Optional[str] = None,
        slack_webhook: Optional[str] = None,
        alert_cooldown_minutes: int = 15,
    ):
        self.pagerduty_key = pagerduty_key or os.getenv("PAGERDUTY_API_KEY")
        self.slack_webhook = slack_webhook or os.getenv("SLACK_WEBHOOK_URL")
        self.alert_cooldown = timedelta(minutes=alert_cooldown_minutes)
        
        # SLO definitions
        self._slos: Dict[str, SLODefinition] = {}
        
        # Metric windows
        self._metrics: Dict[str, MetricWindow] = {}
        
        # Active alerts
        self._alerts: Dict[str, Alert] = {}
        
        # Alert history
        self._alert_history: List[Alert] = []
        
        # Error budgets
        self._error_budgets: Dict[str, ErrorBudget] = {}
        
        # Last alert time (for cooldown)
        self._last_alert: Dict[str, datetime] = {}
        
        # Alert callbacks
        self._callbacks: List[Callable[[Alert], Awaitable[None]]] = []
        
        # Initialize default SLOs
        self._init_default_slos()
    
    def _init_default_slos(self):
        """Initialize default SLO definitions."""
        default_slos = [
            SLODefinition(
                name="availability",
                description="Service availability",
                target=0.9999,  # 99.99%
                measurement="availability",
            ),
            SLODefinition(
                name="latency_p99",
                description="99th percentile latency",
                target=0.95,  # 95% of requests under target
                measurement="latency",
            ),
            SLODefinition(
                name="error_rate",
                description="Error rate",
                target=0.999,  # 99.9% success rate
                measurement="error_rate",
            ),
        ]
        
        for slo in default_slos:
            self._slos[slo.name] = slo
            self._metrics[slo.name] = MetricWindow(window_minutes=5)
    
    def define_slo(self, slo: SLODefinition):
        """Define or update SLO."""
        self._slos[slo.name] = slo
        if slo.name not in self._metrics:
            self._metrics[slo.name] = MetricWindow()
    
    async def record_metric(
        self,
        slo_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ):
        """Record metric value."""
        if slo_name not in self._metrics:
            logger.warning(f"Unknown SLO: {slo_name}")
            return
        
        self._metrics[slo_name].add(value, timestamp)
        
        # Check for violations
        await self._check_slo(slo_name)
    
    async def _check_slo(self, slo_name: str):
        """Check SLO and trigger alerts if needed."""
        slo = self._slos.get(slo_name)
        if not slo:
            return
        
        window = self._metrics[slo_name]
        current = window.average()
        
        # Check if violating SLO
        if current < slo.target:
            # Calculate severity
            gap = slo.target - current
            gap_percentage = gap / slo.target
            
            if gap_percentage > 0.1:  # More than 10% below target
                severity = AlertSeverity.CRITICAL
            elif gap_percentage > 0.05:  # More than 5% below target
                severity = AlertSeverity.WARNING
            else:
                severity = AlertSeverity.INFO
            
            await self._fire_alert(slo, current, severity)
        else:
            # Resolve any existing alert
            await self._resolve_alert(slo_name)
    
    async def _fire_alert(
        self,
        slo: SLODefinition,
        current_value: float,
        severity: AlertSeverity,
    ):
        """Fire or update alert."""
        alert_id = f"slo:{slo.name}"
        
        # Check cooldown
        last = self._last_alert.get(alert_id)
        if last and datetime.now(timezone.utc) - last < self.alert_cooldown:
            return
        
        # Create or update alert
        existing = self._alerts.get(alert_id)
        
        if existing and existing.state == AlertState.FIRING:
            # Update existing alert
            existing.current_value = current_value
            existing.severity = max(existing.severity, severity, key=lambda s: s.value)
            return
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            slo_name=slo.name,
            severity=severity,
            state=AlertState.FIRING,
            title=f"SLO Violation: {slo.name}",
            description=f"{slo.description} is below target. Current: {current_value:.4f}, Target: {slo.target:.4f}",
            current_value=current_value,
            target_value=slo.target,
            started_at=datetime.now(timezone.utc),
        )
        
        self._alerts[alert_id] = alert
        self._last_alert[alert_id] = datetime.now(timezone.utc)
        
        logger.warning(f"Alert fired: {alert.title}")
        
        # Send notifications
        await self._send_notifications(alert)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    async def _resolve_alert(self, slo_name: str):
        """Resolve alert."""
        alert_id = f"slo:{slo_name}"
        alert = self._alerts.get(alert_id)
        
        if alert and alert.state == AlertState.FIRING:
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)
            
            self._alert_history.append(alert)
            del self._alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.title}")
            
            # Send resolution notification
            await self._send_notifications(alert)
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        # PagerDuty for critical alerts
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.PAGE]:
            await self._send_pagerduty(alert)
        
        # Slack for all alerts
        await self._send_slack(alert)
    
    async def _send_pagerduty(self, alert: Alert):
        """Send PagerDuty notification."""
        if not self.pagerduty_key:
            return
        
        try:
            import httpx
            
            payload = {
                "routing_key": self.pagerduty_key,
                "event_action": "trigger" if alert.state == AlertState.FIRING else "resolve",
                "dedup_key": alert.id,
                "payload": {
                    "summary": alert.title,
                    "severity": alert.severity.value,
                    "source": "code-review-platform",
                    "custom_details": {
                        "description": alert.description,
                        "current_value": alert.current_value,
                        "target_value": alert.target_value,
                    },
                },
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    timeout=10,
                )
                
                if response.status_code != 202:
                    logger.error(f"PagerDuty notification failed: {response.text}")
                    
        except Exception as e:
            logger.error(f"PagerDuty notification error: {e}")
    
    async def _send_slack(self, alert: Alert):
        """Send Slack notification."""
        if not self.slack_webhook:
            return
        
        try:
            import httpx
            
            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffcc00",
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.PAGE: "#ff0000",
            }.get(alert.severity, "#808080")
            
            status_emoji = "🔥" if alert.state == AlertState.FIRING else "✅"
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"{status_emoji} {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {"title": "SLO", "value": alert.slo_name, "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Current", "value": f"{alert.current_value:.4f}", "short": True},
                        {"title": "Target", "value": f"{alert.target_value:.4f}", "short": True},
                    ],
                    "footer": "SLO Alert System",
                    "ts": int(alert.started_at.timestamp()),
                }],
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.slack_webhook,
                    json=payload,
                    timeout=10,
                )
                
                if response.status_code != 200:
                    logger.error(f"Slack notification failed: {response.text}")
                    
        except Exception as e:
            logger.error(f"Slack notification error: {e}")
    
    def calculate_error_budget(self, slo_name: str) -> ErrorBudget:
        """Calculate error budget for SLO."""
        slo = self._slos.get(slo_name)
        if not slo:
            raise ValueError(f"Unknown SLO: {slo_name}")
        
        # Total allowed downtime in window
        window_minutes = slo.window_hours * 60
        allowed_failure_rate = 1 - slo.target
        total_budget = window_minutes * allowed_failure_rate
        
        # Calculate consumed budget (simplified)
        window = self._metrics[slo_name]
        current = window.average()
        failure_rate = max(0, slo.target - current)
        
        # Estimate consumed budget
        consumed = failure_rate * window_minutes
        remaining = max(0, total_budget - consumed)
        remaining_pct = (remaining / total_budget) * 100 if total_budget > 0 else 0
        
        # Calculate burn rate
        burn_rate = consumed / (window_minutes / (slo.window_hours / 24)) if window_minutes > 0 else 0
        
        # Project exhaustion
        projected = None
        if burn_rate > 0 and remaining > 0:
            days_until_exhaustion = remaining / (burn_rate * 24 * 60)
            projected = datetime.now(timezone.utc) + timedelta(days=days_until_exhaustion)
        
        budget = ErrorBudget(
            slo_name=slo_name,
            total_budget_minutes=total_budget,
            consumed_minutes=consumed,
            remaining_minutes=remaining,
            remaining_percentage=remaining_pct,
            burn_rate=burn_rate,
            projected_exhaustion=projected,
        )
        
        self._error_budgets[slo_name] = budget
        return budget
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self._alerts.values())
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge alert."""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.state = AlertState.ACKNOWLEDGED
            alert.acknowledged_by = user
            alert.acknowledged_at = datetime.now(timezone.utc)
            return True
        return False
    
    def on_alert(self, callback: Callable[[Alert], Awaitable[None]]):
        """Register alert callback."""
        self._callbacks.append(callback)
    
    def get_alert_history(
        self,
        slo_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alert history."""
        history = self._alert_history
        
        if slo_name:
            history = [a for a in history if a.slo_name == slo_name]
        
        return sorted(history, key=lambda a: a.started_at, reverse=True)[:limit]
