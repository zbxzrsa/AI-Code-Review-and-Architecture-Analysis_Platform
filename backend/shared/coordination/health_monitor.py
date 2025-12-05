"""
Health Monitor

Continuous V2 production health monitoring:
- Performance metrics (latency, throughput)
- Quality metrics (accuracy, false positive/negative rates)
- Cost metrics (API costs, infrastructure)
- User experience metrics
- Auto-remediation triggers
"""

import asyncio
import copy  # FIXED: For deep copy of thresholds
import logging
import uuid  # FIXED: Moved from inside function
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from collections import deque

from .event_types import EventType, Version, VersionEvent, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class AlertThreshold:
    """Alert threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "gt"  # gt, lt, eq
    duration_minutes: int = 5
    auto_remediate: bool = False
    remediation_action: Optional[str] = None


@dataclass
class HealthAlert:
    """Health alert instance."""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    message: str
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    auto_remediated: bool = False


@dataclass 
class HealthMetrics:
    """Current health metrics snapshot."""
    timestamp: datetime
    
    # Performance
    latency_p50_ms: float = 0
    latency_p95_ms: float = 0
    latency_p99_ms: float = 0
    throughput_rps: float = 0
    error_rate: float = 0
    
    # Quality
    accuracy_rate: float = 1.0
    false_positive_rate: float = 0
    false_negative_rate: float = 0
    
    # Cost
    ai_api_cost_hourly: float = 0
    infrastructure_cost_hourly: float = 0
    cost_per_review: float = 0
    
    # User Experience
    user_satisfaction_score: float = 5.0
    feature_adoption_rate: float = 0
    
    # Resources
    cpu_utilization: float = 0
    memory_utilization: float = 0
    gpu_utilization: float = 0


class HealthMonitor:
    """
    Continuous V2 production health monitoring.
    
    Implements:
    - Real-time metric collection
    - Multi-level alerting (info, warning, critical, page)
    - Auto-remediation triggers
    - Historical trend analysis
    """
    
    DEFAULT_THRESHOLDS = [
        # Performance
        AlertThreshold("error_rate", 0.02, 0.05, "gt", 5, True, "rollback"),
        AlertThreshold("latency_p95_ms", 3000, 10000, "gt", 5, True, "scale_up"),
        AlertThreshold("throughput_rps", 50, 20, "lt", 5, False, None),
        
        # Quality
        AlertThreshold("accuracy_rate", 0.90, 0.85, "lt", 15, False, None),
        AlertThreshold("false_positive_rate", 0.20, 0.30, "gt", 15, False, None),
        
        # Cost
        AlertThreshold("cost_per_review", 0.30, 0.50, "gt", 60, False, None),
        
        # Resources
        AlertThreshold("cpu_utilization", 0.70, 0.90, "gt", 10, True, "scale_up"),
        AlertThreshold("memory_utilization", 0.80, 0.95, "gt", 10, True, "scale_up"),
    ]
    
    def __init__(
        self,
        event_bus = None,
        metrics_client = None,
        check_interval_seconds: int = 60,
        history_window_hours: int = 24,
    ):
        self.event_bus = event_bus
        self.metrics_client = metrics_client
        self.check_interval = check_interval_seconds
        self.history_window = history_window_hours
        
        # Thresholds - FIXED: Use deep copy to avoid modifying class defaults
        self._thresholds: List[AlertThreshold] = copy.deepcopy(self.DEFAULT_THRESHOLDS)
        
        # Metric history - FIXED: Calculate maxlen based on check interval
        maxlen = (history_window_hours * 3600) // check_interval_seconds
        self._history: deque = deque(maxlen=maxlen)
        
        # Active alerts
        self._active_alerts: Dict[str, HealthAlert] = {}
        
        # Alert handlers
        self._alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            AlertSeverity.INFO: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.CRITICAL: [],
            AlertSeverity.PAGE: [],
        }
        
        # Auto-remediation handlers
        self._remediation_handlers: Dict[str, Callable] = {}
        
        # Monitoring state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")
    
    async def stop(self):
        """Stop health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                raise  # Re-raise after cleanup
        logger.info("Health monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics - FIXED: Add timeout to prevent hangs
                try:
                    metrics = await asyncio.wait_for(
                        self._collect_metrics(),
                        timeout=max(1, self.check_interval - 5)  # Leave buffer
                    )
                except asyncio.TimeoutError:
                    logger.warning("Metrics collection timed out")
                    await asyncio.sleep(self.check_interval)
                    continue
                self._history.append(metrics)
                
                # Check thresholds
                await self._check_thresholds(metrics)
                
                # Check for resolved alerts
                await self._check_resolved_alerts(metrics)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_metrics(self) -> HealthMetrics:
        """Collect current health metrics."""
        # In production, fetch from Prometheus/metrics service
        if self.metrics_client:
            return await self.metrics_client.get_current_metrics()
        
        # Mock metrics for demo
        return HealthMetrics(
            timestamp=datetime.now(timezone.utc),
            latency_p50_ms=150,
            latency_p95_ms=800,
            latency_p99_ms=1500,
            throughput_rps=100,
            error_rate=0.01,
            accuracy_rate=0.95,
            false_positive_rate=0.10,
            false_negative_rate=0.05,
            ai_api_cost_hourly=5.0,
            infrastructure_cost_hourly=2.0,
            cost_per_review=0.07,
            user_satisfaction_score=4.5,
            cpu_utilization=0.45,
            memory_utilization=0.60,
            gpu_utilization=0.30,
        )
    
    async def _check_thresholds(self, metrics: HealthMetrics):
        """Check metrics against thresholds."""
        for threshold in self._thresholds:
            value = getattr(metrics, threshold.metric_name, None)
            if value is None:
                continue
            
            violated = self._check_violation(value, threshold)
            
            if violated:
                severity = self._determine_severity(value, threshold)
                await self._handle_threshold_violation(
                    threshold, value, severity, metrics
                )
    
    def _check_violation(
        self,
        value: float,
        threshold: AlertThreshold,
    ) -> bool:
        """Check if threshold is violated."""
        if threshold.comparison == "gt":
            return value > threshold.warning_threshold
        elif threshold.comparison == "lt":
            return value < threshold.warning_threshold
        elif threshold.comparison == "eq":
            return abs(value - threshold.warning_threshold) < 0.001
        return False
    
    def _determine_severity(
        self,
        value: float,
        threshold: AlertThreshold,
    ) -> AlertSeverity:
        """Determine alert severity."""
        if threshold.comparison == "gt":
            if value > threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            return AlertSeverity.WARNING
        elif threshold.comparison == "lt":
            if value < threshold.critical_threshold:
                return AlertSeverity.CRITICAL
            return AlertSeverity.WARNING
        return AlertSeverity.WARNING
    
    async def _handle_threshold_violation(
        self,
        threshold: AlertThreshold,
        value: float,
        severity: AlertSeverity,
        metrics: HealthMetrics,  # noqa: ARG002 - Reserved for alert context
    ):
        """Handle threshold violation."""
        alert_key = f"{threshold.metric_name}:{severity.value}"
        
        # Check if alert already exists
        if alert_key in self._active_alerts:
            return
        
        # Create alert - FIXED: uuid import moved to module level
        alert = HealthAlert(
            alert_id=str(uuid.uuid4()),
            metric_name=threshold.metric_name,
            severity=severity,
            current_value=value,
            threshold_value=threshold.warning_threshold,
            message=f"{threshold.metric_name} is {value:.2f}, threshold is {threshold.warning_threshold:.2f}",
            triggered_at=datetime.now(timezone.utc),
        )
        
        self._active_alerts[alert_key] = alert
        
        logger.warning(f"Alert triggered: {alert.message}")
        
        # Emit event
        await self._emit_event(
            EventType.MONITORING_ALERT,
            {
                "alert_id": alert.alert_id,
                "metric_name": threshold.metric_name,
                "severity": severity.value,
                "value": value,
                "threshold": threshold.warning_threshold,
                "message": alert.message,
            },
        )
        
        # Call alert handlers - FIXED: Add timeout for handlers
        for handler in self._alert_handlers[severity]:
            try:
                await asyncio.wait_for(handler(alert), timeout=30)
            except asyncio.TimeoutError:
                logger.error(f"Alert handler timed out for {alert.metric_name}")
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Auto-remediation
        if threshold.auto_remediate and severity == AlertSeverity.CRITICAL:
            await self._trigger_auto_remediation(threshold, alert)
    
    async def _trigger_auto_remediation(
        self,
        threshold: AlertThreshold,
        alert: HealthAlert,
    ):
        """Trigger auto-remediation action."""
        action = threshold.remediation_action
        if not action:
            return
        
        logger.info(f"Triggering auto-remediation: {action}")
        
        handler = self._remediation_handlers.get(action)
        if handler:
            try:
                await handler(alert)
                alert.auto_remediated = True
                
                await self._emit_event(
                    EventType.AUTO_REMEDIATION,
                    {
                        "alert_id": alert.alert_id,
                        "action": action,
                        "metric_name": alert.metric_name,
                    },
                )
            except Exception as e:
                logger.error(f"Auto-remediation failed: {e}")
    
    async def _check_resolved_alerts(self, metrics: HealthMetrics):
        """Check if any alerts are resolved."""
        resolved = []
        
        for key, alert in self._active_alerts.items():
            threshold = next(
                (t for t in self._thresholds if t.metric_name == alert.metric_name),
                None
            )
            if not threshold:
                continue
            
            value = getattr(metrics, alert.metric_name, None)
            if value is None:
                continue
            
            still_violated = self._check_violation(value, threshold)
            
            if not still_violated:
                alert.resolved_at = datetime.now(timezone.utc)
                resolved.append(key)
                
                logger.info(f"Alert resolved: {alert.metric_name}")
                
                await self._emit_event(
                    EventType.MONITORING_RECOVERY,
                    {
                        "alert_id": alert.alert_id,
                        "metric_name": alert.metric_name,
                        "resolved_at": alert.resolved_at.isoformat(),
                    },
                )
        
        for key in resolved:
            del self._active_alerts[key]
    
    def register_alert_handler(
        self,
        severity: AlertSeverity,
        handler: Callable[[HealthAlert], Awaitable[None]],
    ):
        """Register alert handler for severity level."""
        self._alert_handlers[severity].append(handler)
    
    def register_remediation_handler(
        self,
        action: str,
        handler: Callable[[HealthAlert], Awaitable[None]],
    ):
        """Register auto-remediation handler."""
        self._remediation_handlers[action] = handler
    
    def add_threshold(self, threshold: AlertThreshold):
        """Add custom threshold."""
        self._thresholds.append(threshold)
    
    def get_current_health(self) -> Optional[HealthMetrics]:
        """Get most recent health metrics."""
        if self._history:
            return self._history[-1]
        return None
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Get active alerts."""
        return list(self._active_alerts.values())
    
    def get_health_trend(
        self,
        metric_name: str,
        hours: int = 1,
    ) -> List[tuple]:
        """Get metric trend over time."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            (m.timestamp, getattr(m, metric_name, 0))
            for m in self._history
            if m.timestamp >= cutoff
        ]
    
    def calculate_slo_compliance(self) -> Dict[str, Any]:
        """Calculate SLO compliance."""
        if not self._history:
            return {"compliant": True, "details": {}}
        
        recent = list(self._history)[-60:]  # Last hour
        
        # Availability SLO: Error rate < 2%
        avg_error_rate = sum(m.error_rate for m in recent) / len(recent)
        availability_ok = avg_error_rate < 0.02
        
        # Latency SLO: p95 < 3s
        avg_latency = sum(m.latency_p95_ms for m in recent) / len(recent)
        latency_ok = avg_latency < 3000
        
        return {
            "compliant": availability_ok and latency_ok,
            "details": {
                "availability": {
                    "target": 0.98,
                    "actual": 1 - avg_error_rate,
                    "compliant": availability_ok,
                },
                "latency_p95_ms": {
                    "target": 3000,
                    "actual": avg_latency,
                    "compliant": latency_ok,
                },
            },
        }
    
    async def _emit_event(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
    ):
        """Emit event to event bus."""
        event = VersionEvent(
            event_type=event_type,
            version=Version.V2_PRODUCTION,
            payload=payload,
            source="health-monitor",
        )
        
        if self.event_bus:
            await self.event_bus.publish(event_type.value, event.to_dict())
