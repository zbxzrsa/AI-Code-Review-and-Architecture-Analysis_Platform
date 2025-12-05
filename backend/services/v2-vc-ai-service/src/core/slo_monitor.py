"""
V2 VC-AI SLO Monitor

Real-time SLO monitoring and error budget tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from collections import deque
import statistics

from ..models.slo_models import (
    SLOMetrics,
    SLOStatus,
    ErrorBudget,
    BudgetZone,
    SLOState,
    AlertSeverity,
    ActiveAlert,
    AlertConfig,
)
from ..config.slo_config import SLO_DEFINITIONS, ALERTING_CONFIG


logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    """Individual metric sample"""
    timestamp: datetime
    value: float


@dataclass
class MetricWindow:
    """Sliding window of metric samples"""
    samples: deque = field(default_factory=lambda: deque(maxlen=10000))
    window_seconds: float = 300.0  # 5-minute default window
    
    def add(self, value: float) -> None:
        self.samples.append(MetricSample(datetime.now(timezone.utc), value))
    
    def get_values_in_window(self) -> List[float]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)
        return [s.value for s in self.samples if s.timestamp >= cutoff]
    
    def mean(self) -> Optional[float]:
        values = self.get_values_in_window()
        return statistics.mean(values) if values else None
    
    def percentile(self, p: float) -> Optional[float]:
        values = sorted(self.get_values_in_window())
        if not values:
            return None
        idx = int(len(values) * p / 100)
        return values[min(idx, len(values) - 1)]


class SLOMonitor:
    """
    Production-grade SLO monitoring system.
    
    Features:
    - Real-time metric tracking
    - Error budget calculation
    - Alert triggering
    - Trend analysis
    """
    
    def __init__(
        self,
        service_name: str = "v2-vc-ai-service",
        on_alert: Optional[Callable[[ActiveAlert], None]] = None,
    ):
        self.service_name = service_name
        self.on_alert = on_alert
        
        # Metric windows
        self._latency_window = MetricWindow()
        self._error_window = MetricWindow()
        self._availability_window = MetricWindow()
        self._accuracy_window = MetricWindow()
        
        # Counters (30-day window)
        self._total_requests = 0
        self._successful_requests = 0
        self._error_requests = 0
        
        # Error budget tracking
        self._budget_window_start = datetime.now(timezone.utc)
        self._budget_consumed_minutes = 0.0
        
        # Active alerts
        self._active_alerts: Dict[str, ActiveAlert] = {}
        
        # SLO targets from config
        self._slo_targets = SLO_DEFINITIONS
        self._alert_configs = ALERTING_CONFIG
        
        # History (24h snapshots)
        self._history: deque = deque(maxlen=288)  # 5-min intervals for 24h
        
        logger.info(f"SLO Monitor initialized for {service_name}")
    
    async def record_request(
        self,
        latency_ms: float,
        success: bool,
        accurate: Optional[bool] = None,
    ) -> None:
        """Record a request for SLO tracking"""
        self._total_requests += 1
        
        if success:
            self._successful_requests += 1
            self._availability_window.add(1.0)
            self._error_window.add(0.0)
        else:
            self._error_requests += 1
            self._availability_window.add(0.0)
            self._error_window.add(1.0)
        
        self._latency_window.add(latency_ms)
        
        if accurate is not None:
            self._accuracy_window.add(1.0 if accurate else 0.0)
        
        # Check for SLO violations
        await self._check_slos()
    
    async def _check_slos(self) -> None:
        """Check current metrics against SLO thresholds"""
        metrics = self.get_current_metrics()
        
        # Check availability
        if not metrics.availability_compliant:
            self._trigger_alert(
                "availability_violation",
                f"Availability {metrics.availability:.4%} below target {metrics.availability_target:.4%}",
                AlertSeverity.CRITICAL,
                metrics.availability,
            )
        
        # Check latency
        if not metrics.latency_compliant:
            self._trigger_alert(
                "latency_violation",
                f"P99 latency {metrics.latency_p99_ms:.0f}ms exceeds target {metrics.latency_target_p99_ms:.0f}ms",
                AlertSeverity.CRITICAL,
                metrics.latency_p99_ms,
            )
        
        # Check error rate
        if not metrics.error_rate_compliant:
            self._trigger_alert(
                "error_rate_violation",
                f"Error rate {metrics.error_rate:.4%} exceeds target {metrics.error_rate_target:.4%}",
                AlertSeverity.CRITICAL,
                metrics.error_rate,
            )
        
        # Check accuracy
        if not metrics.accuracy_compliant:
            self._trigger_alert(
                "accuracy_violation",
                f"Accuracy {metrics.accuracy:.4%} below target {metrics.accuracy_target:.4%}",
                AlertSeverity.WARNING,
                metrics.accuracy,
            )
    
    def _trigger_alert(
        self,
        alert_id: str,
        message: str,
        severity: AlertSeverity,
        metric_value: float,
    ) -> None:
        """Trigger an alert"""
        if alert_id in self._active_alerts:
            return  # Alert already active
        
        alert = ActiveAlert(
            id=alert_id,
            config=AlertConfig(
                name=alert_id,
                description=message,
                metric=alert_id.replace("_violation", ""),
                condition="threshold_breach",
                threshold=0,
                severity=severity,
            ),
            triggered_at=datetime.now(timezone.utc),
            metric_value=metric_value,
            message=message,
        )
        
        self._active_alerts[alert_id] = alert
        logger.warning(f"SLO Alert triggered: {message}")
        
        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        if alert_id not in self._active_alerts:
            return False
        
        alert = self._active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now(timezone.utc)
        alert.duration_minutes = (alert.resolved_at - alert.triggered_at).total_seconds() / 60
        
        del self._active_alerts[alert_id]
        logger.info(f"Alert resolved: {alert_id}")
        return True
    
    def get_current_metrics(self) -> SLOMetrics:
        """Get current SLO metrics snapshot"""
        # Calculate availability
        availability_values = self._availability_window.get_values_in_window()
        availability = statistics.mean(availability_values) if availability_values else 1.0
        
        # Calculate error rate
        error_values = self._error_window.get_values_in_window()
        error_rate = statistics.mean(error_values) if error_values else 0.0
        
        # Calculate latency percentiles
        latency_p50 = self._latency_window.percentile(50) or 0
        latency_p99 = self._latency_window.percentile(99) or 0
        latency_p999 = self._latency_window.percentile(99.9) or 0
        
        # Calculate accuracy
        accuracy_values = self._accuracy_window.get_values_in_window()
        accuracy = statistics.mean(accuracy_values) if accuracy_values else 1.0
        
        # Get targets
        avail_target = self._slo_targets["availability"]["target"]
        error_target = self._slo_targets["error_rate"]["target"]
        latency_target = self._slo_targets["latency"]["p99"]["target_ms"]
        accuracy_target = self._slo_targets["accuracy"]["target"]
        
        # Check compliance
        avail_compliant = availability >= avail_target
        error_compliant = error_rate <= error_target
        latency_compliant = latency_p99 <= latency_target
        accuracy_compliant = accuracy >= accuracy_target
        
        # Determine overall state
        if not (avail_compliant and error_compliant and latency_compliant):
            overall_state = SLOState.VIOLATION
        elif not accuracy_compliant:
            overall_state = SLOState.WARNING
        else:
            overall_state = SLOState.COMPLIANT
        
        # Calculate compliance percentage
        checks = [avail_compliant, error_compliant, latency_compliant, accuracy_compliant]
        compliance_percentage = sum(checks) / len(checks) * 100
        
        return SLOMetrics(
            availability=availability,
            availability_target=avail_target,
            availability_compliant=avail_compliant,
            latency_p50_ms=latency_p50,
            latency_p99_ms=latency_p99,
            latency_p999_ms=latency_p999,
            latency_target_p99_ms=latency_target,
            latency_compliant=latency_compliant,
            error_rate=error_rate,
            error_rate_target=error_target,
            error_rate_compliant=error_compliant,
            accuracy=accuracy,
            accuracy_target=accuracy_target,
            accuracy_compliant=accuracy_compliant,
            overall_state=overall_state,
            compliance_percentage=compliance_percentage,
        )
    
    def get_error_budget(self) -> ErrorBudget:
        """Calculate current error budget status"""
        window_days = 30
        slo_target = self._slo_targets["availability"]["target"]
        
        # Total budget (minutes of allowed downtime)
        total_budget_minutes = window_days * 24 * 60 * (1 - slo_target)
        
        # Calculate consumed budget
        # For 99.99% SLO over 30 days = 4.32 minutes budget
        error_ratio = self._error_requests / max(1, self._total_requests)
        elapsed_days = (datetime.now(timezone.utc) - self._budget_window_start).days + 1
        consumed_minutes = elapsed_days * 24 * 60 * error_ratio
        
        remaining_minutes = max(0, total_budget_minutes - consumed_minutes)
        remaining_percentage = (remaining_minutes / total_budget_minutes) * 100 if total_budget_minutes > 0 else 100
        
        # Calculate burn rate
        burn_rate_per_day = consumed_minutes / max(1, elapsed_days)
        
        # Projection
        if burn_rate_per_day > 0 and remaining_minutes > 0:
            days_until_exhaustion = remaining_minutes / burn_rate_per_day
            projected_exhaustion = datetime.now(timezone.utc) + timedelta(days=days_until_exhaustion)
        else:
            days_until_exhaustion = None
            projected_exhaustion = None
        
        # Determine zone
        if remaining_percentage > 50:
            zone = BudgetZone.GREEN
        elif remaining_percentage > 25:
            zone = BudgetZone.YELLOW
        else:
            zone = BudgetZone.RED
        
        return ErrorBudget(
            window_days=window_days,
            window_start=self._budget_window_start,
            window_end=self._budget_window_start + timedelta(days=window_days),
            total_budget_minutes=total_budget_minutes,
            consumed_minutes=consumed_minutes,
            remaining_minutes=remaining_minutes,
            remaining_percentage=remaining_percentage,
            burn_rate_per_day=burn_rate_per_day,
            projected_exhaustion_date=projected_exhaustion,
            days_until_exhaustion=days_until_exhaustion,
            zone=zone,
            zone_since=self._budget_window_start,
            deployment_allowed=zone != BudgetZone.RED,
            change_freeze_recommended=zone == BudgetZone.RED,
        )
    
    def get_status(self) -> SLOStatus:
        """Get complete SLO status"""
        metrics = self.get_current_metrics()
        error_budget = self.get_error_budget()
        
        return SLOStatus(
            service_name=self.service_name,
            metrics=metrics,
            error_budget=error_budget,
            slo_history_24h=list(self._history),
            violations_24h=self._count_recent_violations(hours=24),
            violations_7d=self._count_recent_violations(hours=168),
            violations_30d=self._count_recent_violations(hours=720),
            availability_trend=self._calculate_trend("availability"),
            latency_trend=self._calculate_trend("latency"),
            error_rate_trend=self._calculate_trend("error_rate"),
        )
    
    def _count_recent_violations(self, hours: int) -> int:
        """Count violations in recent history"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return sum(
            1 for h in self._history
            if h.get("timestamp", datetime.min) >= cutoff and h.get("violation", False)
        )
    
    def _calculate_trend(self, metric: str) -> str:
        """Calculate trend for a metric"""
        # Simplified trend calculation
        values = [h.get(metric) for h in list(self._history)[-12:] if h.get(metric) is not None]
        if len(values) < 2:
            return "stable"
        
        first_half = statistics.mean(values[:len(values)//2])
        second_half = statistics.mean(values[len(values)//2:])
        
        change = (second_half - first_half) / first_half if first_half != 0 else 0
        
        if abs(change) < 0.05:
            return "stable"
        elif change > 0:
            return "improving" if metric != "error_rate" else "degrading"
        else:
            return "degrading" if metric != "error_rate" else "improving"
    
    def get_active_alerts(self) -> List[ActiveAlert]:
        """Get all active alerts"""
        return list(self._active_alerts.values())
    
    def snapshot_history(self) -> None:
        """Take a snapshot for history tracking"""
        metrics = self.get_current_metrics()
        self._history.append({
            "timestamp": datetime.now(timezone.utc),
            "availability": metrics.availability,
            "latency_p99": metrics.latency_p99_ms,
            "error_rate": metrics.error_rate,
            "accuracy": metrics.accuracy,
            "violation": metrics.overall_state == SLOState.VIOLATION,
        })
    
    def reset_budget_window(self) -> None:
        """Reset error budget window"""
        self._budget_window_start = datetime.now(timezone.utc)
        self._total_requests = 0
        self._successful_requests = 0
        self._error_requests = 0
        self._budget_consumed_minutes = 0.0
        logger.info("Error budget window reset")
