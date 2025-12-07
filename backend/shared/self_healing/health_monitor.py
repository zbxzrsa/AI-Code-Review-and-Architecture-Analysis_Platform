"""
Health Monitor - Real-time System Health Monitoring

Monitors system health across all dimensions and triggers
self-healing actions when issues are detected.

Features:
- Real-time metric collection
- Anomaly detection
- Threshold monitoring
- Alert generation
- Auto-repair triggering
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ERROR = "error"
    AVAILABILITY = "availability"
    CUSTOM = "custom"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    check_id: str
    name: str
    check_type: MetricType

    # Thresholds
    warning_threshold: float
    critical_threshold: float

    # Check function
    check_function: Optional[Callable] = None

    # Status
    last_value: Optional[float] = None
    last_check: Optional[str] = None
    status: HealthStatus = HealthStatus.HEALTHY
    consecutive_failures: int = 0

    # Auto-repair
    auto_repair_enabled: bool = True
    repair_action: Optional[str] = None


@dataclass
class HealthMetrics:
    """System health metrics snapshot."""
    timestamp: str

    # Performance
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    throughput_rps: float = 0.0

    # Resources
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0

    # Errors
    error_rate: float = 0.0
    error_count_5m: int = 0
    timeout_count_5m: int = 0

    # Availability
    uptime_seconds: float = 0.0
    availability_percent: float = 100.0

    # Custom
    queue_size: int = 0
    active_connections: int = 0
    cache_hit_rate: float = 0.0

    def calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0

        # Deduct for high response time
        if self.response_time_p95 > 3000:
            score -= 20
        elif self.response_time_p95 > 2000:
            score -= 10

        # Deduct for high error rate
        if self.error_rate > 0.05:
            score -= 30
        elif self.error_rate > 0.02:
            score -= 15

        # Deduct for high resource usage
        if self.cpu_usage_percent > 90:
            score -= 20
        elif self.cpu_usage_percent > 70:
            score -= 10

        if self.memory_usage_percent > 90:
            score -= 20
        elif self.memory_usage_percent > 70:
            score -= 10

        # Deduct for low availability
        if self.availability_percent < 99.0:
            score -= 25
        elif self.availability_percent < 99.9:
            score -= 10

        return max(0.0, score)

    def get_status(self) -> HealthStatus:
        """Determine health status based on score."""
        score = self.calculate_health_score()

        if score >= 90:
            return HealthStatus.HEALTHY
        elif score >= 70:
            return HealthStatus.DEGRADED
        elif score >= 50:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL


class HealthMonitor:
    """
    Real-time health monitoring system.

    Features:
    - Continuous metric collection
    - Threshold-based alerting
    - Anomaly detection
    - Auto-repair triggering
    - Historical tracking
    """

    def __init__(
        self,
        check_interval: int = 30,
        history_size: int = 1000,
        anomaly_detection_enabled: bool = True
    ):
        self.check_interval = check_interval
        self.history_size = history_size
        self.anomaly_detection_enabled = anomaly_detection_enabled

        # Health checks
        self.checks: Dict[str, HealthCheck] = {}

        # Metrics history
        self.metrics_history: deque = deque(maxlen=history_size)

        # Current state
        self.current_metrics: Optional[HealthMetrics] = None
        self.current_status: HealthStatus = HealthStatus.HEALTHY

        # Monitoring
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Callbacks
        self.on_status_change: Optional[Callable] = None
        self.on_threshold_exceeded: Optional[Callable] = None
        self.on_anomaly_detected: Optional[Callable] = None

        # Statistics
        self.stats = {
            "checks_performed": 0,
            "alerts_generated": 0,
            "repairs_triggered": 0,
            "anomalies_detected": 0
        }

        self._setup_default_checks()

    def _setup_default_checks(self) -> None:
        """Setup default health checks."""
        # Response time check
        self.add_check(HealthCheck(
            check_id="response_time_p95",
            name="Response Time P95",
            check_type=MetricType.PERFORMANCE,
            warning_threshold=2000,  # 2s
            critical_threshold=3000,  # 3s
            auto_repair_enabled=True,
            repair_action="scale_up"
        ))

        # Error rate check
        self.add_check(HealthCheck(
            check_id="error_rate",
            name="Error Rate",
            check_type=MetricType.ERROR,
            warning_threshold=0.02,  # 2%
            critical_threshold=0.05,  # 5%
            auto_repair_enabled=True,
            repair_action="rollback"
        ))

        # Memory usage check
        self.add_check(HealthCheck(
            check_id="memory_usage",
            name="Memory Usage",
            check_type=MetricType.RESOURCE,
            warning_threshold=70.0,  # 70%
            critical_threshold=90.0,  # 90%
            auto_repair_enabled=True,
            repair_action="restart_service"
        ))

        # CPU usage check
        self.add_check(HealthCheck(
            check_id="cpu_usage",
            name="CPU Usage",
            check_type=MetricType.RESOURCE,
            warning_threshold=70.0,
            critical_threshold=90.0,
            auto_repair_enabled=True,
            repair_action="scale_up"
        ))

        # Availability check
        self.add_check(HealthCheck(
            check_id="availability",
            name="System Availability",
            check_type=MetricType.AVAILABILITY,
            warning_threshold=99.5,  # Below 99.5%
            critical_threshold=99.0,  # Below 99%
            auto_repair_enabled=True,
            repair_action="investigate"
        ))

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self.checks[check.check_id] = check
        logger.info(f"Added health check: {check.name}")

    async def start(self) -> None:
        """Start health monitoring."""
        logger.info("Starting health monitor...")
        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Health monitor started (interval: {self.check_interval}s)")

    async def stop(self) -> None:
        """Stop health monitoring."""
        logger.info("Stopping health monitor...")
        self.is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                # Intentionally not re-raised: we initiated the cancellation
                # during shutdown, so propagation is not needed
                logger.debug("Health monitor task cancelled during shutdown")

        logger.info("Health monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()

                # Store in history
                self.metrics_history.append(metrics)
                self.current_metrics = metrics

                # Run health checks
                await self.run_health_checks(metrics)

                # Detect anomalies
                if self.anomaly_detection_enabled:
                    await self.detect_anomalies(metrics)

                # Update overall status
                await self._update_overall_status(metrics)

                self.stats["checks_performed"] += 1

            except Exception as e:
                logger.error(f"Health monitor error: {e}", exc_info=True)

            await asyncio.sleep(self.check_interval)

    async def collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        # In production, collect real metrics from:
        # - Prometheus
        # - System resources (psutil)
        # - Application metrics
        # - Database metrics

        metrics = HealthMetrics(
            timestamp=datetime.now().isoformat(),
            response_time_p50=800,
            response_time_p95=2100,
            response_time_p99=3200,
            throughput_rps=150,
            cpu_usage_percent=45,
            memory_usage_mb=1500,
            memory_usage_percent=37.5,
            disk_usage_percent=55,
            error_rate=0.008,
            error_count_5m=12,
            timeout_count_5m=2,
            uptime_seconds=86400,
            availability_percent=99.95,
            queue_size=120,
            active_connections=25,
            cache_hit_rate=0.85
        )

        return metrics

    async def run_health_checks(self, metrics: HealthMetrics) -> List[Dict[str, Any]]:
        """Run all health checks against current metrics."""
        results = []

        for check in self.checks.values():
            result = await self._run_single_check(check, metrics)
            results.append(result)

            # Trigger repair if needed
            if result["status"] == HealthStatus.CRITICAL and check.auto_repair_enabled:
                await self._trigger_repair(check, result)

        return results

    async def _run_single_check(
        self,
        check: HealthCheck,
        metrics: HealthMetrics
    ) -> Dict[str, Any]:
        """Run a single health check."""
        # Get metric value
        value = getattr(metrics, check.check_id, None)

        if value is None:
            return {
                "check_id": check.check_id,
                "status": HealthStatus.HEALTHY,
                "value": None,
                "message": "Metric not available"
            }

        check.last_value = value
        check.last_check = datetime.now().isoformat()

        # Determine status
        if check.check_id in ["availability"]:
            # For availability, lower is worse
            if value < check.critical_threshold:
                status = HealthStatus.CRITICAL
                check.consecutive_failures += 1
            elif value < check.warning_threshold:
                status = HealthStatus.DEGRADED
                check.consecutive_failures += 1
            else:
                status = HealthStatus.HEALTHY
                check.consecutive_failures = 0
        else:
            # For other metrics, higher is worse
            if value >= check.critical_threshold:
                status = HealthStatus.CRITICAL
                check.consecutive_failures += 1
            elif value >= check.warning_threshold:
                status = HealthStatus.DEGRADED
                check.consecutive_failures += 1
            else:
                status = HealthStatus.HEALTHY
                check.consecutive_failures = 0

        check.status = status

        result = {
            "check_id": check.check_id,
            "name": check.name,
            "status": status,
            "value": value,
            "warning_threshold": check.warning_threshold,
            "critical_threshold": check.critical_threshold,
            "consecutive_failures": check.consecutive_failures,
            "message": self._get_status_message(check, value, status)
        }

        # Trigger callback if threshold exceeded
        if status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
            if self.on_threshold_exceeded:
                await self.on_threshold_exceeded(check, result)

        return result

    def _get_status_message(
        self,
        check: HealthCheck,
        value: float,
        status: HealthStatus
    ) -> str:
        """Generate status message for check."""
        if status == HealthStatus.HEALTHY:
            return f"{check.name} is healthy ({value:.2f})"
        elif status == HealthStatus.DEGRADED:
            return f"{check.name} is degraded ({value:.2f} > {check.warning_threshold})"
        elif status == HealthStatus.CRITICAL:
            return f"{check.name} is critical ({value:.2f} > {check.critical_threshold})"
        else:
            return f"{check.name} status unknown"

    async def _trigger_repair(
        self,
        check: HealthCheck,
        result: Dict[str, Any]
    ) -> None:
        """Trigger auto-repair action."""
        if not check.repair_action:
            return

        logger.warning(
            f"Triggering repair for {check.name}: {check.repair_action}"
        )

        self.stats["repairs_triggered"] += 1

        # In production, trigger actual repair
        # For now, just log
        repair_event = {
            "check_id": check.check_id,
            "repair_action": check.repair_action,
            "triggered_at": datetime.now().isoformat(),
            "reason": result["message"]
        }

        logger.info(f"Repair triggered: {repair_event}")

    async def detect_anomalies(self, metrics: HealthMetrics) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods."""
        if len(self.metrics_history) < 10:
            return []  # Need history for anomaly detection

        anomalies = []

        # Calculate baseline from history
        recent_metrics = list(self.metrics_history)[-100:]

        # Check response time anomaly
        response_times = [m.response_time_p95 for m in recent_metrics]
        avg_response = sum(response_times) / len(response_times)
        std_response = (
            sum((x - avg_response) ** 2 for x in response_times) / len(response_times)
        ) ** 0.5

        if metrics.response_time_p95 > avg_response + 3 * std_response:
            anomalies.append({
                "type": "response_time_spike",
                "severity": "high",
                "current": metrics.response_time_p95,
                "baseline": avg_response,
                "deviation": (metrics.response_time_p95 - avg_response) / avg_response
            })

        # Check error rate anomaly
        error_rates = [m.error_rate for m in recent_metrics]
        avg_error = sum(error_rates) / len(error_rates)

        if metrics.error_rate > avg_error * 3 and metrics.error_rate > 0.01:
            anomalies.append({
                "type": "error_rate_spike",
                "severity": "critical",
                "current": metrics.error_rate,
                "baseline": avg_error,
                "deviation": (metrics.error_rate - avg_error) / avg_error if avg_error > 0 else 999
            })

        if anomalies:
            self.stats["anomalies_detected"] += len(anomalies)

            if self.on_anomaly_detected:
                await self.on_anomaly_detected(anomalies)

        return anomalies

    async def _update_overall_status(self, metrics: HealthMetrics) -> None:
        """Update overall system status."""
        previous_status = self.current_status
        new_status = metrics.get_status()

        if new_status != previous_status:
            logger.info(
                f"System status changed: {previous_status.value} → {new_status.value}"
            )

            if self.on_status_change:
                await self.on_status_change(previous_status, new_status)

        self.current_status = new_status

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        if not self.current_metrics:
            return {"status": "no_data"}

        # Check statuses
        check_statuses = {}
        for check_id, check in self.checks.items():
            check_statuses[check_id] = {
                "name": check.name,
                "status": check.status.value,
                "value": check.last_value,
                "consecutive_failures": check.consecutive_failures
            }

        # Overall health score
        health_score = self.current_metrics.calculate_health_score()

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.current_status.value,
            "health_score": health_score,
            "current_metrics": {
                "response_time_p95": self.current_metrics.response_time_p95,
                "error_rate": self.current_metrics.error_rate,
                "cpu_usage": self.current_metrics.cpu_usage_percent,
                "memory_usage": self.current_metrics.memory_usage_percent,
                "availability": self.current_metrics.availability_percent
            },
            "checks": check_statuses,
            "statistics": self.stats,
            "history_size": len(self.metrics_history)
        }

    def get_trends(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get trend analysis for a metric."""
        if not self.metrics_history:
            return {"status": "no_data"}

        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

        # Filter recent metrics
        recent = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        if not recent:
            return {"status": "insufficient_data"}

        # Extract values
        values = [getattr(m, metric_name, 0) for m in recent]

        if not values:
            return {"status": "metric_not_found"}

        # Calculate trend
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        current = values[-1]

        # Simple trend detection
        if len(values) >= 2:
            first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
            second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
            trend = "increasing" if second_half_avg > first_half_avg * 1.1 else \
                    "decreasing" if second_half_avg < first_half_avg * 0.9 else \
                    "stable"
        else:
            trend = "unknown"

        return {
            "metric": metric_name,
            "window_minutes": window_minutes,
            "data_points": len(values),
            "current": current,
            "average": avg,
            "min": min_val,
            "max": max_val,
            "trend": trend,
            "variance": max_val - min_val
        }
