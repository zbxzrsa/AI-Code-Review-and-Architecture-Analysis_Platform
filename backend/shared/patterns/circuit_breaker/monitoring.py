"""
Circuit Breaker Real-Time Monitoring

Provides real-time monitoring dashboard and alerting for circuit breakers.

Features:
- Real-time status updates
- Metrics collection and aggregation
- Alert triggers
- WebSocket/SSE support for live updates
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque
from enum import Enum

from .enhanced_circuit_breaker import CircuitState, EnhancedCircuitBreaker
from .provider_circuit_breakers import ProviderCircuitBreakerManager, ProviderHealth

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    HIGH_FAILURE_RATE = "high_failure_rate"
    HIGH_LATENCY = "high_latency"
    ALL_PROVIDERS_DOWN = "all_providers_down"
    RECOVERY_FAILED = "recovery_failed"
    THRESHOLD_BREACH = "threshold_breach"


@dataclass
class Alert:
    """Circuit breaker alert."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    provider_name: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "provider_name": self.provider_name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot."""
    timestamp: datetime
    provider_name: str
    state: CircuitState
    failure_rate: float
    avg_latency_ms: float
    requests_per_second: float
    success_count: int
    failure_count: int
    rejected_count: int


class CircuitBreakerMonitor:
    """
    Real-time monitoring for circuit breakers.

    Features:
    - Continuous metrics collection
    - Alert generation
    - Historical data retention
    - Dashboard data generation

    Usage:
        monitor = CircuitBreakerMonitor(manager)
        await monitor.start()

        # Get current status
        status = monitor.get_dashboard_data()

        # Get alerts
        alerts = monitor.get_active_alerts()
    """

    # Alert thresholds
    HIGH_FAILURE_RATE_THRESHOLD = 0.30
    HIGH_LATENCY_THRESHOLD_MS = 5000

    def __init__(
        self,
        manager: ProviderCircuitBreakerManager,
        collection_interval_seconds: float = 5.0,
        history_retention_hours: int = 24,
        alert_callback: Optional[Callable] = None,
    ):
        self.manager = manager
        self.collection_interval = collection_interval_seconds
        self.history_retention = timedelta(hours=history_retention_hours)
        self.alert_callback = alert_callback

        # Metrics history
        self._metrics_history: Dict[str, deque] = {}
        self._max_history_points = int(history_retention_hours * 3600 / collection_interval_seconds)

        # Alerts
        self._alerts: List[Alert] = []
        self._alert_counter = 0

        # Subscribers for real-time updates
        self._subscribers: Set[asyncio.Queue] = set()

        # State
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the monitoring service."""
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Circuit breaker monitor started")

    async def stop(self):
        """Stop the monitoring service."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                # Intentionally not re-raised: we initiated the cancellation
                # during shutdown, so propagation is not needed
                logger.debug("Collection task cancelled during shutdown")
        logger.info("Circuit breaker monitor stopped")

    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self._running:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await self._broadcast_update()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                logger.debug("Collection loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(1)

    async def _collect_metrics(self):
        """Collect current metrics from all providers."""
        now = datetime.now(timezone.utc)

        for provider_name in self.manager._providers:
            breaker = self.manager._breakers.get(provider_name)
            if not breaker:
                continue

            metrics = breaker.metrics

            # Calculate requests per second
            rps = metrics.total_requests / max(
                (now - metrics.last_state_change).total_seconds(), 1
            )

            snapshot = MetricSnapshot(
                timestamp=now,
                provider_name=provider_name,
                state=breaker.state,
                failure_rate=metrics.current_failure_rate,
                avg_latency_ms=metrics.avg_latency_ms,
                requests_per_second=rps,
                success_count=metrics.successful_requests,
                failure_count=metrics.failed_requests,
                rejected_count=metrics.rejected_requests,
            )

            # Store in history
            if provider_name not in self._metrics_history:
                self._metrics_history[provider_name] = deque(maxlen=self._max_history_points)

            self._metrics_history[provider_name].append(snapshot)

    async def _check_alerts(self):
        """Check for alert conditions."""
        for provider_name in self.manager._providers:
            health = self.manager.get_provider_health(provider_name)
            if not health:
                continue

            # Check circuit state change
            await self._check_circuit_state_alert(provider_name, health)

            # Check high failure rate
            await self._check_failure_rate_alert(provider_name, health)

            # Check high latency
            await self._check_latency_alert(provider_name, health)

        # Check all providers down
        await self._check_all_providers_down()

    async def _check_circuit_state_alert(self, provider_name: str, health: ProviderHealth):
        """Check for circuit state change alerts."""
        if health.circuit_state == CircuitState.OPEN:
            # Check if already alerted
            existing = self._find_active_alert(
                provider_name, AlertType.CIRCUIT_OPENED
            )
            if not existing:
                await self._create_alert(
                    AlertType.CIRCUIT_OPENED,
                    AlertSeverity.ERROR,
                    provider_name,
                    f"Circuit breaker opened for {provider_name}",
                    {
                        "failure_rate": health.failure_rate,
                        "consecutive_failures": health.consecutive_failures,
                    }
                )
        elif health.circuit_state == CircuitState.CLOSED:
            # Resolve any open circuit alerts
            self._resolve_alerts(provider_name, AlertType.CIRCUIT_OPENED)

    async def _check_failure_rate_alert(self, provider_name: str, health: ProviderHealth):
        """Check for high failure rate alerts."""
        if health.failure_rate > self.HIGH_FAILURE_RATE_THRESHOLD:
            existing = self._find_active_alert(
                provider_name, AlertType.HIGH_FAILURE_RATE
            )
            if not existing:
                await self._create_alert(
                    AlertType.HIGH_FAILURE_RATE,
                    AlertSeverity.WARNING,
                    provider_name,
                    f"High failure rate detected for {provider_name}: {health.failure_rate:.2%}",
                    {"failure_rate": health.failure_rate}
                )
        else:
            self._resolve_alerts(provider_name, AlertType.HIGH_FAILURE_RATE)

    async def _check_latency_alert(self, provider_name: str, health: ProviderHealth):
        """Check for high latency alerts."""
        if health.avg_latency_ms > self.HIGH_LATENCY_THRESHOLD_MS:
            existing = self._find_active_alert(
                provider_name, AlertType.HIGH_LATENCY
            )
            if not existing:
                await self._create_alert(
                    AlertType.HIGH_LATENCY,
                    AlertSeverity.WARNING,
                    provider_name,
                    f"High latency detected for {provider_name}: {health.avg_latency_ms:.0f}ms",
                    {"avg_latency_ms": health.avg_latency_ms}
                )
        else:
            self._resolve_alerts(provider_name, AlertType.HIGH_LATENCY)

    async def _check_all_providers_down(self):
        """Check if all providers are unavailable."""
        available = self.manager.get_available_providers()

        if not available:
            existing = self._find_active_alert("system", AlertType.ALL_PROVIDERS_DOWN)
            if not existing:
                await self._create_alert(
                    AlertType.ALL_PROVIDERS_DOWN,
                    AlertSeverity.CRITICAL,
                    "system",
                    "All AI providers are unavailable",
                    {"total_providers": len(self.manager._providers)}
                )
        else:
            self._resolve_alerts("system", AlertType.ALL_PROVIDERS_DOWN)

    async def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        provider_name: str,
        message: str,
        details: Dict[str, Any],
    ):
        """Create and store a new alert."""
        self._alert_counter += 1
        alert = Alert(
            alert_id=f"alert_{self._alert_counter}",
            alert_type=alert_type,
            severity=severity,
            provider_name=provider_name,
            message=message,
            details=details,
        )

        async with self._lock:
            self._alerts.append(alert)

        logger.warning(f"Alert created: {message}")

        # Notify callback
        if self.alert_callback:
            try:
                if asyncio.iscoroutinefunction(self.alert_callback):
                    await self.alert_callback(alert)
                else:
                    self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def _find_active_alert(
        self,
        provider_name: str,
        alert_type: AlertType
    ) -> Optional[Alert]:
        """Find an active (unresolved) alert."""
        for alert in self._alerts:
            if (alert.provider_name == provider_name and
                alert.alert_type == alert_type and
                not alert.resolved):
                return alert
        return None

    def _resolve_alerts(self, provider_name: str, alert_type: AlertType):
        """Resolve alerts of a specific type for a provider."""
        for alert in self._alerts:
            if (alert.provider_name == provider_name and
                alert.alert_type == alert_type and
                not alert.resolved):
                alert.resolved = True
                logger.info(f"Alert resolved: {alert.message}")

    async def _broadcast_update(self):
        """Broadcast status update to all subscribers."""
        if not self._subscribers:
            return

        update = {
            "type": "status_update",
            "data": self.get_current_status(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        message = json.dumps(update)

        dead_subscribers = set()
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                dead_subscribers.add(queue)

        # Remove dead subscribers
        self._subscribers -= dead_subscribers

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time updates."""
        queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from real-time updates."""
        self._subscribers.discard(queue)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current status of all circuit breakers."""
        providers = []
        for provider_name in self.manager._providers:
            health = self.manager.get_provider_health(provider_name)
            if health:
                providers.append(health.to_dict())

        manager_metrics = self.manager.get_metrics()

        return {
            "providers": providers,
            "metrics": manager_metrics,
            "healthy_count": len(self.manager.get_healthy_providers()),
            "available_count": len(self.manager.get_available_providers()),
            "total_count": len(self.manager._providers),
        }

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts."""
        return [
            alert.to_dict()
            for alert in self._alerts
            if not alert.resolved
        ]

    def get_all_alerts(
        self,
        limit: int = 100,
        include_resolved: bool = False
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        alerts = self._alerts

        if not include_resolved:
            alerts = [a for a in alerts if not a.resolved]

        return [a.to_dict() for a in alerts[-limit:]]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_metrics_history(
        self,
        provider_name: str,
        minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Get metrics history for a provider."""
        if provider_name not in self._metrics_history:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)

        history = []
        for snapshot in self._metrics_history[provider_name]:
            if snapshot.timestamp >= cutoff:
                history.append({
                    "timestamp": snapshot.timestamp.isoformat(),
                    "state": snapshot.state.value,
                    "failure_rate": round(snapshot.failure_rate, 4),
                    "avg_latency_ms": round(snapshot.avg_latency_ms, 2),
                    "requests_per_second": round(snapshot.requests_per_second, 2),
                })

        return history

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "current_status": self.get_current_status(),
            "active_alerts": self.get_active_alerts(),
            "metrics_summary": self._get_metrics_summary(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        total_requests = 0
        total_failures = 0
        total_latency = 0
        count = 0

        for provider_name in self.manager._providers:
            breaker = self.manager._breakers.get(provider_name)
            if breaker:
                metrics = breaker.metrics
                total_requests += metrics.total_requests
                total_failures += metrics.failed_requests
                total_latency += metrics.avg_latency_ms
                count += 1

        return {
            "total_requests": total_requests,
            "total_failures": total_failures,
            "overall_failure_rate": total_failures / max(total_requests, 1),
            "avg_latency_ms": total_latency / max(count, 1),
            "providers_monitored": count,
        }


# =============================================================================
# WebSocket Handler for Real-Time Updates
# =============================================================================

class CircuitBreakerWebSocketHandler:
    """
    WebSocket handler for real-time circuit breaker updates.

    Usage with FastAPI:
        @app.websocket("/ws/circuit-breakers")
        async def ws_circuit_breakers(websocket: WebSocket):
            handler = CircuitBreakerWebSocketHandler(monitor)
            await handler.handle(websocket)
    """

    def __init__(self, monitor: CircuitBreakerMonitor):
        self.monitor = monitor

    async def handle(self, websocket):
        """Handle WebSocket connection."""
        await websocket.accept()

        # Send initial status
        initial_data = self.monitor.get_dashboard_data()
        await websocket.send_json(initial_data)

        # Subscribe to updates
        queue = self.monitor.subscribe()

        try:
            while True:
                # Wait for update
                message = await queue.get()
                await websocket.send_text(message)
        except Exception as e:
            logger.debug(f"WebSocket closed: {e}")
        finally:
            self.monitor.unsubscribe(queue)
