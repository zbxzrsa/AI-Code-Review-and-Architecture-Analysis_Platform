"""
SelfHealing_V2 - Enhanced Health Monitor

Production health monitoring with SLO integration.
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service: str
    status: HealthStatus
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ServiceConfig:
    """Service health check configuration"""
    name: str
    check_func: Callable[[], Any]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3

    # V2: SLO thresholds
    slo_latency_ms: float = 1000
    slo_error_rate: float = 0.01


class HealthMonitor:
    """
    Production health monitor.

    V2 Features:
    - SLO-based health evaluation
    - Configurable thresholds
    - Health history tracking
    - Dependency health aggregation
    """

    def __init__(self):
        self._services: Dict[str, ServiceConfig] = {}
        self._status: Dict[str, HealthStatus] = {}
        self._history: Dict[str, List[HealthCheckResult]] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._consecutive_successes: Dict[str, int] = {}
        self._running = False
        self._max_history = 100

    def register_service(
        self,
        name: str,
        check_func: Callable[[], Any],
        **kwargs,
    ):
        """Register service for monitoring"""
        config = ServiceConfig(name=name, check_func=check_func, **kwargs)
        self._services[name] = config
        self._status[name] = HealthStatus.UNKNOWN
        self._history[name] = []
        self._consecutive_failures[name] = 0
        self._consecutive_successes[name] = 0

        logger.info(f"Registered service: {name}")

    async def check_service(self, name: str) -> HealthCheckResult:
        """Check health of a service"""
        if name not in self._services:
            raise ValueError(f"Unknown service: {name}")

        config = self._services[name]
        start_time = datetime.now(timezone.utc)

        try:
            # Execute check with timeout
            result = await asyncio.wait_for(
                self._execute_check(config.check_func),
                timeout=config.timeout_seconds,
            )

            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Evaluate SLO
            slo_met = response_time <= config.slo_latency_ms

            if result and slo_met:
                self._consecutive_successes[name] += 1
                self._consecutive_failures[name] = 0

                if self._consecutive_successes[name] >= config.healthy_threshold:
                    status = HealthStatus.HEALTHY
                else:
                    status = self._status[name]  # Keep current
            else:
                self._consecutive_failures[name] += 1
                self._consecutive_successes[name] = 0

                if self._consecutive_failures[name] >= config.unhealthy_threshold:
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.DEGRADED

            check_result = HealthCheckResult(
                service=name,
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.now(timezone.utc),
                details={"slo_met": slo_met, "result": bool(result)},
            )

        except asyncio.TimeoutError:
            self._consecutive_failures[name] += 1
            self._consecutive_successes[name] = 0

            status = HealthStatus.UNHEALTHY if self._consecutive_failures[name] >= config.unhealthy_threshold else HealthStatus.DEGRADED

            check_result = HealthCheckResult(
                service=name,
                status=status,
                response_time_ms=config.timeout_seconds * 1000,
                timestamp=datetime.now(timezone.utc),
                error="Timeout",
            )

        except Exception as e:
            self._consecutive_failures[name] += 1
            self._consecutive_successes[name] = 0

            status = HealthStatus.UNHEALTHY if self._consecutive_failures[name] >= config.unhealthy_threshold else HealthStatus.DEGRADED

            check_result = HealthCheckResult(
                service=name,
                status=status,
                response_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                timestamp=datetime.now(timezone.utc),
                error=str(e),
            )

        # Update status and history
        self._status[name] = status
        self._history[name].append(check_result)

        if len(self._history[name]) > self._max_history:
            self._history[name] = self._history[name][-self._max_history:]

        return check_result

    async def _execute_check(self, check_func: Callable) -> Any:
        """Execute check function (sync or async)"""
        if asyncio.iscoroutinefunction(check_func):
            return await check_func()
        return check_func()

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Check all services"""
        results = {}
        for name in self._services:
            results[name] = await self.check_service(name)
        return results

    def get_status(self, name: str) -> HealthStatus:
        """Get current status of a service"""
        return self._status.get(name, HealthStatus.UNKNOWN)

    def get_all_status(self) -> Dict[str, str]:
        """Get status of all services"""
        return {name: status.value for name, status in self._status.items()}

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health"""
        statuses = list(self._status.values())

        if not statuses:
            return HealthStatus.UNKNOWN

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_history(self, name: str, limit: int = 10) -> List[Dict]:
        """Get health check history"""
        history = self._history.get(name, [])
        return [
            {
                "status": r.status.value,
                "response_time_ms": r.response_time_ms,
                "timestamp": r.timestamp.isoformat(),
                "error": r.error,
            }
            for r in history[-limit:]
        ]

    async def start_monitoring(self):
        """Start continuous monitoring"""
        self._running = True
        logger.info("Health monitoring started")

        while self._running:
            await self.check_all()

            # Use minimum interval
            min_interval = min(
                (c.interval_seconds for c in self._services.values()),
                default=30
            )
            await asyncio.sleep(min_interval)

    def stop_monitoring(self):
        """Stop monitoring"""
        self._running = False
