"""
SelfHealing_V1 - Health Monitor

Monitors system health across services.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Health status of a service"""
    service_name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class SystemHealth:
    """Overall system health"""
    status: HealthStatus
    services: Dict[str, ServiceHealth]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "healthy_count": self.healthy_count,
            "degraded_count": self.degraded_count,
            "unhealthy_count": self.unhealthy_count,
            "services": {k: v.to_dict() for k, v in self.services.items()},
        }


class HealthMonitor:
    """
    System health monitor.

    Features:
    - Service health checks
    - Overall system status
    - Continuous monitoring
    """

    def __init__(
        self,
        check_interval: int = 30,
        degraded_threshold: int = 2,
        unhealthy_threshold: int = 5,
    ):
        self.check_interval = check_interval
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold

        self._services: Dict[str, ServiceHealth] = {}
        self._health_checks: Dict[str, callable] = {}
        self._running = False

    def register_service(
        self,
        service_name: str,
        health_check: callable,
    ):
        """Register service for monitoring"""
        self._health_checks[service_name] = health_check
        self._services[service_name] = ServiceHealth(
            service_name=service_name,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.now(timezone.utc),
        )
        logger.info(f"Registered service: {service_name}")

    async def check_service(self, service_name: str) -> ServiceHealth:
        """Check health of specific service"""
        if service_name not in self._health_checks:
            raise ValueError(f"Unknown service: {service_name}")

        check_func = self._health_checks[service_name]
        service = self._services[service_name]

        start_time = datetime.now(timezone.utc)

        try:
            result = await check_func()
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            if result:
                service.status = HealthStatus.HEALTHY
                service.consecutive_failures = 0
                service.error_message = None
            else:
                service.consecutive_failures += 1
                service.status = self._determine_status(service.consecutive_failures)
                service.error_message = "Health check returned false"

            service.response_time_ms = response_time

        except Exception as e:
            service.consecutive_failures += 1
            service.status = self._determine_status(service.consecutive_failures)
            service.error_message = str(e)
            logger.error(f"Health check failed for {service_name}: {e}")

        service.last_check = datetime.now(timezone.utc)
        return service

    async def check_all(self) -> SystemHealth:
        """Check health of all services"""
        for service_name in self._health_checks:
            await self.check_service(service_name)

        return self._compute_system_health()

    def _determine_status(self, failures: int) -> HealthStatus:
        """Determine status based on failure count"""
        if failures >= self.unhealthy_threshold:
            return HealthStatus.UNHEALTHY
        elif failures >= self.degraded_threshold:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _compute_system_health(self) -> SystemHealth:
        """Compute overall system health"""
        healthy = degraded = unhealthy = 0

        for service in self._services.values():
            if service.status == HealthStatus.HEALTHY:
                healthy += 1
            elif service.status == HealthStatus.DEGRADED:
                degraded += 1
            else:
                unhealthy += 1

        if unhealthy > 0:
            overall = HealthStatus.UNHEALTHY
        elif degraded > 0:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return SystemHealth(
            status=overall,
            services=self._services.copy(),
            healthy_count=healthy,
            degraded_count=degraded,
            unhealthy_count=unhealthy,
        )

    async def start_monitoring(self):
        """Start continuous monitoring"""
        self._running = True
        logger.info("Health monitoring started")

        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)

    def stop_monitoring(self):
        """Stop monitoring"""
        self._running = False
        logger.info("Health monitoring stopped")

    def get_status(self) -> SystemHealth:
        """Get current system health"""
        return self._compute_system_health()
