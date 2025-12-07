"""
AIOrchestration_V2 - Load Balancer

Production load balancing with health awareness.
"""

import logging
import random
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    IP_HASH = "ip_hash"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ProviderEndpoint:
    """Provider endpoint with health tracking"""
    name: str
    weight: int = 1
    max_connections: int = 100
    current_connections: int = 0
    health_status: HealthStatus = HealthStatus.HEALTHY
    avg_response_time_ms: float = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_health_check: Optional[datetime] = None

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.failed_requests / self.total_requests

    @property
    def available_capacity(self) -> int:
        return self.max_connections - self.current_connections


class LoadBalancer:
    """
    Production load balancer.

    V2 Features:
    - Health-aware routing
    - Multiple strategies
    - Connection tracking
    - SLO monitoring
    """

    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_RESPONSE_TIME,
        health_check_interval: int = 30,
        unhealthy_threshold: float = 0.1,
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.unhealthy_threshold = unhealthy_threshold

        self._endpoints: Dict[str, ProviderEndpoint] = {}
        self._round_robin_index = 0

    def add_endpoint(
        self,
        name: str,
        weight: int = 1,
        max_connections: int = 100,
    ):
        """Add provider endpoint"""
        self._endpoints[name] = ProviderEndpoint(
            name=name,
            weight=weight,
            max_connections=max_connections,
        )
        logger.info(f"Added endpoint: {name} (weight={weight})")

    def remove_endpoint(self, name: str):
        """Remove endpoint"""
        self._endpoints.pop(name, None)

    def select_endpoint(self, request_id: Optional[str] = None) -> Optional[str]:
        """Select endpoint based on strategy"""
        available = self._get_available_endpoints()

        if not available:
            logger.warning("No available endpoints")
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(available)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(available)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(available)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time(available)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random(available)
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash(available, request_id)

        return available[0]

    def _get_available_endpoints(self) -> List[str]:
        """Get healthy endpoints with capacity"""
        return [
            name for name, ep in self._endpoints.items()
            if ep.health_status != HealthStatus.UNHEALTHY
            and ep.available_capacity > 0
        ]

    def _round_robin(self, available: List[str]) -> str:
        """Round robin selection"""
        self._round_robin_index = (self._round_robin_index + 1) % len(available)
        return available[self._round_robin_index]

    def _weighted_round_robin(self, available: List[str]) -> str:
        """Weighted round robin"""
        weighted = []
        for name in available:
            weight = self._endpoints[name].weight
            weighted.extend([name] * weight)

        self._round_robin_index = (self._round_robin_index + 1) % len(weighted)
        return weighted[self._round_robin_index]

    def _least_connections(self, available: List[str]) -> str:
        """Select endpoint with least connections"""
        return min(available, key=lambda n: self._endpoints[n].current_connections)

    def _least_response_time(self, available: List[str]) -> str:
        """Select fastest endpoint"""
        return min(available, key=lambda n: self._endpoints[n].avg_response_time_ms)

    def _random(self, available: List[str]) -> str:
        """Random selection"""
        return random.choice(available)

    def _ip_hash(self, available: List[str], request_id: Optional[str]) -> str:
        """Consistent hashing based on request ID"""
        if request_id:
            idx = hash(request_id) % len(available)
            return available[idx]
        return self._random(available)

    def acquire_connection(self, name: str) -> bool:
        """Acquire connection to endpoint"""
        if name not in self._endpoints:
            return False

        ep = self._endpoints[name]
        if ep.available_capacity <= 0:
            return False

        ep.current_connections += 1
        return True

    def release_connection(
        self,
        name: str,
        response_time_ms: float,
        success: bool,
    ):
        """Release connection and record metrics"""
        if name not in self._endpoints:
            return

        ep = self._endpoints[name]
        ep.current_connections = max(0, ep.current_connections - 1)
        ep.total_requests += 1

        if not success:
            ep.failed_requests += 1

        # Update moving average
        alpha = 0.1
        ep.avg_response_time_ms = alpha * response_time_ms + (1 - alpha) * ep.avg_response_time_ms

        # Update health status
        self._update_health(name)

    def _update_health(self, name: str):
        """Update endpoint health based on metrics"""
        ep = self._endpoints[name]

        if ep.error_rate >= self.unhealthy_threshold:
            ep.health_status = HealthStatus.UNHEALTHY
            logger.warning(f"Endpoint {name} marked unhealthy (error_rate={ep.error_rate:.2%})")
        elif ep.error_rate >= self.unhealthy_threshold / 2:
            ep.health_status = HealthStatus.DEGRADED
        else:
            ep.health_status = HealthStatus.HEALTHY

    def force_health_status(self, name: str, status: HealthStatus):
        """Manually set health status"""
        if name in self._endpoints:
            self._endpoints[name].health_status = status

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            "strategy": self.strategy.value,
            "total_endpoints": len(self._endpoints),
            "healthy_endpoints": sum(
                1 for ep in self._endpoints.values()
                if ep.health_status == HealthStatus.HEALTHY
            ),
            "endpoints": {
                name: {
                    "health": ep.health_status.value,
                    "connections": ep.current_connections,
                    "capacity": ep.available_capacity,
                    "avg_response_ms": ep.avg_response_time_ms,
                    "error_rate": ep.error_rate,
                }
                for name, ep in self._endpoints.items()
            }
        }
