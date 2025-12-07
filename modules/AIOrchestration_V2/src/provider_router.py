"""
AIOrchestration_V2 - Provider Router

Production provider routing with health-aware selection.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional


class RoutingStrategy(str, Enum):
    """Routing strategy options."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LATENCY = "latency"
    COST = "cost"
    HEALTH = "health"


class ProviderStatus(str, Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ProviderConfig:
    """Provider configuration."""
    name: str
    weight: int = 1
    priority: int = 1
    max_requests_per_minute: int = 100
    cost_per_token: float = 0.0001
    enabled: bool = True


@dataclass
class ProviderStats:
    """Provider statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    last_error: Optional[str] = None
    last_request: Optional[datetime] = None


class ProviderRouter:
    """
    Production Provider Router.

    Features:
    - Multiple routing strategies
    - Health-aware routing
    - Cost optimization
    - Load balancing
    """

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN):
        self.strategy = strategy

        self._providers: Dict[str, ProviderConfig] = {}
        self._stats: Dict[str, ProviderStats] = {}
        self._status: Dict[str, ProviderStatus] = {}

        self._round_robin_index = 0

    def register_provider(self, config: ProviderConfig):
        """Register a provider."""
        self._providers[config.name] = config
        self._stats[config.name] = ProviderStats()
        self._status[config.name] = ProviderStatus.HEALTHY

    def select_provider(self, strategy: Optional[str] = None) -> Optional[str]:
        """Select provider based on strategy."""
        effective_strategy = RoutingStrategy(strategy) if strategy else self.strategy

        available = [
            name for name, config in self._providers.items()
            if config.enabled and self._status.get(name) != ProviderStatus.UNHEALTHY
        ]

        if not available:
            return None

        if effective_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available)
        elif effective_strategy == RoutingStrategy.WEIGHTED:
            return self._select_weighted(available)
        elif effective_strategy == RoutingStrategy.LATENCY:
            return self._select_by_latency(available)
        elif effective_strategy == RoutingStrategy.COST:
            return self._select_by_cost(available)
        elif effective_strategy == RoutingStrategy.HEALTH:
            return self._select_by_health(available)

        return available[0]

    def _select_round_robin(self, available: List[str]) -> str:
        """Round-robin selection."""
        self._round_robin_index = (self._round_robin_index + 1) % len(available)
        return available[self._round_robin_index]

    def _select_weighted(self, available: List[str]) -> str:
        """Weighted random selection."""
        import random

        weights = [self._providers[name].weight for name in available]
        return random.choices(available, weights=weights, k=1)[0]

    def _select_by_latency(self, available: List[str]) -> str:
        """Select by lowest average latency."""
        def avg_latency(name: str) -> float:
            stats = self._stats[name]
            if stats.successful_requests == 0:
                return float('inf')
            return stats.total_latency_ms / stats.successful_requests

        return min(available, key=avg_latency)

    def _select_by_cost(self, available: List[str]) -> str:
        """Select by lowest cost."""
        return min(available, key=lambda n: self._providers[n].cost_per_token)

    def _select_by_health(self, available: List[str]) -> str:
        """Select by health and success rate."""
        def health_score(name: str) -> float:
            stats = self._stats[name]
            total = stats.total_requests
            if total == 0:
                return 1.0
            return stats.successful_requests / total

        # Prioritize healthy providers
        healthy = [n for n in available if self._status.get(n) == ProviderStatus.HEALTHY]
        if healthy:
            return max(healthy, key=health_score)

        return max(available, key=health_score)

    def update_provider_stats(
        self,
        provider: str,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """Update provider statistics after request."""
        if provider not in self._stats:
            return

        stats = self._stats[provider]
        stats.total_requests += 1
        stats.total_latency_ms += latency_ms
        stats.last_request = datetime.now(timezone.utc)

        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
            stats.last_error = error

            # Update health status
            total = stats.total_requests
            if total >= 10:
                failure_rate = stats.failed_requests / total
                if failure_rate > 0.5:
                    self._status[provider] = ProviderStatus.UNHEALTHY
                elif failure_rate > 0.2:
                    self._status[provider] = ProviderStatus.DEGRADED

    def get_provider_status(self, provider: str) -> Optional[ProviderStatus]:
        """Get provider health status."""
        return self._status.get(provider)

    def set_provider_status(self, provider: str, status: ProviderStatus):
        """Manually set provider status."""
        if provider in self._status:
            self._status[provider] = status

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "providers": len(self._providers),
            "healthy": sum(1 for s in self._status.values() if s == ProviderStatus.HEALTHY),
            "degraded": sum(1 for s in self._status.values() if s == ProviderStatus.DEGRADED),
            "unhealthy": sum(1 for s in self._status.values() if s == ProviderStatus.UNHEALTHY),
            "strategy": self.strategy.value,
        }


__all__ = [
    "RoutingStrategy",
    "ProviderStatus",
    "ProviderConfig",
    "ProviderStats",
    "ProviderRouter",
]
