"""
AIOrchestration_V1 - Provider Router

Intelligent routing to AI providers.
"""

import logging
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LATENCY = "least_latency"
    LEAST_COST = "least_cost"
    WEIGHTED = "weighted"


@dataclass
class ProviderStats:
    """Provider statistics"""
    name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    total_cost: float = 0
    last_used: Optional[datetime] = None
    is_healthy: bool = True
    weight: float = 1.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency_ms / max(1, self.successful_requests)

    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(1, self.total_requests)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency,
            "total_cost": self.total_cost,
            "is_healthy": self.is_healthy,
        }


class ProviderRouter:
    """
    AI provider router with multiple strategies.

    Strategies:
    - Round robin: Rotate through providers
    - Random: Random selection
    - Least latency: Select fastest provider
    - Least cost: Select cheapest provider
    - Weighted: Based on configured weights
    """

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self._providers: Dict[str, ProviderStats] = {}
        self._round_robin_index = 0

    def register_provider(
        self,
        name: str,
        weight: float = 1.0,
    ):
        """Register provider for routing"""
        self._providers[name] = ProviderStats(name=name, weight=weight)
        logger.info(f"Registered provider for routing: {name}")

    def select_provider(
        self,
        exclude: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Select provider based on strategy"""
        exclude = exclude or []

        available = [
            name for name, stats in self._providers.items()
            if stats.is_healthy and name not in exclude
        ]

        if not available:
            return None

        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available)
        elif self.strategy == RoutingStrategy.RANDOM:
            return self._select_random(available)
        elif self.strategy == RoutingStrategy.LEAST_LATENCY:
            return self._select_least_latency(available)
        elif self.strategy == RoutingStrategy.LEAST_COST:
            return self._select_least_cost(available)
        elif self.strategy == RoutingStrategy.WEIGHTED:
            return self._select_weighted(available)

        return available[0]

    def _select_round_robin(self, available: List[str]) -> str:
        """Round robin selection"""
        self._round_robin_index = (self._round_robin_index + 1) % len(available)
        return available[self._round_robin_index]

    def _select_random(self, available: List[str]) -> str:
        """Random selection"""
        return random.choice(available)

    def _select_least_latency(self, available: List[str]) -> str:
        """Select provider with lowest average latency"""
        return min(
            available,
            key=lambda n: self._providers[n].avg_latency
        )

    def _select_least_cost(self, available: List[str]) -> str:
        """Select provider with lowest cost"""
        return min(
            available,
            key=lambda n: self._providers[n].total_cost / max(1, self._providers[n].total_requests)
        )

    def _select_weighted(self, available: List[str]) -> str:
        """Weighted random selection"""
        weights = [self._providers[n].weight for n in available]
        total = sum(weights)

        r = random.random() * total
        cumulative = 0

        for name, weight in zip(available, weights):
            cumulative += weight
            if r <= cumulative:
                return name

        return available[-1]

    def record_success(
        self,
        provider: str,
        latency_ms: float,
        cost: float = 0,
    ):
        """Record successful request"""
        if provider in self._providers:
            stats = self._providers[provider]
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.total_latency_ms += latency_ms
            stats.total_cost += cost
            stats.last_used = datetime.now(timezone.utc)

    def record_failure(self, provider: str):
        """Record failed request"""
        if provider in self._providers:
            stats = self._providers[provider]
            stats.total_requests += 1
            stats.failed_requests += 1

    def mark_unhealthy(self, provider: str):
        """Mark provider as unhealthy"""
        if provider in self._providers:
            self._providers[provider].is_healthy = False
            logger.warning(f"Provider marked unhealthy: {provider}")

    def mark_healthy(self, provider: str):
        """Mark provider as healthy"""
        if provider in self._providers:
            self._providers[provider].is_healthy = True
            logger.info(f"Provider marked healthy: {provider}")

    def set_weight(self, provider: str, weight: float):
        """Set provider weight"""
        if provider in self._providers:
            self._providers[provider].weight = weight

    def get_stats(self, provider: str) -> Optional[ProviderStats]:
        """Get provider statistics"""
        return self._providers.get(provider)

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get all provider statistics"""
        return {
            name: stats.to_dict()
            for name, stats in self._providers.items()
        }
