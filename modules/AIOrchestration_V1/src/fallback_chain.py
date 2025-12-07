"""
AIOrchestration_V1 - Fallback Chain

Automatic failover between AI providers.
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FallbackAttempt:
    """Record of fallback attempt"""
    provider: str
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0


@dataclass
class FallbackConfig:
    """Fallback configuration"""
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_seconds: int = 60


class FallbackChain:
    """
    Automatic failover chain for AI providers.

    Features:
    - Ordered fallback list
    - Automatic retry with backoff
    - Circuit breaker per provider
    - Fallback history tracking
    """

    def __init__(self, config: Optional[FallbackConfig] = None):
        self.config = config or FallbackConfig()

        # Provider chain (ordered)
        self._chain: List[str] = []

        # Provider functions
        self._providers: Dict[str, Callable[..., Awaitable[str]]] = {}

        # Circuit breaker state
        self._failures: Dict[str, int] = {}
        self._circuit_open: Dict[str, datetime] = {}

        # History
        self._history: List[FallbackAttempt] = []

    def add_provider(
        self,
        name: str,
        handler: Callable[..., Awaitable[str]],
        priority: int = 0,
    ):
        """Add provider to fallback chain"""
        self._providers[name] = handler
        self._failures[name] = 0

        # Insert based on priority (lower = higher priority)
        if priority >= len(self._chain):
            self._chain.append(name)
        else:
            self._chain.insert(priority, name)

        logger.info(f"Added provider to fallback chain: {name} (priority {priority})")

    def remove_provider(self, name: str):
        """Remove provider from chain"""
        if name in self._providers:
            del self._providers[name]
            self._chain.remove(name)
            del self._failures[name]
            self._circuit_open.pop(name, None)

    async def execute(
        self,
        prompt: str,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Execute with fallback.

        Returns:
            Tuple of (result, provider_name)
        """
        import time

        available = self._get_available_providers()

        if not available:
            raise RuntimeError("No available providers in fallback chain")

        last_error = None

        for attempt, provider_name in enumerate(available):
            handler = self._providers[provider_name]

            # Calculate delay for retry
            if attempt > 0:
                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)

            start_time = time.time()

            try:
                result = await handler(prompt, **kwargs)
                latency = (time.time() - start_time) * 1000

                # Record success
                self._record_success(provider_name, latency)

                return result, provider_name

            except Exception as e:
                latency = (time.time() - start_time) * 1000
                last_error = str(e)

                # Record failure
                self._record_failure(provider_name, last_error, latency)

                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def _get_available_providers(self) -> List[str]:
        """Get providers not in circuit breaker"""
        now = datetime.now(timezone.utc)
        available = []

        for name in self._chain:
            if name in self._circuit_open:
                open_time = self._circuit_open[name]
                if now - open_time < timedelta(seconds=self.config.circuit_breaker_reset_seconds):
                    continue  # Circuit still open
                else:
                    # Reset circuit
                    del self._circuit_open[name]
                    self._failures[name] = 0

            available.append(name)

        return available

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry"""
        base_delay = self.config.retry_delay_seconds

        if self.config.exponential_backoff:
            delay = base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        else:
            delay = base_delay

        return min(delay, self.config.max_delay_seconds)

    def _record_success(self, provider: str, latency_ms: float):
        """Record successful attempt"""
        self._failures[provider] = 0

        self._history.append(FallbackAttempt(
            provider=provider,
            timestamp=datetime.now(timezone.utc),
            success=True,
            latency_ms=latency_ms,
        ))

    def _record_failure(self, provider: str, error: str, latency_ms: float):
        """Record failed attempt"""
        self._failures[provider] += 1

        # Check circuit breaker
        if self._failures[provider] >= self.config.circuit_breaker_threshold:
            self._circuit_open[provider] = datetime.now(timezone.utc)
            logger.warning(f"Circuit breaker opened for: {provider}")

        self._history.append(FallbackAttempt(
            provider=provider,
            timestamp=datetime.now(timezone.utc),
            success=False,
            error=error,
            latency_ms=latency_ms,
        ))

    def get_chain_status(self) -> Dict[str, Any]:
        """Get fallback chain status"""
        now = datetime.now(timezone.utc)

        status = {}
        for name in self._chain:
            circuit_open = False
            if name in self._circuit_open:
                if now - self._circuit_open[name] < timedelta(seconds=self.config.circuit_breaker_reset_seconds):
                    circuit_open = True

            status[name] = {
                "failures": self._failures[name],
                "circuit_open": circuit_open,
                "available": not circuit_open,
            }

        return status

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get fallback history"""
        return [
            {
                "provider": a.provider,
                "timestamp": a.timestamp.isoformat(),
                "success": a.success,
                "error": a.error,
                "latency_ms": a.latency_ms,
            }
            for a in self._history[-limit:]
        ]

    def reset_circuit(self, provider: str):
        """Manually reset circuit breaker"""
        self._circuit_open.pop(provider, None)
        self._failures[provider] = 0
        logger.info(f"Circuit breaker reset for: {provider}")
