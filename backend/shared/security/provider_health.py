"""
Provider health tracking and monitoring.

Tracks provider availability, response times, error rates, and automatic failover.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ProviderHealthTracker:
    """Track and monitor provider health."""

    def __init__(self, redis_client):
        """Initialize health tracker."""
        self.redis = redis_client

    # ============================================
    # Health Status Management
    # ============================================

    def mark_provider_healthy(
        self,
        provider: str,
        response_time_ms: float = None,
        ttl: int = 300
    ) -> bool:
        """Mark provider as healthy."""
        try:
            key = f"provider:{provider}:health_status"
            health_data = {
                "status": HealthStatus.HEALTHY.value,
                "last_check": datetime.utcnow().isoformat(),
                "consecutive_failures": 0,
                "response_time_ms": response_time_ms
            }

            self.redis.set_global_cache(key, health_data, ttl)
            logger.info(f"Provider {provider} marked as healthy")
            return True
        except Exception as e:
            logger.error(f"Failed to mark provider healthy: {e}")
            return False

    def mark_provider_degraded(
        self,
        provider: str,
        reason: str = None,
        ttl: int = 300
    ) -> bool:
        """Mark provider as degraded."""
        try:
            key = f"provider:{provider}:health_status"
            health_data = {
                "status": HealthStatus.DEGRADED.value,
                "last_check": datetime.utcnow().isoformat(),
                "reason": reason,
                "consecutive_failures": self._get_consecutive_failures(provider) + 1
            }

            self.redis.set_global_cache(key, health_data, ttl)
            logger.warning(f"Provider {provider} marked as degraded: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark provider degraded: {e}")
            return False

    def mark_provider_unhealthy(
        self,
        provider: str,
        reason: str = None,
        duration: int = 300
    ) -> bool:
        """Mark provider as unhealthy."""
        try:
            key = f"provider:{provider}:health_status"
            health_data = {
                "status": HealthStatus.UNHEALTHY.value,
                "last_check": datetime.utcnow().isoformat(),
                "reason": reason,
                "consecutive_failures": self._get_consecutive_failures(provider) + 1,
                "unhealthy_until": (datetime.utcnow() + timedelta(seconds=duration)).isoformat()
            }

            self.redis.set_global_cache(key, health_data, duration)
            logger.error(f"Provider {provider} marked as unhealthy: {reason}")

            # Publish unhealthy event
            self.redis.publish_event(
                channel="events:provider_unhealthy",
                message={
                    "provider": provider,
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            return True
        except Exception as e:
            logger.error(f"Failed to mark provider unhealthy: {e}")
            return False

    # ============================================
    # Health Status Queries
    # ============================================

    def is_provider_healthy(self, provider: str) -> bool:
        """Check if provider is healthy."""
        try:
            status = self.get_provider_status(provider)
            return status and status.get("status") == HealthStatus.HEALTHY.value
        except Exception as e:
            logger.error(f"Failed to check provider health: {e}")
            return False

    def is_provider_available(self, provider: str) -> bool:
        """Check if provider is available (healthy or degraded)."""
        try:
            status = self.get_provider_status(provider)
            if not status:
                return True  # Assume healthy if no status

            health_status = status.get("status")
            return health_status in [HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value]
        except Exception as e:
            logger.error(f"Failed to check provider availability: {e}")
            return True  # Assume available on error

    def get_provider_status(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get detailed provider status."""
        try:
            key = f"provider:{provider}:health_status"
            return self.redis.get_global_cache(key)
        except Exception as e:
            logger.error(f"Failed to get provider status: {e}")
            return None

    # ============================================
    # Failure Tracking
    # ============================================

    def record_provider_failure(
        self,
        provider: str,
        error: str,
        response_time_ms: float = None
    ) -> bool:
        """Record provider failure."""
        try:
            # Increment failure counter
            failure_key = f"provider:{provider}:failures"
            failures = self.redis.client.incr(failure_key)
            self.redis.client.expire(failure_key, 3600)  # 1 hour window

            # Update health status
            consecutive = self._get_consecutive_failures(provider) + 1

            if consecutive >= 5:
                # Mark unhealthy after 5 consecutive failures
                self.mark_provider_unhealthy(
                    provider=provider,
                    reason=f"5 consecutive failures: {error}",
                    duration=300
                )
            elif consecutive >= 3:
                # Mark degraded after 3 consecutive failures
                self.mark_provider_degraded(
                    provider=provider,
                    reason=f"Multiple failures: {error}"
                )

            # Log failure
            logger.warning(
                f"Provider {provider} failure #{failures}: {error} "
                f"(response_time: {response_time_ms}ms)"
            )

            return True
        except Exception as e:
            logger.error(f"Failed to record provider failure: {e}")
            return False

    def record_provider_success(
        self,
        provider: str,
        response_time_ms: float
    ) -> bool:
        """Record provider success."""
        try:
            # Reset failure counter
            failure_key = f"provider:{provider}:failures"
            self.redis.client.delete(failure_key)

            # Update health status
            self.mark_provider_healthy(
                provider=provider,
                response_time_ms=response_time_ms
            )

            logger.debug(f"Provider {provider} success (response_time: {response_time_ms}ms)")
            return True
        except Exception as e:
            logger.error(f"Failed to record provider success: {e}")
            return False

    def _get_consecutive_failures(self, provider: str) -> int:
        """Get consecutive failure count."""
        try:
            failure_key = f"provider:{provider}:failures"
            count = self.redis.client.get(failure_key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"Failed to get failure count: {e}")
            return 0

    # ============================================
    # Health Checks
    # ============================================

    async def perform_health_check(
        self,
        provider: str,
        check_func
    ) -> bool:
        """Perform health check on provider."""
        try:
            start_time = datetime.utcnow()

            # Execute health check
            result = await check_func()

            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            if result:
                self.record_provider_success(provider, response_time)
                return True
            else:
                self.record_provider_failure(provider, "Health check failed", response_time)
                return False
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.record_provider_failure(provider, str(e), response_time)
            return False

    async def schedule_health_checks(
        self,
        providers: Dict[str, callable],
        interval: int = 300
    ) -> None:
        """Schedule periodic health checks."""
        while True:
            try:
                tasks = [
                    self.perform_health_check(provider, check_func)
                    for provider, check_func in providers.items()
                ]
                await asyncio.gather(*tasks)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Health check scheduling failed: {e}")
                await asyncio.sleep(interval)

    # ============================================
    # Statistics
    # ============================================

    def get_provider_stats(self, provider: str) -> Dict[str, Any]:
        """Get provider statistics."""
        try:
            status = self.get_provider_status(provider)
            failures = self._get_consecutive_failures(provider)

            return {
                "provider": provider,
                "status": status.get("status") if status else "unknown",
                "last_check": status.get("last_check") if status else None,
                "consecutive_failures": failures,
                "response_time_ms": status.get("response_time_ms") if status else None,
                "reason": status.get("reason") if status else None
            }
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {}

    def get_all_providers_stats(self, providers: list) -> Dict[str, Any]:
        """Get statistics for all providers."""
        try:
            stats = {}
            for provider in providers:
                stats[provider] = self.get_provider_stats(provider)

            return {
                "providers": stats,
                "timestamp": datetime.utcnow().isoformat(),
                "healthy_count": sum(
                    1 for p in stats.values()
                    if p.get("status") == HealthStatus.HEALTHY.value
                ),
                "degraded_count": sum(
                    1 for p in stats.values()
                    if p.get("status") == HealthStatus.DEGRADED.value
                ),
                "unhealthy_count": sum(
                    1 for p in stats.values()
                    if p.get("status") == HealthStatus.UNHEALTHY.value
                )
            }
        except Exception as e:
            logger.error(f"Failed to get all providers stats: {e}")
            return {}

    # ============================================
    # Failover Management
    # ============================================

    def get_healthy_provider(
        self,
        providers: list,
        prefer_healthy: bool = True
    ) -> Optional[str]:
        """Get a healthy provider from the list."""
        try:
            if prefer_healthy:
                # Try to find a healthy provider
                for provider in providers:
                    if self.is_provider_healthy(provider):
                        return provider

            # Fall back to available provider
            for provider in providers:
                if self.is_provider_available(provider):
                    return provider

            # All providers unavailable, return first
            return providers[0] if providers else None
        except Exception as e:
            logger.error(f"Failed to get healthy provider: {e}")
            return providers[0] if providers else None

    def get_fallback_chain(
        self,
        primary: str,
        fallbacks: list
    ) -> list:
        """Get ordered fallback chain based on health."""
        try:
            chain = [primary]

            # Sort fallbacks by health status
            fallback_stats = [
                (p, self.get_provider_status(p))
                for p in fallbacks
            ]

            # Healthy providers first
            healthy = [
                p for p, status in fallback_stats
                if status and status.get("status") == HealthStatus.HEALTHY.value
            ]

            # Degraded providers next
            degraded = [
                p for p, status in fallback_stats
                if status and status.get("status") == HealthStatus.DEGRADED.value
            ]

            # Unhealthy providers last
            unhealthy = [
                p for p, status in fallback_stats
                if status and status.get("status") == HealthStatus.UNHEALTHY.value
            ]

            chain.extend(healthy)
            chain.extend(degraded)
            chain.extend(unhealthy)

            return chain
        except Exception as e:
            logger.error(f"Failed to get fallback chain: {e}")
            return [primary] + fallbacks
