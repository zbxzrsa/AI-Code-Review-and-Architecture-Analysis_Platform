"""
Caching_V2 - Production Backend Integration

Enhanced caching with SLO-aware TTL, warming strategies, and metrics.
"""

import sys
import asyncio
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

# Add backend path
_backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

# Import backend implementations
try:
    from shared.cache import (
        AIResultCache as BackendAIResultCache,
        CacheConfig as BackendCacheConfig,
        get_ai_cache as backend_get_ai_cache,
    )
    from shared.cache.cache_strategies import CacheStrategy
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int
    misses: int
    hit_rate: float
    avg_latency_ms: float
    evictions: int
    size: int


class ProductionCacheManager:
    """
    V2 Production Cache Manager with SLO-aware TTL.

    Features:
    - Multi-tier caching (L1 memory, L2 Redis)
    - SLO-aware TTL adjustment
    - Cache warming
    - Hit rate monitoring
    """

    def __init__(
        self,
        l1_size: int = 1000,
        l1_ttl: int = 300,
        l2_ttl: int = 3600,
        slo_hit_rate: float = 0.85,
        use_backend: bool = True,
    ):
        self.l1_size = l1_size
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl
        self.slo_hit_rate = slo_hit_rate
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            config = BackendCacheConfig(
                l1_max_size=l1_size,
                l1_ttl_seconds=l1_ttl,
                l2_ttl_seconds=l2_ttl,
            )
            self._backend = BackendAIResultCache(config)
        else:
            from modules.Caching_V1.src.cache_manager import CacheManager
            self._local = CacheManager(l1_ttl=l1_ttl)

        # Metrics tracking
        self._hits = 0
        self._misses = 0
        self._latencies: List[float] = []

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with metrics."""
        import time
        start = time.time()

        if self.use_backend:
            result = await self._backend.get(key)
        else:
            result = self._local.get_cascading(key)

        latency = (time.time() - start) * 1000
        self._latencies.append(latency)

        if result is not None:
            self._hits += 1
        else:
            self._misses += 1

        # Adjust TTL based on hit rate
        self._adjust_ttl_if_needed()

        return result

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        priority: str = "normal",
    ):
        """Set value in cache with priority."""
        effective_ttl = ttl or self.l2_ttl

        # Increase TTL for high-priority items
        if priority == "high":
            effective_ttl = int(effective_ttl * 1.5)
        elif priority == "critical":
            effective_ttl = int(effective_ttl * 2)

        if self.use_backend:
            await self._backend.set(key, value, effective_ttl)
        else:
            self._local.set(key, value)

    async def delete(self, key: str):
        """Delete from cache."""
        if self.use_backend:
            await self._backend.delete(key)
        else:
            self._local.delete(key)

    async def invalidate_pattern(self, pattern: str):
        """Invalidate keys matching pattern."""
        if self.use_backend:
            await self._backend.invalidate_pattern(pattern)
        # Local implementation doesn't support patterns

    def _adjust_ttl_if_needed(self):
        """Adjust TTL based on hit rate SLO."""
        total = self._hits + self._misses
        if total < 100:
            return  # Not enough data

        hit_rate = self._hits / total

        if hit_rate < self.slo_hit_rate:
            # Increase TTL to improve hit rate
            self.l1_ttl = min(self.l1_ttl * 1.1, 600)
            self.l2_ttl = min(self.l2_ttl * 1.1, 7200)

    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0

        return CacheMetrics(
            hits=self._hits,
            misses=self._misses,
            hit_rate=hit_rate,
            avg_latency_ms=avg_latency,
            evictions=0,  # Would need backend support
            size=self.l1_size,
        )

    def get_slo_status(self) -> Dict[str, Any]:
        """Get SLO compliance status."""
        metrics = self.get_metrics()

        return {
            "hit_rate": metrics.hit_rate,
            "slo_target": self.slo_hit_rate,
            "compliant": metrics.hit_rate >= self.slo_hit_rate,
            "samples": self._hits + self._misses,
        }


class ProductionCacheWarmer:
    """
    V2 Production Cache Warmer.
    """

    def __init__(self, cache: ProductionCacheManager):
        self._cache = cache
        self._warming_tasks: Dict[str, Dict] = {}
        self._running = False

    def register_task(
        self,
        key: str,
        loader: Callable[[], Any],
        ttl: int = 3600,
        refresh_interval: int = 300,
    ):
        """Register warming task."""
        self._warming_tasks[key] = {
            "loader": loader,
            "ttl": ttl,
            "refresh_interval": refresh_interval,
            "last_refresh": None,
        }

    async def warm_all(self):
        """Warm all registered caches."""
        for key, task in self._warming_tasks.items():
            try:
                value = await task["loader"]() if asyncio.iscoroutinefunction(task["loader"]) else task["loader"]()
                await self._cache.set(key, value, task["ttl"], priority="high")
                task["last_refresh"] = datetime.now(timezone.utc)
            except Exception as e:
                # Log but continue warming other caches
                pass

    async def start_background_warming(self, interval: int = 60):
        """Start background warming loop."""
        self._running = True

        while self._running:
            now = datetime.now(timezone.utc)

            for key, task in self._warming_tasks.items():
                if task["last_refresh"] is None:
                    await self._warm_single(key)
                else:
                    elapsed = (now - task["last_refresh"]).total_seconds()
                    if elapsed >= task["refresh_interval"]:
                        await self._warm_single(key)

            await asyncio.sleep(interval)

    async def _warm_single(self, key: str):
        """Warm a single cache key."""
        task = self._warming_tasks.get(key)
        if not task:
            return

        try:
            value = await task["loader"]() if asyncio.iscoroutinefunction(task["loader"]) else task["loader"]()
            await self._cache.set(key, value, task["ttl"])
            task["last_refresh"] = datetime.now(timezone.utc)
        except Exception:
            pass

    def stop(self):
        """Stop background warming."""
        self._running = False


class ProductionSemanticCache:
    """
    V2 Production Semantic Cache with deduplication.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if not self.use_backend:
            from modules.Caching_V1.src.semantic_cache import SemanticCache
            self._local = SemanticCache()

        self._duplicates_found = 0

    def get(self, code: str, language: str = "python") -> Optional[Any]:
        """Get cached result by semantic code hash."""
        if self.use_backend:
            from shared.cache import CacheKeyGenerator
            key = CacheKeyGenerator.generate_code_key(code, language)
            return backend_get_ai_cache().get_sync(key)
        return self._local.get(code, language)

    def set(self, code: str, result: Any, language: str = "python", ttl: int = 3600):
        """Cache result with semantic key."""
        if self.use_backend:
            from shared.cache import CacheKeyGenerator
            key = CacheKeyGenerator.generate_code_key(code, language)
            backend_get_ai_cache().set_sync(key, result, ttl)
        else:
            self._local.set(code, result, language)

    def check_duplicate(self, code: str, language: str = "python") -> bool:
        """Check if code is semantically duplicate."""
        existing = self.get(code, language)
        if existing is not None:
            self._duplicates_found += 1
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics."""
        return {
            "duplicates_found": self._duplicates_found,
            "backend_available": self.use_backend,
        }


# Factory functions
def get_cache_manager(
    slo_hit_rate: float = 0.85,
    use_backend: bool = True,
) -> ProductionCacheManager:
    """Get production cache manager."""
    return ProductionCacheManager(slo_hit_rate=slo_hit_rate, use_backend=use_backend)


def get_cache_warmer(cache: ProductionCacheManager) -> ProductionCacheWarmer:
    """Get production cache warmer."""
    return ProductionCacheWarmer(cache)


def get_semantic_cache(use_backend: bool = True) -> ProductionSemanticCache:
    """Get production semantic cache."""
    return ProductionSemanticCache(use_backend)


__all__ = [
    "BACKEND_AVAILABLE",
    "CacheMetrics",
    "ProductionCacheManager",
    "ProductionCacheWarmer",
    "ProductionSemanticCache",
    "get_cache_manager",
    "get_cache_warmer",
    "get_semantic_cache",
]
