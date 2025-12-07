"""
Caching_V1 - Backend Integration Bridge

Integrates with backend/shared/cache implementations.
"""

import sys
from typing import Optional, Dict, Any, Callable
from pathlib import Path

# Add backend path for imports
_backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

# Import backend implementations
try:
    from shared.cache import (
        AIResultCache as BackendAIResultCache,
        CacheConfig as BackendCacheConfig,
        CacheKeyGenerator as BackendCacheKeyGenerator,
        CacheEntry as BackendCacheEntry,
        CacheStats as BackendCacheStats,
        CacheTier as BackendCacheTier,
        L1MemoryCache as BackendL1Cache,
        L2RedisCache as BackendL2Cache,
        cached_ai_call as backend_cached_ai_call,
        get_ai_cache as backend_get_ai_cache,
        init_ai_cache as backend_init_ai_cache,
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    BackendAIResultCache = None
    BackendL1Cache = None
    BackendL2Cache = None


class IntegratedCacheManager:
    """
    V1 Cache Manager with backend multi-tier integration.
    """

    def __init__(
        self,
        l1_size: int = 1000,
        l1_ttl: int = 300,
        redis_url: Optional[str] = None,
        use_backend: bool = True,
    ):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            config = BackendCacheConfig(
                l1_max_size=l1_size,
                l1_ttl_seconds=l1_ttl,
                redis_url=redis_url,
            )
            self._backend = BackendAIResultCache(config)
        else:
            from .cache_manager import CacheManager
            self._local = CacheManager(l1_ttl=l1_ttl)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.use_backend:
            return await self._backend.get(key)
        return self._local.get_cascading(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        if self.use_backend:
            await self._backend.set(key, value, ttl)
        else:
            self._local.set(key, value)

    async def delete(self, key: str):
        """Delete from cache."""
        if self.use_backend:
            await self._backend.delete(key)
        else:
            self._local.delete(key)

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.use_backend:
            stats = await self._backend.get_stats()
            return stats.__dict__ if hasattr(stats, '__dict__') else stats
        return self._local.get_stats()


class IntegratedSemanticCache:
    """
    V1 Semantic Cache with backend integration.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if not self.use_backend:
            from .semantic_cache import SemanticCache
            self._local = SemanticCache()

    def get_by_code(self, code: str, language: str = "python") -> Optional[Any]:
        """Get cached result by code content."""
        if self.use_backend:
            key = BackendCacheKeyGenerator.generate_code_key(code, language)
            return backend_get_ai_cache().get_sync(key)
        return self._local.get(code, language)

    def set_by_code(self, code: str, result: Any, language: str = "python", ttl: int = 3600):
        """Cache result by code content."""
        if self.use_backend:
            key = BackendCacheKeyGenerator.generate_code_key(code, language)
            backend_get_ai_cache().set_sync(key, result, ttl)
        else:
            self._local.set(code, result, language)


class IntegratedRedisClient:
    """
    V1 Redis Client with backend integration.
    """

    def __init__(self, redis_url: Optional[str] = None, use_backend: bool = True):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            try:
                from shared.cache.redis_client import RedisClient as BackendRedisClient
                self._backend = BackendRedisClient(redis_url)
            except ImportError:
                self.use_backend = False

        if not self.use_backend:
            from .redis_client import RedisClient
            self._local = RedisClient(use_mock=True)

    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        if self.use_backend:
            return await self._backend.get(key)
        return self._local.get(key)

    async def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Set value in Redis."""
        if self.use_backend:
            await self._backend.set(key, value, ttl)
        else:
            self._local.set(key, value, ttl)

    async def delete(self, key: str):
        """Delete from Redis."""
        if self.use_backend:
            await self._backend.delete(key)
        else:
            self._local.delete(key)


def cached_ai_call(ttl: int = 3600, cache_key_func: Optional[Callable] = None):
    """
    Decorator to cache AI call results.
    """
    if BACKEND_AVAILABLE:
        return backend_cached_ai_call(ttl, cache_key_func)

    def decorator(func: Callable):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Factory functions
def get_cache_manager(use_backend: bool = True) -> IntegratedCacheManager:
    """Get integrated cache manager."""
    return IntegratedCacheManager(use_backend=use_backend)


def get_semantic_cache(use_backend: bool = True) -> IntegratedSemanticCache:
    """Get integrated semantic cache."""
    return IntegratedSemanticCache(use_backend)


def get_redis_client(redis_url: Optional[str] = None, use_backend: bool = True) -> IntegratedRedisClient:
    """Get integrated Redis client."""
    return IntegratedRedisClient(redis_url, use_backend)


__all__ = [
    "BACKEND_AVAILABLE",
    "IntegratedCacheManager",
    "IntegratedSemanticCache",
    "IntegratedRedisClient",
    "cached_ai_call",
    "get_cache_manager",
    "get_semantic_cache",
    "get_redis_client",
]
