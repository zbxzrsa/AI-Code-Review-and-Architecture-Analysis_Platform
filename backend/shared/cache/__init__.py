"""
Cache modules for Redis-based caching.

Includes:
- AI Result Cache: Multi-tier caching for AI call results
- Redis Client: Distributed caching utilities
"""
from .ai_result_cache import (
    AIResultCache,
    CacheConfig,
    CacheKeyGenerator,
    CacheEntry,
    CacheStats,
    CacheTier,
    L1MemoryCache,
    L2RedisCache,
    cached_ai_call,
    get_ai_cache,
    init_ai_cache,
)
