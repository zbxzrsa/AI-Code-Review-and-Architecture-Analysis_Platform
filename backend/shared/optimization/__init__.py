"""
Performance Optimization Module

Provides enhanced caching, connection pooling, and batch processing utilities.
"""

from .connection_pool import (
    ConnectionPoolManager,
    PoolConfig,
    get_db_pool,
    get_redis_pool,
)
from .cache_manager import (
    CacheManager,
    CacheConfig,
    cache_response,
    cache_query,
    invalidate_cache,
)
from .batch_processor import (
    BatchProcessor,
    BatchConfig,
    batch_process,
)

__all__ = [
    # Connection Pool
    "ConnectionPoolManager",
    "PoolConfig",
    "get_db_pool",
    "get_redis_pool",
    # Cache Manager
    "CacheManager",
    "CacheConfig",
    "cache_response",
    "cache_query",
    "invalidate_cache",
    # Batch Processor
    "BatchProcessor",
    "BatchConfig",
    "batch_process",
]
