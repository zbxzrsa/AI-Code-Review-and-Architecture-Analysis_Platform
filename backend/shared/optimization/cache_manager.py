"""
Expanded Cache Manager

Extends Redis cache usage across the application with:
- Query result caching
- API response caching
- Session caching
- Computed value caching
- Rate limiting
- Distributed locks
"""

import hashlib
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Callable, Union, TypeVar
from datetime import datetime, timezone, timedelta
from functools import wraps
from enum import Enum

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheTier(Enum):
    """Cache tier levels."""
    HOT = "hot"          # 1-5 minutes, very frequent access
    WARM = "warm"        # 5-60 minutes, moderate access
    COLD = "cold"        # 1-24 hours, infrequent access
    PERSISTENT = "persistent"  # 24+ hours, rarely changes


@dataclass
class CacheConfig:
    """
    Cache configuration with optimized defaults.
    
    TTL recommendations by data type:
    - User session: 5-15 minutes
    - API responses: 1-5 minutes
    - Query results: 5-30 minutes
    - Static data: 1-24 hours
    - AI results: 1-24 hours (based on code hash)
    """
    redis_url: str = field(default_factory=lambda: "redis://localhost:6379/0")
    
    # TTL settings by tier (seconds)
    hot_ttl: int = 300          # 5 minutes
    warm_ttl: int = 1800        # 30 minutes
    cold_ttl: int = 86400       # 24 hours
    
    # TTL settings by use case (seconds)
    session_ttl: int = 900      # 15 minutes
    query_ttl: int = 600        # 10 minutes
    response_ttl: int = 300     # 5 minutes
    ai_result_ttl: int = 3600   # 1 hour
    static_ttl: int = 86400     # 24 hours
    
    # Pool settings
    max_connections: int = 100
    
    # Feature flags
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress if > 1KB
    enable_metrics: bool = True
    
    # Key prefixes
    prefix: str = "cache:"
    session_prefix: str = "session:"
    query_prefix: str = "query:"
    response_prefix: str = "response:"
    ai_prefix: str = "ai:"


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    bytes_saved: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class CacheManager:
    """
    Expanded cache manager with comprehensive caching strategies.
    
    Covers additional use cases:
    1. Query result caching - Database query results
    2. API response caching - HTTP response caching
    3. Computed value caching - Expensive calculations
    4. User preference caching - User settings/preferences
    5. Permission caching - Authorization results
    6. Rate limit state - Rate limiting counters
    7. Distributed locks - Cross-service coordination
    
    Usage:
        cache = CacheManager()
        await cache.connect()
        
        # Cache a query result
        await cache.cache_query("users:list", users_data, ttl=600)
        
        # Get cached result
        users = await cache.get_query("users:list")
        
        # Use decorator
        @cache.cached(ttl=300)
        async def get_user(user_id: str):
            return await db.get_user(user_id)
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._redis: Optional[aioredis.Redis] = None
        self._stats = CacheStats()
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return
        
        if aioredis is None:
            logger.warning("Redis not available")
            return
        
        self._redis = aioredis.from_url(
            self.config.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=self.config.max_connections,
        )
        
        await self._redis.ping()
        self._connected = True
        logger.info("Cache manager connected to Redis")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._connected = False
    
    def _make_key(self, prefix: str, key: str) -> str:
        """Generate cache key with prefix."""
        return f"{self.config.prefix}{prefix}{key}"
    
    def _hash_key(self, data: Any) -> str:
        """Generate hash for complex keys."""
        key_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    async def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        data = json.dumps(value, default=str)
        
        # Compress if enabled and large enough
        if (
            self.config.enable_compression
            and len(data) > self.config.compression_threshold
        ):
            import gzip
            compressed = gzip.compress(data.encode())
            # Store with compression marker
            return f"gz:{compressed.hex()}"
        
        return data
    
    async def _deserialize(self, data: str) -> Any:
        """Deserialize value from storage."""
        if data.startswith("gz:"):
            import gzip
            compressed = bytes.fromhex(data[3:])
            data = gzip.decompress(compressed).decode()
        
        return json.loads(data)
    
    # ========================================
    # Core Cache Operations
    # ========================================
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._redis:
            return None
        
        try:
            data = await self._redis.get(key)
            if data:
                self._stats.hits += 1
                return await self._deserialize(data)
            else:
                self._stats.misses += 1
                return None
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tier: CacheTier = CacheTier.WARM
    ) -> bool:
        """Set value in cache."""
        if not self._redis:
            return False
        
        try:
            # Determine TTL
            if ttl is None:
                ttl = {
                    CacheTier.HOT: self.config.hot_ttl,
                    CacheTier.WARM: self.config.warm_ttl,
                    CacheTier.COLD: self.config.cold_ttl,
                    CacheTier.PERSISTENT: self.config.cold_ttl * 7,
                }.get(tier, self.config.warm_ttl)
            
            data = await self._serialize(value)
            await self._redis.setex(key, ttl, data)
            self._stats.sets += 1
            return True
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self._redis:
            return False
        
        try:
            await self._redis.delete(key)
            self._stats.deletes += 1
            return True
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self._redis:
            return 0
        
        try:
            deleted = 0
            async for key in self._redis.scan_iter(match=pattern):
                await self._redis.delete(key)
                deleted += 1
            return deleted
        except Exception as e:
            logger.error(f"Cache delete pattern error: {e}")
            return 0
    
    # ========================================
    # Query Result Caching
    # ========================================
    
    async def cache_query(
        self,
        query_key: str,
        result: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache database query result.
        
        Args:
            query_key: Unique query identifier
            result: Query result to cache
            ttl: Time to live in seconds
        
        Example:
            await cache.cache_query(
                f"users:project:{project_id}",
                users_list,
                ttl=600
            )
        """
        key = self._make_key(self.config.query_prefix, query_key)
        return await self.set(key, result, ttl or self.config.query_ttl)
    
    async def get_query(self, query_key: str) -> Optional[Any]:
        """Get cached query result."""
        key = self._make_key(self.config.query_prefix, query_key)
        return await self.get(key)
    
    async def invalidate_query(self, pattern: str) -> int:
        """Invalidate cached queries matching pattern."""
        full_pattern = self._make_key(self.config.query_prefix, pattern)
        return await self.delete_pattern(full_pattern)
    
    # ========================================
    # API Response Caching
    # ========================================
    
    async def cache_response(
        self,
        endpoint: str,
        params: Dict[str, Any],
        response: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache API response.
        
        Args:
            endpoint: API endpoint path
            params: Request parameters
            response: Response data
            ttl: Time to live in seconds
        
        Example:
            await cache.cache_response(
                "/api/projects",
                {"user_id": "123", "page": 1},
                projects_response,
                ttl=300
            )
        """
        params_hash = self._hash_key(params)
        key = self._make_key(self.config.response_prefix, f"{endpoint}:{params_hash}")
        return await self.set(key, response, ttl or self.config.response_ttl)
    
    async def get_response(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached API response."""
        params_hash = self._hash_key(params)
        key = self._make_key(self.config.response_prefix, f"{endpoint}:{params_hash}")
        return await self.get(key)
    
    async def invalidate_responses(self, endpoint: str) -> int:
        """Invalidate all cached responses for an endpoint."""
        pattern = self._make_key(self.config.response_prefix, f"{endpoint}:*")
        return await self.delete_pattern(pattern)
    
    # ========================================
    # Session Caching
    # ========================================
    
    async def cache_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache user session data."""
        key = self._make_key(self.config.session_prefix, session_id)
        return await self.set(key, data, ttl or self.config.session_ttl)
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data."""
        key = self._make_key(self.config.session_prefix, session_id)
        return await self.get(key)
    
    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update session with new data."""
        current = await self.get_session(session_id)
        if current:
            current.update(updates)
            return await self.cache_session(session_id, current)
        return False
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate user session."""
        key = self._make_key(self.config.session_prefix, session_id)
        return await self.delete(key)
    
    # ========================================
    # AI Result Caching
    # ========================================
    
    async def cache_ai_result(
        self,
        code_hash: str,
        model: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache AI analysis result."""
        key = self._make_key(self.config.ai_prefix, f"{model}:{code_hash}")
        return await self.set(key, result, ttl or self.config.ai_result_ttl, CacheTier.COLD)
    
    async def get_ai_result(
        self,
        code_hash: str,
        model: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached AI result."""
        key = self._make_key(self.config.ai_prefix, f"{model}:{code_hash}")
        return await self.get(key)
    
    # ========================================
    # Permission Caching (New)
    # ========================================
    
    async def cache_permissions(
        self,
        user_id: str,
        permissions: List[str],
        ttl: int = 300
    ) -> bool:
        """Cache user permissions for quick access control checks."""
        key = self._make_key("permissions:", user_id)
        return await self.set(key, permissions, ttl, CacheTier.HOT)
    
    async def get_permissions(self, user_id: str) -> Optional[List[str]]:
        """Get cached user permissions."""
        key = self._make_key("permissions:", user_id)
        return await self.get(key)
    
    async def check_permission(self, user_id: str, permission: str) -> Optional[bool]:
        """Check if user has specific permission from cache."""
        permissions = await self.get_permissions(user_id)
        if permissions is not None:
            return permission in permissions
        return None  # Cache miss, caller should check DB
    
    # ========================================
    # User Preferences Caching (New)
    # ========================================
    
    async def cache_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """Cache user preferences."""
        key = self._make_key("preferences:", user_id)
        return await self.set(key, preferences, ttl, CacheTier.WARM)
    
    async def get_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user preferences."""
        key = self._make_key("preferences:", user_id)
        return await self.get(key)
    
    # ========================================
    # Feature Flag Caching (New)
    # ========================================
    
    async def cache_feature_flags(
        self,
        flags: Dict[str, bool],
        ttl: int = 60
    ) -> bool:
        """Cache feature flags for quick checks."""
        key = self._make_key("", "feature_flags")
        return await self.set(key, flags, ttl, CacheTier.HOT)
    
    async def get_feature_flags(self) -> Optional[Dict[str, bool]]:
        """Get cached feature flags."""
        key = self._make_key("", "feature_flags")
        return await self.get(key)
    
    async def is_feature_enabled(self, feature: str) -> Optional[bool]:
        """Check if feature is enabled from cache."""
        flags = await self.get_feature_flags()
        if flags is not None:
            return flags.get(feature, False)
        return None
    
    # ========================================
    # Project Data Caching (New)
    # ========================================
    
    async def cache_project_metadata(
        self,
        project_id: str,
        metadata: Dict[str, Any],
        ttl: int = 1800
    ) -> bool:
        """Cache project metadata."""
        key = self._make_key("project:meta:", project_id)
        return await self.set(key, metadata, ttl, CacheTier.WARM)
    
    async def get_project_metadata(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get cached project metadata."""
        key = self._make_key("project:meta:", project_id)
        return await self.get(key)
    
    async def cache_project_files(
        self,
        project_id: str,
        file_tree: Dict[str, Any],
        ttl: int = 600
    ) -> bool:
        """Cache project file tree."""
        key = self._make_key("project:files:", project_id)
        return await self.set(key, file_tree, ttl, CacheTier.WARM)
    
    async def get_project_files(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get cached project file tree."""
        key = self._make_key("project:files:", project_id)
        return await self.get(key)
    
    # ========================================
    # Statistics and Metrics
    # ========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "sets": self._stats.sets,
            "deletes": self._stats.deletes,
            "errors": self._stats.errors,
            "hit_rate": round(self._stats.hit_rate, 2),
        }
    
    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        if not self._redis:
            return {}
        
        try:
            info = await self._redis.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}
    
    # ========================================
    # Decorator for Function Caching
    # ========================================
    
    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: Optional[str] = None,
        tier: CacheTier = CacheTier.WARM,
        skip_if: Optional[Callable[..., bool]] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Cache TTL in seconds
            key_prefix: Custom key prefix
            tier: Cache tier for default TTL
            skip_if: Function that returns True to skip caching
        
        Example:
            @cache.cached(ttl=300)
            async def get_user(user_id: str):
                return await db.get_user(user_id)
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Check skip condition
                if skip_if and skip_if(*args, **kwargs):
                    return await func(*args, **kwargs)
                
                # Generate cache key
                prefix = key_prefix or func.__name__
                key_data = {"args": args, "kwargs": kwargs}
                cache_key = self._make_key(prefix + ":", self._hash_key(key_data))
                
                # Try cache
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                
                # Execute and cache
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl, tier)
                
                return result
            
            # Add invalidation helper
            async def invalidate(*args, **kwargs):
                prefix = key_prefix or func.__name__
                key_data = {"args": args, "kwargs": kwargs}
                cache_key = self._make_key(prefix + ":", self._hash_key(key_data))
                await self.delete(cache_key)
            
            wrapper.invalidate = invalidate
            return wrapper
        return decorator


# Global cache instance
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """Get or create the global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.connect()
    return _cache_manager


# Convenience decorators
def cache_response(ttl: int = 300, key_prefix: str = "response"):
    """Decorator for caching API responses."""
    async def get_manager():
        return await get_cache_manager()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await get_manager()
            return await cache.cached(ttl=ttl, key_prefix=key_prefix)(func)(*args, **kwargs)
        return wrapper
    return decorator


def cache_query(ttl: int = 600, key_prefix: str = "query"):
    """Decorator for caching database queries."""
    async def get_manager():
        return await get_cache_manager()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await get_manager()
            return await cache.cached(ttl=ttl, key_prefix=key_prefix)(func)(*args, **kwargs)
        return wrapper
    return decorator


async def invalidate_cache(pattern: str) -> int:
    """Invalidate cache entries matching pattern."""
    cache = await get_cache_manager()
    return await cache.delete_pattern(pattern)
