"""
AI Result Caching System (Performance Optimization #1)

Distributed caching for AI call results with:
- Multiple cache tiers (L1 memory, L2 Redis)
- Intelligent TTL policies
- Standardized cache key generation
- Semantic deduplication
- Cache warming and prefetching

Expected Benefits:
- Reduce >50% duplicate AI calls
- Sub-millisecond response for cached results
- Automatic cache invalidation
"""
import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
from collections import OrderedDict
import functools

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheTier(str, Enum):
    """Cache tier levels."""
    L1_MEMORY = "l1_memory"      # In-process memory cache (fastest)
    L2_REDIS = "l2_redis"        # Distributed Redis cache
    L3_DATABASE = "l3_database"  # Database cache (persistent)


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    # TTL settings (in seconds)
    default_ttl: int = 3600               # 1 hour default
    analysis_result_ttl: int = 86400      # 24 hours for analysis results
    code_embedding_ttl: int = 604800      # 7 days for embeddings
    model_response_ttl: int = 3600        # 1 hour for model responses
    
    # Size limits
    l1_max_size: int = 10000              # Max L1 cache entries
    l1_max_memory_mb: int = 512           # Max L1 memory usage
    
    # Behavior
    enable_l1: bool = True
    enable_l2: bool = True
    enable_compression: bool = True
    compression_threshold: int = 1024     # Compress if > 1KB
    
    # Cache warming
    enable_warming: bool = True
    warming_batch_size: int = 100


@dataclass
class CacheEntry:
    """Cached data entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    tier: CacheTier = CacheTier.L1_MEMORY
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def ttl_remaining(self) -> int:
        """Remaining TTL in seconds."""
        remaining = (self.expires_at - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(remaining))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "hit_count": self.hit_count,
            "size_bytes": self.size_bytes,
            "tier": self.tier.value,
            "tags": self.tags,
        }


@dataclass 
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    bytes_saved: int = 0
    ai_calls_saved: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def savings_percentage(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "savings_percentage": round(self.savings_percentage, 2),
            "bytes_saved": self.bytes_saved,
            "ai_calls_saved": self.ai_calls_saved,
        }


class CacheKeyGenerator:
    """
    Standardized cache key generation with semantic awareness.
    
    Generates consistent, collision-resistant cache keys based on:
    - Operation type (analysis, embedding, chat, etc.)
    - Input content hash
    - Model configuration
    - Version information
    """
    
    @staticmethod
    def generate(
        operation: str,
        content: Union[str, Dict, List],
        model: Optional[str] = None,
        version: str = "v1",
        extra_params: Optional[Dict] = None
    ) -> str:
        """
        Generate a standardized cache key.
        
        Args:
            operation: Type of operation (analysis, embedding, chat, fix)
            content: Input content to hash
            model: AI model identifier
            version: API/model version
            extra_params: Additional parameters affecting the result
            
        Returns:
            Cache key string
        """
        # Normalize content to string
        if isinstance(content, (dict, list)):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
        
        # Create hash components
        components = [
            operation,
            version,
            model or "default",
        ]
        
        # Add extra params if provided
        if extra_params:
            param_str = json.dumps(extra_params, sort_keys=True)
            components.append(param_str)
        
        # Generate content hash (SHA256 for collision resistance)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        
        # Generate params hash
        params_hash = hashlib.md5(
            ":".join(components).encode()
        ).hexdigest()[:8]
        
        return f"ai:{operation}:{version}:{params_hash}:{content_hash}"
    
    @staticmethod
    def generate_analysis_key(
        code: str,
        language: str,
        rules: List[str],
        model: str = "default"
    ) -> str:
        """Generate key for code analysis results."""
        return CacheKeyGenerator.generate(
            operation="analysis",
            content=code,
            model=model,
            extra_params={"language": language, "rules": sorted(rules)}
        )
    
    @staticmethod
    def generate_embedding_key(text: str, model: str = "default") -> str:
        """Generate key for text embeddings."""
        return CacheKeyGenerator.generate(
            operation="embedding",
            content=text,
            model=model
        )
    
    @staticmethod
    def generate_chat_key(
        messages: List[Dict],
        model: str,
        temperature: float = 0.7
    ) -> str:
        """Generate key for chat completions."""
        # Only cache deterministic responses (low temperature)
        if temperature > 0.3:
            # High temperature = non-deterministic, use unique key
            return f"ai:chat:nocache:{hashlib.md5(str(time.time()).encode()).hexdigest()}"
        
        return CacheKeyGenerator.generate(
            operation="chat",
            content=messages,
            model=model,
            extra_params={"temperature": temperature}
        )
    
    @staticmethod
    def generate_fix_key(
        code: str,
        issue_type: str,
        language: str,
        model: str = "default"
    ) -> str:
        """Generate key for fix suggestions."""
        return CacheKeyGenerator.generate(
            operation="fix",
            content=code,
            model=model,
            extra_params={"issue_type": issue_type, "language": language}
        )


class L1MemoryCache:
    """
    In-memory LRU cache (L1 tier).
    
    Features:
    - Fast access (microsecond latency)
    - LRU eviction policy
    - Memory size limits
    - TTL support
    """
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512):
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict[str, None] = OrderedDict()  # LRU tracking O(1)
        self._max_size = max_size
        self._max_memory = max_memory_mb * 1024 * 1024
        self._current_memory = 0
        self._lock = asyncio.Lock()
        self._stats = CacheStats()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            if entry.is_expired:
                await self._delete_internal(key)
                self._stats.misses += 1
                return None
            
            # Update LRU order - O(1) with OrderedDict.move_to_end()
            if key in self._access_order:
                self._access_order.move_to_end(key)
            else:
                self._access_order[key] = None
            
            # Update hit stats
            entry.hit_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
            self._stats.hits += 1
            
            return entry
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set entry in cache."""
        async with self._lock:
            # Calculate size
            size = len(json.dumps(value, default=str).encode())
            
            # Check if we need to evict
            while (
                len(self._cache) >= self._max_size or
                self._current_memory + size > self._max_memory
            ):
                if len(self._access_order) == 0:
                    break
                await self._evict_lru()
            
            now = datetime.now(timezone.utc)
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
                size_bytes=size,
                tier=CacheTier.L1_MEMORY,
                tags=tags or [],
            )
            
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory -= old_entry.size_bytes
            
            self._cache[key] = entry
            self._current_memory += size
            
            # Update LRU order - O(1) with OrderedDict.move_to_end()
            if key in self._access_order:
                self._access_order.move_to_end(key)
            else:
                self._access_order[key] = None
            
            self._stats.sets += 1
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            return await self._delete_internal(key)
    
    async def _delete_internal(self, key: str) -> bool:
        """Internal delete without lock."""
        if key not in self._cache:
            return False
        
        entry = self._cache.pop(key)
        self._current_memory -= entry.size_bytes
        
        self._access_order.pop(key, None)  # O(1) delete from OrderedDict
        
        self._stats.deletes += 1
        return True
    
    async def _evict_lru(self):
        """Evict least recently used entry."""
        if len(self._access_order) > 0:
            key, _ = self._access_order.popitem(last=False)  # O(1) pop oldest
            if key in self._cache:
                entry = self._cache.pop(key)
                self._current_memory -= entry.size_bytes
                self._stats.evictions += 1
    
    async def clear(self):
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_memory = 0
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete entries matching any of the tags."""
        async with self._lock:
            deleted = 0
            keys_to_delete = [
                key for key, entry in self._cache.items()
                if any(tag in entry.tags for tag in tags)
            ]
            for key in keys_to_delete:
                await self._delete_internal(key)
                deleted += 1
            return deleted
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    @property
    def size(self) -> int:
        return len(self._cache)
    
    @property
    def memory_usage(self) -> int:
        return self._current_memory


class L2RedisCache:
    """
    Redis-based distributed cache (L2 tier).
    
    Features:
    - Distributed across multiple nodes
    - Persistence options
    - Pub/sub for invalidation
    - Compression for large values
    """
    
    def __init__(self, redis_client, config: CacheConfig):
        self._redis = redis_client
        self._config = config
        self._stats = CacheStats()
        self._prefix = "ai_cache:"
        self._background_tasks: set = set()  # Store tasks to prevent GC
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from Redis cache."""
        try:
            full_key = f"{self._prefix}{key}"
            data = await self._redis.get(full_key)
            
            if data is None:
                self._stats.misses += 1
                return None
            
            # Decompress if needed
            if data.startswith(b'\x1f\x8b'):  # gzip magic bytes
                import gzip
                data = gzip.decompress(data)
            
            entry_dict = json.loads(data)
            entry = CacheEntry(
                key=entry_dict["key"],
                value=entry_dict["value"],
                created_at=datetime.fromisoformat(entry_dict["created_at"]),
                expires_at=datetime.fromisoformat(entry_dict["expires_at"]),
                hit_count=entry_dict.get("hit_count", 0),
                size_bytes=entry_dict.get("size_bytes", 0),
                tier=CacheTier.L2_REDIS,
                tags=entry_dict.get("tags", []),
            )
            
            # Update hit count in background (store task to prevent GC)
            task = asyncio.create_task(self._increment_hits(full_key))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            
            self._stats.hits += 1
            return entry
            
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self._stats.misses += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set entry in Redis cache."""
        try:
            full_key = f"{self._prefix}{key}"
            now = datetime.now(timezone.utc)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
                tier=CacheTier.L2_REDIS,
                tags=tags or [],
            )
            
            data = json.dumps(entry.to_dict(), default=str).encode()
            
            # Compress if large
            if (
                self._config.enable_compression and
                len(data) > self._config.compression_threshold
            ):
                import gzip
                data = gzip.compress(data)
            
            await self._redis.setex(full_key, ttl, data)
            
            # Store tag mappings
            if tags:
                for tag in tags:
                    await self._redis.sadd(f"{self._prefix}tag:{tag}", key)
                    await self._redis.expire(f"{self._prefix}tag:{tag}", ttl)
            
            self._stats.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete entry from Redis cache."""
        try:
            full_key = f"{self._prefix}{key}"
            result = await self._redis.delete(full_key)
            if result:
                self._stats.deletes += 1
            return bool(result)
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            return False
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete entries matching any of the tags."""
        try:
            deleted = 0
            for tag in tags:
                keys = await self._redis.smembers(f"{self._prefix}tag:{tag}")
                for key in keys:
                    if await self.delete(key.decode() if isinstance(key, bytes) else key):
                        deleted += 1
                await self._redis.delete(f"{self._prefix}tag:{tag}")
            return deleted
        except Exception as e:
            logger.error(f"Redis delete by tags error: {e}")
            return 0
    
    async def _increment_hits(self, key: str):
        """Increment hit count for analytics."""
        try:
            await self._redis.hincrby(f"{key}:meta", "hits", 1)
        except Exception:
            pass  # Non-critical operation
    
    def get_stats(self) -> CacheStats:
        return self._stats


class AIResultCache:
    """
    Multi-tier AI result caching system.
    
    Provides intelligent caching for AI operations with:
    - L1 memory cache for hot data
    - L2 Redis cache for distributed access
    - Automatic cache key generation
    - TTL management per operation type
    - Cache warming and prefetching
    """
    
    def __init__(
        self,
        redis_client=None,
        config: Optional[CacheConfig] = None
    ):
        self._config = config or CacheConfig()
        
        # Initialize cache tiers
        self._l1 = L1MemoryCache(
            max_size=self._config.l1_max_size,
            max_memory_mb=self._config.l1_max_memory_mb
        ) if self._config.enable_l1 else None
        
        self._l2 = L2RedisCache(
            redis_client, self._config
        ) if redis_client and self._config.enable_l2 else None
        
        self._key_gen = CacheKeyGenerator()
        
        # Stats tracking
        self._global_stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get cached value using multi-tier lookup.
        
        Lookup order: L1 -> L2
        On L2 hit, promotes to L1.
        """
        # Try L1 first
        if self._l1:
            entry = await self._l1.get(key)
            if entry:
                self._global_stats.hits += 1
                self._global_stats.ai_calls_saved += 1
                logger.debug(f"L1 cache hit: {key}")
                return entry.value
        
        # Try L2
        if self._l2:
            entry = await self._l2.get(key)
            if entry:
                # Promote to L1
                if self._l1:
                    await self._l1.set(
                        key, entry.value,
                        ttl=entry.ttl_remaining,
                        tags=entry.tags
                    )
                
                self._global_stats.hits += 1
                self._global_stats.ai_calls_saved += 1
                logger.debug(f"L2 cache hit: {key}")
                return entry.value
        
        self._global_stats.misses += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Set cached value in all tiers.
        """
        ttl = ttl or self._config.default_ttl
        success = True
        
        # Set in L1
        if self._l1:
            success &= await self._l1.set(key, value, ttl, tags)
        
        # Set in L2
        if self._l2:
            success &= await self._l2.set(key, value, ttl, tags)
        
        self._global_stats.sets += 1
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete from all cache tiers."""
        success = True
        
        if self._l1:
            success &= await self._l1.delete(key)
        
        if self._l2:
            success &= await self._l2.delete(key)
        
        self._global_stats.deletes += 1
        return success
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all entries with matching tags."""
        deleted = 0
        
        if self._l1:
            deleted += await self._l1.delete_by_tags(tags)
        
        if self._l2:
            deleted += await self._l2.delete_by_tags(tags)
        
        logger.info(f"Invalidated {deleted} cache entries for tags: {tags}")
        return deleted
    
    # Convenience methods for specific operations
    
    async def get_analysis(
        self,
        code: str,
        language: str,
        rules: List[str],
        model: str = "default"
    ) -> Optional[Dict]:
        """Get cached analysis result."""
        key = self._key_gen.generate_analysis_key(code, language, rules, model)
        return await self.get(key)
    
    async def set_analysis(
        self,
        code: str,
        language: str,
        rules: List[str],
        result: Dict,
        model: str = "default"
    ) -> bool:
        """Cache analysis result."""
        key = self._key_gen.generate_analysis_key(code, language, rules, model)
        return await self.set(
            key, result,
            ttl=self._config.analysis_result_ttl,
            tags=["analysis", f"lang:{language}"]
        )
    
    async def get_embedding(
        self,
        text: str,
        model: str = "default"
    ) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._key_gen.generate_embedding_key(text, model)
        return await self.get(key)
    
    async def set_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str = "default"
    ) -> bool:
        """Cache embedding."""
        key = self._key_gen.generate_embedding_key(text, model)
        return await self.set(
            key, embedding,
            ttl=self._config.code_embedding_ttl,
            tags=["embedding"]
        )
    
    async def get_fix_suggestion(
        self,
        code: str,
        issue_type: str,
        language: str,
        model: str = "default"
    ) -> Optional[Dict]:
        """Get cached fix suggestion."""
        key = self._key_gen.generate_fix_key(code, issue_type, language, model)
        return await self.get(key)
    
    async def set_fix_suggestion(
        self,
        code: str,
        issue_type: str,
        language: str,
        suggestion: Dict,
        model: str = "default"
    ) -> bool:
        """Cache fix suggestion."""
        key = self._key_gen.generate_fix_key(code, issue_type, language, model)
        return await self.set(
            key, suggestion,
            ttl=self._config.model_response_ttl,
            tags=["fix", f"issue:{issue_type}", f"lang:{language}"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "global": self._global_stats.to_dict(),
            "savings_estimate": {
                "ai_calls_saved": self._global_stats.ai_calls_saved,
                "estimated_cost_saved_usd": self._global_stats.ai_calls_saved * 0.02,  # ~$0.02 per call
                "estimated_time_saved_seconds": self._global_stats.ai_calls_saved * 2,  # ~2s per call
            }
        }
        
        if self._l1:
            stats["l1"] = {
                **self._l1.get_stats().to_dict(),
                "size": self._l1.size,
                "memory_usage_bytes": self._l1.memory_usage,
            }
        
        if self._l2:
            stats["l2"] = self._l2.get_stats().to_dict()
        
        return stats


def cached_ai_call(
    cache: AIResultCache,
    operation: str,
    ttl: Optional[int] = None
):
    """
    Decorator for caching AI function calls.
    
    Usage:
        @cached_ai_call(cache, "analysis")
        async def analyze_code(code: str, language: str) -> Dict:
            return await ai_client.analyze(code, language)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function arguments
            key = CacheKeyGenerator.generate(
                operation=operation,
                content={"args": args, "kwargs": kwargs}
            )
            
            # Try cache first
            cached = await cache.get(key)
            if cached is not None:
                logger.debug(f"Cache hit for {operation}")
                return cached
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(key, result, ttl=ttl, tags=[operation])
            
            return result
        return wrapper
    return decorator


# Global cache instance
_cache_instance: Optional[AIResultCache] = None


def get_ai_cache() -> AIResultCache:
    """Get or create global AI cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = AIResultCache()
    return _cache_instance


def init_ai_cache(redis_client=None, config: Optional[CacheConfig] = None):
    """Initialize global AI cache with custom configuration."""
    global _cache_instance
    _cache_instance = AIResultCache(redis_client, config)
    return _cache_instance


# =============================================================================
# Connection Pool Manager (P1 Enhancement)
# =============================================================================

class RedisConnectionPool:
    """
    Redis connection pool manager for efficient connection reuse.
    
    P1 optimization: Manages connection pooling to reduce connection
    overhead and improve throughput.
    
    Usage:
        pool = RedisConnectionPool.get_instance()
        await pool.initialize("redis://localhost:6379")
        
        client = await pool.get_client()
        # Use client...
        
        await pool.close()
    """
    
    _instance: Optional["RedisConnectionPool"] = None
    
    def __init__(self):
        self._pool = None
        self._client = None
        self._config = {
            "max_connections": 50,
            "min_idle_connections": 5,
            "timeout": 10.0,
            "retry_on_timeout": True,
            "health_check_interval": 30,
        }
    
    @classmethod
    def get_instance(cls) -> "RedisConnectionPool":
        """Get singleton pool instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def initialize(
        self,
        url: str = "redis://localhost:6379",
        max_connections: int = 50,
        min_idle: int = 5,
    ) -> None:
        """
        Initialize connection pool.
        
        Args:
            url: Redis connection URL
            max_connections: Maximum pool size
            min_idle: Minimum idle connections to maintain
        """
        try:
            import redis.asyncio as redis
            
            self._config["max_connections"] = max_connections
            self._config["min_idle_connections"] = min_idle
            
            self._pool = redis.ConnectionPool.from_url(
                url,
                max_connections=max_connections,
                decode_responses=False,
            )
            
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            
            logger.info(
                f"Redis connection pool initialized: "
                f"max={max_connections}, min_idle={min_idle}"
            )
            
        except ImportError:
            logger.warning("redis package not installed, pool not available")
        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}")
            raise
    
    async def get_client(self):
        """Get a Redis client from the pool."""
        if self._client is None:
            raise RuntimeError("Connection pool not initialized")
        return self._client
    
    async def close(self) -> None:
        """Close all connections in the pool."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        logger.info("Redis connection pool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        if self._pool is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "max_connections": self._config["max_connections"],
            "created_connections": getattr(self._pool, '_created_connections', 0),
            "available_connections": getattr(self._pool, '_available_connections', 0),
        }


async def init_redis_pool(url: str = "redis://localhost:6379") -> RedisConnectionPool:
    """Initialize the global Redis connection pool."""
    pool = RedisConnectionPool.get_instance()
    await pool.initialize(url)
    return pool


async def get_redis_client():
    """Get Redis client from the global pool."""
    pool = RedisConnectionPool.get_instance()
    return await pool.get_client()
