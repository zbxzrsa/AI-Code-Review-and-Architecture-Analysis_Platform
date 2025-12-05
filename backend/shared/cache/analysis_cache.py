"""
Analysis Result Caching

Implements:
- LRU cache for AI model responses
- Code analysis result caching
- Cache invalidation strategies
- CDN-friendly response headers
"""

import asyncio
import hashlib
import logging
import time
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    hits: int = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LRUCache:
    """
    LRU cache for AI model responses.
    
    Features:
    - Configurable max size
    - TTL support
    - Tag-based invalidation
    - Memory-aware eviction
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 512,
        default_ttl: int = 3600,  # 1 hour
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        import sys
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, dict):
                import json
                return len(json.dumps(value).encode('utf-8'))
            else:
                return sys.getsizeof(value)
        except (TypeError, ValueError, UnicodeEncodeError):
            # Size estimation failed, use default
            return 1024  # Default estimate
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            # Check expiration
            if datetime.now(timezone.utc) > entry.expires_at:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1
            
            return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ):
        """Set value in cache."""
        async with self._lock:
            size = self._estimate_size(value)
            ttl = ttl or self.default_ttl
            
            # Evict if necessary
            while (
                len(self._cache) >= self.max_size or
                self._stats.size_bytes + size > self.max_memory_bytes
            ):
                if not self._cache:
                    break
                self._evict_oldest()
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl),
                size_bytes=size,
                tags=tags or [],
            )
            
            # Update if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.size_bytes -= old_entry.size_bytes
            else:
                self._stats.entry_count += 1
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._stats.size_bytes += size
    
    def _evict_oldest(self):
        """Evict oldest entry."""
        if self._cache:
            _, entry = self._cache.popitem(last=False)  # key unused
            self._stats.evictions += 1
            self._stats.size_bytes -= entry.size_bytes
            self._stats.entry_count -= 1
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
                return True
            return False
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag."""
        async with self._lock:
            to_delete = [
                key for key, entry in self._cache.items()
                if tag in entry.tags
            ]
            
            for key in to_delete:
                entry = self._cache.pop(key)
                self._stats.size_bytes -= entry.size_bytes
                self._stats.entry_count -= 1
            
            return len(to_delete)
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class AnalysisCache:
    """
    Specialized cache for code analysis results.
    
    Features:
    - Content-based hashing
    - Project-level invalidation
    - Multi-level caching (memory + Redis)
    """
    
    def __init__(
        self,
        memory_cache: Optional[LRUCache] = None,
        redis_client = None,
        memory_ttl: int = 300,  # 5 minutes
        redis_ttl: int = 86400,  # 24 hours
    ):
        self.memory_cache = memory_cache or LRUCache(
            max_size=500,
            max_memory_mb=256,
            default_ttl=memory_ttl,
        )
        self.redis = redis_client
        self.memory_ttl = memory_ttl
        self.redis_ttl = redis_ttl
    
    def _hash_code(self, code: str) -> str:
        """Generate hash for code content."""
        return hashlib.sha256(code.encode()).hexdigest()
    
    def _make_key(
        self,
        code_hash: str,
        analysis_type: str,
        model: Optional[str] = None,
    ) -> str:
        """Generate cache key."""
        parts = ["analysis", analysis_type, code_hash]
        if model:
            parts.append(model)
        return ":".join(parts)
    
    async def get_analysis(
        self,
        code: str,
        analysis_type: str,
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis result."""
        code_hash = self._hash_code(code)
        key = self._make_key(code_hash, analysis_type, model)
        
        # Check memory cache first (L1)
        result = await self.memory_cache.get(key)
        if result is not None:
            logger.debug(f"Analysis cache hit (memory): {key[:32]}...")
            return result
        
        # Check Redis (L2)
        if self.redis:
            try:
                result = await self.redis.get(f"cache:{key}")
                if result:
                    # Promote to memory cache
                    await self.memory_cache.set(key, result, self.memory_ttl)
                    logger.debug(f"Analysis cache hit (redis): {key[:32]}...")
                    return result
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return None
    
    async def set_analysis(
        self,
        code: str,
        analysis_type: str,
        result: Dict[str, Any],
        model: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Cache analysis result."""
        code_hash = self._hash_code(code)
        key = self._make_key(code_hash, analysis_type, model)
        
        tags = [f"type:{analysis_type}"]
        if project_id:
            tags.append(f"project:{project_id}")
        if model:
            tags.append(f"model:{model}")
        
        # Store in memory cache (L1)
        await self.memory_cache.set(key, result, self.memory_ttl, tags)
        
        # Store in Redis (L2)
        if self.redis:
            try:
                await self.redis.set(
                    f"cache:{key}",
                    result,
                    ex=self.redis_ttl,
                )
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
    
    async def invalidate_project(self, project_id: str) -> int:
        """Invalidate all cached analyses for a project."""
        count = await self.memory_cache.invalidate_by_tag(f"project:{project_id}")
        
        if self.redis:
            try:
                # Delete from Redis by pattern
                pattern = f"cache:analysis:*:project:{project_id}:*"
                # Note: In production, use SCAN instead of KEYS
                await self.redis.delete_pattern(pattern)
            except Exception as e:
                logger.warning(f"Redis invalidation error: {e}")
        
        return count
    
    async def invalidate_model(self, model: str) -> int:
        """Invalidate all cached analyses for a model."""
        return await self.memory_cache.invalidate_by_tag(f"model:{model}")


def cached_analysis(
    analysis_type: str,
    ttl: int = 3600,
    key_func: Optional[Callable] = None,
):
    """
    Decorator for caching analysis functions.
    
    Usage:
        @cached_analysis("code_review", ttl=3600)
        async def analyze_code(code: str, options: dict):
            ...
    """
    def decorator(func):
        cache = LRUCache(default_ttl=ttl)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = hashlib.sha256(
                    f"{analysis_type}:{args}:{kwargs}".encode()
                ).hexdigest()
            
            # Check cache
            result = await cache.get(key)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(key, result, ttl)
            
            return result
        
        # Expose cache for management
        wrapper.cache = cache
        return wrapper
    
    return decorator


class CDNHeaders:
    """
    Generate CDN-friendly cache headers.
    """
    
    @staticmethod
    def public_cache(max_age: int = 3600, stale_while_revalidate: int = 60) -> Dict[str, str]:
        """Headers for public cacheable responses."""
        return {
            "Cache-Control": f"public, max-age={max_age}, stale-while-revalidate={stale_while_revalidate}",
            "Vary": "Accept-Encoding",
        }
    
    @staticmethod
    def private_cache(max_age: int = 300) -> Dict[str, str]:
        """Headers for private (user-specific) cacheable responses."""
        return {
            "Cache-Control": f"private, max-age={max_age}",
            "Vary": "Authorization, Accept-Encoding",
        }
    
    @staticmethod
    def no_cache() -> Dict[str, str]:
        """Headers for non-cacheable responses."""
        return {
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        }
    
    @staticmethod
    def immutable(max_age: int = 31536000) -> Dict[str, str]:
        """Headers for immutable resources (hashed filenames)."""
        return {
            "Cache-Control": f"public, max-age={max_age}, immutable",
        }


# Global cache instance
_analysis_cache: Optional[AnalysisCache] = None


def get_analysis_cache() -> AnalysisCache:
    """Get global analysis cache instance."""
    global _analysis_cache
    if _analysis_cache is None:
        _analysis_cache = AnalysisCache()
    return _analysis_cache
