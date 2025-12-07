"""
LRU Cache Implementation

Memory-efficient caching with:
- LRU eviction policy
- Memory usage tracking
- TTL support
- Statistics
"""

import logging
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0
    memory_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate * 100, 2),
            "current_size": self.current_size,
            "max_size": self.max_size,
            "memory_mb": round(self.memory_bytes / (1024 * 1024), 2),
        }


@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with metadata."""
    value: V
    created_at: float
    accessed_at: float
    expires_at: Optional[float] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU cache with memory limits.
    
    Features:
    - LRU eviction when capacity is reached
    - Optional TTL for entries
    - Memory usage tracking
    - Thread-safe operations
    - Statistics collection
    
    Usage:
        cache = LRUCache(max_size=1000, max_memory_mb=512)
        cache.put("key", value)
        result = cache.get("key")
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: float = 1024,
        default_ttl_seconds: Optional[int] = None,
        size_estimator: Optional[Callable[[V], int]] = None,
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl_seconds: Default TTL for entries
            size_estimator: Function to estimate entry size in bytes
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_ttl = default_ttl_seconds
        self._size_estimator = size_estimator or self._default_size_estimator
        
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats(max_size=max_size)
        self._total_memory = 0
    
    def get(self, key: K) -> Optional[V]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self._stats.misses += 1
                return None
            
            # Update access time and move to end (most recently used)
            entry.accessed_at = time.time()
            self._cache.move_to_end(key)
            
            self._stats.hits += 1
            return entry.value
    
    def put(
        self,
        key: K,
        value: V,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL override
            
        Returns:
            True if successful
        """
        with self._lock:
            now = time.time()
            ttl = ttl_seconds or self.default_ttl
            
            # Calculate entry size
            size_bytes = self._size_estimator(value)
            
            # Check if entry fits
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Entry too large for cache: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict entries if needed
            self._evict_if_needed(size_bytes)
            
            # Create and store entry
            entry = CacheEntry(
                value=value,
                created_at=now,
                accessed_at=now,
                expires_at=now + ttl if ttl else None,
                size_bytes=size_bytes,
            )
            
            self._cache[key] = entry
            self._total_memory += size_bytes
            self._stats.current_size = len(self._cache)
            self._stats.memory_bytes = self._total_memory
            
            return True
    
    def delete(self, key: K) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def contains(self, key: K) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                self._remove_entry(key)
                return False
            return True
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._total_memory = 0
            self._stats.current_size = 0
            self._stats.memory_bytes = 0
    
    def _remove_entry(self, key: K):
        """Remove entry and update memory tracking."""
        entry = self._cache.pop(key, None)
        if entry:
            self._total_memory -= entry.size_bytes
    
    def _evict_if_needed(self, needed_bytes: int):
        """Evict entries if capacity would be exceeded."""
        # Evict by count
        while len(self._cache) >= self.max_size:
            self._evict_one()
        
        # Evict by memory
        while self._total_memory + needed_bytes > self.max_memory_bytes:
            if not self._cache:
                break
            self._evict_one()
    
    def _evict_one(self):
        """Evict the least recently used entry."""
        if not self._cache:
            return
        
        # First item is LRU (oldest)
        key = next(iter(self._cache))
        self._remove_entry(key)
        self._stats.evictions += 1
        self._stats.current_size = len(self._cache)
        self._stats.memory_bytes = self._total_memory
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            self._stats.current_size = len(self._cache)
            self._stats.memory_bytes = self._total_memory
            
            return len(expired_keys)
    
    @staticmethod
    def _default_size_estimator(value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return sys.getsizeof(value)
        except TypeError:
            # Fallback for objects without __sizeof__
            return 1024  # Assume 1KB
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)
    
    @property
    def memory_usage_mb(self) -> float:
        """Current memory usage in MB."""
        return self._total_memory / (1024 * 1024)
    
    @property
    def memory_usage_percent(self) -> float:
        """Memory usage as percentage of max."""
        if self.max_memory_bytes == 0:
            return 0.0
        return (self._total_memory / self.max_memory_bytes) * 100
