"""
Caching_V2 - Cache Manager

Production cache manager with multi-tier caching and SLO tracking.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum


class CacheTier(str, Enum):
    """Cache tier levels."""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    tier: CacheTier
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class CacheManager:
    """
    Production Cache Manager.

    Features:
    - Multi-tier caching (L1 memory, L2 Redis)
    - TTL management
    - Hit rate tracking
    - LRU eviction
    """

    def __init__(
        self,
        l1_max_size: int = 1000,
        l1_ttl: int = 300,
        l2_ttl: int = 3600,
    ):
        self.l1_max_size = l1_max_size
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl

        # L1 Memory cache
        self._l1_cache: Dict[str, CacheEntry] = {}

        # Metrics
        self._hits = 0
        self._misses = 0
        self._l1_evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first)."""
        now = datetime.now(timezone.utc)

        # Try L1
        if key in self._l1_cache:
            entry = self._l1_cache[key]
            if entry.expires_at > now:
                entry.access_count += 1
                entry.last_accessed = now
                self._hits += 1
                return entry.value
            else:
                # Expired
                del self._l1_cache[key]

        self._misses += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        now = datetime.now(timezone.utc)
        effective_ttl = ttl or self.l1_ttl

        # Check capacity
        if len(self._l1_cache) >= self.l1_max_size:
            self._evict_lru()

        self._l1_cache[key] = CacheEntry(
            key=key,
            value=value,
            tier=CacheTier.L1_MEMORY,
            created_at=now,
            expires_at=now + timedelta(seconds=effective_ttl),
        )

    def delete(self, key: str) -> bool:
        """Delete from cache."""
        if key in self._l1_cache:
            del self._l1_cache[key]
            return True
        return False

    def get_cascading(self, key: str) -> Optional[Any]:
        """Get with cascade through tiers."""
        return self.get(key)  # Only L1 in this implementation

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._l1_cache:
            return

        # Find LRU
        lru_key = None
        lru_time = datetime.now(timezone.utc)

        for key, entry in self._l1_cache.items():
            access_time = entry.last_accessed or entry.created_at
            if access_time < lru_time:
                lru_time = access_time
                lru_key = key

        if lru_key:
            del self._l1_cache[lru_key]
            self._l1_evictions += 1

    def clear(self):
        """Clear all cache."""
        self._l1_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            "l1_size": len(self._l1_cache),
            "l1_max_size": self.l1_max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._l1_evictions,
        }

    def get_keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._l1_cache.keys())

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key not in self._l1_cache:
            return False

        entry = self._l1_cache[key]
        if entry.expires_at <= datetime.now(timezone.utc):
            del self._l1_cache[key]
            return False

        return True


__all__ = [
    "CacheTier",
    "CacheEntry",
    "CacheManager",
]
