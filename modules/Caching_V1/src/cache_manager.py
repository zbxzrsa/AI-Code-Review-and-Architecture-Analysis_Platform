"""
Caching_V1 - Cache Manager

Multi-level cache management with TTL support.
"""

import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, TypeVar, Generic
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata"""
    key: str
    value: T
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class CacheLevel:
    """Single cache level"""

    def __init__(self, name: str, ttl_seconds: int, max_size: int = 1000):
        self.name = name
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}

        # Stats
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        entry = self._cache.get(key)

        if entry is None:
            self.misses += 1
            return None

        if entry.is_expired():
            del self._cache[key]
            self.misses += 1
            return None

        entry.access_count += 1
        entry.last_accessed = datetime.now(timezone.utc)
        self.hits += 1

        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._evict_lru()

        ttl = ttl or self.ttl_seconds
        now = datetime.now(timezone.utc)

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl) if ttl else None,
        )

    def delete(self, key: str) -> bool:
        """Delete from cache"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self):
        """Clear all entries"""
        self._cache.clear()

    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._cache:
            return

        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed or self._cache[k].created_at
        )
        del self._cache[lru_key]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    @property
    def size(self) -> int:
        return len(self._cache)


class CacheManager:
    """
    Multi-level cache manager.

    Levels:
    - L1: Session cache (short TTL)
    - L2: Project cache (medium TTL)
    - L3: Global cache (long TTL)
    """

    def __init__(
        self,
        l1_ttl: int = 300,      # 5 minutes
        l2_ttl: int = 3600,     # 1 hour
        l3_ttl: int = 86400,    # 24 hours
    ):
        self.levels = {
            "l1": CacheLevel("session", l1_ttl, max_size=500),
            "l2": CacheLevel("project", l2_ttl, max_size=2000),
            "l3": CacheLevel("global", l3_ttl, max_size=10000),
        }

    def get(self, key: str, level: str = "l2") -> Optional[Any]:
        """Get from specific level"""
        if level not in self.levels:
            return None
        return self.levels[level].get(key)

    def get_cascading(self, key: str) -> Optional[Any]:
        """Get from cache, checking all levels"""
        for level in ["l1", "l2", "l3"]:
            value = self.levels[level].get(key)
            if value is not None:
                # Promote to faster levels
                if level == "l3":
                    self.levels["l2"].set(key, value)
                elif level == "l2":
                    self.levels["l1"].set(key, value)
                return value
        return None

    def set(self, key: str, value: Any, level: str = "l2", ttl: Optional[int] = None):
        """Set in specific level"""
        if level in self.levels:
            self.levels[level].set(key, value, ttl)

    def set_all_levels(self, key: str, value: Any):
        """Set in all levels"""
        for level in self.levels.values():
            level.set(key, value)

    def delete(self, key: str, level: Optional[str] = None):
        """Delete from cache"""
        if level:
            if level in self.levels:
                self.levels[level].delete(key)
        else:
            for lvl in self.levels.values():
                lvl.delete(key)

    def clear(self, level: Optional[str] = None):
        """Clear cache"""
        if level:
            if level in self.levels:
                self.levels[level].clear()
        else:
            for lvl in self.levels.values():
                lvl.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            level_name: {
                "size": level.size,
                "max_size": level.max_size,
                "hits": level.hits,
                "misses": level.misses,
                "hit_rate": level.hit_rate,
            }
            for level_name, level in self.levels.items()
        }

    @staticmethod
    def generate_key(*args) -> str:
        """Generate cache key from arguments"""
        key_str = ":".join(str(arg) for arg in args)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
