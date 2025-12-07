"""
Working Memory Module

Active computation cache with TTL-based expiration. Supports:
- Fast key-value storage
- TTL-based expiration
- Priority-based eviction
- Capacity management
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryItem:
    """An item in working memory."""
    key: str
    value: Any
    created_at: float  # Unix timestamp
    expires_at: float  # Unix timestamp
    priority: float = 1.0
    access_count: int = 0
    last_accessed: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if item has expired."""
        return time.time() > self.expires_at
    
    def time_remaining(self) -> float:
        """Get time remaining until expiration."""
        return max(0, self.expires_at - time.time())


class WorkingMemory:
    """
    Working memory for active computation caching.
    
    Features:
    - TTL-based expiration
    - Capacity limits with LRU eviction
    - Priority-based retention
    - Fast access patterns
    
    Usage:
        memory = WorkingMemory(max_size=1000)
        
        # Store with TTL
        memory.store("key", {"data": "value"}, ttl_seconds=3600)
        
        # Retrieve
        value = memory.get("key")  # Returns None if expired
        
        # Check existence
        if memory.has("key"):
            ...
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: int = 3600,
        cleanup_interval: int = 60,
    ):
        """
        Initialize working memory.
        
        Args:
            max_size: Maximum number of items
            default_ttl_seconds: Default TTL in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        # Storage
        self.items: Dict[str, WorkingMemoryItem] = {}
        
        # Statistics
        self.total_stored = 0
        self.total_hits = 0
        self.total_misses = 0
        self.total_expired = 0
        
        # Last cleanup time
        self._last_cleanup = time.time()
    
    def store(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        priority: float = 1.0,
    ):
        """
        Store an item in working memory.
        
        Args:
            key: Item key
            value: Item value
            ttl_seconds: Time-to-live in seconds (uses default if None)
            priority: Item priority for eviction (higher = more important)
        """
        self._maybe_cleanup()
        
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        
        item = WorkingMemoryItem(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + ttl,
            priority=priority,
            access_count=0,
            last_accessed=now,
        )
        
        # Check capacity
        if key not in self.items and len(self.items) >= self.max_size:
            self._evict_one()
        
        self.items[key] = item
        self.total_stored += 1
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an item from working memory.
        
        Args:
            key: Item key
            default: Default value if not found or expired
            
        Returns:
            Item value or default
        """
        item = self.items.get(key)
        
        if item is None:
            self.total_misses += 1
            return default
        
        if item.is_expired():
            self._remove(key)
            self.total_expired += 1
            self.total_misses += 1
            return default
        
        # Update access info
        item.access_count += 1
        item.last_accessed = time.time()
        self.total_hits += 1
        
        return item.value
    
    def has(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        item = self.items.get(key)
        if item is None:
            return False
        if item.is_expired():
            self._remove(key)
            return False
        return True
    
    def remove(self, key: str) -> bool:
        """Remove an item from working memory."""
        return self._remove(key)
    
    def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """Extend the TTL of an existing item."""
        item = self.items.get(key)
        if item is None or item.is_expired():
            return False
        item.expires_at += additional_seconds
        return True
    
    def get_all_keys(self) -> List[str]:
        """Get all non-expired keys."""
        self._maybe_cleanup()
        return [k for k, v in self.items.items() if not v.is_expired()]
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple items at once."""
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def store_many(
        self,
        items: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ):
        """Store multiple items at once."""
        for key, value in items.items():
            self.store(key, value, ttl_seconds=ttl_seconds)
    
    def _remove(self, key: str) -> bool:
        """Remove an item."""
        if key in self.items:
            del self.items[key]
            return True
        return False
    
    def _evict_one(self):
        """Evict one item based on priority and access pattern."""
        if not self.items:
            return
        
        # First, remove expired items
        expired = [k for k, v in self.items.items() if v.is_expired()]
        if expired:
            self._remove(expired[0])
            return
        
        # Find item with lowest score (priority * recency)
        now = time.time()
        
        def eviction_score(item: WorkingMemoryItem) -> float:
            recency = 1.0 / (1.0 + now - item.last_accessed)
            return item.priority * recency * (1 + item.access_count * 0.1)
        
        min_key = min(self.items, key=lambda k: eviction_score(self.items[k]))
        self._remove(min_key)
    
    def _maybe_cleanup(self):
        """Perform cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup >= self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now
    
    def _cleanup_expired(self):
        """Remove all expired items."""
        expired = [k for k, v in self.items.items() if v.is_expired()]
        for key in expired:
            self._remove(key)
        self.total_expired += len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        self._cleanup_expired()
        
        hit_rate = (
            self.total_hits / (self.total_hits + self.total_misses)
            if (self.total_hits + self.total_misses) > 0 else 0.0
        )
        
        return {
            "current_size": len(self.items),
            "max_size": self.max_size,
            "total_stored": self.total_stored,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "total_expired": self.total_expired,
            "hit_rate": round(hit_rate, 3),
        }
    
    def clear(self):
        """Clear all items."""
        self.items.clear()
