"""
Storage Manager

Orchestrates caching and persistent storage with:
- Memory-first access with LRU cache
- Automatic persistence to sharded storage
- Data lifecycle management
- Archive and cleanup operations
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..collectors.base import CollectedItem
from ..config import MemoryConfig, RetentionPolicy, StorageConfig
from .cache import LRUCache
from .sharded import ShardedStorage

logger = logging.getLogger(__name__)


@dataclass
class StorageStats:
    """Combined storage statistics."""
    cache_hit_rate: float
    cache_size: int
    cache_memory_mb: float
    shard_count: int
    total_items: int
    total_size_gb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_hit_rate": round(self.cache_hit_rate * 100, 1),
            "cache_size": self.cache_size,
            "cache_memory_mb": round(self.cache_memory_mb, 2),
            "shard_count": self.shard_count,
            "total_items": self.total_items,
            "total_size_gb": round(self.total_size_gb, 2),
        }


class StorageManager:
    """
    Unified storage manager for networked learning.
    
    Architecture:
    - L1: In-memory LRU cache (hot data)
    - L2: Sharded persistent storage (cold data)
    - Archive: Compressed long-term storage
    
    Features:
    - Write-through caching
    - Automatic data lifecycle management
    - Retention policy enforcement
    - Archive and cleanup
    """
    
    def __init__(
        self,
        memory_config: MemoryConfig,
        storage_config: StorageConfig,
        retention_policy: RetentionPolicy,
        base_path: str = "/data/networked_learning",
    ):
        """
        Initialize storage manager.
        
        Args:
            memory_config: Memory/cache configuration
            storage_config: Persistent storage configuration
            retention_policy: Data retention policy
            base_path: Base path for storage
        """
        self.memory_config = memory_config
        self.storage_config = storage_config
        self.retention_policy = retention_policy
        self.base_path = base_path
        
        # Initialize cache
        self._cache = LRUCache[str, CollectedItem](
            max_size=10000,
            max_memory_mb=memory_config.cache_size_mb,
            default_ttl_seconds=3600,  # 1 hour cache TTL
        )
        
        # Initialize sharded storage
        self._storage = ShardedStorage(
            base_path=f"{base_path}/shards",
            shard_size_gb=storage_config.shard_size_gb,
        )
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start storage manager and background tasks."""
        self._running = True
        
        # Start cleanup scheduler
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Storage manager started")
    
    async def stop(self):
        """Stop storage manager."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self._storage.close()
        logger.info("Storage manager stopped")
    
    async def store(self, item: CollectedItem) -> bool:
        """
        Store a collected item.
        
        Uses write-through caching:
        1. Write to cache (hot access)
        2. Write to persistent storage
        
        Args:
            item: Item to store
            
        Returns:
            True if successful
        """
        unique_id = item.unique_id
        
        # Write to cache
        self._cache.put(unique_id, item)
        
        # Write to persistent storage
        success = await self._storage.store(item)
        
        if not success:
            logger.warning(f"Failed to persist item: {unique_id}")
        
        return success
    
    async def store_batch(self, items: List[CollectedItem]) -> int:
        """
        Store multiple items.
        
        Args:
            items: Items to store
            
        Returns:
            Number of items stored
        """
        stored = 0
        for item in items:
            if await self.store(item):
                stored += 1
        return stored
    
    async def retrieve(self, unique_id: str) -> Optional[CollectedItem]:
        """
        Retrieve an item by ID.
        
        Checks cache first, then persistent storage.
        
        Args:
            unique_id: Item unique ID
            
        Returns:
            CollectedItem or None
        """
        # Check cache first
        item = self._cache.get(unique_id)
        if item:
            return item
        
        # Check persistent storage
        item = await self._storage.retrieve(unique_id)
        
        if item:
            # Populate cache
            self._cache.put(unique_id, item)
        
        return item
    
    async def delete(self, unique_id: str) -> bool:
        """
        Delete an item.
        
        Args:
            unique_id: Item unique ID
            
        Returns:
            True if deleted
        """
        # Remove from cache
        self._cache.delete(unique_id)
        
        # Remove from storage
        return await self._storage.delete(unique_id)
    
    async def _cleanup_loop(self):
        """Background cleanup task."""
        while self._running:
            try:
                # Calculate next cleanup time
                now = datetime.now(timezone.utc)
                cleanup_hour = self.retention_policy.cleanup_schedule_hour
                
                next_cleanup = now.replace(
                    hour=cleanup_hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                
                if next_cleanup <= now:
                    next_cleanup += timedelta(days=1)
                
                # Wait until cleanup time
                wait_seconds = (next_cleanup - now).total_seconds()
                await asyncio.sleep(min(wait_seconds, 3600))  # Check hourly
                
                if now.hour == cleanup_hour:
                    await self._run_cleanup()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _run_cleanup(self):
        """Run cleanup operations."""
        logger.info("Starting scheduled cleanup")
        
        # Clean expired cache entries
        expired = self._cache.cleanup_expired()
        logger.info(f"Removed {expired} expired cache entries")
        
        # Archive and delete old data based on retention policy
        retention_days = self.retention_policy.raw_data_retention_days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        # TODO: Implement archive and delete for sharded storage
        # This would iterate shards and move/delete old items
        
        logger.info("Cleanup completed")
    
    async def archive_deprecated_technology(self, technology_id: str) -> int:
        """
        Archive all data related to a deprecated technology.
        
        Args:
            technology_id: Technology identifier
            
        Returns:
            Number of items archived
        """
        archived = 0
        
        # In a real implementation, this would:
        # 1. Query items by technology
        # 2. Move to archive storage
        # 3. Optionally compress
        # 4. Delete from active storage
        
        logger.info(f"Archived {archived} items for deprecated technology: {technology_id}")
        return archived
    
    def check_memory_usage(self) -> bool:
        """
        Check if memory usage is within limits.
        
        Returns:
            True if within limits
        """
        import psutil
        
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100
            
            if usage_percent > self.memory_config.max_memory_percent:
                logger.warning(
                    f"Memory usage {usage_percent:.1%} exceeds limit "
                    f"{self.memory_config.max_memory_percent:.1%}"
                )
                return False
            
            # Trigger GC if above threshold
            if usage_percent > self.memory_config.gc_threshold_percent:
                import gc
                gc.collect()
            
            return True
            
        except ImportError:
            return True  # psutil not available
    
    def get_stats(self) -> StorageStats:
        """Get combined storage statistics."""
        cache_stats = self._cache.stats
        storage_stats = self._storage.get_stats()
        
        return StorageStats(
            cache_hit_rate=cache_stats.hit_rate,
            cache_size=cache_stats.current_size,
            cache_memory_mb=self._cache.memory_usage_mb,
            shard_count=storage_stats["shard_count"],
            total_items=storage_stats["total_items"],
            total_size_gb=storage_stats["total_size_gb"],
        )
