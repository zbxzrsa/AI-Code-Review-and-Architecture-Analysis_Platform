"""
无限学习管理器 (Infinite Learning Manager)

模块功能描述:
    支持无上限的持续学习，通过智能内存管理防止 OOM。

主要功能:
    - 支持持续学习，无上限
    - 智能内存管理，防止 OOM
    - 数据轮换和归档
    - 学习检查点

特性说明:
    - 分层存储: 热（内存）→ 温（磁盘）→ 冷（归档）
    - 自动内存压力检测和处理
    - 基于检查点的恢复
    - 基于 LRU 的驱逐
    - 可配置的保留策略
    - 冷存储压缩

架构说明:
    ┌─────────────────┐
    │   热（内存）   │ ← 新数据
    │   7 天         │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │   温（磁盘）   │
    │   30 天        │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │   冷（归档）   │
    │   90 天        │
    └────────┬────────┘
             ▼
           删除

最后修改日期: 2024-12-07
"""

import asyncio
import gzip
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from collections import OrderedDict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Models
# =============================================================================

class StorageTier(Enum):
    """
    存储层级枚举
    
    定义数据存储的三个层级。
    
    层级说明:
        - HOT: 热存储（内存），快速访问
        - WARM: 温存储（磁盘），中速访问
        - COLD: 冷存储（归档），慢速访问
    """
    HOT = "hot"      # In-memory, fast access
    WARM = "warm"    # On-disk, medium access
    COLD = "cold"    # Archived, slow access


class MemoryPressureLevel(Enum):
    """
    内存压力级别枚举
    
    定义内存使用的严重程度级别。
    
    级别说明:
        - NORMAL: 正常
        - WARNING: 警告
        - CRITICAL: 严重
        - EMERGENCY: 紧急
    """
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MemoryConfig:
    """
    内存配置数据类
    
    功能描述:
        控制内存使用和数据生命周期。
    
    配置项:
        - max_memory_mb: 最大内存限制（MB）
        - warning_threshold: 警告阈值（70%）
        - critical_threshold: 严重阈值（85%）
        - emergency_threshold: 紧急阈值（95%）
    """
    # Memory limits
    max_memory_mb: int = 4096
    warning_threshold: float = 0.7   # 70% - start warning
    critical_threshold: float = 0.85  # 85% - aggressive cleanup
    emergency_threshold: float = 0.95  # 95% - emergency eviction
    
    # Cleanup settings
    cleanup_batch_size: int = 1000
    eviction_strategy: str = "lru"  # lru, fifo, random
    
    # Data retention (days)
    hot_data_days: int = 7
    warm_data_days: int = 30
    cold_data_days: int = 90
    
    # Checkpoint settings
    checkpoint_interval_minutes: int = 30
    max_checkpoints: int = 10
    checkpoint_on_stop: bool = True
    
    # Compression
    compress_warm: bool = False
    compress_cold: bool = True
    
    # Monitoring
    memory_check_interval_seconds: int = 60


@dataclass
class LearningCheckpoint:
    """
    学习检查点 / Learning Checkpoint
    
    Represents a snapshot of learning state.
    """
    checkpoint_id: str
    created_at: datetime
    total_items_learned: int
    model_state_path: str
    memory_usage_mb: float
    hot_data_count: int
    warm_files_count: int
    cold_files_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "total_items_learned": self.total_items_learned,
            "model_state_path": self.model_state_path,
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "hot_data_count": self.hot_data_count,
            "warm_files_count": self.warm_files_count,
            "cold_files_count": self.cold_files_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningCheckpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            total_items_learned=data["total_items_learned"],
            model_state_path=data["model_state_path"],
            memory_usage_mb=data["memory_usage_mb"],
            hot_data_count=data.get("hot_data_count", 0),
            warm_files_count=data.get("warm_files_count", 0),
            cold_files_count=data.get("cold_files_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LearningItem:
    """
    学习数据项 / Learning Item Wrapper
    """
    item_id: str
    data: Any
    added_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "data": self.data if isinstance(self.data, (dict, list, str)) else str(self.data),
            "added_at": self.added_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningItem":
        return cls(
            item_id=data["item_id"],
            data=data["data"],
            added_at=datetime.fromisoformat(data["added_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            size_bytes=data.get("size_bytes", 0),
            source=data.get("source", ""),
        )


# =============================================================================
# Memory Monitor
# =============================================================================

class MemoryMonitor:
    """
    内存监控器 / Memory Monitor
    
    Monitors system memory usage and detects pressure.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._psutil_available = self._check_psutil()
    
    def _check_psutil(self) -> bool:
        try:
            import psutil
            return True
        except ImportError:
            logger.warning("psutil not available, using estimation")
            return False
    
    def get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB."""
        if self._psutil_available:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        return 0.0
    
    def get_system_memory_percent(self) -> float:
        """Get system memory usage percentage."""
        if self._psutil_available:
            import psutil
            return psutil.virtual_memory().percent / 100
        return 0.5  # Assume 50% if unknown
    
    def get_pressure_level(self, estimated_items: int = 0) -> MemoryPressureLevel:
        """
        Determine current memory pressure level.
        
        Args:
            estimated_items: Number of items in memory for estimation
            
        Returns:
            MemoryPressureLevel
        """
        if self._psutil_available:
            current_mb = self.get_memory_usage_mb()
            ratio = current_mb / self.config.max_memory_mb
        else:
            # Estimate: ~10KB per item
            estimated_mb = estimated_items * 0.01
            ratio = estimated_mb / self.config.max_memory_mb
        
        if ratio >= self.config.emergency_threshold:
            return MemoryPressureLevel.EMERGENCY
        elif ratio >= self.config.critical_threshold:
            return MemoryPressureLevel.CRITICAL
        elif ratio >= self.config.warning_threshold:
            return MemoryPressureLevel.WARNING
        else:
            return MemoryPressureLevel.NORMAL
    
    def estimate_item_size(self, item: Any) -> int:
        """Estimate size of an item in bytes."""
        try:
            import sys
            return sys.getsizeof(item)
        except Exception:
            return 1024  # Default 1KB


# =============================================================================
# Storage Manager
# =============================================================================

class TieredStorageManager:
    """
    分层存储管理器 / Tiered Storage Manager
    
    Manages data across hot, warm, and cold storage tiers.
    """
    
    def __init__(self, storage_path: Path, config: MemoryConfig):
        self.storage_path = storage_path
        self.config = config
        
        # Create directories
        self.hot_path = storage_path / "hot"
        self.warm_path = storage_path / "warm"
        self.cold_path = storage_path / "cold"
        self.checkpoints_path = storage_path / "checkpoints"
        
        for path in [self.hot_path, self.warm_path, self.cold_path, self.checkpoints_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save_to_warm(self, items: List[LearningItem], batch_id: str) -> Path:
        """Save items to warm storage."""
        filename = f"batch_{batch_id}.json"
        if self.config.compress_warm:
            filename += ".gz"
        
        filepath = self.warm_path / filename
        data = [item.to_dict() for item in items]
        
        if self.config.compress_warm:
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f)
        
        logger.info(f"Saved {len(items)} items to warm storage: {filename}")
        return filepath
    
    def load_from_warm(self, filepath: Path) -> List[LearningItem]:
        """Load items from warm storage."""
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        return [LearningItem.from_dict(item) for item in data]
    
    def move_to_cold(self, filepath: Path) -> Path:
        """Move file from warm to cold storage with compression."""
        dest_filename = filepath.name
        if not dest_filename.endswith(".gz") and self.config.compress_cold:
            dest_filename += ".gz"
        
        dest_path = self.cold_path / dest_filename
        
        if self.config.compress_cold and not filepath.suffix == ".gz":
            # Compress while moving
            with open(filepath, "rb") as f_in:
                with gzip.open(dest_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            filepath.unlink()
        else:
            shutil.move(str(filepath), str(dest_path))
        
        logger.info(f"Archived to cold storage: {dest_filename}")
        return dest_path
    
    def delete_cold_file(self, filepath: Path):
        """Delete file from cold storage."""
        filepath.unlink(missing_ok=True)
        logger.info(f"Deleted expired cold data: {filepath.name}")
    
    def list_warm_files(self) -> List[Path]:
        """List all files in warm storage."""
        return list(self.warm_path.glob("batch_*"))
    
    def list_cold_files(self) -> List[Path]:
        """List all files in cold storage."""
        return list(self.cold_path.glob("batch_*"))
    
    def get_file_age(self, filepath: Path) -> timedelta:
        """Get age of a file."""
        stat = filepath.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        return datetime.now(timezone.utc) - mtime
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        def get_dir_size(path: Path) -> int:
            return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
        
        return {
            "warm_files": len(self.list_warm_files()),
            "warm_size_mb": round(get_dir_size(self.warm_path) / (1024 * 1024), 2),
            "cold_files": len(self.list_cold_files()),
            "cold_size_mb": round(get_dir_size(self.cold_path) / (1024 * 1024), 2),
        }


# =============================================================================
# Main Manager
# =============================================================================

class InfiniteLearningManager:
    """
    无限学习管理器 / Infinite Learning Manager
    
    Manages unlimited learning data with intelligent memory management.
    
    Features:
    - Tiered storage (hot/warm/cold)
    - Automatic memory pressure handling
    - LRU-based eviction
    - Checkpoint-based recovery
    - Configurable retention policies
    
    Usage:
        config = MemoryConfig(max_memory_mb=4096)
        
        async with InfiniteLearningManager("./data", config) as manager:
            # Add data
            await manager.add_learning_data(item)
            
            # Get stats
            stats = manager.get_stats()
    """
    
    def __init__(
        self,
        storage_path: str = "./learning_data",
        config: Optional[MemoryConfig] = None,
    ):
        """
        Initialize the learning manager.
        
        Args:
            storage_path: Path for data storage
            config: Memory configuration
        """
        self.storage_path = Path(storage_path)
        self.config = config or MemoryConfig()
        
        # Components
        self._storage = TieredStorageManager(self.storage_path, self.config)
        self._memory_monitor = MemoryMonitor(self.config)
        
        # Hot data (in-memory)
        self._hot_data: Dict[str, LearningItem] = {}
        self._access_order: OrderedDict[str, None] = OrderedDict()  # For LRU - O(1)
        
        # Statistics
        self.total_learned: int = 0
        self.total_evicted: int = 0
        self.total_archived: int = 0
        
        # Checkpoints
        self.checkpoints: List[LearningCheckpoint] = []
        
        # State
        self._running = False
        self._maintenance_task: Optional[asyncio.Task] = None
        self._memory_check_task: Optional[asyncio.Task] = None
        
        # Load existing checkpoints
        self._load_checkpoints()
    
    def _load_checkpoints(self):
        """Load existing checkpoints from disk."""
        checkpoint_files = sorted(
            self._storage.checkpoints_path.glob("cp_*.json"),
            key=lambda p: p.stat().st_mtime
        )
        
        for filepath in checkpoint_files[-self.config.max_checkpoints:]:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                self.checkpoints.append(LearningCheckpoint.from_dict(data))
            except Exception as e:
                logger.error(f"Error loading checkpoint {filepath}: {e}")
    
    async def start(self):
        """启动管理器 / Start the manager."""
        if self._running:
            return
        
        self._running = True
        
        # Start maintenance loop
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        # Start memory monitoring
        self._memory_check_task = asyncio.create_task(self._memory_check_loop())
        
        logger.info(f"Infinite Learning Manager started (max_memory: {self.config.max_memory_mb}MB)")
    
    async def stop(self):
        """停止管理器 / Stop the manager."""
        self._running = False
        
        # Cancel tasks
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        if self._memory_check_task:
            self._memory_check_task.cancel()
            try:
                await self._memory_check_task
            except asyncio.CancelledError:
                pass
        
        # Create final checkpoint
        if self.config.checkpoint_on_stop:
            await self.create_checkpoint()
        
        logger.info("Infinite Learning Manager stopped")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False
    
    # =========================================================================
    # Data Management
    # =========================================================================
    
    async def add_learning_data(
        self,
        data: Any,
        item_id: Optional[str] = None,
        source: str = "",
    ) -> str:
        """
        添加学习数据 / Add learning data.
        
        Args:
            data: Data to add
            item_id: Optional custom ID
            source: Data source identifier
            
        Returns:
            Item ID
        """
        now = datetime.now(timezone.utc)
        
        # Generate ID if not provided
        if item_id is None:
            item_id = hashlib.sha256(
                f"{now.isoformat()}{id(data)}".encode()
            ).hexdigest()[:16]
        
        # Create wrapper
        item = LearningItem(
            item_id=item_id,
            data=data,
            added_at=now,
            last_accessed=now,
            access_count=1,
            size_bytes=self._memory_monitor.estimate_item_size(data),
            source=source,
        )
        
        # Add to hot storage
        self._hot_data[item_id] = item
        self._access_order[item_id] = None  # O(1) insert
        self.total_learned += 1
        
        # Check memory pressure
        pressure = self._memory_monitor.get_pressure_level(len(self._hot_data))
        
        if pressure != MemoryPressureLevel.NORMAL:
            await self._handle_memory_pressure(pressure)
        
        return item_id
    
    async def add_batch(self, data_list: List[Any], source: str = "") -> List[str]:
        """
        批量添加 / Add batch of data.
        
        Args:
            data_list: List of data items
            source: Data source identifier
            
        Returns:
            List of item IDs
        """
        ids = []
        for data in data_list:
            item_id = await self.add_learning_data(data, source=source)
            ids.append(item_id)
        return ids
    
    def get_item(self, item_id: str) -> Optional[Any]:
        """
        获取数据项 / Get data item.
        
        Updates access time for LRU.
        """
        item = self._hot_data.get(item_id)
        
        if item:
            item.last_accessed = datetime.now(timezone.utc)
            item.access_count += 1
            
            # Update LRU order - O(1) with OrderedDict.move_to_end()
            if item_id in self._access_order:
                self._access_order.move_to_end(item_id)
            else:
                self._access_order[item_id] = None
            
            return item.data
        
        return None
    
    def get_recent_items(self, limit: int = 100) -> List[LearningItem]:
        """Get most recently added items."""
        items = sorted(
            self._hot_data.values(),
            key=lambda x: x.added_at,
            reverse=True
        )
        return items[:limit]
    
    # =========================================================================
    # Memory Management
    # =========================================================================
    
    async def _memory_check_loop(self):
        """Continuous memory monitoring loop."""
        while self._running:
            try:
                pressure = self._memory_monitor.get_pressure_level(len(self._hot_data))
                
                if pressure != MemoryPressureLevel.NORMAL:
                    logger.warning(f"Memory pressure: {pressure.value}")
                    await self._handle_memory_pressure(pressure)
                
            except Exception as e:
                logger.error(f"Memory check error: {e}")
            
            await asyncio.sleep(self.config.memory_check_interval_seconds)
    
    async def _handle_memory_pressure(self, pressure: MemoryPressureLevel):
        """
        处理内存压力 / Handle memory pressure.
        """
        if pressure == MemoryPressureLevel.EMERGENCY:
            # Emergency: Evict 50% of data
            await self._evict_data(int(len(self._hot_data) * 0.5))
        elif pressure == MemoryPressureLevel.CRITICAL:
            # Critical: Evict 25% of data
            await self._evict_data(int(len(self._hot_data) * 0.25))
        elif pressure == MemoryPressureLevel.WARNING:
            # Warning: Rotate old data to warm storage
            await self._rotate_old_data()
    
    async def _evict_data(self, count: int):
        """
        Evict data based on LRU strategy.
        """
        if count <= 0 or len(self._access_order) == 0:
            return
        
        # Get oldest accessed items (LRU) - O(1) per pop with OrderedDict
        evicted_items = []
        
        for _ in range(min(count, len(self._access_order))):
            item_id, _ = self._access_order.popitem(last=False)  # O(1) pop oldest
            if item_id in self._hot_data:
                evicted_items.append(self._hot_data.pop(item_id))
        
        # Save to warm storage before evicting
        if evicted_items:
            batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._storage.save_to_warm(evicted_items, batch_id)
        
        self.total_evicted += len(evicted_items)
        logger.info(f"Evicted {len(evicted_items)} items to warm storage")
    
    async def _rotate_old_data(self):
        """
        Rotate old hot data to warm storage.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.hot_data_days)
        
        to_rotate = [
            item for item in self._hot_data.values()
            if item.added_at < cutoff
        ]
        
        if not to_rotate:
            return
        
        # Save to warm storage
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._storage.save_to_warm(to_rotate, batch_id)
        
        # Remove from hot storage
        for item in to_rotate:
            self._hot_data.pop(item.item_id, None)
            self._access_order.pop(item.item_id, None)  # O(1) delete from OrderedDict
        
        self.total_evicted += len(to_rotate)
        logger.info(f"Rotated {len(to_rotate)} items to warm storage")
    
    # =========================================================================
    # Maintenance
    # =========================================================================
    
    async def _maintenance_loop(self):
        """Periodic maintenance loop."""
        while self._running:
            try:
                # Create checkpoint
                await self.create_checkpoint()
                
                # Archive old warm data to cold
                await self._archive_warm_to_cold()
                
                # Cleanup expired cold data
                await self._cleanup_expired_cold()
                
            except Exception as e:
                logger.error(f"Maintenance error: {e}")
            
            await asyncio.sleep(self.config.checkpoint_interval_minutes * 60)
    
    async def _archive_warm_to_cold(self):
        """Archive old warm data to cold storage."""
        cutoff = timedelta(days=self.config.warm_data_days)
        
        for filepath in self._storage.list_warm_files():
            try:
                age = self._storage.get_file_age(filepath)
                if age > cutoff:
                    self._storage.move_to_cold(filepath)
                    self.total_archived += 1
            except Exception as e:
                logger.error(f"Error archiving {filepath}: {e}")
    
    async def _cleanup_expired_cold(self):
        """Delete expired cold data."""
        cutoff = timedelta(days=self.config.cold_data_days)
        
        for filepath in self._storage.list_cold_files():
            try:
                age = self._storage.get_file_age(filepath)
                if age > cutoff:
                    self._storage.delete_cold_file(filepath)
            except Exception as e:
                logger.error(f"Error deleting {filepath}: {e}")
    
    # =========================================================================
    # Checkpoints
    # =========================================================================
    
    async def create_checkpoint(self) -> LearningCheckpoint:
        """
        创建检查点 / Create a learning checkpoint.
        """
        checkpoint_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self._storage.checkpoints_path / f"cp_{checkpoint_id}.json"
        
        # Get memory usage
        memory_mb = self._memory_monitor.get_memory_usage_mb()
        
        # Get storage stats
        storage_stats = self._storage.get_storage_stats()
        
        checkpoint = LearningCheckpoint(
            checkpoint_id=checkpoint_id,
            created_at=datetime.now(timezone.utc),
            total_items_learned=self.total_learned,
            model_state_path=str(checkpoint_path),
            memory_usage_mb=memory_mb,
            hot_data_count=len(self._hot_data),
            warm_files_count=storage_stats["warm_files"],
            cold_files_count=storage_stats["cold_files"],
            metadata={
                "total_evicted": self.total_evicted,
                "total_archived": self.total_archived,
                "storage_stats": storage_stats,
            }
        )
        
        # Save checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        
        self.checkpoints.append(checkpoint)
        
        # Limit checkpoint count
        while len(self.checkpoints) > self.config.max_checkpoints:
            old = self.checkpoints.pop(0)
            Path(old.model_state_path).unlink(missing_ok=True)
        
        logger.info(f"Checkpoint created: {checkpoint_id}")
        return checkpoint
    
    async def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore state from a checkpoint.
        """
        checkpoint = next(
            (c for c in self.checkpoints if c.checkpoint_id == checkpoint_id),
            None
        )
        
        if not checkpoint:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False
        
        logger.info(f"Restoring from checkpoint: {checkpoint_id}")
        
        # Reset state
        self.total_learned = checkpoint.total_items_learned
        
        return True
    
    def get_latest_checkpoint(self) -> Optional[LearningCheckpoint]:
        """Get the most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息 / Get statistics."""
        storage_stats = self._storage.get_storage_stats()
        memory_mb = self._memory_monitor.get_memory_usage_mb()
        pressure = self._memory_monitor.get_pressure_level(len(self._hot_data))
        
        return {
            "total_learned": self.total_learned,
            "total_evicted": self.total_evicted,
            "total_archived": self.total_archived,
            "hot_data_count": len(self._hot_data),
            "memory_usage_mb": round(memory_mb, 2),
            "memory_limit_mb": self.config.max_memory_mb,
            "memory_pressure": pressure.value,
            "checkpoints_count": len(self.checkpoints),
            "storage_path": str(self.storage_path),
            **storage_stats,
        }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get detailed memory status."""
        memory_mb = self._memory_monitor.get_memory_usage_mb()
        pressure = self._memory_monitor.get_pressure_level(len(self._hot_data))
        
        return {
            "current_mb": round(memory_mb, 2),
            "max_mb": self.config.max_memory_mb,
            "usage_percent": round((memory_mb / self.config.max_memory_mb) * 100, 1),
            "pressure_level": pressure.value,
            "thresholds": {
                "warning": self.config.warning_threshold,
                "critical": self.config.critical_threshold,
                "emergency": self.config.emergency_threshold,
            },
            "hot_items": len(self._hot_data),
        }
