"""
数据生命周期管理器 (Data Lifecycle Manager)

模块功能描述:
    管理学习数据的完整生命周期，从创建到最终删除。

主要功能:
    - 自动识别过期数据
    - 安全删除机制
    - 归档支持
    - 删除调度
    - 与技术淘汰系统集成

数据生命周期状态:
    ACTIVE → OBSOLETE → ARCHIVED → PENDING_DELETE → DELETED
       ↓        ↓           ↓            ↓
      30天      7天         90天        确认后

特性说明:
    - 自动过期检测
    - 带确认的安全删除
    - 删除前归档
    - 与技术淘汰集成
    - 保留策略执行
    - 删除审计跟踪

最后修改日期: 2024-12-07
"""

import asyncio
import gzip
import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Iterator
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Models
# =============================================================================

class DataState(Enum):
    """
    数据状态枚举

    定义数据在生命周期中的各个状态。

    状态说明:
        - ACTIVE: 活跃状态
        - OBSOLETE: 已过时
        - ARCHIVED: 已归档
        - PENDING_DELETE: 等待删除
        - DELETED: 已删除
    """
    ACTIVE = "active"
    OBSOLETE = "obsolete"
    ARCHIVED = "archived"
    PENDING_DELETE = "pending_delete"
    DELETED = "deleted"


class DeletionReason(Enum):
    """
    删除原因枚举

    定义数据被删除的各种原因。

    原因说明:
        - RETENTION_EXPIRED: 保留期已过
        - TECH_ELIMINATED: 技术已淘汰
        - MANUAL_DELETE: 手动删除
        - QUALITY_FAILURE: 质量不合格
        - DUPLICATE: 重复内容
        - POLICY_VIOLATION: 违反策略
    """
    RETENTION_EXPIRED = "retention_expired"
    TECH_ELIMINATED = "tech_eliminated"
    MANUAL_DELETE = "manual_delete"
    QUALITY_FAILURE = "quality_failure"
    DUPLICATE = "duplicate"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class DataLifecycleConfig:
    """
    数据生命周期配置类

    功能描述:
        控制数据保留和删除策略。

    配置项:
        - 保留期: active_retention_days, obsolete_retention_days, archive_retention_days
        - 删除宽限期: deletion_grace_period_hours
        - 删除策略: soft_delete_first, require_archive_before_delete, compress_archives
    """
    # Retention periods (days)
    active_retention_days: int = 30
    obsolete_retention_days: int = 7
    archive_retention_days: int = 90

    # Grace period before final deletion
    deletion_grace_period_hours: int = 24

    # Deletion policies
    soft_delete_first: bool = True
    require_archive_before_delete: bool = True
    compress_archives: bool = True

    # Scheduling
    cleanup_interval_hours: int = 6
    batch_size: int = 1000
    max_deletions_per_cycle: int = 5000

    # Safety
    delete_confirmation_required: bool = False
    min_items_to_keep: int = 100
    protect_recent_hours: int = 24

    # Notifications
    notify_on_archive: bool = True
    notify_on_delete: bool = True
    notify_threshold_items: int = 100


@dataclass
class DataEntry:
    """
    数据条目 / Data Entry

    Represents a tracked data item with lifecycle metadata.
    """
    data_id: str
    created_at: datetime
    state: DataState
    source: str = ""
    tech_id: Optional[str] = None
    size_bytes: int = 0
    last_accessed: Optional[datetime] = None
    obsolete_at: Optional[datetime] = None
    obsolete_reason: Optional[str] = None
    archived_at: Optional[datetime] = None
    archive_path: Optional[str] = None
    pending_delete_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_id": self.data_id,
            "created_at": self.created_at.isoformat(),
            "state": self.state.value,
            "source": self.source,
            "tech_id": self.tech_id,
            "size_bytes": self.size_bytes,
            "obsolete_reason": self.obsolete_reason,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataEntry":
        return cls(
            data_id=data["data_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            state=DataState(data.get("state", "active")),
            source=data.get("source", ""),
            tech_id=data.get("tech_id"),
            size_bytes=data.get("size_bytes", 0),
            obsolete_reason=data.get("obsolete_reason"),
            archived_at=datetime.fromisoformat(data["archived_at"]) if data.get("archived_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class DeletionRecord:
    """
    删除记录 / Deletion Record

    Audit trail for data deletions.
    """
    record_id: str
    data_ids: List[str]
    deleted_at: datetime
    reason: DeletionReason
    reason_detail: str = ""
    archived_path: Optional[str] = None
    total_size_bytes: int = 0
    confirmed: bool = False
    confirmed_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "data_count": len(self.data_ids),
            "deleted_at": self.deleted_at.isoformat(),
            "reason": self.reason.value,
            "reason_detail": self.reason_detail,
            "archived_path": self.archived_path,
            "total_size_bytes": self.total_size_bytes,
            "confirmed": self.confirmed,
        }


@dataclass
class LifecycleStats:
    """Lifecycle statistics."""
    total_registered: int = 0
    by_state: Dict[str, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    total_archived: int = 0
    total_deleted: int = 0
    pending_deletion: int = 0
    protected_items: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_registered": self.total_registered,
            "by_state": self.by_state,
            "total_size_mb": round(self.total_size_bytes / (1024 * 1024), 2),
            "total_archived": self.total_archived,
            "total_deleted": self.total_deleted,
            "pending_deletion": self.pending_deletion,
            "protected_items": self.protected_items,
        }


# =============================================================================
# Archive Manager
# =============================================================================

class ArchiveManager:
    """
    归档管理器 / Archive Manager

    Handles archiving of obsolete data before deletion.
    """

    def __init__(self, archive_path: Path, compress: bool = True):
        self.archive_path = archive_path
        self.compress = compress
        self.archive_path.mkdir(parents=True, exist_ok=True)

    def archive_items(
        self,
        items: List[DataEntry],
        batch_id: str,
        chunk_size: int = 1000,
    ) -> Tuple[Path, int]:
        """
        Archive a batch of items with streaming support for large datasets.

        P1 optimization: Uses chunked processing to reduce memory usage
        for large batches.

        Args:
            items: List of data entries to archive
            batch_id: Unique identifier for this archive batch
            chunk_size: Number of items to process at a time (default: 1000)

        Returns:
            Tuple of (archive_path, bytes_written)
        """
        filename = f"archive_{batch_id}.json"
        if self.compress:
            filename += ".gz"

        filepath = self.archive_path / filename

        # For small batches, use direct serialization
        if len(items) <= chunk_size:
            archive_data = {
                "batch_id": batch_id,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "item_count": len(items),
                "items": [item.to_dict() for item in items],
            }

            if self.compress:
                with gzip.open(filepath, "wt", encoding="utf-8") as f:
                    json.dump(archive_data, f)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(archive_data, f)
        else:
            # For large batches, use streaming write to reduce memory
            self._archive_items_streaming(items, filepath, batch_id, chunk_size)

        bytes_written = filepath.stat().st_size
        logger.info(f"Archived {len(items)} items to {filename} ({bytes_written} bytes)")

        return filepath, bytes_written

    def _archive_items_streaming(
        self,
        items: List[DataEntry],
        filepath: Path,
        batch_id: str,
        chunk_size: int,
    ) -> None:
        """
        Stream-write large batches to avoid memory spikes.

        Writes JSON manually to avoid building large in-memory structures.
        """
        open_func = gzip.open if self.compress else open

        with open_func(filepath, "wt", encoding="utf-8") as f:
            # Write header
            f.write('{\n')
            f.write(f'  "batch_id": "{batch_id}",\n')
            f.write(f'  "archived_at": "{datetime.now(timezone.utc).isoformat()}",\n')
            f.write(f'  "item_count": {len(items)},\n')
            f.write('  "items": [\n')

            # Write items in chunks
            for i, item in enumerate(items):
                item_json = json.dumps(item.to_dict())
                if i > 0:
                    f.write(',\n')
                f.write(f'    {item_json}')

                # Flush periodically for very large batches
                if (i + 1) % chunk_size == 0:
                    f.flush()

            # Write footer
            f.write('\n  ]\n}')

    def load_archive(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load items from an archive."""
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

        return data.get("items", [])

    def load_archive_streaming(self, filepath: Path) -> Iterator[Dict[str, Any]]:
        """
        Load items from archive using streaming to reduce memory.

        Yields items one at a time for memory-efficient processing.
        """
        # For streaming, we still need to load the full JSON due to format
        # In a production system, consider using JSON Lines format instead
        items = self.load_archive(filepath)
        yield from items

    def delete_archive(self, filepath: Path):
        """Delete an archive file."""
        if filepath.exists():
            filepath.unlink()
            logger.debug(f"Deleted archive: {filepath.name}")

    def list_archives(self) -> List[Tuple[Path, datetime]]:
        """List all archives with creation times."""
        archives = []
        for f in self.archive_path.glob("archive_*.json*"):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                archives.append((f, mtime))
            except Exception:
                continue
        return sorted(archives, key=lambda x: x[1], reverse=True)

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        archives = self.list_archives()
        total_size = sum(f.stat().st_size for f, _ in archives)

        return {
            "archive_count": len(archives),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest": archives[-1][1].isoformat() if archives else None,
            "newest": archives[0][1].isoformat() if archives else None,
        }


# =============================================================================
# Main Manager
# =============================================================================

class DataLifecycleManager:
    """
    数据生命周期管理器 / Data Lifecycle Manager

    Manages the complete lifecycle of learning data:
    - Registration and tracking
    - Automatic expiration detection
    - Safe archival and deletion
    - Integration with tech elimination

    Lifecycle:
    ACTIVE (30d) → OBSOLETE (7d) → ARCHIVED (90d) → DELETED

    Usage:
        config = DataLifecycleConfig(active_retention_days=30)

        async with DataLifecycleManager("./data", config) as manager:
            # Register data
            manager.register_data("item_1", source="github")

            # Mark obsolete
            manager.mark_obsolete("item_1", "Outdated")

            # Handle tech elimination
            manager.mark_for_technology_elimination("old_framework")
    """

    def __init__(
        self,
        storage_base: str = "./data_storage",
        config: Optional[DataLifecycleConfig] = None,
        on_before_delete: Optional[Callable[[List[str]], bool]] = None,
        on_after_delete: Optional[Callable[[DeletionRecord], None]] = None,
    ):
        """
        Initialize the lifecycle manager.

        Args:
            storage_base: Base path for data storage
            config: Lifecycle configuration
            on_before_delete: Callback before deletion (return False to cancel)
            on_after_delete: Callback after deletion
        """
        self.storage_base = Path(storage_base)
        self.config = config or DataLifecycleConfig()
        self.on_before_delete = on_before_delete
        self.on_after_delete = on_after_delete

        # Create directories
        self.storage_base.mkdir(parents=True, exist_ok=True)
        self.archive_path = self.storage_base / "archive"
        self.pending_path = self.storage_base / "pending_delete"
        self.archive_path.mkdir(exist_ok=True)
        self.pending_path.mkdir(exist_ok=True)

        # Archive manager
        self._archive_manager = ArchiveManager(
            self.archive_path,
            compress=self.config.compress_archives,
        )

        # Data registry
        self.data_registry: Dict[str, DataEntry] = {}

        # Deletion history
        self.deletion_history: List[DeletionRecord] = []

        # Protected items (cannot be deleted)
        self._protected_items: Set[str] = set()

        # State
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_archived = 0
        self._total_deleted = 0

    async def start(self):
        """启动生命周期管理 / Start lifecycle management."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(f"Data Lifecycle Manager started (cleanup every {self.config.cleanup_interval_hours}h)")

    async def stop(self):
        """停止生命周期管理 / Stop lifecycle management."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                # Intentionally not re-raised: we initiated the cancellation
                # during shutdown, so propagation is not needed
                logger.debug("Cleanup task cancelled during shutdown")

        logger.info("Data Lifecycle Manager stopped")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        return False

    # =========================================================================
    # Data Registration
    # =========================================================================

    def register_data(
        self,
        data_id: str,
        source: str = "",
        tech_id: Optional[str] = None,
        size_bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DataEntry:
        """
        注册数据项 / Register a data item.

        Args:
            data_id: Unique identifier
            source: Data source
            tech_id: Associated technology ID
            size_bytes: Data size
            metadata: Additional metadata

        Returns:
            Created DataEntry
        """
        now = datetime.now(timezone.utc)

        entry = DataEntry(
            data_id=data_id,
            created_at=now,
            state=DataState.ACTIVE,
            source=source,
            tech_id=tech_id,
            size_bytes=size_bytes,
            last_accessed=now,
            metadata=metadata or {},
        )

        self.data_registry[data_id] = entry
        return entry

    def register_batch(
        self,
        data_ids: List[str],
        source: str = "",
        tech_id: Optional[str] = None,
    ) -> int:
        """Register multiple data items."""
        count = 0
        for data_id in data_ids:
            self.register_data(data_id, source=source, tech_id=tech_id)
            count += 1
        return count

    def update_access(self, data_id: str):
        """Update last access time."""
        if data_id in self.data_registry:
            self.data_registry[data_id].last_accessed = datetime.now(timezone.utc)

    def protect_item(self, data_id: str):
        """Protect an item from deletion."""
        self._protected_items.add(data_id)

    def unprotect_item(self, data_id: str):
        """Remove protection from an item."""
        self._protected_items.discard(data_id)

    # =========================================================================
    # State Transitions
    # =========================================================================

    def mark_obsolete(
        self,
        data_id: str,
        reason: str = "",
        deletion_reason: DeletionReason = DeletionReason.RETENTION_EXPIRED,
    ):
        """
        标记为过时 / Mark data as obsolete.

        Args:
            data_id: Data identifier
            reason: Human-readable reason
            deletion_reason: Deletion reason enum
        """
        if data_id not in self.data_registry:
            return

        entry = self.data_registry[data_id]

        if entry.state == DataState.ACTIVE:
            entry.state = DataState.OBSOLETE
            entry.obsolete_at = datetime.now(timezone.utc)
            entry.obsolete_reason = reason or deletion_reason.value

            logger.debug(f"Marked obsolete: {data_id} ({reason})")

    def mark_for_technology_elimination(self, tech_id: str) -> int:
        """
        标记与淘汰技术相关的数据 / Mark data for eliminated technology.

        Args:
            tech_id: Technology identifier

        Returns:
            Number of items marked
        """
        count = 0

        for data_id, entry in self.data_registry.items():
            if entry.tech_id == tech_id and entry.state == DataState.ACTIVE:
                self.mark_obsolete(
                    data_id,
                    f"Technology {tech_id} eliminated",
                    DeletionReason.TECH_ELIMINATED,
                )
                count += 1

        if count > 0:
            logger.info(f"Marked {count} items obsolete for tech: {tech_id}")

        return count

    def mark_pending_delete(self, data_id: str):
        """Mark item as pending deletion."""
        if data_id in self.data_registry:
            entry = self.data_registry[data_id]
            entry.state = DataState.PENDING_DELETE
            entry.pending_delete_at = datetime.now(timezone.utc)

    # =========================================================================
    # Cleanup Loop
    # =========================================================================

    async def _cleanup_loop(self):
        """清理循环 / Cleanup loop."""
        while self._running:
            try:
                await self._execute_cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

            await asyncio.sleep(self.config.cleanup_interval_hours * 3600)

    async def _execute_cleanup(self):
        """执行清理 / Execute cleanup cycle."""
        now = datetime.now(timezone.utc)

        # Categorize items
        to_mark_obsolete: List[str] = []
        to_archive: List[str] = []
        to_delete: List[str] = []

        for data_id, entry in list(self.data_registry.items()):
            # Skip protected items
            if data_id in self._protected_items:
                continue

            # Skip recently accessed items
            if entry.last_accessed:
                if now - entry.last_accessed < timedelta(hours=self.config.protect_recent_hours):
                    continue

            if entry.state == DataState.ACTIVE:
                # Check if active retention expired
                if now - entry.created_at > timedelta(days=self.config.active_retention_days):
                    to_mark_obsolete.append(data_id)

            elif entry.state == DataState.OBSOLETE:
                # Check if obsolete retention expired
                obsolete_at = entry.obsolete_at or entry.created_at
                if now - obsolete_at > timedelta(days=self.config.obsolete_retention_days):
                    to_archive.append(data_id)

            elif entry.state == DataState.ARCHIVED:
                # Check if archive retention expired
                archived_at = entry.archived_at or entry.created_at
                if now - archived_at > timedelta(days=self.config.archive_retention_days):
                    to_delete.append(data_id)

            elif entry.state == DataState.PENDING_DELETE:
                # Check if grace period passed
                pending_at = entry.pending_delete_at or now
                if now - pending_at > timedelta(hours=self.config.deletion_grace_period_hours):
                    to_delete.append(data_id)

        # Execute transitions
        for data_id in to_mark_obsolete[:self.config.batch_size]:
            self.mark_obsolete(data_id, "Active retention expired")

        if to_archive:
            await self._archive_batch(to_archive[:self.config.batch_size])

        if to_delete:
            await self._delete_batch(
                to_delete[:min(self.config.batch_size, self.config.max_deletions_per_cycle)]
            )

        logger.info(
            f"Cleanup: marked_obsolete={len(to_mark_obsolete)}, "
            f"archived={len(to_archive)}, deleted={len(to_delete)}"
        )

    async def _archive_batch(self, data_ids: List[str]) -> Optional[Path]:
        """
        批量归档 / Archive a batch of items.
        """
        if not data_ids:
            return None

        # Check if archiving required
        if not self.config.require_archive_before_delete:
            # Skip archiving, move directly to pending delete
            for data_id in data_ids:
                self.mark_pending_delete(data_id)
            return None

        # Collect items to archive
        items_to_archive = []
        for data_id in data_ids:
            if data_id in self.data_registry:
                items_to_archive.append(self.data_registry[data_id])

        if not items_to_archive:
            return None

        # Archive
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        archive_path, _ = self._archive_manager.archive_items(
            items_to_archive,
            batch_id,
        )

        # Update entries
        for entry in items_to_archive:
            entry.state = DataState.ARCHIVED
            entry.archived_at = datetime.now(timezone.utc)
            entry.archive_path = str(archive_path)

        self._total_archived += len(items_to_archive)

        return archive_path

    async def _delete_batch(self, data_ids: List[str]) -> Optional[DeletionRecord]:
        """
        批量删除 / Delete a batch of items.
        """
        if not data_ids:
            return None

        # Safety check: maintain minimum items
        active_count = sum(
            1 for entry in self.data_registry.values()
            if entry.state == DataState.ACTIVE
        )

        if active_count < self.config.min_items_to_keep:
            logger.warning(
                f"Deletion skipped: only {active_count} active items "
                f"(min: {self.config.min_items_to_keep})"
            )
            return None

        # Before-delete callback
        if self.on_before_delete:
            try:
                if not await self._call_before_delete(data_ids):
                    logger.info("Deletion cancelled by callback")
                    return None
            except Exception as e:
                logger.error(f"Before-delete callback error: {e}")

        # Confirmation required?
        if self.config.delete_confirmation_required:
            # Move to pending delete state
            for data_id in data_ids:
                self.mark_pending_delete(data_id)
            logger.info(f"Moved {len(data_ids)} items to pending delete (confirmation required)")
            return None

        # Execute deletion
        deleted_ids = []
        total_size = 0
        archive_paths: Set[str] = set()

        for data_id in data_ids:
            if data_id in self._protected_items:
                continue

            if data_id in self.data_registry:
                entry = self.data_registry[data_id]
                total_size += entry.size_bytes

                if entry.archive_path:
                    archive_paths.add(entry.archive_path)

                del self.data_registry[data_id]
                deleted_ids.append(data_id)

        if not deleted_ids:
            return None

        # Create deletion record
        record = DeletionRecord(
            record_id=datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}",
            data_ids=deleted_ids,
            deleted_at=datetime.now(timezone.utc),
            reason=DeletionReason.RETENTION_EXPIRED,
            reason_detail=f"Deleted {len(deleted_ids)} items after retention period",
            total_size_bytes=total_size,
            confirmed=True,
        )

        self.deletion_history.append(record)
        self._total_deleted += len(deleted_ids)

        # After-delete callback
        if self.on_after_delete:
            try:
                await self._call_after_delete(record)
            except Exception as e:
                logger.error(f"After-delete callback error: {e}")

        logger.info(f"Deleted {len(deleted_ids)} items ({total_size} bytes)")
        return record

    async def _call_before_delete(self, data_ids: List[str]) -> bool:
        """Call before-delete callback."""
        if asyncio.iscoroutinefunction(self.on_before_delete):
            return await self.on_before_delete(data_ids)
        return self.on_before_delete(data_ids)

    async def _call_after_delete(self, record: DeletionRecord):
        """Call after-delete callback."""
        if asyncio.iscoroutinefunction(self.on_after_delete):
            await self.on_after_delete(record)
        else:
            self.on_after_delete(record)

    # =========================================================================
    # Manual Operations
    # =========================================================================

    async def force_archive(self, data_ids: List[str]) -> Optional[Path]:
        """Force archive specific items."""
        return await self._archive_batch(data_ids)

    async def force_delete(
        self,
        data_ids: List[str],
        reason: str = "Manual deletion",
        confirmed_by: Optional[str] = None,
    ) -> Optional[DeletionRecord]:
        """
        Force delete specific items (bypass retention).

        Args:
            data_ids: Items to delete
            reason: Deletion reason
            confirmed_by: User who confirmed deletion

        Returns:
            DeletionRecord or None
        """
        # Archive first if required
        if self.config.require_archive_before_delete:
            await self._archive_batch(data_ids)

        # Delete
        deleted_ids = []
        total_size = 0

        for data_id in data_ids:
            if data_id in self._protected_items:
                logger.warning(f"Cannot delete protected item: {data_id}")
                continue

            if data_id in self.data_registry:
                entry = self.data_registry[data_id]
                total_size += entry.size_bytes
                del self.data_registry[data_id]
                deleted_ids.append(data_id)

        if not deleted_ids:
            return None

        record = DeletionRecord(
            record_id=datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}",
            data_ids=deleted_ids,
            deleted_at=datetime.now(timezone.utc),
            reason=DeletionReason.MANUAL_DELETE,
            reason_detail=reason,
            total_size_bytes=total_size,
            confirmed=True,
            confirmed_by=confirmed_by,
        )

        self.deletion_history.append(record)
        self._total_deleted += len(deleted_ids)

        logger.warning(f"Force deleted {len(deleted_ids)} items by {confirmed_by}: {reason}")
        return record

    def confirm_pending_deletions(self, confirmed_by: str) -> List[str]:
        """Confirm all pending deletions."""
        confirmed = []

        for data_id, entry in list(self.data_registry.items()):
            if entry.state == DataState.PENDING_DELETE:
                del self.data_registry[data_id]
                confirmed.append(data_id)

        if confirmed:
            record = DeletionRecord(
                record_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                data_ids=confirmed,
                deleted_at=datetime.now(timezone.utc),
                reason=DeletionReason.RETENTION_EXPIRED,
                reason_detail="Confirmed pending deletions",
                confirmed=True,
                confirmed_by=confirmed_by,
            )
            self.deletion_history.append(record)
            self._total_deleted += len(confirmed)

        return confirmed

    def cancel_pending_deletion(self, data_id: str) -> bool:
        """Cancel a pending deletion."""
        if data_id in self.data_registry:
            entry = self.data_registry[data_id]
            if entry.state == DataState.PENDING_DELETE:
                entry.state = DataState.ARCHIVED
                entry.pending_delete_at = None
                return True
        return False

    # =========================================================================
    # Queries
    # =========================================================================

    def get_entry(self, data_id: str) -> Optional[DataEntry]:
        """Get a data entry."""
        return self.data_registry.get(data_id)

    def get_entries_by_state(self, state: DataState) -> List[DataEntry]:
        """Get all entries in a specific state."""
        return [
            entry for entry in self.data_registry.values()
            if entry.state == state
        ]

    def get_entries_by_tech(self, tech_id: str) -> List[DataEntry]:
        """Get all entries for a technology."""
        return [
            entry for entry in self.data_registry.values()
            if entry.tech_id == tech_id
        ]

    def get_obsolete_items(self) -> List[Dict[str, Any]]:
        """Get all obsolete items."""
        return [
            {
                "data_id": entry.data_id,
                "obsolete_at": entry.obsolete_at.isoformat() if entry.obsolete_at else None,
                "reason": entry.obsolete_reason,
                "days_until_archive": max(
                    0,
                    self.config.obsolete_retention_days - (
                        (datetime.now(timezone.utc) - (entry.obsolete_at or entry.created_at)).days
                    )
                ),
            }
            for entry in self.data_registry.values()
            if entry.state == DataState.OBSOLETE
        ]

    def get_pending_deletions(self) -> List[Dict[str, Any]]:
        """Get all pending deletions."""
        now = datetime.now(timezone.utc)
        return [
            {
                "data_id": entry.data_id,
                "pending_since": entry.pending_delete_at.isoformat() if entry.pending_delete_at else None,
                "hours_until_deletion": max(
                    0,
                    self.config.deletion_grace_period_hours - (
                        (now - (entry.pending_delete_at or now)).total_seconds() / 3600
                    )
                ),
            }
            for entry in self.data_registry.values()
            if entry.state == DataState.PENDING_DELETE
        ]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息 / Get statistics."""
        state_counts: Dict[str, int] = {}
        total_size = 0

        for entry in self.data_registry.values():
            state = entry.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
            total_size += entry.size_bytes

        stats = LifecycleStats(
            total_registered=len(self.data_registry),
            by_state=state_counts,
            total_size_bytes=total_size,
            total_archived=self._total_archived,
            total_deleted=self._total_deleted,
            pending_deletion=state_counts.get("pending_delete", 0),
            protected_items=len(self._protected_items),
        )

        result = stats.to_dict()
        result["deletion_history_count"] = len(self.deletion_history)
        result["archive_stats"] = self._archive_manager.get_archive_stats()

        return result

    def get_deletion_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get deletion history."""
        return [
            record.to_dict()
            for record in self.deletion_history[-limit:]
        ]
