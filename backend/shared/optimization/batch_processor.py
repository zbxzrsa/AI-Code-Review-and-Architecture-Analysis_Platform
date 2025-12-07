"""
Batch Processing Utilities

Provides efficient batch processing for:
- Database operations (bulk inserts, updates, deletes)
- AI inference requests
- Online learning updates
- Data pipeline processing
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import (
    Optional, Any, Dict, List, Callable, TypeVar, Generic,
    Awaitable, AsyncIterator, Union
)
from datetime import datetime, timezone
from collections import deque
from functools import wraps
from enum import Enum
import time

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class BatchStrategy(Enum):
    """Batch processing strategies."""
    SIZE = "size"          # Batch by count
    TIME = "time"          # Batch by time window
    HYBRID = "hybrid"      # Batch by size or time, whichever comes first
    ADAPTIVE = "adaptive"  # Adjust batch size based on performance


@dataclass
class BatchConfig:
    """
    Batch processing configuration.
    
    Recommended settings by use case:
    - Database inserts: batch_size=100-500, max_wait_ms=100
    - AI inference: batch_size=8-32, max_wait_ms=50
    - Online learning: batch_size=50-100, max_wait_ms=1000
    - Data pipeline: batch_size=1000-5000, max_wait_ms=5000
    """
    # Batch size settings
    batch_size: int = 100
    min_batch_size: int = 1
    max_batch_size: int = 1000
    
    # Timing settings
    max_wait_ms: int = 100
    flush_interval_ms: int = 1000
    
    # Strategy
    strategy: BatchStrategy = BatchStrategy.HYBRID
    
    # Retry settings
    max_retries: int = 3
    retry_delay_ms: int = 100
    
    # Performance settings
    enable_adaptive: bool = True
    target_latency_ms: float = 50.0
    
    # Concurrency
    max_concurrent_batches: int = 4


@dataclass
class BatchStats:
    """Batch processing statistics."""
    total_items: int = 0
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    avg_batch_size: float = 0.0
    avg_latency_ms: float = 0.0
    items_per_second: float = 0.0
    start_time: Optional[datetime] = None


class BatchBuffer(Generic[T]):
    """
    Thread-safe buffer for collecting batch items.
    """
    
    def __init__(self, max_size: int = 1000):
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()
    
    async def add(self, item: T) -> None:
        """Add item to buffer."""
        async with self._lock:
            self._buffer.append(item)
            self._event.set()
    
    async def add_many(self, items: List[T]) -> None:
        """Add multiple items to buffer."""
        async with self._lock:
            self._buffer.extend(items)
            self._event.set()
    
    async def take(self, count: int) -> List[T]:
        """Take up to count items from buffer."""
        async with self._lock:
            items = []
            for _ in range(min(count, len(self._buffer))):
                items.append(self._buffer.popleft())
            if not self._buffer:
                self._event.clear()
            return items
    
    async def take_all(self) -> List[T]:
        """Take all items from buffer."""
        async with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            self._event.clear()
            return items
    
    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self._buffer)
    
    async def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for items to be available."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False


class BatchProcessor(Generic[T, R]):
    """
    Efficient batch processor for async operations.
    
    Supports multiple processing patterns:
    - Collect items and process in batches
    - Automatic flushing based on size or time
    - Adaptive batch sizing for optimal performance
    - Retry handling for failed batches
    
    Usage:
        # Create processor with handler function
        processor = BatchProcessor(
            handler=bulk_insert_handler,
            config=BatchConfig(batch_size=100, max_wait_ms=100)
        )
        
        # Start processor
        await processor.start()
        
        # Add items (automatically batched)
        await processor.add(item1)
        await processor.add(item2)
        await processor.add_many([item3, item4, item5])
        
        # Stop and flush remaining
        await processor.stop()
    
    Example handlers:
        # Database bulk insert
        async def bulk_insert_handler(items: List[Dict]) -> int:
            async with db.session() as session:
                session.add_all([Model(**item) for item in items])
                await session.commit()
            return len(items)
        
        # AI batch inference
        async def ai_inference_handler(items: List[str]) -> List[Dict]:
            return await ai_model.batch_infer(items)
    """
    
    def __init__(
        self,
        handler: Callable[[List[T]], Awaitable[R]],
        config: Optional[BatchConfig] = None,
        error_handler: Optional[Callable[[List[T], Exception], Awaitable[None]]] = None
    ):
        self.handler = handler
        self.config = config or BatchConfig()
        self.error_handler = error_handler
        
        self._buffer = BatchBuffer[T](self.config.max_batch_size * 2)
        self._stats = BatchStats()
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        
        # Adaptive sizing state
        self._current_batch_size = self.config.batch_size
        self._latency_history: deque = deque(maxlen=10)
    
    async def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return
        
        self._running = True
        self._stats.start_time = datetime.now(timezone.utc)
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"BatchProcessor started with batch_size={self.config.batch_size}")
    
    async def stop(self, flush: bool = True) -> None:
        """Stop the batch processor."""
        self._running = False
        
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        
        if flush:
            await self._flush()
        
        logger.info(f"BatchProcessor stopped. Stats: {self.get_stats()}")
    
    async def add(self, item: T) -> None:
        """Add a single item for processing."""
        await self._buffer.add(item)
        self._stats.total_items += 1
    
    async def add_many(self, items: List[T]) -> None:
        """Add multiple items for processing."""
        await self._buffer.add_many(items)
        self._stats.total_items += len(items)
    
    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Wait for items or timeout
                timeout = self.config.max_wait_ms / 1000.0
                has_items = await self._buffer.wait(timeout)
                
                # Check if we should process
                if self._should_process():
                    await self._process_batch()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"BatchProcessor loop error: {e}")
                await asyncio.sleep(0.1)
    
    def _should_process(self) -> bool:
        """Check if we should process a batch."""
        if self._buffer.size == 0:
            return False
        
        if self.config.strategy == BatchStrategy.SIZE:
            return self._buffer.size >= self._current_batch_size
        
        elif self.config.strategy == BatchStrategy.TIME:
            return True  # Always process on timeout
        
        elif self.config.strategy == BatchStrategy.HYBRID:
            return self._buffer.size >= self._current_batch_size or True
        
        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            return self._buffer.size >= self._current_batch_size
        
        return self._buffer.size >= self._current_batch_size
    
    async def _process_batch(self) -> None:
        """Process a single batch."""
        async with self._semaphore:
            items = await self._buffer.take(self._current_batch_size)
            if not items:
                return
            
            start_time = time.monotonic()
            
            try:
                # Execute handler
                await self._execute_with_retry(items)
                
                # Update stats
                latency_ms = (time.monotonic() - start_time) * 1000
                self._stats.successful_batches += 1
                self._latency_history.append(latency_ms)
                
                # Adaptive sizing
                if self.config.enable_adaptive:
                    self._adjust_batch_size(latency_ms)
                
            except Exception as e:
                self._stats.failed_batches += 1
                logger.error(f"Batch processing failed: {e}")
                
                if self.error_handler:
                    await self.error_handler(items, e)
            
            finally:
                self._stats.total_batches += 1
                self._update_stats()
    
    async def _execute_with_retry(self, items: List[T]) -> R:
        """Execute handler with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return await self.handler(items)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_ms / 1000.0 * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        raise last_error
    
    def _adjust_batch_size(self, latency_ms: float) -> None:
        """Adjust batch size based on performance."""
        if len(self._latency_history) < 3:
            return
        
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        
        if avg_latency < self.config.target_latency_ms * 0.5:
            # Under target, increase batch size
            self._current_batch_size = min(
                int(self._current_batch_size * 1.2),
                self.config.max_batch_size
            )
        elif avg_latency > self.config.target_latency_ms * 1.5:
            # Over target, decrease batch size
            self._current_batch_size = max(
                int(self._current_batch_size * 0.8),
                self.config.min_batch_size
            )
    
    def _update_stats(self) -> None:
        """Update running statistics."""
        if self._stats.total_batches > 0:
            self._stats.avg_batch_size = self._stats.total_items / self._stats.total_batches
        
        if self._latency_history:
            self._stats.avg_latency_ms = sum(self._latency_history) / len(self._latency_history)
        
        if self._stats.start_time:
            elapsed = (datetime.now(timezone.utc) - self._stats.start_time).total_seconds()
            if elapsed > 0:
                self._stats.items_per_second = self._stats.total_items / elapsed
    
    async def _flush(self) -> None:
        """Flush all remaining items."""
        while self._buffer.size > 0:
            await self._process_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_items": self._stats.total_items,
            "total_batches": self._stats.total_batches,
            "successful_batches": self._stats.successful_batches,
            "failed_batches": self._stats.failed_batches,
            "avg_batch_size": round(self._stats.avg_batch_size, 1),
            "avg_latency_ms": round(self._stats.avg_latency_ms, 2),
            "items_per_second": round(self._stats.items_per_second, 1),
            "current_batch_size": self._current_batch_size,
            "buffer_size": self._buffer.size,
        }


class DatabaseBatchProcessor:
    """
    Specialized batch processor for database operations.
    
    Usage:
        db_batch = DatabaseBatchProcessor(session_factory)
        
        # Bulk insert
        await db_batch.bulk_insert(Model, records)
        
        # Bulk update
        await db_batch.bulk_update(Model, updates)
        
        # Bulk upsert
        await db_batch.bulk_upsert(Model, records, conflict_keys=["id"])
    """
    
    def __init__(
        self,
        session_factory,
        batch_size: int = 500,
        max_retries: int = 3
    ):
        self.session_factory = session_factory
        self.batch_size = batch_size
        self.max_retries = max_retries
    
    async def bulk_insert(
        self,
        model_class,
        records: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> int:
        """
        Bulk insert records in batches.
        
        Args:
            model_class: SQLAlchemy model class
            records: List of record dictionaries
            batch_size: Override default batch size
        
        Returns:
            Number of records inserted
        """
        batch_size = batch_size or self.batch_size
        inserted = 0
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            async with self.session_factory() as session:
                try:
                    objects = [model_class(**record) for record in batch]
                    session.add_all(objects)
                    await session.commit()
                    inserted += len(batch)
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Bulk insert failed for batch {i // batch_size}: {e}")
                    raise
        
        logger.info(f"Bulk inserted {inserted} records into {model_class.__tablename__}")
        return inserted
    
    async def bulk_update(
        self,
        model_class,
        updates: List[Dict[str, Any]],
        id_field: str = "id",
        batch_size: Optional[int] = None
    ) -> int:
        """
        Bulk update records in batches.
        
        Args:
            model_class: SQLAlchemy model class
            updates: List of update dictionaries (must include id_field)
            id_field: Primary key field name
            batch_size: Override default batch size
        
        Returns:
            Number of records updated
        """
        from sqlalchemy import update
        
        batch_size = batch_size or self.batch_size
        updated = 0
        
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            
            async with self.session_factory() as session:
                try:
                    for record in batch:
                        record_id = record.pop(id_field)
                        stmt = (
                            update(model_class)
                            .where(getattr(model_class, id_field) == record_id)
                            .values(**record)
                        )
                        await session.execute(stmt)
                    
                    await session.commit()
                    updated += len(batch)
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Bulk update failed for batch {i // batch_size}: {e}")
                    raise
        
        logger.info(f"Bulk updated {updated} records in {model_class.__tablename__}")
        return updated
    
    async def bulk_delete(
        self,
        model_class,
        ids: List[Any],
        id_field: str = "id",
        batch_size: Optional[int] = None
    ) -> int:
        """
        Bulk delete records in batches.
        
        Args:
            model_class: SQLAlchemy model class
            ids: List of IDs to delete
            id_field: Primary key field name
            batch_size: Override default batch size
        
        Returns:
            Number of records deleted
        """
        from sqlalchemy import delete
        
        batch_size = batch_size or self.batch_size
        deleted = 0
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            
            async with self.session_factory() as session:
                try:
                    stmt = (
                        delete(model_class)
                        .where(getattr(model_class, id_field).in_(batch_ids))
                    )
                    result = await session.execute(stmt)
                    await session.commit()
                    deleted += result.rowcount
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Bulk delete failed for batch {i // batch_size}: {e}")
                    raise
        
        logger.info(f"Bulk deleted {deleted} records from {model_class.__tablename__}")
        return deleted


class OnlineLearningBatchProcessor:
    """
    Batch processor optimized for online learning scenarios.
    
    Supports:
    - Incremental model updates
    - Experience replay batching
    - Gradient accumulation
    - Memory consolidation
    
    Usage:
        learner = OnlineLearningBatchProcessor(
            update_fn=model.update,
            batch_size=50
        )
        
        await learner.start()
        await learner.add_experience(experience)
        await learner.stop()
    """
    
    def __init__(
        self,
        update_fn: Callable[[List[Any]], Awaitable[None]],
        batch_size: int = 50,
        max_wait_ms: int = 1000,
        replay_buffer_size: int = 10000
    ):
        self.update_fn = update_fn
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        
        self._replay_buffer: deque = deque(maxlen=replay_buffer_size)
        self._current_batch: List = []
        self._lock = asyncio.Lock()
        self._running = False
        self._process_task = None
        
        # Stats
        self._total_experiences = 0
        self._total_updates = 0
    
    async def start(self) -> None:
        """Start the learning processor."""
        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info(f"OnlineLearningBatchProcessor started with batch_size={self.batch_size}")
    
    async def stop(self, flush: bool = True) -> None:
        """Stop the learning processor."""
        self._running = False
        
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
        
        if flush and self._current_batch:
            await self._process_batch(self._current_batch)
        
        logger.info(f"OnlineLearningBatchProcessor stopped. Updates: {self._total_updates}")
    
    async def add_experience(self, experience: Any) -> None:
        """Add a learning experience."""
        async with self._lock:
            self._current_batch.append(experience)
            self._replay_buffer.append(experience)
            self._total_experiences += 1
    
    async def add_experiences(self, experiences: List[Any]) -> None:
        """Add multiple learning experiences."""
        async with self._lock:
            self._current_batch.extend(experiences)
            self._replay_buffer.extend(experiences)
            self._total_experiences += len(experiences)
    
    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                await asyncio.sleep(self.max_wait_ms / 1000.0)
                
                async with self._lock:
                    if len(self._current_batch) >= self.batch_size:
                        batch = self._current_batch[:self.batch_size]
                        self._current_batch = self._current_batch[self.batch_size:]
                        await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OnlineLearning loop error: {e}")
    
    async def _process_batch(self, batch: List[Any]) -> None:
        """Process a learning batch."""
        if not batch:
            return
        
        try:
            await self.update_fn(batch)
            self._total_updates += 1
            logger.debug(f"Processed learning batch: size={len(batch)}")
        except Exception as e:
            logger.error(f"Learning batch processing failed: {e}")
    
    async def sample_replay(self, size: int) -> List[Any]:
        """Sample from replay buffer for experience replay."""
        import random
        
        async with self._lock:
            if len(self._replay_buffer) < size:
                return list(self._replay_buffer)
            return random.sample(list(self._replay_buffer), size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_experiences": self._total_experiences,
            "total_updates": self._total_updates,
            "replay_buffer_size": len(self._replay_buffer),
            "pending_batch_size": len(self._current_batch),
            "avg_experiences_per_update": (
                self._total_experiences / self._total_updates
                if self._total_updates > 0 else 0
            ),
        }


# Convenience decorator
def batch_process(
    batch_size: int = 100,
    max_wait_ms: int = 100,
    max_retries: int = 3
):
    """
    Decorator to batch function calls.
    
    Usage:
        @batch_process(batch_size=50)
        async def process_items(items: List[Item]):
            await bulk_insert(items)
    """
    def decorator(func: Callable[[List[T]], Awaitable[R]]):
        config = BatchConfig(
            batch_size=batch_size,
            max_wait_ms=max_wait_ms,
            max_retries=max_retries
        )
        processor = BatchProcessor(handler=func, config=config)
        
        @wraps(func)
        async def wrapper(*items: T) -> None:
            if not processor._running:
                await processor.start()
            for item in items:
                await processor.add(item)
        
        wrapper.start = processor.start
        wrapper.stop = processor.stop
        wrapper.add = processor.add
        wrapper.add_many = processor.add_many
        wrapper.stats = processor.get_stats
        
        return wrapper
    return decorator
