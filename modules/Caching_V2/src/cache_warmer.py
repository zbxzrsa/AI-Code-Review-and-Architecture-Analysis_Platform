"""
Caching_V2 - Cache Warmer

Proactive cache warming and preloading.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


@dataclass
class WarmingTask:
    """Cache warming task"""
    key: str
    loader: Callable[[], Awaitable[Any]]
    ttl_seconds: int
    priority: int = 0
    refresh_before_expiry_seconds: int = 60
    last_warmed: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def needs_refresh(self) -> bool:
        if self.expires_at is None:
            return True

        refresh_at = self.expires_at - timedelta(seconds=self.refresh_before_expiry_seconds)
        return datetime.now(timezone.utc) >= refresh_at


class CacheWarmer:
    """
    Proactive cache warming service.

    V2 Features:
    - Scheduled warming
    - Priority-based refresh
    - Refresh-ahead pattern
    - Warm on startup
    """

    def __init__(
        self,
        cache_set_func: Callable[[str, Any, int], Awaitable[None]],
        max_concurrent_warmers: int = 5,
    ):
        """
        Initialize cache warmer.

        Args:
            cache_set_func: Function to set cache values
            max_concurrent_warmers: Max parallel warming tasks
        """
        self.cache_set = cache_set_func
        self.max_concurrent = max_concurrent_warmers

        self._tasks: Dict[str, WarmingTask] = {}
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent_warmers)

        # Stats
        self._warm_count = 0
        self._warm_failures = 0

    def register_task(
        self,
        key: str,
        loader: Callable[[], Awaitable[Any]],
        ttl_seconds: int,
        priority: int = 0,
        refresh_before_seconds: int = 60,
    ):
        """Register warming task"""
        self._tasks[key] = WarmingTask(
            key=key,
            loader=loader,
            ttl_seconds=ttl_seconds,
            priority=priority,
            refresh_before_expiry_seconds=refresh_before_seconds,
        )
        logger.debug(f"Registered warming task: {key}")

    def unregister_task(self, key: str):
        """Remove warming task"""
        self._tasks.pop(key, None)

    async def warm_key(self, key: str) -> bool:
        """Warm a specific key"""
        task = self._tasks.get(key)
        if not task:
            return False

        return await self._execute_warm(task)

    async def warm_all(self):
        """Warm all registered keys"""
        tasks_to_warm = sorted(
            self._tasks.values(),
            key=lambda t: t.priority,
            reverse=True
        )

        for task in tasks_to_warm:
            await self._execute_warm(task)

    async def warm_expired(self):
        """Warm only keys that need refresh"""
        tasks_to_warm = [
            task for task in self._tasks.values()
            if task.needs_refresh()
        ]

        # Sort by priority
        tasks_to_warm.sort(key=lambda t: t.priority, reverse=True)

        for task in tasks_to_warm:
            await self._execute_warm(task)

    async def _execute_warm(self, task: WarmingTask) -> bool:
        """Execute warming task"""
        async with self._semaphore:
            try:
                # Load data
                data = await task.loader()

                # Set in cache
                await self.cache_set(task.key, data, task.ttl_seconds)

                # Update task
                now = datetime.now(timezone.utc)
                task.last_warmed = now
                task.expires_at = now + timedelta(seconds=task.ttl_seconds)

                self._warm_count += 1
                logger.debug(f"Warmed cache key: {task.key}")
                return True

            except Exception as e:
                self._warm_failures += 1
                logger.error(f"Failed to warm {task.key}: {e}")
                return False

    async def start_background_warming(self, interval_seconds: int = 30):
        """Start background warming loop"""
        self._running = True
        logger.info("Cache warmer started")

        # Initial warm
        await self.warm_all()

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)
                await self.warm_expired()
            except asyncio.CancelledError:
                # Re-raise to properly propagate cancellation
                raise
            except Exception as e:
                logger.error(f"Background warming error: {e}")
                await asyncio.sleep(5)

        logger.info("Cache warmer stopped")

    def stop(self):
        """Stop background warming"""
        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get warmer statistics"""
        expired_count = sum(
            1 for t in self._tasks.values()
            if t.needs_refresh()
        )

        return {
            "registered_tasks": len(self._tasks),
            "expired_tasks": expired_count,
            "total_warms": self._warm_count,
            "warm_failures": self._warm_failures,
            "success_rate": self._warm_count / max(1, self._warm_count + self._warm_failures),
            "running": self._running,
        }

    def get_task_status(self) -> List[Dict[str, Any]]:
        """Get status of all tasks"""
        return [
            {
                "key": task.key,
                "priority": task.priority,
                "ttl_seconds": task.ttl_seconds,
                "last_warmed": task.last_warmed.isoformat() if task.last_warmed else None,
                "expires_at": task.expires_at.isoformat() if task.expires_at else None,
                "needs_refresh": task.needs_refresh(),
            }
            for task in self._tasks.values()
        ]
