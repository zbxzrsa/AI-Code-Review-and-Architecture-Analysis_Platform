"""
Networked Learning System

Main orchestrator for automatic networked learning.

Features:
1. Automatic data collection from prioritized sources (GitHub, ArXiv, Tech Blogs)
2. Data cleaning pipeline with quality scoring >= 0.8
3. Infinite learning with LRU caching (max 70% memory)
4. Technology deprecation (accuracy < 75%, 3 consecutive failures)
5. Data retention (raw 90 days, processed permanent)
6. User review workflow for critical operations
7. Real-time monitoring and alerting
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .collectors import BaseCollector, GitHubCollector, ArXivCollector, TechBlogCollector
from .collectors.base import CollectedItem, CollectionResult
from .config import (
    NetworkedLearningConfig,
    DeprecationCriteria,
    DataSourcePriority,
)
from .monitoring import AlertManager, MetricsCollector, SystemMonitor
from .pipeline import DataCleaningPipeline, CleaningResult, QualityScore
from .storage import StorageManager

logger = logging.getLogger(__name__)


class SystemState(str, Enum):
    """System lifecycle states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class TechnologyStatus:
    """Status of a tracked technology."""
    technology_id: str
    name: str
    accuracy: float = 1.0
    consecutive_failures: int = 0
    deprecated: bool = False
    deprecated_at: Optional[datetime] = None
    last_evaluation: Optional[datetime] = None
    pending_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "technology_id": self.technology_id,
            "name": self.name,
            "accuracy": round(self.accuracy, 3),
            "consecutive_failures": self.consecutive_failures,
            "deprecated": self.deprecated,
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "pending_review": self.pending_review,
        }


@dataclass
class PendingApproval:
    """Pending user approval request."""
    approval_id: str
    action_type: str  # "new_technology", "deprecate_technology", "parameter_change"
    description: str
    details: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    approved: Optional[bool] = None
    approved_by: Optional[str] = None


class NetworkedLearningSystem:
    """
    Automatic networked learning system for V1/V3 versions.

    Collection Schedule:
    - Polls each data source every hour
    - Priority: GitHub (1) > ArXiv (2) > Tech Blogs (3)

    Quality Threshold: >= 0.8

    Performance Targets:
    - Processing latency < 500ms
    - Availability > 99.9%
    - Daily capacity >= 1TB

    Usage:
        config = NetworkedLearningConfig(target_version="v1")

        async with NetworkedLearningSystem(config) as system:
            await system.run_forever()
    """

    def __init__(self, config: NetworkedLearningConfig):
        """
        Initialize networked learning system.

        Args:
            config: System configuration
        """
        self.config = config
        self.state = SystemState.STOPPED

        # Initialize collectors (ordered by priority)
        self._collectors: List[BaseCollector] = []
        self._init_collectors()

        # Initialize pipeline
        self._pipeline = DataCleaningPipeline(
            config,
            on_item_processed=self._on_item_processed,
        )

        # Initialize storage
        self._storage = StorageManager(
            memory_config=config.memory,
            storage_config=config.storage,
            retention_policy=config.retention,
        )

        # Initialize monitoring
        self._metrics = MetricsCollector(config.monitoring.metrics_interval_seconds)
        self._alerts = AlertManager()
        self._monitor = SystemMonitor(config.monitoring, self._metrics, self._alerts)

        # Technology tracking
        self._technologies: Dict[str, TechnologyStatus] = {}

        # User review queue
        self._pending_approvals: Dict[str, PendingApproval] = {}
        self._approval_counter = 0

        # Collection state
        self._last_collection: Dict[str, datetime] = {}
        self._collection_task: Optional[asyncio.Task] = None

    def _init_collectors(self):
        """Initialize data collectors from config."""
        for source_config in self.config.get_sources_by_priority():
            if not source_config.enabled:
                continue

            if source_config.priority == DataSourcePriority.GITHUB:
                collector = GitHubCollector(source_config, self.config.collection)
            elif source_config.priority == DataSourcePriority.ARXIV:
                collector = ArXivCollector(source_config, self.config.collection)
            elif source_config.priority == DataSourcePriority.TECH_BLOGS:
                collector = TechBlogCollector(source_config, self.config.collection)
            else:
                logger.warning(f"Unknown source priority: {source_config.priority}")
                continue

            self._collectors.append(collector)

        logger.info(f"Initialized {len(self._collectors)} collectors")

    async def start(self):
        """Start the learning system."""
        if self.state == SystemState.RUNNING:
            return

        self.state = SystemState.STARTING
        logger.info("Starting networked learning system...")

        try:
            # Start collectors
            for collector in self._collectors:
                await collector.start()

            # Start pipeline
            await self._pipeline.start()

            # Start storage
            await self._storage.start()

            # Start monitoring
            await self._monitor.start()

            # Register health checkers
            self._register_health_checkers()

            self.state = SystemState.RUNNING
            logger.info("Networked learning system started")

        except Exception as e:
            self.state = SystemState.ERROR
            logger.error(f"Failed to start system: {e}")
            raise

    async def stop(self):
        """Stop the learning system."""
        if self.state == SystemState.STOPPED:
            return

        self.state = SystemState.STOPPING
        logger.info("Stopping networked learning system...")

        # Cancel collection task
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                # Intentionally not re-raised: we initiated the cancellation
                # during shutdown, so propagation is not needed
                logger.debug("Collection task cancelled during shutdown")

        # Stop components
        await self._monitor.stop()
        await self._storage.stop()
        await self._pipeline.stop()

        for collector in self._collectors:
            await collector.stop()

        self.state = SystemState.STOPPED
        logger.info("Networked learning system stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False

    def _register_health_checkers(self):
        """Register component health checkers."""
        self._monitor.register_health_checker(
            "storage",
            lambda: self._storage.check_memory_usage(),
        )

        for collector in self._collectors:
            self._monitor.register_health_checker(
                f"collector_{collector.name}",
                lambda c=collector: not c.is_rate_limited,
            )

    def _on_item_processed(self, item: CollectedItem, score: QualityScore):
        """Callback when an item is processed."""
        # Track quality metrics
        self._metrics.record_processing(
            items_processed=1,
            latency_ms=0,  # Updated in batch processing
            quality_scores=[score.overall],
        )

    async def run_collection_cycle(self) -> Dict[str, CollectionResult]:
        """
        Run a single collection cycle across all sources.

        Returns:
            Dictionary of source -> CollectionResult
        """
        import time

        results: Dict[str, CollectionResult] = {}
        all_items: List[CollectedItem] = []

        # Collect from each source in priority order
        for collector in self._collectors:
            # Check if it's time to collect
            last_time = self._last_collection.get(collector.name)
            interval = timedelta(seconds=self.config.collection.interval_seconds)

            if last_time and datetime.now(timezone.utc) - last_time < interval:
                continue

            logger.info(f"Collecting from {collector.name}...")

            try:
                # Get items since last collection
                since = last_time if last_time else None

                items = []
                async for item in collector.collect(since=since):
                    items.append(item)

                result = CollectionResult(
                    source=collector.name,
                    success=True,
                    items_collected=len(items),
                )

                all_items.extend(items)
                self._last_collection[collector.name] = datetime.now(timezone.utc)

            except Exception as e:
                result = CollectionResult(
                    source=collector.name,
                    success=False,
                    errors=[str(e)],
                )
                logger.error(f"Collection failed for {collector.name}: {e}")

            results[collector.name] = result
            self._metrics.record_collection(
                items_collected=result.items_collected,
                items_filtered=result.items_filtered,
                errors=len(result.errors),
            )

        # Process collected items through pipeline
        if all_items:
            start_time = time.time()
            cleaning_result = await self._pipeline.process(all_items)
            latency_ms = (time.time() - start_time) * 1000 / len(all_items)

            self._metrics.record_processing(
                items_processed=cleaning_result.output_count,
                latency_ms=latency_ms,
            )

            # Store cleaned items
            if cleaning_result.output_count > 0:
                # Get cleaned items from pipeline (they've been indexed)
                # In practice, you'd pass them through more explicitly
                pass

        return results

    async def run_forever(self):
        """Run continuous collection cycles."""
        self._collection_task = asyncio.create_task(self._collection_loop())
        await self._collection_task

    async def _collection_loop(self):
        """Main collection loop."""
        while self.state == SystemState.RUNNING:
            try:
                results = await self.run_collection_cycle()

                # Log summary
                total_collected = sum(r.items_collected for r in results.values())
                logger.info(f"Collection cycle complete: {total_collected} items collected")

                # Wait before next cycle
                await asyncio.sleep(self.config.collection.interval_seconds)

            except asyncio.CancelledError:
                # Re-raise to properly propagate cancellation
                raise
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    # =========================================================================
    # Technology Deprecation
    # =========================================================================

    def evaluate_technology(
        self,
        technology_id: str,
        accuracy: float,
        success: bool,
    ) -> Optional[TechnologyStatus]:
        """
        Evaluate a technology and check for deprecation.

        Deprecation criteria:
        - Accuracy < 75%
        - 3 consecutive evaluation failures

        Args:
            technology_id: Technology identifier
            accuracy: Current accuracy score
            success: Whether evaluation was successful

        Returns:
            Updated technology status
        """
        status = self._technologies.get(technology_id)

        if not status:
            status = TechnologyStatus(
                technology_id=technology_id,
                name=technology_id,
            )
            self._technologies[technology_id] = status

        # Update metrics
        status.accuracy = accuracy
        status.last_evaluation = datetime.now(timezone.utc)

        if success:
            status.consecutive_failures = 0
        else:
            status.consecutive_failures += 1

        # Check deprecation criteria
        criteria = self.config.deprecation

        should_deprecate = (
            accuracy < criteria.min_accuracy or
            status.consecutive_failures >= criteria.max_consecutive_failures
        )

        if should_deprecate and not status.deprecated:
            if criteria.require_user_confirmation:
                # Queue for user review
                self._request_deprecation_approval(status)
            else:
                # Auto-deprecate
                self._deprecate_technology(status)

        return status

    def _request_deprecation_approval(self, status: TechnologyStatus):
        """Request user approval for deprecation."""
        status.pending_review = True

        _approval = self._create_approval(  # Stored in approval system
            action_type="deprecate_technology",
            description=f"Deprecate technology: {status.name}",
            details={
                "technology_id": status.technology_id,
                "accuracy": status.accuracy,
                "consecutive_failures": status.consecutive_failures,
            },
        )

        logger.warning(
            f"Technology deprecation pending approval: {status.name} "
            f"(accuracy={status.accuracy:.2%}, failures={status.consecutive_failures})"
        )

    def _deprecate_technology(self, status: TechnologyStatus):
        """Mark technology as deprecated and cleanup."""
        status.deprecated = True
        status.deprecated_at = datetime.now(timezone.utc)
        status.pending_review = False

        # Stop related learning tasks
        if self.config.deprecation.stop_learning_tasks:
            logger.info(f"Stopping learning tasks for: {status.name}")

        # Trigger data cleanup
        if self.config.deprecation.trigger_data_cleanup:
            asyncio.create_task(
                self._storage.archive_deprecated_technology(status.technology_id)
            )

        logger.warning(f"Technology deprecated: {status.name}")

    # =========================================================================
    # User Review Workflow
    # =========================================================================

    def _create_approval(
        self,
        action_type: str,
        description: str,
        details: Dict[str, Any],
    ) -> PendingApproval:
        """Create a pending approval request."""
        self._approval_counter += 1
        approval_id = f"approval_{self._approval_counter:06d}"

        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=self.config.user_review.approval_timeout_hours)

        approval = PendingApproval(
            approval_id=approval_id,
            action_type=action_type,
            description=description,
            details=details,
            created_at=now,
            expires_at=expires,
        )

        self._pending_approvals[approval_id] = approval
        return approval

    def approve(self, approval_id: str, user: str) -> bool:
        """
        Approve a pending request.

        Args:
            approval_id: Approval to approve
            user: User approving

        Returns:
            True if successful
        """
        approval = self._pending_approvals.get(approval_id)
        if not approval:
            return False

        if datetime.now(timezone.utc) > approval.expires_at:
            logger.warning(f"Approval {approval_id} has expired")
            return False

        approval.approved = True
        approval.approved_by = user

        # Execute the approved action
        self._execute_approved_action(approval)

        logger.info(f"Approval {approval_id} approved by {user}")
        return True

    def reject(self, approval_id: str, user: str) -> bool:
        """
        Reject a pending request.

        Args:
            approval_id: Approval to reject
            user: User rejecting

        Returns:
            True if successful
        """
        approval = self._pending_approvals.get(approval_id)
        if not approval:
            return False

        approval.approved = False
        approval.approved_by = user

        # Handle rejection
        if approval.action_type == "deprecate_technology":
            tech_id = approval.details.get("technology_id")
            if tech_id and tech_id in self._technologies:
                self._technologies[tech_id].pending_review = False

        logger.info(f"Approval {approval_id} rejected by {user}")
        return True

    def _execute_approved_action(self, approval: PendingApproval):
        """Execute an approved action."""
        if approval.action_type == "deprecate_technology":
            tech_id = approval.details.get("technology_id")
            if tech_id and tech_id in self._technologies:
                self._deprecate_technology(self._technologies[tech_id])

        elif approval.action_type == "new_technology":
            # Handle new technology introduction
            pass

        elif approval.action_type == "parameter_change":
            # Handle parameter changes
            pass

    def get_pending_approvals(self) -> List[PendingApproval]:
        """Get all pending approvals."""
        now = datetime.now(timezone.utc)
        return [
            a for a in self._pending_approvals.values()
            if a.approved is None and a.expires_at > now
        ]

    # =========================================================================
    # Status and Metrics
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "state": self.state.value,
            "target_version": self.config.target_version,
            "collectors": [
                {
                    "name": c.name,
                    "priority": c.priority.value,
                    "rate_limited": c.is_rate_limited,
                    "last_collection": self._last_collection.get(c.name, None),
                }
                for c in self._collectors
            ],
            "storage": self._storage.get_stats().to_dict(),
            "monitoring": self._monitor.get_dashboard_data(),
            "technologies": {
                "total": len(self._technologies),
                "deprecated": sum(1 for t in self._technologies.values() if t.deprecated),
                "pending_review": sum(1 for t in self._technologies.values() if t.pending_review),
            },
            "pending_approvals": len(self.get_pending_approvals()),
        }
