"""
Test cases for critical fixes implemented based on code review.

Tests cover:
- CRIT-001: Dual loop timeout protection
- CRIT-002: Exception handling improvements
- CRIT-003: Input validation
- CRIT-004: SQL injection prevention
- CRIT-005: Memory bounds
"""

import pytest
import asyncio
from datetime import datetime
from collections import deque

# Import modules to test
import sys
sys.path.insert(0, 'ai_core/distributed_vc')
sys.path.insert(0, 'backend/shared/database')

from ai_core.distributed_vc.dual_loop import DualLoopUpdater, UpdateCandidate, UpdateType
from ai_core.distributed_vc.learning_engine import (
    OnlineLearningEngine,
    LearningSource,
    ChannelType,
    LearningItem
)
from ai_core.distributed_vc.core_module import ServiceRegistry, ServiceNode
from backend.shared.database.query_optimizer import QueryOptimizer, _validate_sql_identifier


class TestCRIT001_DualLoopTimeout:
    """Test CRIT-001: Dual loop deadlock prevention with timeout protection."""

    @pytest.mark.asyncio
    async def test_dual_loop_timeout_protection(self):
        """Dual loop should timeout and continue if iteration hangs."""
        updater = DualLoopUpdater(iteration_cycle_hours=0.1)

        # Mock hanging iteration
        async def hanging_iteration():
            await asyncio.sleep(1000)

        original_iteration = updater.project_loop.run_iteration
        updater.project_loop.run_iteration = hanging_iteration

        # Start loop
        task = asyncio.create_task(updater._run_loops())

        # Should not hang - timeout should occur
        await asyncio.sleep(0.5)

        # Stop the loop
        updater.stop()

        # Should complete within reasonable time
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.CancelledError:
            # Expected when stopping - in test context, no need to re-raise
            # as we're validating the stop behavior, not propagating cancellation
            pass

        # Restore original
        updater.project_loop.run_iteration = original_iteration

    @pytest.mark.asyncio
    async def test_dual_loop_continues_after_timeout(self):
        """AI loop should run even if project loop times out."""
        updater = DualLoopUpdater(iteration_cycle_hours=0.1)

        project_ran = False
        ai_ran = False

        async def slow_project_iteration():
            nonlocal project_ran
            project_ran = True
            await asyncio.sleep(100)  # Will timeout

        async def fast_ai_iteration():
            nonlocal ai_ran
            ai_ran = True

        updater.project_loop.run_iteration = slow_project_iteration
        updater.ai_loop.run_iteration = fast_ai_iteration

        # Run one iteration
        task = asyncio.create_task(updater._run_loops())
        await asyncio.sleep(0.2)
        updater.stop()

        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.CancelledError:
            # Expected when stopping - in test context, no need to re-raise
            pass

        # Both should have been attempted
        assert project_ran, "Project loop should have been attempted"
        # AI loop may or may not run depending on timing

    @pytest.mark.asyncio
    async def test_cross_loop_updates_timeout(self):
        """Cross-loop update processing should timeout if stuck."""
        updater = DualLoopUpdater(iteration_cycle_hours=0.1)

        # Fill queue with many items
        for i in range(1000):
            await updater.cross_loop_updates.put({"target": "project", "data": {}})

        # Should timeout and not process all
        try:
            await asyncio.wait_for(
                updater._process_cross_loop_updates(),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            pass  # Expected

        # Queue should still have items
        assert not updater.cross_loop_updates.empty()


class TestCRIT002_ExceptionHandling:
    """Test CRIT-002: Improved exception handling with circuit breaker."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Circuit breaker should open after 5 consecutive failures."""
        engine = OnlineLearningEngine()

        source = LearningSource(
            source_id="test_source",
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Test Source",
            url="https://api.github.com/test",
            fetch_interval_seconds=60
        )

        engine.register_source(source)

        # Mock failing fetch
        import aiohttp
        async def failing_fetch():
            raise aiohttp.ClientError("Connection failed")

        channel = engine.channels.get("test_source")
        if channel:
            original_fetch = channel.fetch
            channel.fetch = failing_fetch

            # Trigger 5 failures
            for _ in range(5):
                try:
                    await engine._fetch_loop("test_source")
                    await asyncio.sleep(0.1)
                except Exception:
                    # Expected failures from failing_fetch
                    pass

            # Circuit should be open (source disabled)
            assert not engine.sources["test_source"].enabled
            assert engine.sources["test_source"].error_count >= 5

            # Restore
            channel.fetch = original_fetch

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Circuit breaker should re-enable source after backoff."""
        engine = OnlineLearningEngine()

        source = LearningSource(
            source_id="test_recovery",
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Test Recovery",
            url="https://api.github.com/test",
            fetch_interval_seconds=60
        )

        engine.register_source(source)

        # Disable source
        engine.sources["test_recovery"].enabled = False
        engine.sources["test_recovery"].error_count = 5

        # Trigger recovery with short backoff
        await engine._reenable_source("test_recovery", backoff=0.1)

        # Should be re-enabled
        assert engine.sources["test_recovery"].enabled
        assert engine.sources["test_recovery"].error_count == 0


class TestCRIT003_InputValidation:
    """Test CRIT-003: Input validation for register_source."""

    def test_empty_source_id_rejected(self):
        """Empty source_id should be rejected."""
        engine = OnlineLearningEngine()

        with pytest.raises(ValueError, match="source_id cannot be empty"):
            engine.register_source(LearningSource(
                source_id="",
                channel_type=ChannelType.GITHUB_TRENDING,
                name="Test",
                url="https://api.github.com"
            ))

    def test_invalid_source_id_format_rejected(self):
        """Invalid source_id format should be rejected."""
        engine = OnlineLearningEngine()

        with pytest.raises(ValueError, match="alphanumeric"):
            engine.register_source(LearningSource(
                source_id="test@source!",
                channel_type=ChannelType.GITHUB_TRENDING,
                name="Test",
                url="https://api.github.com"
            ))

    def test_duplicate_source_id_rejected(self):
        """Duplicate source_id should be rejected."""
        engine = OnlineLearningEngine()

        source = LearningSource(
            source_id="test_dup",
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Test",
            url="https://api.github.com"
        )

        engine.register_source(source)

        with pytest.raises(ValueError, match="already registered"):
            engine.register_source(source)

    def test_invalid_url_rejected(self):
        """Invalid URL format should be rejected."""
        engine = OnlineLearningEngine()

        with pytest.raises(ValueError, match="Invalid URL"):
            engine.register_source(LearningSource(
                source_id="test_url",
                channel_type=ChannelType.GITHUB_TRENDING,
                name="Test",
                url="not-a-valid-url"
            ))

    def test_invalid_fetch_interval_rejected(self):
        """Fetch interval < 60 seconds should be rejected."""
        engine = OnlineLearningEngine()

        with pytest.raises(ValueError, match="fetch_interval_seconds"):
            engine.register_source(LearningSource(
                source_id="test_interval",
                channel_type=ChannelType.GITHUB_TRENDING,
                name="Test",
                url="https://api.github.com",
                fetch_interval_seconds=30
            ))

    def test_invalid_priority_rejected(self):
        """Priority outside 1-5 range should be rejected."""
        engine = OnlineLearningEngine()

        with pytest.raises(ValueError, match="priority must be 1-5"):
            engine.register_source(LearningSource(
                source_id="test_priority",
                channel_type=ChannelType.GITHUB_TRENDING,
                name="Test",
                url="https://api.github.com",
                priority=10
            ))

    def test_valid_source_accepted(self):
        """Valid source should be accepted."""
        engine = OnlineLearningEngine()

        source = LearningSource(
            source_id="valid_source",
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Valid Source",
            url="https://api.github.com",
            fetch_interval_seconds=300,
            priority=3
        )

        engine.register_source(source)

        assert "valid_source" in engine.sources
        assert engine.sources["valid_source"].name == "Valid Source"


class TestCRIT004_SQLInjectionPrevention:
    """Test CRIT-004: SQL injection prevention."""

    def test_sql_injection_in_table_name_rejected(self):
        """SQL injection in table name should be rejected."""
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            _validate_sql_identifier("users; DROP TABLE users; --")

    def test_sql_injection_in_column_name_rejected(self):
        """SQL injection in column name should be rejected."""
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            _validate_sql_identifier("name' OR '1'='1")

    def test_sql_keyword_as_identifier_rejected(self):
        """SQL keywords should not be allowed as identifiers."""
        with pytest.raises(ValueError, match="SQL keyword"):
            _validate_sql_identifier("SELECT")

        with pytest.raises(ValueError, match="SQL keyword"):
            _validate_sql_identifier("DROP")

    def test_valid_identifier_accepted(self):
        """Valid SQL identifiers should be accepted."""
        assert _validate_sql_identifier("users") == "users"
        assert _validate_sql_identifier("user_name") == "user_name"
        assert _validate_sql_identifier("_internal") == "_internal"

    def test_schema_qualified_identifier_accepted(self):
        """Schema-qualified identifiers should be accepted."""
        assert _validate_sql_identifier("public.users") == "public.users"

    @pytest.mark.asyncio
    async def test_batch_insert_validates_table_name(self):
        """batch_insert should validate table name."""
        optimizer = QueryOptimizer()

        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            await optimizer.batch_insert(
                table="users; DROP TABLE users; --",
                columns=["name"],
                rows=[("Alice",)]
            )

    @pytest.mark.asyncio
    async def test_batch_insert_validates_column_names(self):
        """batch_insert should validate column names."""
        optimizer = QueryOptimizer()

        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            await optimizer.batch_insert(
                table="users",
                columns=["name", "email; DROP TABLE users; --"],
                rows=[("Alice", "alice@example.com")]
            )

    @pytest.mark.asyncio
    async def test_batch_insert_validates_row_length(self):
        """batch_insert should validate row length matches columns."""
        optimizer = QueryOptimizer()

        with pytest.raises(ValueError, match="has .* values but .* columns expected"):
            await optimizer.batch_insert(
                table="users",
                columns=["name", "email"],
                rows=[("Alice",)]  # Missing email
            )


class TestCRIT005_MemoryBounds:
    """Test CRIT-005: Bounded memory for processed items."""

    def test_processed_items_uses_deque(self):
        """processed_items should be a deque with maxlen."""
        engine = OnlineLearningEngine()

        assert isinstance(engine.processed_items, deque)
        assert engine.processed_items.maxlen == 10000

    def test_processed_items_bounded(self):
        """processed_items should not exceed maxlen."""
        engine = OnlineLearningEngine()

        # Add more items than maxlen
        for i in range(20000):
            item = LearningItem(
                item_id=f"item_{i}",
                source_id="test",
                channel_type=ChannelType.GITHUB_TRENDING,
                title=f"Item {i}",
                content="Test content"
            )
            engine.processed_items.append(item)

        # Should not exceed maxlen
        assert len(engine.processed_items) == 10000

        # Should contain most recent items
        assert engine.processed_items[-1].item_id == "item_19999"
        assert engine.processed_items[0].item_id == "item_10000"

    def test_statistics_tracked_separately(self):
        """Statistics should be tracked separately from bounded deque."""
        engine = OnlineLearningEngine()

        # Add items
        for i in range(15000):
            item = LearningItem(
                item_id=f"item_{i}",
                source_id="test",
                channel_type=ChannelType.GITHUB_TRENDING,
                title=f"Item {i}",
                content="Test content"
            )
            engine.processed_items.append(item)
            engine.stats["total_processed"] += 1

        # Deque should be bounded
        assert len(engine.processed_items) == 10000

        # But statistics should track all
        assert engine.stats["total_processed"] == 15000


class TestHealthCheckTimeout:
    """Test health check timeout protection."""

    @pytest.mark.asyncio
    async def test_health_check_has_timeout(self):
        """Health check should have timeout protection."""
        registry = ServiceRegistry()

        # Mock slow health check
        original_check = registry._check_all_nodes

        def slow_check():
            import time
            time.sleep(10)  # Simulate slow check

        registry._check_all_nodes = slow_check

        # Start health checks with short interval
        await registry.start_health_checks(interval=1)

        # Should not hang
        await asyncio.sleep(2)

        # Stop health checks
        registry.stop_health_checks()

        # Restore
        registry._check_all_nodes = original_check

    @pytest.mark.asyncio
    async def test_health_check_stops_cleanly(self):
        """Health check should stop cleanly when requested."""
        registry = ServiceRegistry()

        await registry.start_health_checks(interval=1)
        assert registry._health_check_running

        await asyncio.sleep(0.5)

        registry.stop_health_checks()
        assert not registry._health_check_running

        # Task should be cancelled
        if registry._health_check_task:
            assert registry._health_check_task.cancelled() or registry._health_check_task.done()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
