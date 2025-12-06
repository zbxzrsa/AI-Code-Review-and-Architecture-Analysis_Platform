"""
Unit Tests for CQRS Pattern Implementation

Tests cover:
- Command handling and validation
- Query execution with caching
- Event sourcing and replay
- Read model synchronization
"""
import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, 'd:/Desktop/AI-Code-Review-and-Architecture-Analysis_Platform')

from backend.shared.patterns.cqrs.commands import (
    Command, CommandMetadata, CommandResult, CommandBus,
    CreateAnalysisCommand, PromoteVersionCommand,
    DomainEvent, logging_middleware, audit_middleware,
)
from backend.shared.patterns.cqrs.queries import (
    Query, QueryMetadata, QueryResult, QueryBus, QueryCache,
    GetAnalysisQuery, ListAnalysesQuery, GetVersionStatusQuery,
    PaginationParams, SortParams, SortOrder,
)
from backend.shared.patterns.cqrs.event_sourcing import (
    StoredEvent, Snapshot, InMemoryEventStore, EventPublisher,
    EventReplayer, AnalysisProjector,
)
from backend.shared.patterns.cqrs.read_models import (
    InMemoryReadModelStore, AnalysisReadModel, VersionStatusReadModel,
    ReadModelSynchronizer,
)


# =============================================================================
# Command Tests
# =============================================================================

class TestCommandMetadata:
    """Tests for CommandMetadata."""
    
    def test_metadata_has_unique_id(self):
        """Each command should have a unique ID."""
        meta1 = CommandMetadata()
        meta2 = CommandMetadata()
        assert meta1.command_id != meta2.command_id
    
    def test_metadata_has_timestamp(self):
        """Metadata should have a timestamp."""
        meta = CommandMetadata()
        assert meta.timestamp is not None
        assert isinstance(meta.timestamp, datetime)


class TestCreateAnalysisCommand:
    """Tests for CreateAnalysisCommand."""
    
    def test_create_command(self):
        """Should create a valid command."""
        cmd = CreateAnalysisCommand(
            code="print('hello')",
            language="python",
            rules=["security", "quality"],
            project_id="proj-123"
        )
        
        assert cmd.code == "print('hello')"
        assert cmd.language == "python"
        assert "security" in cmd.rules
        assert cmd.command_type == "CreateAnalysisCommand"
    
    def test_command_to_dict(self):
        """Should convert to dictionary."""
        cmd = CreateAnalysisCommand(code="x=1", language="python")
        data = cmd.to_dict()
        
        assert "command_type" in data
        assert "command_id" in data
        assert data["command_type"] == "CreateAnalysisCommand"


class TestCommandBus:
    """Tests for CommandBus."""
    
    @pytest.fixture
    def command_bus(self):
        return CommandBus()
    
    @pytest.mark.asyncio
    async def test_dispatch_without_handler_returns_error(self, command_bus):
        """Should return error when no handler registered."""
        cmd = CreateAnalysisCommand(code="x=1", language="python")
        result = await command_bus.dispatch(cmd)
        
        assert result.success is False
        assert "No handler" in result.error
    
    @pytest.mark.asyncio
    async def test_middleware_is_applied(self, command_bus):
        """Middleware should be applied to commands."""
        applied = []
        
        async def test_middleware(cmd):
            applied.append(cmd.command_type)
            return cmd
        
        command_bus.add_middleware(test_middleware)
        
        # Need to register a mock handler
        mock_handler = MagicMock()
        mock_handler.can_handle = MagicMock(return_value=True)
        mock_handler.handle = AsyncMock(return_value=CommandResult(
            success=True, command_id="test"
        ))
        
        command_bus.register_handler(CreateAnalysisCommand, mock_handler)
        
        cmd = CreateAnalysisCommand(code="x=1", language="python")
        await command_bus.dispatch(cmd)
        
        assert "CreateAnalysisCommand" in applied
    
    def test_get_metrics(self, command_bus):
        """Should return metrics."""
        metrics = command_bus.get_metrics()
        
        assert "total_commands" in metrics
        assert "successful_commands" in metrics
        assert "failed_commands" in metrics
        assert "registered_handlers" in metrics


class TestDomainEvent:
    """Tests for DomainEvent."""
    
    def test_event_has_unique_id(self):
        """Each event should have a unique ID."""
        event1 = DomainEvent(event_type="TestEvent", aggregate_id="agg-1", aggregate_type="Test")
        event2 = DomainEvent(event_type="TestEvent", aggregate_id="agg-1", aggregate_type="Test")
        
        assert event1.event_id != event2.event_id
    
    def test_event_to_dict(self):
        """Should convert to dictionary."""
        event = DomainEvent(
            event_type="TestEvent",
            aggregate_id="agg-1",
            aggregate_type="Test",
            data={"key": "value"}
        )
        data = event.to_dict()
        
        assert data["event_type"] == "TestEvent"
        assert data["aggregate_id"] == "agg-1"
        assert data["data"]["key"] == "value"


# =============================================================================
# Query Tests
# =============================================================================

class TestQueryMetadata:
    """Tests for QueryMetadata."""
    
    def test_metadata_has_default_cache_ttl(self):
        """Metadata should have default cache TTL."""
        meta = QueryMetadata()
        assert meta.cache_ttl == 300


class TestPaginationParams:
    """Tests for PaginationParams."""
    
    def test_offset_calculation(self):
        """Offset should be calculated from page and page_size."""
        params = PaginationParams(page=3, page_size=20)
        assert params.offset == 40  # (3-1) * 20
    
    def test_default_values(self):
        """Should have sensible defaults."""
        params = PaginationParams()
        assert params.page == 1
        assert params.page_size == 50


class TestGetAnalysisQuery:
    """Tests for GetAnalysisQuery."""
    
    def test_query_type(self):
        """Should return correct query type."""
        query = GetAnalysisQuery(analysis_id="analysis-123")
        assert query.query_type == "GetAnalysisQuery"
    
    def test_cache_key_generation(self):
        """Should generate consistent cache keys."""
        query1 = GetAnalysisQuery(analysis_id="analysis-123")
        query2 = GetAnalysisQuery(analysis_id="analysis-123")
        
        # Same query should generate same cache key
        key1 = query1.get_cache_key()
        key2 = query2.get_cache_key()
        assert key1 == key2


class TestQueryBus:
    """Tests for QueryBus."""
    
    @pytest.fixture
    def query_bus(self):
        return QueryBus()
    
    @pytest.mark.asyncio
    async def test_execute_without_handler_returns_error(self, query_bus):
        """Should return error when no handler registered."""
        query = GetAnalysisQuery(analysis_id="test")
        result = await query_bus.execute(query)
        
        assert result.success is False
        assert "No handler" in result.error
    
    def test_response_time_target(self, query_bus):
        """Should have response time target defined."""
        assert query_bus.RESPONSE_TIME_TARGET_MS == 200
    
    def test_get_metrics(self, query_bus):
        """Should return metrics."""
        metrics = query_bus.get_metrics()
        
        assert "total_queries" in metrics
        assert "cache_hits" in metrics
        assert "slow_queries" in metrics
        assert "avg_execution_time_ms" in metrics


class TestQueryCache:
    """Tests for QueryCache."""
    
    @pytest.fixture
    def cache(self):
        return QueryCache(max_size=100, default_ttl=300)
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Should store and retrieve values."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, cache):
        """Should return None for nonexistent keys."""
        result = await cache.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Should delete values."""
        await cache.set("key1", "value1")
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Should clear all values."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        assert cache.size() == 0


# =============================================================================
# Event Sourcing Tests
# =============================================================================

class TestStoredEvent:
    """Tests for StoredEvent."""
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        event = StoredEvent(
            event_id="evt-1",
            event_type="TestEvent",
            aggregate_id="agg-1",
            aggregate_type="Test",
            data={"key": "value"},
            metadata={},
            timestamp=datetime.now(timezone.utc),
            version=1,
            sequence_number=1
        )
        
        data = event.to_dict()
        assert data["event_id"] == "evt-1"
        assert data["event_type"] == "TestEvent"
    
    def test_from_dict(self):
        """Should reconstruct from dictionary."""
        data = {
            "event_id": "evt-1",
            "event_type": "TestEvent",
            "aggregate_id": "agg-1",
            "aggregate_type": "Test",
            "data": {"key": "value"},
            "metadata": {},
            "timestamp": "2024-01-15T10:00:00+00:00",
            "version": 1,
            "sequence_number": 1
        }
        
        event = StoredEvent.from_dict(data)
        assert event.event_id == "evt-1"
        assert event.data["key"] == "value"


class TestInMemoryEventStore:
    """Tests for InMemoryEventStore."""
    
    @pytest.fixture
    def event_store(self):
        return InMemoryEventStore()
    
    @pytest.mark.asyncio
    async def test_append_event(self, event_store):
        """Should append events and return sequence number."""
        event = DomainEvent(
            event_type="TestEvent",
            aggregate_id="agg-1",
            aggregate_type="Test"
        )
        
        seq = await event_store.append(event)
        assert seq == 1
        
        seq2 = await event_store.append(event)
        assert seq2 == 2
    
    @pytest.mark.asyncio
    async def test_get_events_for_aggregate(self, event_store):
        """Should get events for specific aggregate."""
        event1 = DomainEvent(event_type="Event1", aggregate_id="agg-1", aggregate_type="Test")
        event2 = DomainEvent(event_type="Event2", aggregate_id="agg-2", aggregate_type="Test")
        event3 = DomainEvent(event_type="Event3", aggregate_id="agg-1", aggregate_type="Test")
        
        await event_store.append(event1)
        await event_store.append(event2)
        await event_store.append(event3)
        
        events = await event_store.get_events("agg-1")
        assert len(events) == 2
    
    @pytest.mark.asyncio
    async def test_get_all_events(self, event_store):
        """Should get all events from sequence."""
        for i in range(5):
            event = DomainEvent(
                event_type=f"Event{i}",
                aggregate_id=f"agg-{i}",
                aggregate_type="Test"
            )
            await event_store.append(event)
        
        events = await event_store.get_all_events(from_sequence=3, limit=10)
        assert len(events) == 3  # Events 3, 4, 5
    
    @pytest.mark.asyncio
    async def test_snapshot_save_and_get(self, event_store):
        """Should save and retrieve snapshots."""
        snapshot = Snapshot(
            aggregate_id="agg-1",
            aggregate_type="Test",
            state={"counter": 10},
            version=5,
            timestamp=datetime.now(timezone.utc)
        )
        
        await event_store.save_snapshot(snapshot)
        retrieved = await event_store.get_snapshot("agg-1")
        
        assert retrieved is not None
        assert retrieved.state["counter"] == 10
        assert retrieved.version == 5


class TestEventPublisher:
    """Tests for EventPublisher."""
    
    @pytest.fixture
    def publisher(self):
        store = InMemoryEventStore()
        return EventPublisher(store)
    
    @pytest.mark.asyncio
    async def test_publish_notifies_subscribers(self, publisher):
        """Should notify subscribers when publishing."""
        received_events = []
        
        async def handler(event):
            received_events.append(event)
        
        publisher.subscribe("TestEvent", handler)
        
        event = DomainEvent(
            event_type="TestEvent",
            aggregate_id="agg-1",
            aggregate_type="Test"
        )
        
        await publisher.publish(event)
        assert len(received_events) == 1
    
    @pytest.mark.asyncio
    async def test_subscribe_all(self, publisher):
        """Should receive all events with subscribe_all."""
        received = []
        
        async def handler(event):
            received.append(event)
        
        publisher.subscribe_all(handler)
        
        event1 = DomainEvent(event_type="Event1", aggregate_id="agg-1", aggregate_type="Test")
        event2 = DomainEvent(event_type="Event2", aggregate_id="agg-2", aggregate_type="Test")
        
        await publisher.publish(event1)
        await publisher.publish(event2)
        
        assert len(received) == 2


# =============================================================================
# Read Model Tests
# =============================================================================

class TestInMemoryReadModelStore:
    """Tests for InMemoryReadModelStore."""
    
    @pytest.fixture
    def store(self):
        return InMemoryReadModelStore()
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, store):
        """Should store and retrieve data."""
        await store.set("key1", {"name": "test", "value": 123})
        data = await store.get("key1")
        
        assert data["name"] == "test"
        assert data["value"] == 123
    
    @pytest.mark.asyncio
    async def test_query_with_filters(self, store):
        """Should filter results."""
        await store.set("item1", {"status": "active", "type": "a"})
        await store.set("item2", {"status": "inactive", "type": "b"})
        await store.set("item3", {"status": "active", "type": "c"})
        
        results, total = await store.query({"status": "active"})
        
        assert total == 2
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_query_with_pagination(self, store):
        """Should paginate results."""
        for i in range(10):
            await store.set(f"item{i}", {"index": i})
        
        results, total = await store.query({}, offset=3, limit=3)
        
        assert total == 10
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Should delete data."""
        await store.set("key1", {"value": 1})
        await store.delete("key1")
        data = await store.get("key1")
        
        assert data is None


class TestAnalysisReadModel:
    """Tests for AnalysisReadModel."""
    
    @pytest.fixture
    def read_model(self):
        store = InMemoryReadModelStore()
        return AnalysisReadModel(store)
    
    @pytest.mark.asyncio
    async def test_create_and_get_analysis(self, read_model):
        """Should create and retrieve analysis."""
        await read_model.create_analysis({
            "id": "analysis-1",
            "language": "python",
            "project_id": "proj-1",
            "status": "completed"
        })
        
        data = await read_model.get_analysis("analysis-1")
        
        assert data is not None
        assert data["language"] == "python"
    
    @pytest.mark.asyncio
    async def test_update_analysis(self, read_model):
        """Should update analysis."""
        await read_model.create_analysis({
            "id": "analysis-1",
            "status": "running"
        })
        
        await read_model.update_analysis("analysis-1", {"status": "completed"})
        
        data = await read_model.get_analysis("analysis-1")
        assert data["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_add_issue(self, read_model):
        """Should add issues to analysis."""
        await read_model.create_analysis({"id": "analysis-1"})
        await read_model.add_issue("analysis-1", {
            "id": "issue-1",
            "severity": "high",
            "message": "Test issue"
        })
        
        data = await read_model.get_analysis("analysis-1")
        assert len(data["issues"]) == 1
        assert data["metrics"]["issues_count"] == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestCQRSIntegration:
    """Integration tests for CQRS pattern."""
    
    @pytest.mark.asyncio
    async def test_command_produces_event_updates_read_model(self):
        """Full flow: command -> event -> read model update."""
        # Setup
        event_store = InMemoryEventStore()
        read_model_store = InMemoryReadModelStore()
        analysis_read_model = AnalysisReadModel(read_model_store)
        
        # Simulate command handling
        analysis_id = "analysis-123"
        
        # Create event
        event = DomainEvent(
            event_type="AnalysisCreated",
            aggregate_id=analysis_id,
            aggregate_type="Analysis",
            data={"language": "python", "project_id": "proj-1"}
        )
        
        # Store event
        await event_store.append(event)
        
        # Update read model (simulating projector)
        await analysis_read_model.create_analysis({
            "id": analysis_id,
            "language": event.data["language"],
            "project_id": event.data["project_id"],
            "status": "created"
        })
        
        # Query read model
        data = await analysis_read_model.get_analysis(analysis_id)
        
        assert data is not None
        assert data["language"] == "python"
        assert data["project_id"] == "proj-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
