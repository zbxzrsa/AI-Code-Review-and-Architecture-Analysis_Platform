"""
Event Sourcing for CQRS

Implements event store and event-driven synchronization between
command and query models.

Features:
- Event store with persistence
- Event replay for read model rebuilding
- Async event publishing
- Event versioning and snapshots
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class StoredEvent:
    """Event as stored in event store."""
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    version: int
    sequence_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "sequence_number": self.sequence_number,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StoredEvent':
        return cls(
            event_id=d["event_id"],
            event_type=d["event_type"],
            aggregate_id=d["aggregate_id"],
            aggregate_type=d["aggregate_type"],
            data=d["data"],
            metadata=d.get("metadata", {}),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            version=d["version"],
            sequence_number=d.get("sequence_number", 0),
        )


@dataclass
class Snapshot:
    """Aggregate snapshot for faster replay."""
    aggregate_id: str
    aggregate_type: str
    state: Dict[str, Any]
    version: int
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "state": self.state,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
        }


class EventStore(ABC):
    """Abstract base class for event stores."""
    
    @abstractmethod
    async def append(self, event: 'DomainEvent') -> int:
        """Append event to store. Returns sequence number."""
        pass
    
    @abstractmethod
    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[StoredEvent]:
        """Get events for an aggregate."""
        pass
    
    @abstractmethod
    async def get_all_events(
        self,
        from_sequence: int = 0,
        limit: int = 1000
    ) -> List[StoredEvent]:
        """Get all events from a sequence number."""
        pass
    
    @abstractmethod
    async def save_snapshot(self, snapshot: Snapshot):
        """Save aggregate snapshot."""
        pass
    
    @abstractmethod
    async def get_snapshot(self, aggregate_id: str) -> Optional[Snapshot]:
        """Get latest snapshot for aggregate."""
        pass


class InMemoryEventStore(EventStore):
    """In-memory event store for development/testing."""
    
    def __init__(self):
        self._events: List[StoredEvent] = []
        self._snapshots: Dict[str, Snapshot] = {}
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._subscribers: List[Callable] = []
    
    async def append(self, event) -> int:
        """Append event to store."""
        async with self._lock:
            self._sequence += 1
            
            stored_event = StoredEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                aggregate_id=event.aggregate_id,
                aggregate_type=event.aggregate_type,
                data=event.data,
                metadata=event.metadata,
                timestamp=event.timestamp,
                version=event.version,
                sequence_number=self._sequence,
            )
            
            self._events.append(stored_event)
            
            logger.debug(
                f"Event stored: {event.event_type} for {event.aggregate_id}, "
                f"seq={self._sequence}"
            )
            
            # Notify subscribers
            for subscriber in self._subscribers:
                try:
                    await subscriber(stored_event)
                except Exception as e:
                    logger.error(f"Event subscriber error: {e}")
            
            return self._sequence
    
    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[StoredEvent]:
        """Get events for an aggregate."""
        async with self._lock:
            events = [
                e for e in self._events
                if e.aggregate_id == aggregate_id
                and e.version >= from_version
                and (to_version is None or e.version <= to_version)
            ]
            return sorted(events, key=lambda e: e.version)
    
    async def get_all_events(
        self,
        from_sequence: int = 0,
        limit: int = 1000
    ) -> List[StoredEvent]:
        """Get all events from a sequence number."""
        async with self._lock:
            events = [
                e for e in self._events
                if e.sequence_number >= from_sequence
            ]
            return sorted(events, key=lambda e: e.sequence_number)[:limit]
    
    async def save_snapshot(self, snapshot: Snapshot):
        """Save aggregate snapshot."""
        async with self._lock:
            self._snapshots[snapshot.aggregate_id] = snapshot
    
    async def get_snapshot(self, aggregate_id: str) -> Optional[Snapshot]:
        """Get latest snapshot for aggregate."""
        return self._snapshots.get(aggregate_id)
    
    def subscribe(self, callback: Callable):
        """Subscribe to new events."""
        self._subscribers.append(callback)
    
    def get_event_count(self) -> int:
        """Get total event count."""
        return len(self._events)


class PostgresEventStore(EventStore):
    """PostgreSQL-based event store for production."""
    
    def __init__(self, db_pool):
        self.db = db_pool
        self._subscribers: List[Callable] = []
    
    async def append(self, event) -> int:
        """Append event to PostgreSQL."""
        query = """
            INSERT INTO events.event_store
            (event_id, event_type, aggregate_id, aggregate_type, data, metadata, timestamp, version)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING sequence_number
        """
        
        result = await self.db.fetchval(
            query,
            event.event_id,
            event.event_type,
            event.aggregate_id,
            event.aggregate_type,
            json.dumps(event.data),
            json.dumps(event.metadata),
            event.timestamp,
            event.version,
        )
        
        sequence_number = result
        
        # Create stored event for subscribers
        stored_event = StoredEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            aggregate_id=event.aggregate_id,
            aggregate_type=event.aggregate_type,
            data=event.data,
            metadata=event.metadata,
            timestamp=event.timestamp,
            version=event.version,
            sequence_number=sequence_number,
        )
        
        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                await subscriber(stored_event)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}")
        
        return sequence_number
    
    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[StoredEvent]:
        """Get events for an aggregate from PostgreSQL."""
        if to_version:
            query = """
                SELECT * FROM events.event_store
                WHERE aggregate_id = $1 AND version >= $2 AND version <= $3
                ORDER BY version
            """
            rows = await self.db.fetch(query, aggregate_id, from_version, to_version)
        else:
            query = """
                SELECT * FROM events.event_store
                WHERE aggregate_id = $1 AND version >= $2
                ORDER BY version
            """
            rows = await self.db.fetch(query, aggregate_id, from_version)
        
        return [
            StoredEvent(
                event_id=row["event_id"],
                event_type=row["event_type"],
                aggregate_id=row["aggregate_id"],
                aggregate_type=row["aggregate_type"],
                data=json.loads(row["data"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                timestamp=row["timestamp"],
                version=row["version"],
                sequence_number=row["sequence_number"],
            )
            for row in rows
        ]
    
    async def get_all_events(
        self,
        from_sequence: int = 0,
        limit: int = 1000
    ) -> List[StoredEvent]:
        """Get all events from PostgreSQL."""
        query = """
            SELECT * FROM events.event_store
            WHERE sequence_number >= $1
            ORDER BY sequence_number
            LIMIT $2
        """
        rows = await self.db.fetch(query, from_sequence, limit)
        
        return [
            StoredEvent(
                event_id=row["event_id"],
                event_type=row["event_type"],
                aggregate_id=row["aggregate_id"],
                aggregate_type=row["aggregate_type"],
                data=json.loads(row["data"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                timestamp=row["timestamp"],
                version=row["version"],
                sequence_number=row["sequence_number"],
            )
            for row in rows
        ]
    
    async def save_snapshot(self, snapshot: Snapshot):
        """Save snapshot to PostgreSQL."""
        query = """
            INSERT INTO events.snapshots
            (aggregate_id, aggregate_type, state, version, timestamp)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (aggregate_id) DO UPDATE
            SET state = $3, version = $4, timestamp = $5
        """
        await self.db.execute(
            query,
            snapshot.aggregate_id,
            snapshot.aggregate_type,
            json.dumps(snapshot.state),
            snapshot.version,
            snapshot.timestamp,
        )
    
    async def get_snapshot(self, aggregate_id: str) -> Optional[Snapshot]:
        """Get snapshot from PostgreSQL."""
        query = "SELECT * FROM events.snapshots WHERE aggregate_id = $1"
        row = await self.db.fetchrow(query, aggregate_id)
        
        if row:
            return Snapshot(
                aggregate_id=row["aggregate_id"],
                aggregate_type=row["aggregate_type"],
                state=json.loads(row["state"]),
                version=row["version"],
                timestamp=row["timestamp"],
            )
        return None
    
    def subscribe(self, callback: Callable):
        """Subscribe to new events."""
        self._subscribers.append(callback)


# =============================================================================
# Event Publisher
# =============================================================================

class EventPublisher:
    """
    Publishes events to subscribers for read model sync.
    
    Supports:
    - Multiple subscribers per event type
    - Async event delivery
    - Retry on failure
    """
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._global_subscribers: List[Callable] = []
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to specific event type."""
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed to {event_type}")
    
    def subscribe_all(self, handler: Callable):
        """Subscribe to all events."""
        self._global_subscribers.append(handler)
    
    async def publish(self, event) -> int:
        """
        Publish event to store and notify subscribers.
        
        Args:
            event: Domain event to publish
            
        Returns:
            Event sequence number
        """
        # Store event
        sequence = await self.event_store.append(event)
        
        # Create stored event
        stored_event = StoredEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            aggregate_id=event.aggregate_id,
            aggregate_type=event.aggregate_type,
            data=event.data,
            metadata=event.metadata,
            timestamp=event.timestamp,
            version=event.version,
            sequence_number=sequence,
        )
        
        # Notify type-specific subscribers
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(stored_event)
            except Exception as e:
                logger.error(f"Event handler failed for {event.event_type}: {e}")
        
        # Notify global subscribers
        for handler in self._global_subscribers:
            try:
                await handler(stored_event)
            except Exception as e:
                logger.error(f"Global event handler failed: {e}")
        
        return sequence


# =============================================================================
# Read Model Projector
# =============================================================================

class ReadModelProjector(ABC):
    """
    Base class for read model projectors.
    
    Projectors listen to events and update read models.
    """
    
    @abstractmethod
    async def project(self, event: StoredEvent):
        """Project event to read model."""
        pass
    
    @abstractmethod
    async def rebuild(self, events: List[StoredEvent]):
        """Rebuild read model from events."""
        pass
    
    @abstractmethod
    def handles_event(self, event_type: str) -> bool:
        """Check if projector handles this event type."""
        pass


class AnalysisProjector(ReadModelProjector):
    """Projector for analysis read model."""
    
    HANDLED_EVENTS = {
        "AnalysisCreated",
        "AnalysisCompleted",
        "IssueDetected",
        "FixApplied",
    }
    
    def __init__(self, read_model_store):
        self.store = read_model_store
    
    def handles_event(self, event_type: str) -> bool:
        return event_type in self.HANDLED_EVENTS
    
    async def project(self, event: StoredEvent):
        """Project analysis events to read model."""
        if event.event_type == "AnalysisCreated":
            await self.store.create_analysis({
                "id": event.aggregate_id,
                "language": event.data.get("language"),
                "project_id": event.data.get("project_id"),
                "status": "created",
                "created_at": event.timestamp.isoformat(),
            })
        
        elif event.event_type == "AnalysisCompleted":
            await self.store.update_analysis(event.aggregate_id, {
                "status": "completed",
                "issues_count": event.data.get("issues_count", 0),
                "completed_at": event.timestamp.isoformat(),
            })
        
        elif event.event_type == "IssueDetected":
            await self.store.add_issue(event.aggregate_id, event.data)
        
        elif event.event_type == "FixApplied":
            await self.store.update_issue(
                event.aggregate_id,
                event.data.get("issue_id"),
                {"status": "fixed"}
            )
    
    async def rebuild(self, events: List[StoredEvent]):
        """Rebuild analysis read model from events."""
        for event in events:
            if self.handles_event(event.event_type):
                await self.project(event)


class VersionStatusProjector(ReadModelProjector):
    """Projector for version status read model."""
    
    HANDLED_EVENTS = {
        "VersionPromoted",
        "VersionDemoted",
        "ExperimentCreated",
        "ExperimentCompleted",
        "MetricsUpdated",
    }
    
    def __init__(self, read_model_store):
        self.store = read_model_store
    
    def handles_event(self, event_type: str) -> bool:
        return event_type in self.HANDLED_EVENTS
    
    async def project(self, event: StoredEvent):
        """Project version events to read model."""
        if event.event_type == "VersionPromoted":
            await self.store.update_version_status({
                "last_promotion": event.timestamp.isoformat(),
                "promoted_experiment": event.aggregate_id,
                "v2_version": event.data.get("promotion_id"),
            })
        
        elif event.event_type == "VersionDemoted":
            await self.store.update_version_status({
                "last_demotion": event.timestamp.isoformat(),
                "demoted_version": event.aggregate_id,
            })
        
        elif event.event_type == "ExperimentCreated":
            await self.store.add_experiment({
                "id": event.aggregate_id,
                "name": event.data.get("name"),
                "status": "created",
                "zone": "v1",
            })
        
        elif event.event_type == "MetricsUpdated":
            await self.store.update_metrics(event.data)
    
    async def rebuild(self, events: List[StoredEvent]):
        """Rebuild version status read model from events."""
        for event in events:
            if self.handles_event(event.event_type):
                await self.project(event)


# =============================================================================
# Event Replayer
# =============================================================================

class EventReplayer:
    """
    Replays events to rebuild read models.
    
    Used for:
    - Read model recovery
    - New projector initialization
    - Data migration
    """
    
    def __init__(
        self,
        event_store: EventStore,
        projectors: List[ReadModelProjector]
    ):
        self.event_store = event_store
        self.projectors = projectors
    
    async def replay_all(self, batch_size: int = 1000):
        """Replay all events to all projectors."""
        sequence = 0
        total_events = 0
        
        logger.info("Starting full event replay")
        
        while True:
            events = await self.event_store.get_all_events(
                from_sequence=sequence,
                limit=batch_size
            )
            
            if not events:
                break
            
            for event in events:
                for projector in self.projectors:
                    if projector.handles_event(event.event_type):
                        try:
                            await projector.project(event)
                        except Exception as e:
                            logger.error(
                                f"Replay error for {event.event_type}: {e}"
                            )
            
            total_events += len(events)
            sequence = events[-1].sequence_number + 1
            
            logger.info(f"Replayed {total_events} events")
        
        logger.info(f"Event replay complete. Total: {total_events} events")
    
    async def replay_aggregate(self, aggregate_id: str):
        """Replay events for a specific aggregate."""
        events = await self.event_store.get_events(aggregate_id)
        
        for event in events:
            for projector in self.projectors:
                if projector.handles_event(event.event_type):
                    await projector.project(event)
        
        logger.info(f"Replayed {len(events)} events for aggregate {aggregate_id}")


# =============================================================================
# Database Migration SQL
# =============================================================================

EVENT_STORE_MIGRATION = """
-- Event store schema
CREATE SCHEMA IF NOT EXISTS events;

-- Event store table
CREATE TABLE IF NOT EXISTS events.event_store (
    sequence_number BIGSERIAL PRIMARY KEY,
    event_id VARCHAR(64) NOT NULL UNIQUE,
    event_type VARCHAR(100) NOT NULL,
    aggregate_id VARCHAR(64) NOT NULL,
    aggregate_type VARCHAR(100) NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    version INTEGER NOT NULL DEFAULT 1,
    CONSTRAINT unique_aggregate_version UNIQUE (aggregate_id, version)
);

-- Snapshots table
CREATE TABLE IF NOT EXISTS events.snapshots (
    aggregate_id VARCHAR(64) PRIMARY KEY,
    aggregate_type VARCHAR(100) NOT NULL,
    state JSONB NOT NULL,
    version INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_event_store_aggregate 
    ON events.event_store(aggregate_id, version);
CREATE INDEX IF NOT EXISTS idx_event_store_type 
    ON events.event_store(event_type);
CREATE INDEX IF NOT EXISTS idx_event_store_timestamp 
    ON events.event_store(timestamp);
CREATE INDEX IF NOT EXISTS idx_event_store_sequence 
    ON events.event_store(sequence_number);
"""
