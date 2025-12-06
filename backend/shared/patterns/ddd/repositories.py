"""
Repository Pattern Implementation

Provides persistence abstraction for aggregates.

Features:
- Generic repository interface
- Unit of Work pattern
- Optimistic concurrency control
- Specification pattern for queries
"""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar
from uuid import uuid4

from .domain_models import (
    AggregateRoot, Analysis, Experiment, DomainEvent,
    EntityNotFoundError
)
from .aggregates import AnalysisFactory, ExperimentFactory

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=AggregateRoot)


# =============================================================================
# Specification Pattern
# =============================================================================

class Specification(ABC, Generic[T]):
    """Base class for specifications."""
    
    @abstractmethod
    def is_satisfied_by(self, entity: T) -> bool:
        """Check if entity satisfies the specification."""
        pass
    
    def and_(self, other: 'Specification[T]') -> 'Specification[T]':
        """AND combination of specifications."""
        return AndSpecification(self, other)
    
    def or_(self, other: 'Specification[T]') -> 'Specification[T]':
        """OR combination of specifications."""
        return OrSpecification(self, other)
    
    def not_(self) -> 'Specification[T]':
        """NOT of specification."""
        return NotSpecification(self)


class AndSpecification(Specification[T]):
    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, entity: T) -> bool:
        return self.left.is_satisfied_by(entity) and self.right.is_satisfied_by(entity)


class OrSpecification(Specification[T]):
    def __init__(self, left: Specification[T], right: Specification[T]):
        self.left = left
        self.right = right
    
    def is_satisfied_by(self, entity: T) -> bool:
        return self.left.is_satisfied_by(entity) or self.right.is_satisfied_by(entity)


class NotSpecification(Specification[T]):
    def __init__(self, spec: Specification[T]):
        self.spec = spec
    
    def is_satisfied_by(self, entity: T) -> bool:
        return not self.spec.is_satisfied_by(entity)


# Analysis Specifications
class AnalysisByProjectSpec(Specification[Analysis]):
    def __init__(self, project_id: str):
        self.project_id = project_id
    
    def is_satisfied_by(self, analysis: Analysis) -> bool:
        return analysis.project_id == self.project_id


class AnalysisByStatusSpec(Specification[Analysis]):
    def __init__(self, status: str):
        self.status = status
    
    def is_satisfied_by(self, analysis: Analysis) -> bool:
        return analysis.status.value == self.status


class AnalysisWithCriticalIssuesSpec(Specification[Analysis]):
    def is_satisfied_by(self, analysis: Analysis) -> bool:
        return any(i.severity.value == "critical" for i in analysis.issues)


# Experiment Specifications
class ExperimentByStatusSpec(Specification[Experiment]):
    def __init__(self, status: str):
        self.status = status
    
    def is_satisfied_by(self, experiment: Experiment) -> bool:
        return experiment.status.value == self.status


class ExperimentReadyForPromotionSpec(Specification[Experiment]):
    def __init__(self, min_accuracy: float = 0.85):
        self.min_accuracy = min_accuracy
    
    def is_satisfied_by(self, experiment: Experiment) -> bool:
        if experiment.status.value != "completed":
            return False
        
        latest = experiment.latest_evaluation
        if not latest:
            return False
        
        return latest.metrics.meets_promotion_criteria(min_accuracy=self.min_accuracy)


# =============================================================================
# Repository Interface
# =============================================================================

class Repository(ABC, Generic[T]):
    """
    Base repository interface.
    
    Provides CRUD operations and query capabilities for aggregates.
    """
    
    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        """Get aggregate by ID."""
        pass
    
    @abstractmethod
    async def save(self, aggregate: T) -> None:
        """Save (insert or update) aggregate."""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> None:
        """Delete aggregate by ID."""
        pass
    
    @abstractmethod
    async def find(self, spec: Specification[T]) -> List[T]:
        """Find aggregates matching specification."""
        pass
    
    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all aggregates with pagination."""
        pass
    
    @abstractmethod
    async def count(self, spec: Optional[Specification[T]] = None) -> int:
        """Count aggregates matching optional specification."""
        pass


# =============================================================================
# In-Memory Repository (for testing)
# =============================================================================

class InMemoryRepository(Repository[T]):
    """In-memory repository implementation for testing."""
    
    def __init__(self):
        self._store: Dict[str, T] = {}
        self._event_handlers: List[Callable[[DomainEvent], None]] = []
    
    async def get(self, id: str) -> Optional[T]:
        return self._store.get(id)
    
    async def save(self, aggregate: T) -> None:
        self._store[aggregate.id] = aggregate
        
        # Dispatch domain events
        events = aggregate.clear_domain_events()
        for event in events:
            for handler in self._event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
    
    async def delete(self, id: str) -> None:
        if id in self._store:
            del self._store[id]
    
    async def find(self, spec: Specification[T]) -> List[T]:
        return [
            entity for entity in self._store.values()
            if spec.is_satisfied_by(entity)
        ]
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        items = list(self._store.values())
        return items[offset:offset + limit]
    
    async def count(self, spec: Optional[Specification[T]] = None) -> int:
        if spec is None:
            return len(self._store)
        return len(await self.find(spec))
    
    def register_event_handler(self, handler: Callable[[DomainEvent], None]):
        """Register a handler for domain events."""
        self._event_handlers.append(handler)


# =============================================================================
# PostgreSQL Repository Implementation
# =============================================================================

class PostgresAnalysisRepository(Repository[Analysis]):
    """PostgreSQL repository for Analysis aggregates."""
    
    def __init__(self, db_pool):
        self.db = db_pool
    
    async def get(self, id: str) -> Optional[Analysis]:
        query = """
            SELECT a.*, 
                   COALESCE(json_agg(i.*) FILTER (WHERE i.id IS NOT NULL), '[]') as issues
            FROM production.analyses a
            LEFT JOIN production.issues i ON i.analysis_id = a.id
            WHERE a.id = $1
            GROUP BY a.id
        """
        
        row = await self.db.fetchrow(query, id)
        if not row:
            return None
        
        return AnalysisFactory.reconstitute(
            analysis_id=row["id"],
            project_id=row["project_id"],
            code_hash=row["code_hash"],
            language=row["language"],
            status=row["status"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            issues=json.loads(row["issues"]) if row["issues"] else [],
            version=row["version"],
        )
    
    async def save(self, aggregate: Analysis) -> None:
        # Upsert analysis
        analysis_query = """
            INSERT INTO production.analyses 
            (id, project_id, code_hash, language, status, created_at, completed_at, version)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                status = $5,
                completed_at = $7,
                version = $8
            WHERE production.analyses.version < $8
        """
        
        await self.db.execute(
            analysis_query,
            aggregate.id,
            aggregate.project_id,
            aggregate.code_hash.value,
            aggregate.language,
            aggregate.status.value,
            aggregate.created_at,
            aggregate.completed_at,
            aggregate.version,
        )
        
        # Upsert issues
        for issue in aggregate.issues:
            issue_query = """
                INSERT INTO production.issues
                (id, analysis_id, type, severity, message, file_path, 
                 line_start, line_end, rule_id, is_resolved)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    is_resolved = $10
            """
            
            await self.db.execute(
                issue_query,
                issue.id,
                aggregate.id,
                issue.issue_type.value,
                issue.severity.value,
                issue.message,
                issue.location.file_path,
                issue.location.line_start,
                issue.location.line_end,
                issue.rule_id,
                issue.is_resolved,
            )
    
    async def delete(self, id: str) -> None:
        # Delete issues first (foreign key)
        await self.db.execute(
            "DELETE FROM production.issues WHERE analysis_id = $1", id
        )
        await self.db.execute(
            "DELETE FROM production.analyses WHERE id = $1", id
        )
    
    async def find(self, spec: Specification[Analysis]) -> List[Analysis]:
        # For complex specifications, load all and filter in memory
        # In production, translate specification to SQL
        all_analyses = await self.find_all(limit=10000)
        return [a for a in all_analyses if spec.is_satisfied_by(a)]
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Analysis]:
        query = """
            SELECT id FROM production.analyses
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """
        
        rows = await self.db.fetch(query, limit, offset)
        analyses = []
        
        for row in rows:
            analysis = await self.get(row["id"])
            if analysis:
                analyses.append(analysis)
        
        return analyses
    
    async def count(self, spec: Optional[Specification[Analysis]] = None) -> int:
        if spec is None:
            row = await self.db.fetchrow(
                "SELECT COUNT(*) as count FROM production.analyses"
            )
            return row["count"]
        
        # For complex specs, count filtered results
        matching = await self.find(spec)
        return len(matching)


# =============================================================================
# Unit of Work Pattern
# =============================================================================

class UnitOfWork:
    """
    Unit of Work pattern for managing transactions.
    
    Ensures all changes are committed together or rolled back.
    
    Usage:
        async with UnitOfWork(db) as uow:
            analysis = await uow.analyses.get(id)
            analysis.add_issue(issue)
            await uow.analyses.save(analysis)
            await uow.commit()
    """
    
    def __init__(self, db_pool):
        self.db = db_pool
        self._connection = None
        self._transaction = None
        
        # Repositories (initialized with transaction connection)
        self.analyses: Optional[Repository[Analysis]] = None
        self.experiments: Optional[Repository[Experiment]] = None
    
    async def __aenter__(self) -> 'UnitOfWork':
        self._connection = await self.db.acquire()
        self._transaction = self._connection.transaction()
        await self._transaction.start()
        
        # Initialize repositories with transaction connection
        # (In production, these would use the transaction connection)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.rollback()
        
        if self._connection:
            await self.db.release(self._connection)
    
    async def commit(self):
        """Commit the transaction."""
        if self._transaction:
            await self._transaction.commit()
            logger.debug("Transaction committed")
    
    async def rollback(self):
        """Rollback the transaction."""
        if self._transaction:
            await self._transaction.rollback()
            logger.debug("Transaction rolled back")


# =============================================================================
# Domain Event Dispatcher
# =============================================================================

class DomainEventDispatcher:
    """
    Dispatches domain events to registered handlers.
    
    Used after repository saves to publish events.
    """
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
    
    def register(self, event_type: str, handler: Callable):
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def dispatch(self, event: DomainEvent):
        """Dispatch event to all registered handlers."""
        event_type = event.event_type
        handlers = self._handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if hasattr(handler, '__call__'):
                    import asyncio
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")
    
    async def dispatch_all(self, events: List[DomainEvent]):
        """Dispatch multiple events."""
        for event in events:
            await self.dispatch(event)


# Global event dispatcher
_event_dispatcher: Optional[DomainEventDispatcher] = None


def get_event_dispatcher() -> DomainEventDispatcher:
    """Get or create the global event dispatcher."""
    global _event_dispatcher
    if _event_dispatcher is None:
        _event_dispatcher = DomainEventDispatcher()
    return _event_dispatcher
