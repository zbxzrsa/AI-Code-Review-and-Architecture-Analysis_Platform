"""
CQRS Read Models

Optimized read-only data copies for high-frequency query operations.

Features:
- Denormalized data for fast reads
- Multiple view optimizations
- <200ms query response time
- Eventually consistent with write model
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ReadModelStore(ABC):
    """Abstract base class for read model stores."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get entity by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, data: Dict[str, Any]):
        """Set entity data."""
        pass
    
    @abstractmethod
    async def delete(self, key: str):
        """Delete entity."""
        pass
    
    @abstractmethod
    async def query(
        self,
        filters: Dict[str, Any],
        offset: int = 0,
        limit: int = 50,
        sort_field: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Query with filters and pagination."""
        pass


class InMemoryReadModelStore(ReadModelStore):
    """In-memory read model store for development/testing."""
    
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._indexes: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._data.get(key)
    
    async def set(self, key: str, data: Dict[str, Any]):
        async with self._lock:
            self._data[key] = data
            
            # Update indexes
            for field, value in data.items():
                if isinstance(value, (str, int, bool)):
                    self._indexes[field][str(value)].append(key)
    
    async def delete(self, key: str):
        async with self._lock:
            if key in self._data:
                del self._data[key]
    
    async def query(
        self,
        filters: Dict[str, Any],
        offset: int = 0,
        limit: int = 50,
        sort_field: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[Dict[str, Any]], int]:
        async with self._lock:
            # Filter
            results = []
            for key, data in self._data.items():
                match = True
                for field, value in filters.items():
                    if field.endswith("_gte"):
                        actual_field = field[:-4]
                        if data.get(actual_field, "") < value:
                            match = False
                            break
                    elif field.endswith("_lte"):
                        actual_field = field[:-4]
                        if data.get(actual_field, "") > value:
                            match = False
                            break
                    elif data.get(field) != value:
                        match = False
                        break
                
                if match:
                    results.append(data)
            
            # Sort
            reverse = sort_order == "desc"
            results.sort(key=lambda x: x.get(sort_field, ""), reverse=reverse)
            
            total = len(results)
            
            # Paginate
            results = results[offset:offset + limit]
            
            return results, total


class PostgresReadModelStore(ReadModelStore):
    """PostgreSQL-based read model store for production."""
    
    def __init__(self, db_pool, table_name: str):
        self.db = db_pool
        self.table_name = table_name
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        query = f"SELECT data FROM {self.table_name} WHERE id = $1"
        row = await self.db.fetchrow(query, key)
        if row:
            return json.loads(row["data"])
        return None
    
    async def set(self, key: str, data: Dict[str, Any]):
        query = f"""
            INSERT INTO {self.table_name} (id, data, updated_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (id) DO UPDATE
            SET data = $2, updated_at = NOW()
        """
        await self.db.execute(query, key, json.dumps(data))
    
    async def delete(self, key: str):
        query = f"DELETE FROM {self.table_name} WHERE id = $1"
        await self.db.execute(query, key)
    
    async def query(
        self,
        filters: Dict[str, Any],
        offset: int = 0,
        limit: int = 50,
        sort_field: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[Dict[str, Any]], int]:
        # Build WHERE clause
        conditions = []
        params = []
        param_idx = 1
        
        for field, value in filters.items():
            if field.endswith("_gte"):
                actual_field = field[:-4]
                conditions.append(f"data->>'{actual_field}' >= ${param_idx}")
            elif field.endswith("_lte"):
                actual_field = field[:-4]
                conditions.append(f"data->>'{actual_field}' <= ${param_idx}")
            else:
                conditions.append(f"data->>'{field}' = ${param_idx}")
            params.append(str(value))
            param_idx += 1
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Count query
        count_query = f"SELECT COUNT(*) FROM {self.table_name} WHERE {where_clause}"
        total = await self.db.fetchval(count_query, *params)
        
        # Data query
        data_query = f"""
            SELECT data FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY data->>'{sort_field}' {sort_order.upper()}
            OFFSET ${param_idx} LIMIT ${param_idx + 1}
        """
        params.extend([offset, limit])
        
        rows = await self.db.fetch(data_query, *params)
        results = [json.loads(row["data"]) for row in rows]
        
        return results, total


# =============================================================================
# Concrete Read Models
# =============================================================================

class AnalysisReadModel:
    """
    Denormalized read model for code analyses.
    
    Optimized for:
    - Single analysis lookup by ID
    - List analyses with filtering
    - Analysis statistics
    """
    
    def __init__(self, store: ReadModelStore):
        self.store = store
    
    async def get_analysis(
        self,
        analysis_id: str,
        include_issues: bool = True,
        include_metrics: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get analysis by ID."""
        data = await self.store.get(f"analysis:{analysis_id}")
        
        if data and not include_issues:
            data.pop("issues", None)
        
        if data and not include_metrics:
            data.pop("metrics", None)
        
        return data
    
    async def list_analyses(
        self,
        filters: Dict[str, Any] = None,
        offset: int = 0,
        limit: int = 50,
        sort_field: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List analyses with filtering."""
        return await self.store.query(
            filters=filters or {},
            offset=offset,
            limit=limit,
            sort_field=sort_field,
            sort_order=sort_order
        )
    
    async def create_analysis(self, data: Dict[str, Any]):
        """Create new analysis record."""
        analysis_id = data.get("id")
        data["issues"] = []
        data["metrics"] = {}
        await self.store.set(f"analysis:{analysis_id}", data)
    
    async def update_analysis(self, analysis_id: str, updates: Dict[str, Any]):
        """Update analysis record."""
        existing = await self.store.get(f"analysis:{analysis_id}")
        if existing:
            existing.update(updates)
            await self.store.set(f"analysis:{analysis_id}", existing)
    
    async def add_issue(self, analysis_id: str, issue: Dict[str, Any]):
        """Add issue to analysis."""
        existing = await self.store.get(f"analysis:{analysis_id}")
        if existing:
            existing.setdefault("issues", []).append(issue)
            existing.setdefault("metrics", {})["issues_count"] = len(existing["issues"])
            await self.store.set(f"analysis:{analysis_id}", existing)
    
    async def update_issue(self, analysis_id: str, issue_id: str, updates: Dict[str, Any]):
        """Update specific issue."""
        existing = await self.store.get(f"analysis:{analysis_id}")
        if existing:
            for issue in existing.get("issues", []):
                if issue.get("id") == issue_id:
                    issue.update(updates)
                    break
            await self.store.set(f"analysis:{analysis_id}", existing)
    
    async def get_statistics(
        self,
        project_id: Optional[str] = None,
        time_range: str = "24h"
    ) -> Dict[str, Any]:
        """Get analysis statistics."""
        filters = {}
        if project_id:
            filters["project_id"] = project_id
        
        analyses, total = await self.store.query(filters, limit=10000)
        
        total_issues = sum(len(a.get("issues", [])) for a in analyses)
        critical_issues = sum(
            len([i for i in a.get("issues", []) if i.get("severity") == "critical"])
            for a in analyses
        )
        
        return {
            "total_analyses": total,
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "avg_issues_per_analysis": total_issues / max(total, 1),
        }


class VersionStatusReadModel:
    """
    Denormalized read model for three-version status.
    
    Optimized for:
    - Quick status overview
    - Experiment listings by zone
    - Metrics aggregation
    """
    
    def __init__(self, store: ReadModelStore):
        self.store = store
    
    async def get_version_status(
        self,
        include_metrics: bool = True,
        include_experiments: bool = False
    ) -> Dict[str, Any]:
        """Get current version status."""
        status = await self.store.get("version_status:current") or {}
        
        # Add zone-specific data
        v1_data = await self.store.get("version_status:v1") or {}
        v2_data = await self.store.get("version_status:v2") or {}
        v3_data = await self.store.get("version_status:v3") or {}
        
        status["v1"] = v1_data
        status["v2"] = v2_data
        status["v3"] = v3_data
        
        if include_experiments:
            experiments, _ = await self.store.query({"type": "experiment"}, limit=100)
            status["experiments"] = experiments
        
        if not include_metrics:
            for zone in ["v1", "v2", "v3"]:
                if zone in status:
                    status[zone].pop("metrics", None)
        
        return status
    
    async def update_version_status(self, updates: Dict[str, Any]):
        """Update version status."""
        existing = await self.store.get("version_status:current") or {}
        existing.update(updates)
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        await self.store.set("version_status:current", existing)
    
    async def add_experiment(self, experiment: Dict[str, Any]):
        """Add experiment to read model."""
        experiment["type"] = "experiment"
        await self.store.set(f"experiment:{experiment['id']}", experiment)
        
        # Update zone count
        zone = experiment.get("zone", "v1")
        zone_data = await self.store.get(f"version_status:{zone}") or {}
        zone_data["experiment_count"] = zone_data.get("experiment_count", 0) + 1
        await self.store.set(f"version_status:{zone}", zone_data)
    
    async def update_experiment(self, experiment_id: str, updates: Dict[str, Any]):
        """Update experiment."""
        existing = await self.store.get(f"experiment:{experiment_id}")
        if existing:
            old_zone = existing.get("zone")
            existing.update(updates)
            new_zone = existing.get("zone")
            
            # Handle zone change
            if old_zone != new_zone:
                # Decrement old zone count
                old_zone_data = await self.store.get(f"version_status:{old_zone}") or {}
                old_zone_data["experiment_count"] = max(
                    old_zone_data.get("experiment_count", 1) - 1, 0
                )
                await self.store.set(f"version_status:{old_zone}", old_zone_data)
                
                # Increment new zone count
                new_zone_data = await self.store.get(f"version_status:{new_zone}") or {}
                new_zone_data["experiment_count"] = new_zone_data.get("experiment_count", 0) + 1
                await self.store.set(f"version_status:{new_zone}", new_zone_data)
            
            await self.store.set(f"experiment:{experiment_id}", existing)
    
    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics for a zone."""
        zone = metrics.pop("zone", "v2")
        zone_data = await self.store.get(f"version_status:{zone}") or {}
        zone_data.setdefault("metrics", {}).update(metrics)
        zone_data["metrics_updated_at"] = datetime.now(timezone.utc).isoformat()
        await self.store.set(f"version_status:{zone}", zone_data)
    
    async def get_experiments(
        self,
        zone: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get experiments with filtering."""
        filters = {"type": "experiment"}
        if zone:
            filters["zone"] = zone
        if status:
            filters["status"] = status
        
        experiments, _ = await self.store.query(filters, limit=limit)
        return experiments


class AuditLogReadModel:
    """
    Read model for audit logs.
    
    Optimized for:
    - Time-range queries
    - Entity/action filtering
    - Actor history
    """
    
    def __init__(self, store: ReadModelStore):
        self.store = store
    
    async def get_logs(
        self,
        entity: Optional[str] = None,
        action: Optional[str] = None,
        actor_id: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        offset: int = 0,
        limit: int = 100
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get audit logs with filtering."""
        filters = {"type": "audit_log"}
        
        if entity:
            filters["entity"] = entity
        if action:
            filters["action"] = action
        if actor_id:
            filters["actor_id"] = actor_id
        if from_date:
            filters["timestamp_gte"] = from_date.isoformat()
        if to_date:
            filters["timestamp_lte"] = to_date.isoformat()
        
        return await self.store.query(
            filters,
            offset=offset,
            limit=limit,
            sort_field="timestamp",
            sort_order="desc"
        )
    
    async def add_log(self, log: Dict[str, Any]):
        """Add audit log entry."""
        log["type"] = "audit_log"
        log_id = log.get("id") or str(__import__('uuid').uuid4())
        await self.store.set(f"audit:{log_id}", log)
    
    async def get_actor_history(
        self,
        actor_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get activity history for an actor."""
        logs, _ = await self.store.query(
            {"type": "audit_log", "actor_id": actor_id},
            limit=limit,
            sort_field="timestamp",
            sort_order="desc"
        )
        return logs


# =============================================================================
# Read Model Synchronizer
# =============================================================================

class ReadModelSynchronizer:
    """
    Synchronizes read models with event store.
    
    Features:
    - Async event processing
    - Batch updates
    - Error recovery
    - Lag monitoring
    """
    
    def __init__(
        self,
        event_store,
        read_models: List,
        batch_size: int = 100,
        sync_interval_ms: int = 100
    ):
        self.event_store = event_store
        self.read_models = read_models
        self.batch_size = batch_size
        self.sync_interval_ms = sync_interval_ms
        
        self._last_sequence = 0
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._events_processed = 0
        self._sync_lag_ms = 0
    
    async def start(self):
        """Start synchronization."""
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Read model synchronizer started")
    
    async def stop(self):
        """Stop synchronization."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                logger.debug("Sync task cancelled")
        logger.info("Read model synchronizer stopped")
    
    async def _sync_loop(self):
        """Main synchronization loop."""
        while self._running:
            try:
                start_time = datetime.now(timezone.utc)
                
                # Get new events
                events = await self.event_store.get_all_events(
                    from_sequence=self._last_sequence + 1,
                    limit=self.batch_size
                )
                
                if events:
                    # Process events
                    for event in events:
                        for read_model in self.read_models:
                            if hasattr(read_model, 'handles_event'):
                                if read_model.handles_event(event.event_type):
                                    await read_model.project(event)
                        
                        self._last_sequence = event.sequence_number
                        self._events_processed += 1
                    
                    # Calculate lag
                    if events:
                        latest_event_time = events[-1].timestamp
                        self._sync_lag_ms = (
                            datetime.now(timezone.utc) - latest_event_time
                        ).total_seconds() * 1000
                
                await asyncio.sleep(self.sync_interval_ms / 1000)
                
            except asyncio.CancelledError:
                logger.debug("Sync loop cancelled")
                raise
            except Exception as e:
                logger.error(f"Sync error: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get synchronizer metrics."""
        return {
            "events_processed": self._events_processed,
            "last_sequence": self._last_sequence,
            "sync_lag_ms": self._sync_lag_ms,
            "running": self._running,
        }


# =============================================================================
# Read Model Database Migration
# =============================================================================

READ_MODEL_MIGRATION = """
-- Read model schema
CREATE SCHEMA IF NOT EXISTS read_models;

-- Analysis read model table
CREATE TABLE IF NOT EXISTS read_models.analyses (
    id VARCHAR(64) PRIMARY KEY,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Version status read model table
CREATE TABLE IF NOT EXISTS read_models.version_status (
    id VARCHAR(64) PRIMARY KEY,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Experiments read model table
CREATE TABLE IF NOT EXISTS read_models.experiments (
    id VARCHAR(64) PRIMARY KEY,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Audit logs read model table
CREATE TABLE IF NOT EXISTS read_models.audit_logs (
    id VARCHAR(64) PRIMARY KEY,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_analyses_project 
    ON read_models.analyses((data->>'project_id'));
CREATE INDEX IF NOT EXISTS idx_analyses_status 
    ON read_models.analyses((data->>'status'));
CREATE INDEX IF NOT EXISTS idx_analyses_created 
    ON read_models.analyses((data->>'created_at'));

CREATE INDEX IF NOT EXISTS idx_experiments_zone 
    ON read_models.experiments((data->>'zone'));
CREATE INDEX IF NOT EXISTS idx_experiments_status 
    ON read_models.experiments((data->>'status'));

CREATE INDEX IF NOT EXISTS idx_audit_entity 
    ON read_models.audit_logs((data->>'entity'));
CREATE INDEX IF NOT EXISTS idx_audit_actor 
    ON read_models.audit_logs((data->>'actor_id'));
CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
    ON read_models.audit_logs((data->>'timestamp'));

-- Sync tracking table
CREATE TABLE IF NOT EXISTS read_models.sync_state (
    id VARCHAR(64) PRIMARY KEY DEFAULT 'default',
    last_sequence BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""
