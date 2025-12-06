"""
CQRS Query Layer

Handles all read operations with optimized read models.

Features:
- Query definitions with filtering/pagination
- Query handlers with caching
- Query bus for routing
- Read model access with <200ms response time target
"""
import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class SortOrder(str, Enum):
    """Sort order for queries."""
    ASC = "asc"
    DESC = "desc"


@dataclass
class QueryMetadata:
    """Metadata for query tracking."""
    query_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: Optional[str] = None
    cache_key: Optional[str] = None
    cache_ttl: int = 300  # 5 minutes default


@dataclass
class PaginationParams:
    """Pagination parameters."""
    page: int = 1
    page_size: int = 50
    offset: int = 0
    
    def __post_init__(self):
        if self.offset == 0:
            self.offset = (self.page - 1) * self.page_size


@dataclass
class SortParams:
    """Sorting parameters."""
    field: str = "created_at"
    order: SortOrder = SortOrder.DESC


@dataclass
class Query(ABC):
    """Base class for all queries."""
    metadata: QueryMetadata = field(default_factory=QueryMetadata)
    
    @property
    def query_type(self) -> str:
        return self.__class__.__name__
    
    def get_cache_key(self) -> str:
        """Generate cache key for this query."""
        query_data = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(query_data.encode()).hexdigest()[:32]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "query_id": self.metadata.query_id,
        }


@dataclass
class QueryResult(Generic[T]):
    """Result of query execution."""
    success: bool
    query_id: str
    data: Optional[T] = None
    error: Optional[str] = None
    total_count: int = 0
    page: int = 1
    page_size: int = 50
    execution_time_ms: float = 0
    from_cache: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "query_id": self.query_id,
            "data": self.data,
            "error": self.error,
            "total_count": self.total_count,
            "page": self.page,
            "page_size": self.page_size,
            "execution_time_ms": self.execution_time_ms,
            "from_cache": self.from_cache,
        }


class QueryHandler(ABC, Generic[T, R]):
    """Base class for query handlers."""
    
    @abstractmethod
    async def handle(self, query: T) -> QueryResult[R]:
        """Handle the query and return result."""
        pass
    
    @abstractmethod
    def can_handle(self, query: Query) -> bool:
        """Check if this handler can handle the given query."""
        pass


# =============================================================================
# Concrete Query Definitions
# =============================================================================

@dataclass
class GetAnalysisQuery(Query):
    """Query to get a single analysis by ID."""
    analysis_id: str = ""
    include_issues: bool = True
    include_metrics: bool = True


@dataclass
class ListAnalysesQuery(Query):
    """Query to list analyses with filtering."""
    project_id: Optional[str] = None
    language: Optional[str] = None
    status: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    pagination: PaginationParams = field(default_factory=PaginationParams)
    sort: SortParams = field(default_factory=SortParams)


@dataclass
class GetVersionStatusQuery(Query):
    """Query to get three-version system status."""
    include_metrics: bool = True
    include_experiments: bool = False


@dataclass
class GetExperimentQuery(Query):
    """Query to get experiment details."""
    experiment_id: str = ""
    include_evaluations: bool = True


@dataclass
class ListExperimentsQuery(Query):
    """Query to list experiments with filtering."""
    zone: Optional[str] = None  # v1, v2, v3
    status: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    pagination: PaginationParams = field(default_factory=PaginationParams)
    sort: SortParams = field(default_factory=SortParams)


@dataclass
class GetAuditLogsQuery(Query):
    """Query to get audit logs."""
    entity: Optional[str] = None
    action: Optional[str] = None
    actor_id: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    pagination: PaginationParams = field(default_factory=PaginationParams)
    sort: SortParams = field(default_factory=SortParams)


@dataclass
class GetMetricsQuery(Query):
    """Query to get system metrics."""
    metric_type: str = "all"  # all, performance, accuracy, cost
    time_range: str = "24h"  # 1h, 24h, 7d, 30d
    aggregation: str = "avg"  # avg, sum, min, max


@dataclass
class SearchCodeQuery(Query):
    """Query to search code patterns."""
    pattern: str = ""
    language: Optional[str] = None
    project_id: Optional[str] = None
    pagination: PaginationParams = field(default_factory=PaginationParams)


# =============================================================================
# Query Handlers
# =============================================================================

class GetAnalysisHandler(QueryHandler[GetAnalysisQuery, Dict]):
    """Handler for GetAnalysisQuery."""
    
    def __init__(self, read_model, cache=None):
        self.read_model = read_model
        self.cache = cache
    
    def can_handle(self, query: Query) -> bool:
        return isinstance(query, GetAnalysisQuery)
    
    async def handle(self, query: GetAnalysisQuery) -> QueryResult[Dict]:
        start_time = time.perf_counter()
        
        try:
            # Check cache
            if self.cache:
                cache_key = f"analysis:{query.analysis_id}"
                cached = await self.cache.get(cache_key)
                if cached:
                    return QueryResult(
                        success=True,
                        query_id=query.metadata.query_id,
                        data=json.loads(cached),
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        from_cache=True
                    )
            
            # Query read model
            data = await self.read_model.get_analysis(
                query.analysis_id,
                include_issues=query.include_issues,
                include_metrics=query.include_metrics
            )
            
            # Cache result
            if self.cache and data:
                await self.cache.set(
                    f"analysis:{query.analysis_id}",
                    json.dumps(data),
                    ttl=query.metadata.cache_ttl
                )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return QueryResult(
                success=True,
                query_id=query.metadata.query_id,
                data=data,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"GetAnalysis query failed: {e}")
            return QueryResult(
                success=False,
                query_id=query.metadata.query_id,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )


class ListAnalysesHandler(QueryHandler[ListAnalysesQuery, List[Dict]]):
    """Handler for ListAnalysesQuery."""
    
    def __init__(self, read_model, cache=None):
        self.read_model = read_model
        self.cache = cache
    
    def can_handle(self, query: Query) -> bool:
        return isinstance(query, ListAnalysesQuery)
    
    async def handle(self, query: ListAnalysesQuery) -> QueryResult[List[Dict]]:
        start_time = time.perf_counter()
        
        try:
            # Build filter
            filters = {}
            if query.project_id:
                filters["project_id"] = query.project_id
            if query.language:
                filters["language"] = query.language
            if query.status:
                filters["status"] = query.status
            if query.from_date:
                filters["from_date"] = query.from_date
            if query.to_date:
                filters["to_date"] = query.to_date
            
            # Query read model
            data, total = await self.read_model.list_analyses(
                filters=filters,
                offset=query.pagination.offset,
                limit=query.pagination.page_size,
                sort_field=query.sort.field,
                sort_order=query.sort.order.value
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return QueryResult(
                success=True,
                query_id=query.metadata.query_id,
                data=data,
                total_count=total,
                page=query.pagination.page,
                page_size=query.pagination.page_size,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"ListAnalyses query failed: {e}")
            return QueryResult(
                success=False,
                query_id=query.metadata.query_id,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )


class GetVersionStatusHandler(QueryHandler[GetVersionStatusQuery, Dict]):
    """Handler for GetVersionStatusQuery."""
    
    def __init__(self, read_model, cache=None):
        self.read_model = read_model
        self.cache = cache
    
    def can_handle(self, query: Query) -> bool:
        return isinstance(query, GetVersionStatusQuery)
    
    async def handle(self, query: GetVersionStatusQuery) -> QueryResult[Dict]:
        start_time = time.perf_counter()
        
        try:
            # Check cache
            if self.cache:
                cached = await self.cache.get("version_status")
                if cached:
                    return QueryResult(
                        success=True,
                        query_id=query.metadata.query_id,
                        data=json.loads(cached),
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        from_cache=True
                    )
            
            # Query read model
            data = await self.read_model.get_version_status(
                include_metrics=query.include_metrics,
                include_experiments=query.include_experiments
            )
            
            # Cache for shorter time (30s) as status changes frequently
            if self.cache and data:
                await self.cache.set("version_status", json.dumps(data), ttl=30)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return QueryResult(
                success=True,
                query_id=query.metadata.query_id,
                data=data,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"GetVersionStatus query failed: {e}")
            return QueryResult(
                success=False,
                query_id=query.metadata.query_id,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )


# =============================================================================
# Query Bus
# =============================================================================

class QueryBus:
    """
    Query bus for routing queries to handlers.
    
    Features:
    - Handler registration
    - Caching middleware
    - Performance monitoring
    - <200ms response time target
    """
    
    # Performance threshold in ms
    RESPONSE_TIME_TARGET_MS = 200
    
    def __init__(self, cache=None):
        self._handlers: Dict[Type[Query], QueryHandler] = {}
        self._cache = cache
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_queries = 0
        self._cache_hits = 0
        self._slow_queries = 0
        self._total_execution_time_ms = 0
    
    def register_handler(self, query_type: Type[Query], handler: QueryHandler):
        """Register a handler for a query type."""
        self._handlers[query_type] = handler
        logger.info(f"Registered query handler for {query_type.__name__}")
    
    async def execute(self, query: Query) -> QueryResult:
        """
        Execute a query through its handler.
        
        Args:
            query: Query to execute
            
        Returns:
            QueryResult with data or error
        """
        async with self._lock:
            self._total_queries += 1
        
        query_type = type(query)
        start_time = time.perf_counter()
        
        # Find handler
        handler = self._handlers.get(query_type)
        if not handler:
            logger.error(f"No handler registered for {query_type.__name__}")
            return QueryResult(
                success=False,
                query_id=query.metadata.query_id,
                error=f"No handler for {query_type.__name__}"
            )
        
        try:
            result = await handler.handle(query)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            result.execution_time_ms = execution_time
            
            async with self._lock:
                self._total_execution_time_ms += execution_time
                
                if result.from_cache:
                    self._cache_hits += 1
                
                if execution_time > self.RESPONSE_TIME_TARGET_MS:
                    self._slow_queries += 1
                    logger.warning(
                        f"Slow query detected: {query_type.__name__} "
                        f"took {execution_time:.2f}ms (target: {self.RESPONSE_TIME_TARGET_MS}ms)"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return QueryResult(
                success=False,
                query_id=query.metadata.query_id,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def execute_batch(self, queries: List[Query]) -> List[QueryResult]:
        """Execute multiple queries in parallel."""
        tasks = [self.execute(query) for query in queries]
        return await asyncio.gather(*tasks)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get query bus metrics."""
        avg_execution_time = (
            self._total_execution_time_ms / max(self._total_queries, 1)
        )
        
        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(self._total_queries, 1),
            "slow_queries": self._slow_queries,
            "slow_query_rate": self._slow_queries / max(self._total_queries, 1),
            "avg_execution_time_ms": avg_execution_time,
            "target_response_time_ms": self.RESPONSE_TIME_TARGET_MS,
            "registered_handlers": len(self._handlers),
        }


# =============================================================================
# Query Cache
# =============================================================================

class QueryCache:
    """
    In-memory query cache with TTL support.
    
    For production, integrate with Redis.
    """
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        self._cache: Dict[str, tuple] = {}  # (value, expires_at)
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        async with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if datetime.now(timezone.utc).timestamp() < expires_at:
                    return value
                else:
                    del self._cache[key]
        return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Set value in cache."""
        ttl = ttl or self._default_ttl
        expires_at = datetime.now(timezone.utc).timestamp() + ttl
        
        async with self._lock:
            # Evict if needed
            if len(self._cache) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            self._cache[key] = (value, expires_at)
    
    async def delete(self, key: str):
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)
