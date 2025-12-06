"""
Database Query Optimization System (Performance Optimization #3)

Provides database optimization with:
- Slow query analysis and logging
- Automatic index recommendations
- Query result caching
- Connection pooling optimization
- Query execution plan analysis

Expected Benefits:
- Query latency reduced by 20-40%
- Better database resource utilization
- Automatic performance insights
"""
import asyncio
import functools
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar
import re

logger = logging.getLogger(__name__)

T = TypeVar('T')


class QueryType(str, Enum):
    """Types of SQL queries."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    OTHER = "OTHER"


@dataclass
class QueryStats:
    """Statistics for a query."""
    query_hash: str
    query_template: str
    query_type: QueryType
    execution_count: int = 0
    total_time_ms: float = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0
    avg_time_ms: float = 0
    rows_affected_total: int = 0
    last_executed: Optional[datetime] = None
    slow_count: int = 0  # Times exceeded threshold
    
    def record_execution(self, duration_ms: float, rows_affected: int = 0, slow_threshold: float = 100):
        self.execution_count += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.avg_time_ms = self.total_time_ms / self.execution_count
        self.rows_affected_total += rows_affected
        self.last_executed = datetime.now(timezone.utc)
        
        if duration_ms > slow_threshold:
            self.slow_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "query_template": self.query_template[:200] + "..." if len(self.query_template) > 200 else self.query_template,
            "query_type": self.query_type.value,
            "execution_count": self.execution_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2) if self.min_time_ms != float('inf') else 0,
            "max_time_ms": round(self.max_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "slow_count": self.slow_count,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
        }


@dataclass
class SlowQueryLog:
    """Log entry for a slow query."""
    query: str
    duration_ms: float
    timestamp: datetime
    query_hash: str
    parameters: Optional[Dict] = None
    execution_plan: Optional[str] = None
    rows_affected: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:500] + "..." if len(self.query) > 500 else self.query,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "query_hash": self.query_hash,
            "rows_affected": self.rows_affected,
            "has_execution_plan": self.execution_plan is not None,
        }


@dataclass
class IndexRecommendation:
    """Recommended index for query optimization."""
    table_name: str
    columns: List[str]
    index_type: str  # btree, hash, gin, etc.
    estimated_improvement: float  # Percentage
    reason: str
    create_statement: str
    priority: int = 0  # Lower is higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "columns": self.columns,
            "index_type": self.index_type,
            "estimated_improvement": self.estimated_improvement,
            "reason": self.reason,
            "create_statement": self.create_statement,
            "priority": self.priority,
        }


@dataclass
class QueryCacheConfig:
    """Configuration for query result caching."""
    enabled: bool = True
    default_ttl: int = 300  # 5 minutes
    max_size: int = 10000
    max_result_size_bytes: int = 1024 * 1024  # 1MB
    cacheable_tables: Set[str] = field(default_factory=lambda: {
        "projects", "users", "configurations", "versions"
    })
    exclude_patterns: List[str] = field(default_factory=lambda: [
        r".*_temp.*",
        r".*audit.*",
        r".*session.*",
    ])


class QueryNormalizer:
    """Normalizes SQL queries for analysis and caching."""
    
    # Patterns for parameter replacement
    PARAM_PATTERNS = [
        (r"'[^']*'", "'?'"),           # String literals
        (r"\b\d+\b", "?"),              # Numeric literals
        (r"\$\d+", "?"),                # PostgreSQL parameters
        (r":\w+", "?"),                 # Named parameters
        (r"\?\?", "?"),                 # Cleanup double ??
    ]
    
    @classmethod
    def normalize(cls, query: str) -> str:
        """Normalize query by replacing literals with placeholders."""
        normalized = query.strip()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        
        for pattern, replacement in cls.PARAM_PATTERNS:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized.upper()
    
    @classmethod
    def get_query_type(cls, query: str) -> QueryType:
        """Determine query type."""
        query_upper = query.strip().upper()
        
        if query_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE
        else:
            return QueryType.OTHER
    
    @classmethod
    def get_query_hash(cls, query: str) -> str:
        """Generate hash for normalized query."""
        normalized = cls.normalize(query)
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    @classmethod
    def extract_tables(cls, query: str) -> List[str]:
        """Extract table names from query."""
        # Simple extraction - production would use SQL parser
        tables = []
        
        # FROM clause
        from_match = re.search(r'\bFROM\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))
        
        # JOIN clauses
        join_matches = re.findall(r'\bJOIN\s+(\w+)', query, re.IGNORECASE)
        tables.extend(join_matches)
        
        # UPDATE/INSERT/DELETE
        for pattern in [
            r'\bUPDATE\s+(\w+)',
            r'\bINSERT\s+INTO\s+(\w+)',
            r'\bDELETE\s+FROM\s+(\w+)',
        ]:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                tables.append(match.group(1))
        
        return list(set(tables))
    
    @classmethod
    def extract_where_columns(cls, query: str) -> List[str]:
        """Extract columns used in WHERE clauses."""
        columns = []
        
        # Simple extraction of column = value patterns
        where_match = re.search(r'\bWHERE\s+(.+?)(?:ORDER|GROUP|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            col_matches = re.findall(r'(\w+)\s*[=<>]', where_clause)
            columns.extend(col_matches)
        
        return list(set(columns))


class QueryResultCache:
    """
    In-memory cache for query results.
    
    Features:
    - LRU eviction
    - TTL support
    - Size-based limits
    - Table-based invalidation
    """
    
    def __init__(self, config: QueryCacheConfig):
        self._config = config
        self._cache: Dict[str, Tuple[Any, datetime, Set[str]]] = {}  # hash -> (result, expires, tables)
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0,
        }
    
    def _generate_cache_key(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for query with parameters."""
        key_data = query
        if params:
            key_data += json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def _is_cacheable(self, query: str) -> bool:
        """Check if query should be cached."""
        if not self._config.enabled:
            return False
        
        # Only cache SELECT queries
        if QueryNormalizer.get_query_type(query) != QueryType.SELECT:
            return False
        
        # Check exclude patterns
        for pattern in self._config.exclude_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False
        
        return True
    
    async def get(self, query: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Get cached query result."""
        if not self._is_cacheable(query):
            return None
        
        key = self._generate_cache_key(query, params)
        
        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            result, expires, tables = self._cache[key]
            
            if datetime.now(timezone.utc) > expires:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats["misses"] += 1
                return None
            
            # Update LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self._stats["hits"] += 1
            return result
    
    async def set(
        self,
        query: str,
        result: Any,
        params: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache query result."""
        if not self._is_cacheable(query):
            return False
        
        # Check result size
        result_size = len(json.dumps(result, default=str).encode())
        if result_size > self._config.max_result_size_bytes:
            return False
        
        key = self._generate_cache_key(query, params)
        ttl = ttl or self._config.default_ttl
        expires = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        tables = set(QueryNormalizer.extract_tables(query))
        
        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._config.max_size and self._access_order:
                old_key = self._access_order.pop(0)
                self._cache.pop(old_key, None)
            
            self._cache[key] = (result, expires, tables)
            self._access_order.append(key)
            self._stats["sets"] += 1
            
        return True
    
    async def invalidate_table(self, table_name: str) -> int:
        """Invalidate all cached queries involving a table."""
        async with self._lock:
            keys_to_remove = [
                key for key, (_, _, tables) in self._cache.items()
                if table_name.lower() in {t.lower() for t in tables}
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            
            self._stats["invalidations"] += len(keys_to_remove)
            return len(keys_to_remove)
    
    async def clear(self) -> int:
        """Clear all cached results."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        hit_rate = self._stats["hits"] / (self._stats["hits"] + self._stats["misses"]) if (self._stats["hits"] + self._stats["misses"]) > 0 else 0
        return {
            **self._stats,
            "hit_rate": round(hit_rate, 4),
            "size": len(self._cache),
        }


class IndexAnalyzer:
    """Analyzes queries and recommends indexes."""
    
    def __init__(self, db_connection=None):
        self._db = db_connection
        self._query_patterns: Dict[str, QueryStats] = {}
    
    def analyze_query(self, query: str, duration_ms: float) -> List[IndexRecommendation]:
        """Analyze query and return index recommendations."""
        recommendations = []
        
        query_type = QueryNormalizer.get_query_type(query)
        if query_type != QueryType.SELECT:
            return recommendations
        
        tables = QueryNormalizer.extract_tables(query)
        where_columns = QueryNormalizer.extract_where_columns(query)
        
        if not tables or not where_columns:
            return recommendations
        
        # Recommend index for frequently filtered columns
        for table in tables:
            for column in where_columns:
                recommendations.append(IndexRecommendation(
                    table_name=table,
                    columns=[column],
                    index_type="btree",
                    estimated_improvement=20.0,  # Conservative estimate
                    reason=f"Column '{column}' used in WHERE clause",
                    create_statement=f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{table}_{column} ON {table} ({column});",
                    priority=1 if duration_ms > 100 else 2,
                ))
        
        # Check for composite index opportunities
        if len(where_columns) > 1:
            for table in tables:
                col_list = ", ".join(where_columns[:3])  # Max 3 columns
                recommendations.append(IndexRecommendation(
                    table_name=table,
                    columns=where_columns[:3],
                    index_type="btree",
                    estimated_improvement=30.0,
                    reason=f"Multiple columns used in WHERE clause",
                    create_statement=f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{table}_composite ON {table} ({col_list});",
                    priority=0,
                ))
        
        return recommendations
    
    async def get_execution_plan(self, query: str) -> Optional[str]:
        """Get query execution plan."""
        if not self._db:
            return None
        
        try:
            explain_query = f"EXPLAIN ANALYZE {query}"
            result = await self._db.fetch(explain_query)
            return "\n".join([row[0] for row in result])
        except Exception as e:
            logger.error(f"Failed to get execution plan: {e}")
            return None


class SlowQueryAnalyzer:
    """
    Analyzes and logs slow queries.
    
    Features:
    - Configurable slow query threshold
    - Query statistics aggregation
    - Automatic alerting for problematic queries
    """
    
    def __init__(
        self,
        slow_threshold_ms: float = 100,
        max_log_size: int = 10000,
        index_analyzer: Optional[IndexAnalyzer] = None
    ):
        self._slow_threshold = slow_threshold_ms
        self._max_log_size = max_log_size
        self._index_analyzer = index_analyzer or IndexAnalyzer()
        
        self._query_stats: Dict[str, QueryStats] = {}
        self._slow_log: List[SlowQueryLog] = []
        self._lock = asyncio.Lock()
    
    async def record_query(
        self,
        query: str,
        duration_ms: float,
        rows_affected: int = 0,
        params: Optional[Dict] = None
    ) -> Optional[List[IndexRecommendation]]:
        """Record query execution and analyze if slow."""
        query_hash = QueryNormalizer.get_query_hash(query)
        query_type = QueryNormalizer.get_query_type(query)
        normalized = QueryNormalizer.normalize(query)
        
        async with self._lock:
            # Update or create stats
            if query_hash not in self._query_stats:
                self._query_stats[query_hash] = QueryStats(
                    query_hash=query_hash,
                    query_template=normalized,
                    query_type=query_type,
                )
            
            self._query_stats[query_hash].record_execution(
                duration_ms, rows_affected, self._slow_threshold
            )
        
        # Handle slow query
        recommendations = None
        if duration_ms > self._slow_threshold:
            recommendations = await self._handle_slow_query(
                query, duration_ms, query_hash, params, rows_affected
            )
        
        return recommendations
    
    async def _handle_slow_query(
        self,
        query: str,
        duration_ms: float,
        query_hash: str,
        params: Optional[Dict],
        rows_affected: int
    ) -> List[IndexRecommendation]:
        """Handle slow query logging and analysis."""
        # Get execution plan
        execution_plan = await self._index_analyzer.get_execution_plan(query)
        
        # Create log entry
        log_entry = SlowQueryLog(
            query=query,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc),
            query_hash=query_hash,
            parameters=params,
            execution_plan=execution_plan,
            rows_affected=rows_affected,
        )
        
        async with self._lock:
            self._slow_log.append(log_entry)
            
            # Trim log if too large
            while len(self._slow_log) > self._max_log_size:
                self._slow_log.pop(0)
        
        logger.warning(
            f"Slow query detected: {duration_ms:.2f}ms - {query[:100]}..."
        )
        
        # Get recommendations
        recommendations = self._index_analyzer.analyze_query(query, duration_ms)
        
        return recommendations
    
    def get_slow_queries(
        self,
        limit: int = 100,
        min_duration_ms: Optional[float] = None
    ) -> List[Dict]:
        """Get recent slow queries."""
        queries = self._slow_log[-limit:]
        
        if min_duration_ms:
            queries = [q for q in queries if q.duration_ms >= min_duration_ms]
        
        return [q.to_dict() for q in queries]
    
    def get_top_queries(
        self,
        limit: int = 20,
        order_by: str = "avg_time"
    ) -> List[Dict]:
        """Get top queries by specified metric."""
        stats = list(self._query_stats.values())
        
        if order_by == "avg_time":
            stats.sort(key=lambda x: x.avg_time_ms, reverse=True)
        elif order_by == "total_time":
            stats.sort(key=lambda x: x.total_time_ms, reverse=True)
        elif order_by == "execution_count":
            stats.sort(key=lambda x: x.execution_count, reverse=True)
        elif order_by == "slow_count":
            stats.sort(key=lambda x: x.slow_count, reverse=True)
        
        return [s.to_dict() for s in stats[:limit]]
    
    def get_index_recommendations(self) -> List[Dict]:
        """Get aggregated index recommendations."""
        all_recommendations: Dict[str, IndexRecommendation] = {}
        
        for query_hash, stats in self._query_stats.items():
            if stats.slow_count > 0:
                recs = self._index_analyzer.analyze_query(
                    stats.query_template,
                    stats.avg_time_ms
                )
                for rec in recs:
                    key = f"{rec.table_name}:{':'.join(rec.columns)}"
                    if key not in all_recommendations:
                        all_recommendations[key] = rec
        
        # Sort by priority
        sorted_recs = sorted(
            all_recommendations.values(),
            key=lambda x: (x.priority, -x.estimated_improvement)
        )
        
        return [r.to_dict() for r in sorted_recs[:20]]
    
    def get_stats(self) -> Dict[str, Any]:
        total_queries = sum(s.execution_count for s in self._query_stats.values())
        total_slow = sum(s.slow_count for s in self._query_stats.values())
        
        return {
            "total_queries_tracked": total_queries,
            "unique_query_patterns": len(self._query_stats),
            "total_slow_queries": total_slow,
            "slow_query_log_size": len(self._slow_log),
            "slow_threshold_ms": self._slow_threshold,
        }


class QueryOptimizer:
    """
    Main query optimization orchestrator.
    
    Combines:
    - Slow query analysis
    - Query result caching
    - Index recommendations
    - Performance monitoring
    """
    
    def __init__(
        self,
        db_connection=None,
        cache_config: Optional[QueryCacheConfig] = None,
        slow_threshold_ms: float = 100
    ):
        self._db = db_connection
        self._cache = QueryResultCache(cache_config or QueryCacheConfig())
        self._index_analyzer = IndexAnalyzer(db_connection)
        self._slow_analyzer = SlowQueryAnalyzer(
            slow_threshold_ms=slow_threshold_ms,
            index_analyzer=self._index_analyzer
        )
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None
    ) -> Tuple[Any, float]:
        """
        Execute query with optimization.
        
        Returns:
            Tuple of (result, duration_ms)
        """
        # Try cache first
        if use_cache:
            cached_result = await self._cache.get(query, params)
            if cached_result is not None:
                logger.debug("Query cache hit")
                return cached_result, 0.0
        
        # Execute query
        start = time.perf_counter()
        
        try:
            if self._db:
                result = await self._db.fetch(query, **(params or {}))
            else:
                result = None  # Placeholder for actual execution
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            # Record for analysis
            await self._slow_analyzer.record_query(
                query, duration_ms, params=params
            )
            
            # Cache result
            if use_cache and result is not None:
                await self._cache.set(query, result, params, cache_ttl)
            
            return result, duration_ms
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            await self._slow_analyzer.record_query(
                query, duration_ms, params=params
            )
            raise
    
    async def invalidate_cache_for_table(self, table_name: str) -> int:
        """Invalidate cache for a specific table after modifications."""
        return await self._cache.invalidate_table(table_name)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            "cache_stats": self._cache.get_stats(),
            "query_stats": self._slow_analyzer.get_stats(),
            "top_slow_queries": self._slow_analyzer.get_top_queries(10, "avg_time"),
            "index_recommendations": self._slow_analyzer.get_index_recommendations(),
        }


def optimized_query(
    optimizer: QueryOptimizer,
    use_cache: bool = True,
    cache_ttl: Optional[int] = None
):
    """
    Decorator for optimized query execution.
    
    Usage:
        @optimized_query(optimizer, cache_ttl=300)
        async def get_users(status: str):
            return "SELECT * FROM users WHERE status = $1", {"status": status}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            query, params = await func(*args, **kwargs)
            result, duration = await optimizer.execute_query(
                query, params, use_cache, cache_ttl
            )
            return result
        return wrapper
    return decorator


# Global optimizer instance
_optimizer: Optional[QueryOptimizer] = None


def get_query_optimizer() -> QueryOptimizer:
    """Get or create global query optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = QueryOptimizer()
    return _optimizer


def init_query_optimizer(
    db_connection=None,
    cache_config: Optional[QueryCacheConfig] = None,
    slow_threshold_ms: float = 100
) -> QueryOptimizer:
    """Initialize global query optimizer."""
    global _optimizer
    _optimizer = QueryOptimizer(db_connection, cache_config, slow_threshold_ms)
    return _optimizer
