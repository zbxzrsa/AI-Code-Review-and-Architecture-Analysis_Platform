"""
Secure Database Query Module

Implements:
- Parameterized queries to prevent SQL injection
- Query builder with automatic escaping
- Connection pooling with limits
- Audit logging for sensitive operations
"""

import logging
import hashlib
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from contextlib import asynccontextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Query type classification for auditing."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    DDL = "DDL"


@dataclass
class QueryAuditEntry:
    """Audit entry for database query."""
    query_hash: str
    query_type: QueryType
    table: str
    user_id: Optional[str]
    timestamp: datetime
    duration_ms: float
    rows_affected: int
    success: bool
    error: Optional[str] = None


class SecureQueryBuilder:
    """
    Secure query builder with parameterized queries.
    
    Prevents SQL injection by enforcing parameterization.
    """
    
    def __init__(self):
        self._query_parts: List[str] = []
        self._params: Dict[str, Any] = {}
        self._param_counter = 0
        self._table: Optional[str] = None
        self._query_type: Optional[QueryType] = None
    
    def select(self, *columns: str) -> "SecureQueryBuilder":
        """Build SELECT query."""
        self._query_type = QueryType.SELECT
        cols = ", ".join(self._sanitize_identifier(c) for c in columns) or "*"
        self._query_parts.append(f"SELECT {cols}")
        return self
    
    def insert_into(self, table: str) -> "SecureQueryBuilder":
        """Build INSERT query."""
        self._query_type = QueryType.INSERT
        self._table = self._sanitize_identifier(table)
        self._query_parts.append(f"INSERT INTO {self._table}")
        return self
    
    def update(self, table: str) -> "SecureQueryBuilder":
        """Build UPDATE query."""
        self._query_type = QueryType.UPDATE
        self._table = self._sanitize_identifier(table)
        self._query_parts.append(f"UPDATE {self._table}")
        return self
    
    def delete_from(self, table: str) -> "SecureQueryBuilder":
        """Build DELETE query."""
        self._query_type = QueryType.DELETE
        self._table = self._sanitize_identifier(table)
        self._query_parts.append(f"DELETE FROM {self._table}")
        return self
    
    def from_table(self, table: str) -> "SecureQueryBuilder":
        """Add FROM clause."""
        self._table = self._sanitize_identifier(table)
        self._query_parts.append(f"FROM {self._table}")
        return self
    
    def columns(self, *cols: str) -> "SecureQueryBuilder":
        """Add column names for INSERT."""
        sanitized = [self._sanitize_identifier(c) for c in cols]
        self._query_parts.append(f"({', '.join(sanitized)})")
        return self
    
    def values(self, **kwargs) -> "SecureQueryBuilder":
        """Add VALUES clause with parameters."""
        placeholders = []
        for key, value in kwargs.items():
            param_name = self._add_param(value)
            placeholders.append(f"${param_name}")
        self._query_parts.append(f"VALUES ({', '.join(placeholders)})")
        return self
    
    def set(self, **kwargs) -> "SecureQueryBuilder":
        """Add SET clause for UPDATE."""
        assignments = []
        for key, value in kwargs.items():
            col = self._sanitize_identifier(key)
            param_name = self._add_param(value)
            assignments.append(f"{col} = ${param_name}")
        self._query_parts.append(f"SET {', '.join(assignments)}")
        return self
    
    def where(self, column: str, operator: str, value: Any) -> "SecureQueryBuilder":
        """Add WHERE clause."""
        col = self._sanitize_identifier(column)
        op = self._sanitize_operator(operator)
        param_name = self._add_param(value)
        
        if "WHERE" in " ".join(self._query_parts):
            self._query_parts.append(f"AND {col} {op} ${param_name}")
        else:
            self._query_parts.append(f"WHERE {col} {op} ${param_name}")
        return self
    
    def where_in(self, column: str, values: List[Any]) -> "SecureQueryBuilder":
        """Add WHERE IN clause."""
        col = self._sanitize_identifier(column)
        placeholders = []
        for v in values:
            param_name = self._add_param(v)
            placeholders.append(f"${param_name}")
        
        if "WHERE" in " ".join(self._query_parts):
            self._query_parts.append(f"AND {col} IN ({', '.join(placeholders)})")
        else:
            self._query_parts.append(f"WHERE {col} IN ({', '.join(placeholders)})")
        return self
    
    def order_by(self, column: str, direction: str = "ASC") -> "SecureQueryBuilder":
        """Add ORDER BY clause."""
        col = self._sanitize_identifier(column)
        dir = "DESC" if direction.upper() == "DESC" else "ASC"
        self._query_parts.append(f"ORDER BY {col} {dir}")
        return self
    
    def limit(self, count: int) -> "SecureQueryBuilder":
        """Add LIMIT clause."""
        param_name = self._add_param(count)
        self._query_parts.append(f"LIMIT ${param_name}")
        return self
    
    def offset(self, count: int) -> "SecureQueryBuilder":
        """Add OFFSET clause."""
        param_name = self._add_param(count)
        self._query_parts.append(f"OFFSET ${param_name}")
        return self
    
    def returning(self, *columns: str) -> "SecureQueryBuilder":
        """Add RETURNING clause."""
        cols = ", ".join(self._sanitize_identifier(c) for c in columns) or "*"
        self._query_parts.append(f"RETURNING {cols}")
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build final query and parameters."""
        query = " ".join(self._query_parts)
        return query, self._params
    
    def _add_param(self, value: Any) -> str:
        """Add parameter and return placeholder name."""
        self._param_counter += 1
        name = str(self._param_counter)
        self._params[name] = value
        return name
    
    def _sanitize_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifier (table/column name)."""
        # Only allow alphanumeric, underscore, and dot
        clean = "".join(c for c in identifier if c.isalnum() or c in "._")
        if clean != identifier:
            logger.warning(f"Sanitized identifier: {identifier} -> {clean}")
        return clean
    
    def _sanitize_operator(self, operator: str) -> str:
        """Sanitize SQL operator."""
        allowed = {"=", "!=", "<>", "<", ">", "<=", ">=", "LIKE", "ILIKE", "IS", "IS NOT"}
        op = operator.upper()
        if op not in allowed:
            raise ValueError(f"Invalid operator: {operator}")
        return op


class SecureDatabaseConnection:
    """
    Secure database connection with:
    - Connection pooling limits
    - Query auditing
    - Automatic parameterization
    """
    
    def __init__(
        self,
        dsn: str,
        min_connections: int = 5,
        max_connections: int = 20,
        connection_timeout: float = 10.0,
        audit_enabled: bool = True,
    ):
        self.dsn = dsn
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.audit_enabled = audit_enabled
        
        self._pool = None
        self._audit_log: List[QueryAuditEntry] = []
    
    async def connect(self):
        """Create connection pool."""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=self.connection_timeout,
                # Security settings
                ssl="require",  # Require SSL in production
            )
            logger.info(f"Database pool created: {self.min_connections}-{self.max_connections} connections")
        except ImportError:
            logger.warning("asyncpg not available, using mock connection")
            self._pool = MockPool()
    
    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Database pool closed")
    
    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def execute(
        self,
        query: str,
        params: Dict[str, Any] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Execute query and return rows affected."""
        start_time = datetime.utcnow()
        
        try:
            async with self._pool.acquire() as conn:
                # Convert named params to positional
                sql, args = self._convert_params(query, params or {})
                result = await conn.execute(sql, *args)
                
                # Parse rows affected
                rows = int(result.split()[-1]) if result else 0
                
                if self.audit_enabled:
                    self._log_query(query, QueryType.UPDATE, user_id, start_time, rows, True)
                
                return rows
                
        except Exception as e:
            if self.audit_enabled:
                self._log_query(query, QueryType.UPDATE, user_id, start_time, 0, False, str(e))
            raise
    
    async def fetch_one(
        self,
        query: str,
        params: Dict[str, Any] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch single row."""
        start_time = datetime.utcnow()
        
        try:
            async with self._pool.acquire() as conn:
                sql, args = self._convert_params(query, params or {})
                row = await conn.fetchrow(sql, *args)
                
                if self.audit_enabled:
                    self._log_query(query, QueryType.SELECT, user_id, start_time, 1 if row else 0, True)
                
                return dict(row) if row else None
                
        except Exception as e:
            if self.audit_enabled:
                self._log_query(query, QueryType.SELECT, user_id, start_time, 0, False, str(e))
            raise
    
    async def fetch_all(
        self,
        query: str,
        params: Dict[str, Any] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all rows."""
        start_time = datetime.utcnow()
        
        try:
            async with self._pool.acquire() as conn:
                sql, args = self._convert_params(query, params or {})
                rows = await conn.fetch(sql, *args)
                
                if self.audit_enabled:
                    self._log_query(query, QueryType.SELECT, user_id, start_time, len(rows), True)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            if self.audit_enabled:
                self._log_query(query, QueryType.SELECT, user_id, start_time, 0, False, str(e))
            raise
    
    def _convert_params(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """Convert named parameters to positional."""
        args = []
        for i, (key, value) in enumerate(params.items(), 1):
            query = query.replace(f"${key}", f"${i}")
            args.append(value)
        return query, args
    
    def _log_query(
        self,
        query: str,
        query_type: QueryType,
        user_id: Optional[str],
        start_time: datetime,
        rows: int,
        success: bool,
        error: Optional[str] = None,
    ):
        """Log query for audit."""
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Hash query for grouping
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        # Extract table name (simple heuristic)
        table = "unknown"
        for word in ["FROM", "INTO", "UPDATE"]:
            if word in query.upper():
                parts = query.upper().split(word)
                if len(parts) > 1:
                    table = parts[1].strip().split()[0].lower()
                    break
        
        entry = QueryAuditEntry(
            query_hash=query_hash,
            query_type=query_type,
            table=table,
            user_id=user_id,
            timestamp=start_time,
            duration_ms=duration,
            rows_affected=rows,
            success=success,
            error=error,
        )
        
        self._audit_log.append(entry)
        
        # Log to file/database for sensitive operations
        if query_type in [QueryType.DELETE, QueryType.UPDATE] or not success:
            logger.info(
                f"Query audit: {query_type.value} on {table} by {user_id}, "
                f"rows={rows}, duration={duration:.2f}ms, success={success}"
            )
    
    def get_audit_log(self) -> List[QueryAuditEntry]:
        """Get query audit log."""
        return self._audit_log.copy()


class MockPool:
    """Mock connection pool for testing."""
    
    async def acquire(self):
        return MockConnection()
    
    async def close(self):
        pass


class MockConnection:
    """Mock database connection."""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    async def execute(self, *args):
        return "UPDATE 0"
    
    async def fetchrow(self, *args):
        return None
    
    async def fetch(self, *args):
        return []
    
    def transaction(self):
        return MockTransaction()


class MockTransaction:
    """Mock transaction."""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass


# Helper function to create secure queries
def query() -> SecureQueryBuilder:
    """Create new query builder."""
    return SecureQueryBuilder()
