"""
Shared Database Module

Provides:
- Async connection pooling with asyncpg
- SQLAlchemy async session management
- Health checks and connection monitoring
- Automatic reconnection with exponential backoff
"""

from .connection import (
    DatabaseManager,
    get_database,
    get_async_session,
    init_database,
    close_database,
)
from .secure_queries import SecureQueryBuilder, QueryType

__all__ = [
    "DatabaseManager",
    "get_database",
    "get_async_session",
    "init_database",
    "close_database",
    "SecureQueryBuilder",
    "QueryType",
]
