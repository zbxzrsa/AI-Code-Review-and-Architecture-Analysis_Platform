"""
Database Connection Manager

Production-ready async database connection with:
- Connection pooling (asyncpg + SQLAlchemy)
- Automatic reconnection with exponential backoff
- Health monitoring and metrics
- Transaction management
- Multi-tenant support via schemas
"""

import os
import logging
import asyncio
from typing import Optional, AsyncGenerator, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy import text, event
from pydantic import BaseModel
import asyncpg

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    # Connection settings
    host: str = os.getenv("DATABASE_HOST", "localhost")
    port: int = int(os.getenv("DATABASE_PORT", "5432"))
    database: str = os.getenv("DATABASE_NAME", "coderev")
    user: str = os.getenv("DATABASE_USER", "coderev")
    password: str = os.getenv("DATABASE_PASSWORD", "coderev_secret")
    
    # Pool settings
    pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
    max_overflow: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
    pool_timeout: int = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
    pool_recycle: int = int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))
    
    # SSL settings
    ssl_mode: str = os.getenv("DATABASE_SSL_MODE", "prefer")
    ssl_ca_cert: Optional[str] = os.getenv("DATABASE_SSL_CA_CERT")
    
    # Schema settings (for multi-tenancy)
    default_schema: str = "public"
    
    @property
    def async_url(self) -> str:
        """Get async database URL for SQLAlchemy."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def sync_url(self) -> str:
        """Get sync database URL for migrations."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class ConnectionStats(BaseModel):
    """Connection pool statistics."""
    pool_size: int = 0
    checked_out: int = 0
    overflow: int = 0
    checkedin: int = 0
    total_connections: int = 0
    last_health_check: Optional[datetime] = None
    is_healthy: bool = False


class DatabaseManager:
    """
    Manages database connections with pooling and health monitoring.
    
    Usage:
        db = DatabaseManager()
        await db.initialize()
        
        async with db.session() as session:
            result = await session.execute(text("SELECT 1"))
            
        await db.close()
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._raw_pool: Optional[asyncpg.Pool] = None
        self._is_initialized = False
        self._stats = ConnectionStats()
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 1.0  # seconds
    
    async def initialize(self) -> None:
        """Initialize database connection pool."""
        if self._is_initialized:
            return
        
        try:
            # Create SQLAlchemy async engine
            self._engine = create_async_engine(
                self.config.async_url,
                poolclass=AsyncAdaptedQueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,  # Enable connection health checks
                echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            
            # Create raw asyncpg pool for high-performance queries
            self._raw_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=2,
                max_size=self.config.pool_size,
                command_timeout=60,
            )
            
            # Test connection
            await self._test_connection()
            
            self._is_initialized = True
            self._reconnect_attempts = 0
            logger.info(f"Database initialized: {self.config.host}:{self.config.port}/{self.config.database}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            await self._handle_connection_error(e)
    
    async def _test_connection(self) -> None:
        """Test database connection."""
        async with self._engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
            self._stats.is_healthy = True
            self._stats.last_health_check = datetime.utcnow()
    
    async def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection errors with exponential backoff."""
        self._reconnect_attempts += 1
        
        if self._reconnect_attempts > self._max_reconnect_attempts:
            raise ConnectionError(
                f"Failed to connect after {self._max_reconnect_attempts} attempts: {error}"
            )
        
        delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))
        logger.warning(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)
        await self.initialize()
    
    async def close(self) -> None:
        """Close all connections."""
        if self._raw_pool:
            await self._raw_pool.close()
            self._raw_pool = None
        
        if self._engine:
            await self._engine.dispose()
            self._engine = None
        
        self._session_factory = None
        self._is_initialized = False
        logger.info("Database connections closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async session with automatic transaction management."""
        if not self._is_initialized:
            await self.initialize()
        
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Session error, rolled back: {e}")
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def raw_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a raw asyncpg connection for high-performance queries."""
        if not self._is_initialized:
            await self.initialize()
        
        async with self._raw_pool.acquire() as conn:
            yield conn
    
    async def execute(self, query: str, *args) -> Any:
        """Execute a query using raw connection."""
        async with self.raw_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def execute_one(self, query: str, *args) -> Optional[Dict]:
        """Execute a query and return single row."""
        async with self.raw_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def execute_scalar(self, query: str, *args) -> Any:
        """Execute a query and return scalar value."""
        async with self.raw_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            start = datetime.utcnow()
            await self._test_connection()
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            
            # Update stats
            if self._engine:
                pool = self._engine.pool
                self._stats.pool_size = pool.size()
                self._stats.checked_out = pool.checkedout()
                self._stats.overflow = pool.overflow()
                self._stats.checkedin = pool.checkedin()
            
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "pool": self._stats.dict(),
            }
        except Exception as e:
            self._stats.is_healthy = False
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    def get_stats(self) -> ConnectionStats:
        """Get connection pool statistics."""
        return self._stats


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database() -> DatabaseManager:
    """Get the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    return _db_manager


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get async session."""
    db = await get_database()
    async with db.session() as session:
        yield session


async def init_database(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(config)
    await _db_manager.initialize()
    return _db_manager


async def close_database() -> None:
    """Close the global database manager."""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
        _db_manager = None
