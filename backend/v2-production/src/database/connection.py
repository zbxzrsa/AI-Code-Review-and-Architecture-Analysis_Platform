"""
Database connection and session management for V2 production.
"""
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from config.settings import settings

logger = logging.getLogger(__name__)

# Global engine and session factory
engine = None
SessionLocal = None


async def init_db():
    """Initialize database connection."""
    global engine, SessionLocal

    try:
        # Create async engine
        engine = create_async_engine(
            settings.database.connection_string.replace("postgresql://", "postgresql+asyncpg://"),
            echo=settings.database.echo,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
        )

        # Create session factory
        SessionLocal = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("Database initialized", database=settings.database.database)

    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


async def get_db_session() -> AsyncSession:
    """Get database session."""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized")

    async with SessionLocal() as session:
        yield session


async def close_db():
    """Close database connection."""
    global engine

    if engine:
        await engine.dispose()
        logger.info("Database connection closed")
