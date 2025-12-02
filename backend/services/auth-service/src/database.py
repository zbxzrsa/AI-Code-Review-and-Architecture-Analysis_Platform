"""
Database module for Auth Service.
"""
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from src.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.DEBUG,
    poolclass=NullPool,  # Use NullPool for better async performance
)

# Create session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initialize database tables."""
    try:
        async with engine.begin() as conn:
            # Create all tables (in production, use Alembic migrations)
            # await conn.run_sync(Base.metadata.create_all)
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db() -> None:
    """Close database connection."""
    await engine.dispose()
    logger.info("Database connection closed")
