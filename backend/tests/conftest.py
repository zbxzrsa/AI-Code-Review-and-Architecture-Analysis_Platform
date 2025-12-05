"""
Pytest Configuration and Fixtures

Provides shared fixtures for all service tests:
- Test database connections
- Redis test client
- Mock services
- Test data factories
"""

import asyncio
import os
from typing import AsyncGenerator, Generator
from datetime, timezone import datetime, timezone
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://test:test@localhost:5432/test_db"
)
os.environ["REDIS_URL"] = os.environ.get(
    "TEST_REDIS_URL",
    "redis://localhost:6379/15"
)


# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def test_db():
    """
    Provide a test database connection
    
    Creates tables before tests, rolls back after each test
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    # Use test database URL
    database_url = os.environ["DATABASE_URL"].replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    
    engine = create_async_engine(database_url, echo=False)
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()
    
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def clean_db(test_db):
    """
    Provide a clean database for each test
    
    Truncates all tables after each test
    """
    yield test_db
    
    # Clean up tables after test
    await test_db.execute("TRUNCATE TABLE users CASCADE")
    await test_db.commit()


# =============================================================================
# Redis Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def test_redis():
    """Provide a test Redis connection"""
    import redis.asyncio as redis
    
    client = redis.from_url(
        os.environ["REDIS_URL"],
        encoding="utf-8",
        decode_responses=True
    )
    
    yield client
    
    # Clean up test keys
    await client.flushdb()
    await client.close()


# =============================================================================
# HTTP Client Fixtures
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def http_client():
    """Provide an HTTP client for API tests"""
    import httpx
    
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture(scope="function")
def test_client():
    """Provide a FastAPI test client"""
    from fastapi.testclient import TestClient
    
    # Import the app - adjust path based on service
    # from src.main import app
    # return TestClient(app)
    
    return MagicMock()


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_auth_service():
    """Mock auth service for testing other services"""
    mock = AsyncMock()
    mock.verify_token.return_value = {
        "user_id": "test-user-123",
        "email": "test@example.com",
        "role": "user"
    }
    return mock


@pytest.fixture
def mock_ai_provider():
    """Mock AI provider for testing analysis"""
    mock = AsyncMock()
    mock.analyze.return_value = {
        "issues": [
            {
                "type": "security",
                "severity": "high",
                "message": "SQL injection vulnerability",
                "line": 10
            }
        ],
        "suggestions": ["Use parameterized queries"],
        "tokens_used": 500
    }
    return mock


@pytest.fixture
def mock_database():
    """Mock database for unit tests"""
    mock = MagicMock()
    mock.execute = AsyncMock(return_value=MagicMock())
    mock.commit = AsyncMock()
    mock.rollback = AsyncMock()
    return mock


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.ping = AsyncMock(return_value=True)
    return mock


# =============================================================================
# Test Data Factories
# =============================================================================

@pytest.fixture
def user_factory():
    """Factory for creating test users"""
    def _create_user(
        id: str = "user-123",
        email: str = "test@example.com",
        name: str = "Test User",
        role: str = "user",
        **kwargs
    ):
        return {
            "id": id,
            "email": email,
            "name": name,
            "role": role,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
    return _create_user


@pytest.fixture
def project_factory():
    """Factory for creating test projects"""
    def _create_project(
        id: str = "project-123",
        name: str = "Test Project",
        owner_id: str = "user-123",
        **kwargs
    ):
        return {
            "id": id,
            "name": name,
            "owner_id": owner_id,
            "description": "Test project description",
            "language": "python",
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
    return _create_project


@pytest.fixture
def analysis_factory():
    """Factory for creating test analysis results"""
    def _create_analysis(
        id: str = "analysis-123",
        project_id: str = "project-123",
        **kwargs
    ):
        return {
            "id": id,
            "project_id": project_id,
            "status": "completed",
            "issues_count": 5,
            "duration_ms": 1500,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
    return _create_analysis


# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration"""
    return {
        "database_url": os.environ["DATABASE_URL"],
        "redis_url": os.environ["REDIS_URL"],
        "jwt_secret": "test-secret-key",
        "environment": "test",
    }


# =============================================================================
# Async Helpers
# =============================================================================

@pytest.fixture
def async_mock():
    """Create an async mock helper"""
    def _async_mock(return_value=None):
        mock = AsyncMock()
        mock.return_value = return_value
        return mock
    return _async_mock


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks after each test"""
    yield
    # Cleanup happens automatically


@pytest_asyncio.fixture(autouse=True)
async def cleanup_async():
    """Cleanup async resources after each test"""
    yield
    # Allow pending tasks to complete
    await asyncio.sleep(0)
