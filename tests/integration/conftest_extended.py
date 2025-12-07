"""
Extended Integration Test Fixtures

Provides fixtures for integration tests that require:
- Real database connections
- Redis connections
- HTTP clients
- Test data in database
"""

import pytest
import asyncio
import os
from typing import Dict, Any, List, AsyncGenerator
from datetime import datetime, timezone
from uuid import uuid4
from httpx import AsyncClient, ASGITransport


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="session")
async def db_pool():
    """Create database connection pool for tests."""
    try:
        import asyncpg
        
        pool = await asyncpg.create_pool(
            os.getenv("TEST_DATABASE_URL", "postgresql://test:test@localhost:5432/test_db"),
            min_size=2,
            max_size=10,
        )
        
        yield pool
        
        await pool.close()
    except ImportError:
        pytest.skip("asyncpg not installed")


@pytest.fixture
async def db_session(db_pool):
    """Get database connection from pool."""
    async with db_pool.acquire() as connection:
        # Start transaction
        transaction = connection.transaction()
        await transaction.start()
        
        yield connection
        
        # Rollback to keep database clean
        await transaction.rollback()


@pytest.fixture
async def db_session_2(db_pool):
    """Get second database connection for concurrent tests."""
    async with db_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        
        yield connection
        
        await transaction.rollback()


# ============================================================================
# Redis Fixtures
# ============================================================================

@pytest.fixture(scope="session")
async def redis_pool():
    """Create Redis connection pool."""
    try:
        import redis.asyncio as redis
        
        pool = redis.ConnectionPool.from_url(
            os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15"),
            max_connections=10,
        )
        
        yield pool
        
        await pool.disconnect()
    except ImportError:
        pytest.skip("redis not installed")


@pytest.fixture
async def redis_client(redis_pool):
    """Get Redis client from pool."""
    import redis.asyncio as redis
    
    client = redis.Redis(connection_pool=redis_pool)
    
    # Clear test database
    await client.flushdb()
    
    yield client
    
    await client.flushdb()


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for API testing."""
    from backend.app.main import app
    
    transport = ASGITransport(app=app)
    
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client


# ============================================================================
# Test User Fixtures
# ============================================================================

@pytest.fixture
async def test_user_in_db(db_session) -> Dict[str, Any]:
    """Create test user in database."""
    user_id = str(uuid4())
    email = f"test-{user_id[:8]}@example.com"
    
    await db_session.execute(
        """
        INSERT INTO auth.users (id, email, name, password_hash, role, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        [
            user_id,
            email,
            "Test User",
            "$argon2id$v=19$m=65536,t=3,p=4$test_hash",  # Pre-hashed password
            "user",
            datetime.now(timezone.utc),
        ]
    )
    
    return {
        "id": user_id,
        "email": email,
        "name": "Test User",
        "role": "user",
    }


@pytest.fixture
async def admin_user(db_session) -> Dict[str, Any]:
    """Create admin user in database."""
    user_id = str(uuid4())
    email = f"admin-{user_id[:8]}@example.com"
    
    await db_session.execute(
        """
        INSERT INTO auth.users (id, email, name, password_hash, role, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        [
            user_id,
            email,
            "Admin User",
            "$argon2id$v=19$m=65536,t=3,p=4$admin_hash",
            "admin",
            datetime.now(timezone.utc),
        ]
    )
    
    # Create access token
    from backend.shared.security.secure_auth import SecureAuthManager, AuthConfig
    
    auth_manager = SecureAuthManager(AuthConfig(secret_key="test-secret"))
    access_token = auth_manager.create_access_token({
        "sub": user_id,
        "role": "admin",
    })
    
    return {
        "id": user_id,
        "email": email,
        "name": "Admin User",
        "role": "admin",
        "access_token": access_token,
    }


@pytest.fixture
async def authenticated_user(test_user_in_db, db_session) -> Dict[str, Any]:
    """Create authenticated user with tokens."""
    from backend.shared.security.secure_auth import SecureAuthManager, AuthConfig
    
    auth_manager = SecureAuthManager(AuthConfig(secret_key="test-secret"))
    
    access_token = auth_manager.create_access_token({
        "sub": test_user_in_db["id"],
        "role": test_user_in_db["role"],
    })
    
    refresh_token = auth_manager.create_refresh_token({
        "sub": test_user_in_db["id"],
    })
    
    return {
        **test_user_in_db,
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


# ============================================================================
# Project Fixtures
# ============================================================================

@pytest.fixture
async def test_project_in_db(db_session, test_user_in_db) -> Dict[str, Any]:
    """Create test project in database."""
    project_id = str(uuid4())
    
    await db_session.execute(
        """
        INSERT INTO projects.projects (id, name, description, language, owner_id, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        [
            project_id,
            "Test Project",
            "Test project for integration tests",
            "python",
            test_user_in_db["id"],
            datetime.now(timezone.utc),
        ]
    )
    
    return {
        "id": project_id,
        "name": "Test Project",
        "description": "Test project for integration tests",
        "language": "python",
        "owner_id": test_user_in_db["id"],
    }


@pytest.fixture
async def other_user_project(db_session) -> Dict[str, Any]:
    """Create project owned by another user."""
    user_id = str(uuid4())
    project_id = str(uuid4())
    
    # Create other user
    await db_session.execute(
        """
        INSERT INTO auth.users (id, email, name, password_hash, role, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        [user_id, f"other-{user_id[:8]}@example.com", "Other User", "hash", "user", datetime.now(timezone.utc)]
    )
    
    # Create project
    await db_session.execute(
        """
        INSERT INTO projects.projects (id, name, description, owner_id, created_at)
        VALUES ($1, $2, $3, $4, $5)
        """,
        [project_id, "Other User Project", "Not accessible", user_id, datetime.now(timezone.utc)]
    )
    
    return {"id": project_id, "owner_id": user_id}


@pytest.fixture
async def many_projects_in_db(db_session, test_user_in_db) -> List[Dict[str, Any]]:
    """Create many projects for pagination tests."""
    projects = []
    
    for i in range(25):
        project_id = str(uuid4())
        await db_session.execute(
            """
            INSERT INTO projects.projects (id, name, description, language, owner_id, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            [
                project_id,
                f"Project {i}",
                f"Test project {i} - python framework",
                "python",
                test_user_in_db["id"],
                datetime.now(timezone.utc),
            ]
        )
        projects.append({"id": project_id, "name": f"Project {i}"})
    
    return projects


# ============================================================================
# Analysis Fixtures
# ============================================================================

@pytest.fixture
async def test_analysis_in_db(db_session, test_project_in_db) -> Dict[str, Any]:
    """Create test analysis in database."""
    analysis_id = str(uuid4())
    
    await db_session.execute(
        """
        INSERT INTO production.analysis_sessions (id, project_id, status, created_at)
        VALUES ($1, $2, $3, $4)
        """,
        [analysis_id, test_project_in_db["id"], "pending", datetime.now(timezone.utc)]
    )
    
    return {
        "id": analysis_id,
        "project_id": test_project_in_db["id"],
        "status": "pending",
    }


@pytest.fixture
async def completed_analysis_in_db(db_session, test_project_in_db) -> Dict[str, Any]:
    """Create completed analysis in database."""
    analysis_id = str(uuid4())
    now = datetime.now(timezone.utc)
    
    await db_session.execute(
        """
        INSERT INTO production.analysis_sessions (
            id, project_id, status, created_at, completed_at,
            issue_count, suggestion_count
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        [analysis_id, test_project_in_db["id"], "completed", now, now, 3, 5]
    )
    
    return {
        "id": analysis_id,
        "project_id": test_project_in_db["id"],
        "status": "completed",
        "issue_count": 3,
        "suggestion_count": 5,
    }


@pytest.fixture
async def many_analyses_in_db(db_session, test_project_in_db) -> List[Dict[str, Any]]:
    """Create many analyses for aggregation tests."""
    analyses = []
    
    for i in range(50):
        analysis_id = str(uuid4())
        now = datetime.now(timezone.utc)
        status = "completed" if i % 5 != 0 else "failed"
        
        await db_session.execute(
            """
            INSERT INTO production.analysis_sessions (
                id, project_id, status, created_at, completed_at, issue_count
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            [
                analysis_id,
                test_project_in_db["id"],
                status,
                now,
                now if status == "completed" else None,
                i % 10,
            ]
        )
        
        analyses.append({
            "id": analysis_id,
            "project_id": test_project_in_db["id"],
            "status": status,
        })
    
    return analyses


@pytest.fixture
async def test_project_with_analyses(db_session, test_user_in_db) -> Dict[str, Any]:
    """Create project with multiple analyses."""
    project_id = str(uuid4())
    now = datetime.now(timezone.utc)
    
    # Create project
    await db_session.execute(
        """
        INSERT INTO projects.projects (id, name, owner_id, created_at)
        VALUES ($1, $2, $3, $4)
        """,
        [project_id, "Project With Analyses", test_user_in_db["id"], now]
    )
    
    # Create analyses
    for i in range(10):
        await db_session.execute(
            """
            INSERT INTO production.analysis_sessions (
                id, project_id, status, created_at, completed_at, issue_count
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            [str(uuid4()), project_id, "completed", now, now, i]
        )
    
    return {"id": project_id, "name": "Project With Analyses"}


# ============================================================================
# Three-Version Fixtures
# ============================================================================

@pytest.fixture
async def v1_experiment_in_db(db_session) -> Dict[str, Any]:
    """Create V1 experiment in database."""
    experiment_id = str(uuid4())
    
    await db_session.execute(
        """
        INSERT INTO experiments_v1.experiments (id, name, status, created_at)
        VALUES ($1, $2, $3, $4)
        """,
        [experiment_id, "Test Experiment", "ready_for_promotion", datetime.now(timezone.utc)]
    )
    
    return {"id": experiment_id, "name": "Test Experiment", "status": "ready_for_promotion"}


@pytest.fixture
async def v2_version_in_db(db_session) -> Dict[str, Any]:
    """Create V2 production version in database."""
    version_id = str(uuid4())
    
    await db_session.execute(
        """
        INSERT INTO projects.version_history (id, version, status, promoted_at, created_at)
        VALUES ($1, $2, $3, $4, $5)
        """,
        [version_id, "2.0.0", "production", datetime.now(timezone.utc), datetime.now(timezone.utc)]
    )
    
    return {"id": version_id, "version": "2.0.0", "status": "production"}


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_code() -> str:
    """Sample Python code for analysis."""
    return '''
def process_data(items):
    """Process a list of items."""
    results = []
    for item in items:
        if item is not None:
            processed = item.strip().lower()
            if len(processed) > 0:
                results.append(processed)
    return results

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def validate(self, data):
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        return True
'''
