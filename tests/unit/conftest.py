"""
Unit Test Configuration and Fixtures

Provides shared fixtures and configuration for all unit tests.
Target Coverage: 80%+
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ============================================================
# Async Event Loop Configuration
# ============================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================
# Mock Data Fixtures
# ============================================================

@dataclass
class MockUser:
    """Mock user for testing."""
    id: str = "user-123"
    email: str = "test@example.com"
    name: str = "Test User"
    role: str = "user"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockProject:
    """Mock project for testing."""
    id: str = "project-456"
    name: str = "Test Project"
    owner_id: str = "user-123"
    language: str = "python"
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "owner_id": self.owner_id,
            "language": self.language,
            "status": self.status,
        }


@dataclass
class MockAnalysisResult:
    """Mock analysis result for testing."""
    id: str = "analysis-789"
    project_id: str = "project-456"
    status: str = "completed"
    issues: List[Dict] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = [
                {
                    "id": "issue-1",
                    "type": "security",
                    "severity": "high",
                    "message": "SQL injection vulnerability",
                    "line": 42,
                }
            ]
        if self.metrics is None:
            self.metrics = {
                "lines_analyzed": 1000,
                "issues_found": 1,
                "duration_ms": 1500,
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "status": self.status,
            "issues": self.issues,
            "metrics": self.metrics,
        }


@pytest.fixture
def mock_user() -> MockUser:
    """Provide mock user."""
    return MockUser()


@pytest.fixture
def mock_admin_user() -> MockUser:
    """Provide mock admin user."""
    return MockUser(id="admin-001", email="admin@example.com", name="Admin", role="admin")


@pytest.fixture
def mock_project() -> MockProject:
    """Provide mock project."""
    return MockProject()


@pytest.fixture
def mock_analysis_result() -> MockAnalysisResult:
    """Provide mock analysis result."""
    return MockAnalysisResult()


# ============================================================
# Database Fixtures
# ============================================================

@pytest.fixture
def mock_db_session():
    """Provide mock database session."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_db_pool():
    """Provide mock database connection pool."""
    pool = MagicMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    return pool


# ============================================================
# Redis Fixtures
# ============================================================

@pytest.fixture
def mock_redis():
    """Provide mock Redis client."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.setex = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.incr = AsyncMock(return_value=1)
    redis.hincrby = AsyncMock(return_value=1)
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=True)
    redis.sadd = AsyncMock(return_value=1)
    redis.smembers = AsyncMock(return_value=set())
    redis.publish = AsyncMock(return_value=1)
    return redis


# ============================================================
# HTTP Client Fixtures
# ============================================================

@pytest.fixture
def mock_http_client():
    """Provide mock HTTP client."""
    client = MagicMock()
    
    async def mock_request(*args, **kwargs):
        return MagicMock(
            status_code=200,
            json=MagicMock(return_value={"success": True}),
            text="OK",
            headers={},
        )
    
    client.get = AsyncMock(side_effect=mock_request)
    client.post = AsyncMock(side_effect=mock_request)
    client.put = AsyncMock(side_effect=mock_request)
    client.delete = AsyncMock(side_effect=mock_request)
    return client


# ============================================================
# AI Service Fixtures
# ============================================================

@pytest.fixture
def mock_ai_client():
    """Provide mock AI client."""
    client = MagicMock()
    
    client.analyze = AsyncMock(return_value={
        "issues": [
            {"type": "security", "severity": "high", "message": "Test issue"}
        ],
        "metrics": {"lines": 100},
    })
    
    client.chat = AsyncMock(return_value={
        "response": "This is a test response",
        "tokens_used": 100,
    })
    
    client.embed = AsyncMock(return_value={
        "embedding": [0.1] * 768,
    })
    
    return client


@pytest.fixture
def mock_openai_client():
    """Provide mock OpenAI client."""
    client = MagicMock()
    
    # Mock chat completion
    completion = MagicMock()
    completion.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    completion.usage = MagicMock(total_tokens=100)
    
    client.chat.completions.create = AsyncMock(return_value=completion)
    
    # Mock embeddings
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
    
    client.embeddings.create = AsyncMock(return_value=embedding_response)
    
    return client


# ============================================================
# Configuration Fixtures
# ============================================================

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "environment": "test",
        "database": {
            "url": "postgresql://test:test@localhost:5432/test_db",
            "pool_size": 5,
        },
        "redis": {
            "url": "redis://localhost:6379/0",
        },
        "ai": {
            "provider": "mock",
            "model": "test-model",
        },
        "auth": {
            "jwt_secret": "test-secret-key-for-testing-only",
            "token_expire_minutes": 30,
        },
    }


@pytest.fixture
def env_vars(monkeypatch):
    """Set environment variables for testing."""
    test_env = {
        "ENVIRONMENT": "test",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
        "REDIS_URL": "redis://localhost:6379/0",
        "JWT_SECRET": "test-secret-key-for-testing-only",
        "OPENAI_API_KEY": "sk-test-key",
        "MOCK_MODE": "true",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env


# ============================================================
# Request/Response Fixtures
# ============================================================

@pytest.fixture
def mock_request():
    """Provide mock HTTP request."""
    request = MagicMock()
    request.headers = {"Authorization": "Bearer test-token"}
    request.json = AsyncMock(return_value={})
    request.query_params = {}
    request.path_params = {}
    request.state = MagicMock()
    request.state.user = None
    return request


@pytest.fixture
def mock_response():
    """Provide mock HTTP response."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    return response


# ============================================================
# Test Utilities
# ============================================================

class TestHelper:
    """Helper utilities for tests."""
    
    @staticmethod
    def generate_token(user_id: str = "user-123", role: str = "user") -> str:
        """Generate test JWT token."""
        import jwt
        payload = {
            "sub": user_id,
            "role": role,
            "exp": datetime.now(timezone.utc).timestamp() + 3600,
        }
        return jwt.encode(payload, "test-secret-key-for-testing-only", algorithm="HS256")
    
    @staticmethod
    def create_test_file(content: str = "test content") -> bytes:
        """Create test file content."""
        return content.encode("utf-8")
    
    @staticmethod
    def generate_code_sample(language: str = "python", lines: int = 100) -> str:
        """Generate sample code for testing."""
        if language == "python":
            return "\n".join([
                f"def function_{i}():",
                f"    '''Function {i}'''",
                f"    return {i}",
                ""
            ] for i in range(lines // 4))
        elif language == "javascript":
            return "\n".join([
                f"function function{i}() {{",
                f"    // Function {i}",
                f"    return {i};",
                "}",
                ""
            ] for i in range(lines // 5))
        else:
            return "// Test code\n" * lines


@pytest.fixture
def test_helper():
    """Provide test helper utilities."""
    return TestHelper()


# ============================================================
# Cleanup Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    # Add cleanup logic here if needed


# ============================================================
# Markers
# ============================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")
