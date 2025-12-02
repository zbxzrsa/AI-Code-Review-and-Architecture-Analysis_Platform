"""
Pytest configuration and shared fixtures.

This file provides common fixtures and configuration for all tests.
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may require external services)"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip integration tests unless explicitly requested
    if not config.getoption("--run-integration", default=False):
        skip_integration = pytest.mark.skip(reason="Integration tests not requested (use --run-integration)")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DEBUG"] = "true"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-testing-only"
    os.environ["CSRF_SECRET_KEY"] = "test-csrf-secret-for-testing-only"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test_db"
    os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # Use DB 15 for tests
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def mock_database():
    """Create mock database session."""
    db = AsyncMock()
    db.execute = AsyncMock(return_value=Mock(scalars=Mock(return_value=Mock(all=Mock(return_value=[])))))
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.close = AsyncMock()
    return db


@pytest.fixture
async def test_database():
    """Create test database connection (integration)."""
    # This would connect to actual test database
    # For unit tests, use mock_database instead
    pytest.skip("Requires database connection")


# ============================================================================
# Redis Fixtures
# ============================================================================

@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)
    redis.zadd = AsyncMock(return_value=1)
    redis.zremrangebyscore = AsyncMock(return_value=0)
    redis.zcard = AsyncMock(return_value=0)
    redis.pipeline = Mock(return_value=AsyncMock())
    return redis


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture
def mock_http_client():
    """Create mock HTTP client."""
    client = AsyncMock()
    
    response = Mock()
    response.status_code = 200
    response.json = Mock(return_value={})
    response.text = ""
    response.raise_for_status = Mock()
    
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.put = AsyncMock(return_value=response)
    client.delete = AsyncMock(return_value=response)
    
    return client


# ============================================================================
# Request Fixtures
# ============================================================================

@pytest.fixture
def mock_request():
    """Create mock FastAPI request."""
    request = Mock()
    request.url.path = "/api/test"
    request.method = "GET"
    request.client.host = "127.0.0.1"
    request.headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }
    request.cookies = {}
    request.query_params = {}
    request.path_params = {}
    return request


@pytest.fixture
def mock_response():
    """Create mock FastAPI response."""
    response = Mock()
    response.headers = {}
    response.set_cookie = Mock()
    response.delete_cookie = Mock()
    return response


# ============================================================================
# User Fixtures
# ============================================================================

@pytest.fixture
def test_user():
    """Create test user data."""
    return {
        "id": "user-123",
        "email": "test@example.com",
        "name": "Test User",
        "role": "user",
        "created_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def admin_user():
    """Create admin user data."""
    return {
        "id": "admin-123",
        "email": "admin@example.com",
        "name": "Admin User",
        "role": "admin",
        "created_at": "2024-01-01T00:00:00Z",
    }


# ============================================================================
# Auth Fixtures
# ============================================================================

@pytest.fixture
def auth_manager():
    """Create auth manager for tests."""
    from backend.shared.security.secure_auth import SecureAuthManager, AuthConfig
    
    config = AuthConfig(
        secret_key="test-secret-key-for-testing-only",
        cookie_secure=False,
    )
    return SecureAuthManager(config)


@pytest.fixture
def valid_access_token(auth_manager, test_user):
    """Create valid access token."""
    return auth_manager.create_access_token({"sub": test_user["id"], "role": test_user["role"]})


@pytest.fixture
def valid_refresh_token(auth_manager, test_user):
    """Create valid refresh token."""
    return auth_manager.create_refresh_token({"sub": test_user["id"]})


@pytest.fixture
def valid_csrf_token(auth_manager, test_user):
    """Create valid CSRF token."""
    return auth_manager.generate_csrf_token(test_user["id"])


# ============================================================================
# AI Provider Fixtures
# ============================================================================

@pytest.fixture
def mock_ollama_provider():
    """Create mock Ollama provider."""
    from backend.shared.utils.ai_provider_factory import AIResponse
    
    provider = AsyncMock()
    provider.name = "ollama:codellama:7b"
    provider.is_free = True
    provider._healthy = True
    
    provider.health_check = AsyncMock(return_value=True)
    provider.analyze = AsyncMock(return_value=AIResponse(
        content="Mock analysis result",
        model="codellama:7b",
        provider="ollama",
        tokens_used=100,
        latency_ms=500.0,
        cost=0.0,
    ))
    
    return provider


@pytest.fixture
def ai_provider_factory(mock_ollama_provider):
    """Create AI provider factory with mock provider."""
    from backend.shared.utils.ai_provider_factory import AIProviderFactory
    
    factory = AIProviderFactory()
    factory._providers[mock_ollama_provider.name] = mock_ollama_provider
    factory._priority_chain = [mock_ollama_provider]
    
    return factory


# ============================================================================
# Rate Limiter Fixtures
# ============================================================================

@pytest.fixture
def rate_limiter(mock_redis):
    """Create rate limiter with mock Redis."""
    from backend.shared.middleware.rate_limiter import SlidingWindowRateLimiter, RateLimitConfig
    
    config = RateLimitConfig(
        default_rpm=60,
        default_rph=1000,
    )
    limiter = SlidingWindowRateLimiter(config)
    limiter._redis = mock_redis
    
    return limiter


# ============================================================================
# Project Fixtures
# ============================================================================

@pytest.fixture
def test_project():
    """Create test project data."""
    return {
        "id": "project-123",
        "name": "test-project",
        "description": "Test project for testing",
        "language": "python",
        "framework": "fastapi",
        "owner_id": "user-123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def test_code():
    """Create test code sample."""
    return '''
def calculate_sum(numbers):
    """Calculate sum of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total

def find_maximum(numbers):
    """Find maximum number."""
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers[1:]:
        if num > max_num:
            max_num = num
    return max_num
'''


# ============================================================================
# Analysis Fixtures
# ============================================================================

@pytest.fixture
def test_analysis_result():
    """Create test analysis result."""
    return {
        "id": "analysis-123",
        "project_id": "project-123",
        "status": "completed",
        "issues": [
            {
                "type": "performance",
                "severity": "warning",
                "line_start": 3,
                "line_end": 6,
                "description": "Consider using sum() instead of manual loop",
                "fix": "return sum(numbers)",
            }
        ],
        "suggestions": [
            "Use built-in functions for better performance",
            "Add input validation",
        ],
        "metrics": {
            "complexity": 5,
            "maintainability": 85,
            "test_coverage": 0,
        },
    }


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Clean up after each test."""
    yield
    
    # Cleanup code here (if needed)
    # e.g., clear caches, reset mocks, etc.
