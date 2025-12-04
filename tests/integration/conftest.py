"""
Pytest Configuration for Integration Tests

Provides fixtures and configuration for integration testing
the three-version architecture components.
"""

import asyncio
import os
from typing import AsyncGenerator, Generator

import httpx
import pytest
import pytest_asyncio

# Environment configuration
EVALUATION_URL = os.getenv("EVALUATION_URL", "http://localhost:8080")
LIFECYCLE_URL = os.getenv("LIFECYCLE_URL", "http://localhost:8003")
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
OPA_URL = os.getenv("OPA_URL", "http://localhost:8181")


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client for API calls"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client


@pytest_asyncio.fixture
async def evaluation_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client configured for evaluation service"""
    async with httpx.AsyncClient(
        base_url=EVALUATION_URL,
        timeout=60.0
    ) as client:
        yield client


@pytest_asyncio.fixture
async def lifecycle_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client configured for lifecycle controller"""
    async with httpx.AsyncClient(
        base_url=LIFECYCLE_URL,
        timeout=30.0
    ) as client:
        yield client


@pytest.fixture
def sample_code_python() -> str:
    """Sample Python code for testing"""
    return '''
def process_user_input(user_data):
    """Process user input with potential vulnerabilities"""
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_data['id']}"
    
    # Command injection vulnerability
    os.system(f"echo {user_data['message']}")
    
    return query
'''


@pytest.fixture
def sample_code_javascript() -> str:
    """Sample JavaScript code for testing"""
    return '''
function displayUserContent(userInput) {
    // XSS vulnerability
    document.getElementById('output').innerHTML = userInput;
    
    // Eval vulnerability
    eval(userInput);
}
'''


@pytest.fixture
def sample_code_safe() -> str:
    """Sample safe code for testing"""
    return '''
def get_user(user_id: int) -> Optional[User]:
    """Safely retrieve user by ID using parameterized query"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE id = %s",
            (user_id,)
        )
        return cursor.fetchone()
'''


@pytest.fixture
def promotion_thresholds() -> dict:
    """Standard promotion thresholds"""
    return {
        "p95_latency_ms": 3000,
        "error_rate": 0.02,
        "accuracy_delta": 0.02,
        "security_pass_rate": 0.99,
        "cost_increase_max": 0.10,
        "statistical_significance_p": 0.05
    }


@pytest.fixture
def passing_evaluation_metrics() -> dict:
    """Metrics that should pass evaluation"""
    return {
        "p95_latency_ms": 2500,
        "error_rate": 0.01,
        "accuracy": 0.92,
        "accuracy_delta": 0.04,
        "security_pass_rate": 0.995,
        "cost_per_request": 0.004,
        "cost_increase": 0.05,
        "total_tests": 100,
        "passed_tests": 95,
        "failed_tests": 5
    }


@pytest.fixture
def failing_evaluation_metrics() -> dict:
    """Metrics that should fail evaluation"""
    return {
        "p95_latency_ms": 4500,
        "error_rate": 0.035,
        "accuracy": 0.82,
        "accuracy_delta": -0.02,
        "security_pass_rate": 0.92,
        "cost_per_request": 0.008,
        "cost_increase": 0.15,
        "total_tests": 100,
        "passed_tests": 70,
        "failed_tests": 30
    }


@pytest.fixture
def mock_comparison_request() -> dict:
    """Mock comparison request data"""
    return {
        "requestId": "test-req-123",
        "code": "function test() { return 1; }",
        "language": "javascript",
        "timestamp": "2024-01-01T00:00:00Z",
        "v1Output": {
            "version": "v1",
            "versionId": "v1-test",
            "modelVersion": "gpt-4o",
            "promptVersion": "code-review-v4",
            "timestamp": "2024-01-01T00:00:01Z",
            "latencyMs": 2500,
            "cost": 0.004,
            "issues": [],
            "rawOutput": "{}",
            "confidence": 0.95,
            "securityPassed": True
        },
        "v2Output": {
            "version": "v2",
            "versionId": "v2-stable",
            "modelVersion": "gpt-4o",
            "promptVersion": "code-review-v3",
            "timestamp": "2024-01-01T00:00:01Z",
            "latencyMs": 2800,
            "cost": 0.0038,
            "issues": [],
            "rawOutput": "{}",
            "confidence": 0.88,
            "securityPassed": True
        }
    }


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_services: mark test as requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if services not available"""
    if os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true":
        skip_integration = pytest.mark.skip(
            reason="SKIP_INTEGRATION_TESTS is set"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
