"""
Pytest Configuration / Pytest 配置

Shared fixtures and configuration for tests.
测试的共享夹具和配置。
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture(scope="session")
def app_config():
    """Application configuration for tests / 测试用应用配置"""
    return {
        "MOCK_MODE": True,
        "ENVIRONMENT": "test"
    }


@pytest.fixture
def sample_code():
    """Sample code for analysis tests / 分析测试用示例代码"""
    return """
function calculateSum(a, b) {
    var result = a + b;
    console.log('Result:', result);
    return result;
}

const numbers = [1, 2, 3, 4, 5];
let total = 0;
for (var i = 0; i < numbers.length; i++) {
    total += numbers[i];
}
"""


@pytest.fixture
def sample_project_data():
    """Sample project data for tests / 测试用示例项目数据"""
    return {
        "name": "Test Project",
        "language": "TypeScript",
        "description": "A test project for unit testing",
        "framework": "React"
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for tests / 测试用示例用户数据"""
    return {
        "email": "test@example.com",
        "name": "Test User",
        "role": "user"
    }
