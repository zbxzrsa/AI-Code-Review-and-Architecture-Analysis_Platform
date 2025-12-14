"""
Pytest配置和共享fixtures
"""
import pytest
import asyncio
import os
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """设置测试环境变量"""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5433/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6380/15")
    monkeypatch.setenv("KAFKA_BROKERS", "localhost:9092")


@pytest.fixture
def project_root():
    """项目根目录"""
    return Path(__file__).parent.parent
