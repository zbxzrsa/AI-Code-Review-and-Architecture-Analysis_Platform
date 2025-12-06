"""
Unit Tests for Health Check Module
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from shared.health import (
    HealthCheckRegistry,
    HealthStatus,
    ComponentHealth,
    HealthResponse,
    check_database,
    check_redis,
    check_http_service,
)


class TestHealthCheckRegistry:
    """Test HealthCheckRegistry class"""
    
    def test_create_registry(self):
        """Test registry initialization"""
        registry = HealthCheckRegistry("test-service", "1.0.0")
        
        assert registry.service_name == "test-service"
        assert registry.version == "1.0.0"
        assert registry._is_ready is False
        assert registry._is_shutting_down is False
    
    def test_set_ready(self):
        """Test setting ready status"""
        registry = HealthCheckRegistry("test-service")
        
        assert registry._is_ready is False
        registry.set_ready(True)
        assert registry._is_ready is True
        registry.set_ready(False)
        assert registry._is_ready is False
    
    def test_set_shutting_down(self):
        """Test setting shutdown status"""
        registry = HealthCheckRegistry("test-service")
        
        assert registry._is_shutting_down is False
        registry.set_shutting_down(True)
        assert registry._is_shutting_down is True
    
    def test_uptime_seconds(self):
        """Test uptime calculation"""
        registry = HealthCheckRegistry("test-service")
        
        # Should have positive uptime
        assert registry.uptime_seconds >= 0
    
    def test_register_check(self):
        """Test registering health check"""
        registry = HealthCheckRegistry("test-service")
        
        @registry.register("database")
        async def check_db():
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY
            )
        
        assert "database" in registry._checks
    
    def test_add_check(self):
        """Test adding health check programmatically"""
        registry = HealthCheckRegistry("test-service")
        
        async def check_db():
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY
            )
        
        registry.add_check("database", check_db)
        assert "database" in registry._checks
    
    def test_remove_check(self):
        """Test removing health check"""
        registry = HealthCheckRegistry("test-service")
        
        async def check_db():
            return ComponentHealth(name="database", status=HealthStatus.HEALTHY)
        
        registry.add_check("database", check_db)
        assert "database" in registry._checks
        
        registry.remove_check("database")
        assert "database" not in registry._checks
    
    @pytest.mark.asyncio
    async def test_check_component_healthy(self):
        """Test checking healthy component"""
        registry = HealthCheckRegistry("test-service")
        
        async def healthy_check():
            return ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK"
            )
        
        registry.add_check("test", healthy_check)
        result = await registry.check_component("test")
        
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms is not None
    
    @pytest.mark.asyncio
    async def test_check_component_unhealthy(self):
        """Test checking unhealthy component"""
        registry = HealthCheckRegistry("test-service")
        
        async def unhealthy_check():
            raise Exception("Connection failed")
        
        registry.add_check("test", unhealthy_check)
        result = await registry.check_component("test")
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.message
    
    @pytest.mark.asyncio
    async def test_check_component_timeout(self):
        """Test component check timeout"""
        registry = HealthCheckRegistry("test-service")
        
        async def slow_check():
            import asyncio
            await asyncio.sleep(10)  # Will timeout
            return ComponentHealth(name="test", status=HealthStatus.HEALTHY)
        
        registry.add_check("test", slow_check)
        result = await registry.check_component("test")
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_run_all_checks_healthy(self):
        """Test running all checks when healthy"""
        registry = HealthCheckRegistry("test-service")
        
        async def db_check():
            return ComponentHealth(name="database", status=HealthStatus.HEALTHY)
        
        async def redis_check():
            return ComponentHealth(name="redis", status=HealthStatus.HEALTHY)
        
        registry.add_check("database", db_check)
        registry.add_check("redis", redis_check)
        
        result = await registry.run_all_checks()
        
        assert result.status == HealthStatus.HEALTHY
        assert len(result.components) == 2
    
    @pytest.mark.asyncio
    async def test_run_all_checks_degraded(self):
        """Test running all checks when degraded"""
        registry = HealthCheckRegistry("test-service")
        
        async def db_check():
            return ComponentHealth(name="database", status=HealthStatus.HEALTHY)
        
        async def redis_check():
            return ComponentHealth(name="redis", status=HealthStatus.DEGRADED)
        
        registry.add_check("database", db_check)
        registry.add_check("redis", redis_check)
        
        result = await registry.run_all_checks()
        
        assert result.status == HealthStatus.DEGRADED
    
    @pytest.mark.asyncio
    async def test_run_all_checks_unhealthy(self):
        """Test running all checks when unhealthy"""
        registry = HealthCheckRegistry("test-service")
        
        async def db_check():
            return ComponentHealth(name="database", status=HealthStatus.UNHEALTHY)
        
        registry.add_check("database", db_check)
        
        result = await registry.run_all_checks()
        
        assert result.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_run_all_checks_shutting_down(self):
        """Test health check during shutdown"""
        registry = HealthCheckRegistry("test-service")
        registry.set_shutting_down(True)
        
        async def db_check():
            return ComponentHealth(name="database", status=HealthStatus.HEALTHY)
        
        registry.add_check("database", db_check)
        
        result = await registry.run_all_checks()
        
        assert result.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_check_readiness(self):
        """Test readiness check"""
        registry = HealthCheckRegistry("test-service")
        registry.set_ready(True)
        
        async def db_check():
            return ComponentHealth(name="database", status=HealthStatus.HEALTHY)
        
        registry.add_check("database", db_check)
        
        result = await registry.check_readiness()
        
        assert result.ready is True
        assert result.checks["database"] is True
    
    @pytest.mark.asyncio
    async def test_check_readiness_not_ready(self):
        """Test readiness when not ready"""
        registry = HealthCheckRegistry("test-service")
        registry.set_ready(False)
        
        result = await registry.check_readiness()
        
        assert result.ready is False


class TestComponentHealthChecks:
    """Test individual component health check functions"""
    
    @pytest.mark.asyncio
    async def test_check_database_healthy(self):
        """Test database check when healthy"""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        
        async def get_connection():
            class MockContext:
                async def __aenter__(self):
                    return mock_conn
                async def __aexit__(self, *args):  # noqa: ARG002
                    pass  # Mock - no cleanup needed
            return MockContext()
        
        result = await check_database(get_connection)
        
        assert result.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_check_database_unhealthy(self):
        """Test database check when unhealthy"""
        async def get_connection():
            class MockContext:
                async def __aenter__(self):
                    raise ConnectionError("Connection refused")
                async def __aexit__(self, *args):  # noqa: ARG002
                    pass  # Mock - no cleanup needed
            return MockContext()
        
        result = await check_database(get_connection)
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection" in result.message
    
    @pytest.mark.asyncio
    async def test_check_redis_healthy(self):
        """Test Redis check when healthy"""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        
        result = await check_redis(mock_redis)
        
        assert result.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_check_redis_unhealthy(self):
        """Test Redis check when unhealthy"""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection refused"))
        
        result = await check_redis(mock_redis)
        
        assert result.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_check_http_service_healthy(self):
        """Test HTTP service check when healthy"""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock()
            
            mock_client.return_value = mock_instance
            
            result = await check_http_service(
                "http://service:8000/health",
                "test-service"
            )
            
            assert result.status == HealthStatus.HEALTHY


class TestHealthResponse:
    """Test HealthResponse model"""
    
    def test_create_response(self):
        """Test creating health response"""
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            service="test-service",
            version="1.0.0",
            uptime_seconds=3600,
            components=[]
        )
        
        assert response.status == HealthStatus.HEALTHY
        assert response.service == "test-service"
        assert response.version == "1.0.0"
        assert response.uptime_seconds == 3600
    
    def test_response_with_components(self):
        """Test response with component health"""
        components = [
            ComponentHealth(name="database", status=HealthStatus.HEALTHY),
            ComponentHealth(name="redis", status=HealthStatus.HEALTHY),
        ]
        
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            service="test-service",
            components=components
        )
        
        assert len(response.components) == 2


class TestComponentHealth:
    """Test ComponentHealth model"""
    
    def test_create_component_health(self):
        """Test creating component health"""
        health = ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=5.5,
            message="Connected"
        )
        
        assert health.name == "database"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms == 5.5
        assert health.message == "Connected"
    
    def test_component_health_defaults(self):
        """Test component health defaults"""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY
        )
        
        assert health.latency_ms is None
        assert health.message is None
        assert health.last_check is not None
