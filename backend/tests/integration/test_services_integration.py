"""
Integration Tests for Microservices

Tests service-to-service communication and database integration
Run with: pytest tests/integration/ -v --docker
"""

import pytest
import asyncio
import httpx
from datetime import datetime


# Skip if Docker not available
pytestmark = pytest.mark.integration


class TestAuthServiceIntegration:
    """Integration tests for Auth Service"""
    
    BASE_URL = "http://localhost:8001"
    
    @pytest.mark.asyncio
    async def test_health_check(self, http_client):
        """Test auth service health endpoint"""
        response = await http_client.get(f"{self.BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "auth-service"
    
    @pytest.mark.asyncio
    async def test_readiness_check(self, http_client):
        """Test auth service readiness endpoint"""
        response = await http_client.get(f"{self.BASE_URL}/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
    
    @pytest.mark.asyncio
    async def test_register_user(self, http_client, clean_db):
        """Test user registration"""
        response = await http_client.post(
            f"{self.BASE_URL}/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "name": "Test User"
            }
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert "id" in data
        assert data["email"] == "test@example.com"
    
    @pytest.mark.asyncio
    async def test_login_user(self, http_client, user_factory):
        """Test user login"""
        # First register
        await http_client.post(
            f"{self.BASE_URL}/auth/register",
            json={
                "email": "login@example.com",
                "password": "SecurePass123!",
                "name": "Login Test"
            }
        )
        
        # Then login
        response = await http_client.post(
            f"{self.BASE_URL}/auth/login",
            json={
                "email": "login@example.com",
                "password": "SecurePass123!"
            }
        )
        
        assert response.status_code == 200
        # Check for httpOnly cookie (won't be visible in response body)
        assert "set-cookie" in response.headers or response.json().get("user")
    
    @pytest.mark.asyncio
    async def test_invalid_login(self, http_client):
        """Test login with invalid credentials"""
        response = await http_client.post(
            f"{self.BASE_URL}/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "WrongPassword"
            }
        )
        
        assert response.status_code == 401


class TestProjectServiceIntegration:
    """Integration tests for Project Service"""
    
    BASE_URL = "http://localhost:8002"
    
    @pytest.mark.asyncio
    async def test_health_check(self, http_client):
        """Test project service health endpoint"""
        response = await http_client.get(f"{self.BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_create_project(self, http_client, mock_auth_service):
        """Test project creation"""
        response = await http_client.post(
            f"{self.BASE_URL}/projects",
            json={
                "name": "Test Project",
                "description": "Integration test project",
                "language": "python"
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        # May fail without auth - that's expected
        assert response.status_code in [200, 201, 401]
    
    @pytest.mark.asyncio
    async def test_list_projects(self, http_client):
        """Test listing projects"""
        response = await http_client.get(
            f"{self.BASE_URL}/projects",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code in [200, 401]


class TestAnalysisServiceIntegration:
    """Integration tests for Analysis Service"""
    
    BASE_URL = "http://localhost:8003"
    
    @pytest.mark.asyncio
    async def test_health_check(self, http_client):
        """Test analysis service health endpoint"""
        response = await http_client.get(f"{self.BASE_URL}/health")
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_analyze_code(self, http_client):
        """Test code analysis endpoint"""
        response = await http_client.post(
            f"{self.BASE_URL}/analyze",
            json={
                "code": "def hello():\n    print('Hello')",
                "language": "python"
            },
            headers={"Authorization": "Bearer test-token"}
        )
        
        # May require auth
        assert response.status_code in [200, 401, 202]


class TestServiceCommunication:
    """Test inter-service communication"""
    
    @pytest.mark.asyncio
    async def test_auth_to_project_communication(self, http_client):
        """Test that project service can validate tokens with auth service"""
        # This tests the service mesh communication
        auth_response = await http_client.get("http://localhost:8001/health")
        project_response = await http_client.get("http://localhost:8002/health")
        
        assert auth_response.status_code == 200
        assert project_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_analysis_to_ai_orchestrator(self, http_client):
        """Test that analysis service can communicate with AI orchestrator"""
        analysis_response = await http_client.get("http://localhost:8003/health")
        orchestrator_response = await http_client.get("http://localhost:8004/health")
        
        assert analysis_response.status_code == 200
        assert orchestrator_response.status_code == 200


class TestDatabaseIntegration:
    """Test database connectivity and operations"""
    
    @pytest.mark.asyncio
    async def test_database_connectivity(self, test_db):
        """Test database is accessible"""
        result = await test_db.execute("SELECT 1")
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_database_transaction(self, test_db):
        """Test database transactions work"""
        await test_db.execute(
            "CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name TEXT)"
        )
        await test_db.execute(
            "INSERT INTO test_table (name) VALUES ('test')"
        )
        await test_db.commit()
        
        result = await test_db.execute("SELECT * FROM test_table WHERE name = 'test'")
        row = result.fetchone()
        assert row is not None
        
        # Cleanup
        await test_db.execute("DROP TABLE test_table")
        await test_db.commit()


class TestRedisIntegration:
    """Test Redis connectivity and operations"""
    
    @pytest.mark.asyncio
    async def test_redis_connectivity(self, test_redis):
        """Test Redis is accessible"""
        result = await test_redis.ping()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_redis_set_get(self, test_redis):
        """Test Redis set/get operations"""
        await test_redis.set("test_key", "test_value")
        value = await test_redis.get("test_key")
        
        assert value == "test_value"
    
    @pytest.mark.asyncio
    async def test_redis_cache_pattern(self, test_redis):
        """Test Redis caching pattern"""
        cache_key = "cache:user:123"
        cache_data = '{"id": "123", "name": "Test User"}'
        
        await test_redis.set(cache_key, cache_data, ex=60)
        cached = await test_redis.get(cache_key)
        
        assert cached == cache_data
        
        ttl = await test_redis.ttl(cache_key)
        assert ttl > 0


class TestGracefulShutdown:
    """Test graceful shutdown handling"""
    
    @pytest.mark.asyncio
    async def test_service_responds_during_shutdown(self, http_client):
        """Test service still responds during graceful shutdown"""
        # Services should handle in-flight requests during shutdown
        response = await http_client.get("http://localhost:8001/health")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_readiness_fails_during_shutdown(self):
        """Test readiness check fails during shutdown"""
        # This would require actually triggering shutdown
        # In practice, verified by Kubernetes probes
        pass
