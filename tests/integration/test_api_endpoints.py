"""
Comprehensive Integration Tests for API Endpoints

Tests the complete request-response cycle including:
- Authentication and authorization
- Database operations
- Redis caching
- AI provider integration
- Error handling
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, Mock, patch
from httpx import AsyncClient


# ============================================================================
# Auth Endpoint Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_db
class TestAuthEndpoints:
    """Integration tests for authentication endpoints."""
    
    async def test_login_success(
        self,
        async_client: AsyncClient,
        test_user_in_db: Dict[str, Any],
    ):
        """Test successful login flow."""
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": test_user_in_db["email"],
                "password": "testpassword123",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["email"] == test_user_in_db["email"]
    
    async def test_login_invalid_credentials(
        self,
        async_client: AsyncClient,
    ):
        """Test login with invalid credentials."""
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "wrongpassword",
            }
        )
        
        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]
    
    async def test_login_rate_limiting(
        self,
        async_client: AsyncClient,
    ):
        """Test rate limiting on login attempts."""
        # Make multiple failed login attempts
        for _ in range(10):
            await async_client.post(
                "/api/auth/login",
                json={
                    "email": "test@example.com",
                    "password": "wrongpassword",
                }
            )
        
        # Should be rate limited now
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": "test@example.com",
                "password": "wrongpassword",
            }
        )
        
        assert response.status_code == 429
    
    async def test_token_refresh(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
    ):
        """Test token refresh flow."""
        response = await async_client.post(
            "/api/auth/refresh",
            headers={"Authorization": f"Bearer {authenticated_user['refresh_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
    
    async def test_logout(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
    ):
        """Test logout flow."""
        response = await async_client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 200
        
        # Token should no longer work
        response = await async_client.get(
            "/api/user/me",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 401


# ============================================================================
# Project Endpoint Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_db
class TestProjectEndpoints:
    """Integration tests for project management endpoints."""
    
    async def test_create_project(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
    ):
        """Test project creation."""
        response = await async_client.post(
            "/api/projects",
            json={
                "name": "integration-test-project",
                "description": "Created by integration test",
                "language": "python",
            },
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "integration-test-project"
        assert "id" in data
    
    async def test_list_projects(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        test_project_in_db: Dict[str, Any],
    ):
        """Test listing user projects."""
        response = await async_client.get(
            "/api/projects",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
    
    async def test_get_project_details(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        test_project_in_db: Dict[str, Any],
    ):
        """Test getting project details."""
        response = await async_client.get(
            f"/api/projects/{test_project_in_db['id']}",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_project_in_db["id"]
    
    async def test_update_project(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        test_project_in_db: Dict[str, Any],
    ):
        """Test updating project."""
        response = await async_client.put(
            f"/api/projects/{test_project_in_db['id']}",
            json={
                "description": "Updated description",
            },
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated description"
    
    async def test_delete_project(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        test_project_in_db: Dict[str, Any],
    ):
        """Test deleting project."""
        response = await async_client.delete(
            f"/api/projects/{test_project_in_db['id']}",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 204
        
        # Verify deletion
        response = await async_client.get(
            f"/api/projects/{test_project_in_db['id']}",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 404
    
    async def test_project_access_control(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        other_user_project: Dict[str, Any],
    ):
        """Test that users cannot access other users' projects."""
        response = await async_client.get(
            f"/api/projects/{other_user_project['id']}",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 403


# ============================================================================
# Analysis Endpoint Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_services
class TestAnalysisEndpoints:
    """Integration tests for code analysis endpoints."""
    
    async def test_submit_analysis(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        test_project_in_db: Dict[str, Any],
        sample_code: str,
    ):
        """Test submitting code for analysis."""
        response = await async_client.post(
            f"/api/projects/{test_project_in_db['id']}/analyze",
            json={
                "code": sample_code,
                "filename": "test.py",
                "language": "python",
            },
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "pending"
    
    async def test_get_analysis_status(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        test_analysis_in_db: Dict[str, Any],
    ):
        """Test getting analysis status."""
        response = await async_client.get(
            f"/api/analysis/{test_analysis_in_db['id']}/status",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["pending", "running", "completed", "failed"]
    
    async def test_get_analysis_results(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        completed_analysis_in_db: Dict[str, Any],
    ):
        """Test getting completed analysis results."""
        response = await async_client.get(
            f"/api/analysis/{completed_analysis_in_db['id']}",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "issues" in data
        assert "suggestions" in data
        assert "metrics" in data
    
    async def test_analysis_caching(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        test_project_in_db: Dict[str, Any],
        sample_code: str,
    ):
        """Test that identical code submissions use cache."""
        headers = {"Authorization": f"Bearer {authenticated_user['access_token']}"}
        
        # First submission
        response1 = await async_client.post(
            f"/api/projects/{test_project_in_db['id']}/analyze",
            json={"code": sample_code, "filename": "test.py"},
            headers=headers
        )
        
        # Wait for completion
        await asyncio.sleep(1)
        
        # Second identical submission
        response2 = await async_client.post(
            f"/api/projects/{test_project_in_db['id']}/analyze",
            json={"code": sample_code, "filename": "test.py"},
            headers=headers
        )
        
        # Should return cached result
        data2 = response2.json()
        assert data2.get("cached", False) is True


# ============================================================================
# Three-Version Cycle Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_services
class TestThreeVersionCycleEndpoints:
    """Integration tests for three-version evolution cycle."""
    
    async def test_get_cycle_status(
        self,
        async_client: AsyncClient,
        admin_user: Dict[str, Any],
    ):
        """Test getting evolution cycle status."""
        response = await async_client.get(
            "/api/three-version/status",
            headers={"Authorization": f"Bearer {admin_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "v1" in data
        assert "v2" in data
        assert "v3" in data
    
    async def test_promote_experiment(
        self,
        async_client: AsyncClient,
        admin_user: Dict[str, Any],
        v1_experiment_in_db: Dict[str, Any],
    ):
        """Test promoting V1 experiment to V2."""
        response = await async_client.post(
            f"/api/three-version/promote/{v1_experiment_in_db['id']}",
            headers={"Authorization": f"Bearer {admin_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["promoted"] is True
    
    async def test_quarantine_version(
        self,
        async_client: AsyncClient,
        admin_user: Dict[str, Any],
        v2_version_in_db: Dict[str, Any],
    ):
        """Test quarantining V2 to V3."""
        response = await async_client.post(
            f"/api/three-version/demote/{v2_version_in_db['id']}",
            json={"reason": "Failed SLO requirements"},
            headers={"Authorization": f"Bearer {admin_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["quarantined"] is True


# ============================================================================
# Provider Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_services
class TestProviderEndpoints:
    """Integration tests for AI provider management."""
    
    async def test_list_providers(
        self,
        async_client: AsyncClient,
        admin_user: Dict[str, Any],
    ):
        """Test listing available providers."""
        response = await async_client.get(
            "/api/admin/providers",
            headers={"Authorization": f"Bearer {admin_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    async def test_provider_health_check(
        self,
        async_client: AsyncClient,
        admin_user: Dict[str, Any],
    ):
        """Test provider health check endpoint."""
        response = await async_client.get(
            "/api/admin/providers/health",
            headers={"Authorization": f"Bearer {admin_user['access_token']}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        
        for provider in data["providers"]:
            assert "name" in provider
            assert "status" in provider
            assert provider["status"] in ["healthy", "degraded", "unhealthy"]


# ============================================================================
# Database Transaction Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_db
class TestDatabaseTransactions:
    """Integration tests for database transaction handling."""
    
    async def test_transaction_rollback_on_error(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        db_session,
    ):
        """Test that transactions are rolled back on error."""
        # Start a transaction that will fail
        initial_count = await db_session.execute(
            "SELECT COUNT(*) FROM projects WHERE owner_id = $1",
            [authenticated_user["id"]]
        )
        
        # Try to create project with invalid data (should fail)
        response = await async_client.post(
            "/api/projects",
            json={
                "name": "",  # Invalid: empty name
                "description": "Test",
            },
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        assert response.status_code == 422
        
        # Verify no project was created
        final_count = await db_session.execute(
            "SELECT COUNT(*) FROM projects WHERE owner_id = $1",
            [authenticated_user["id"]]
        )
        
        assert initial_count == final_count
    
    async def test_concurrent_updates(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        test_project_in_db: Dict[str, Any],
    ):
        """Test handling of concurrent updates."""
        headers = {"Authorization": f"Bearer {authenticated_user['access_token']}"}
        
        # Simulate concurrent updates
        tasks = [
            async_client.put(
                f"/api/projects/{test_project_in_db['id']}",
                json={"description": f"Update {i}"},
                headers=headers
            )
            for i in range(5)
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least one should succeed
        success_count = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        assert success_count >= 1


# ============================================================================
# Redis Cache Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.requires_redis
class TestRedisCacheIntegration:
    """Integration tests for Redis caching."""
    
    async def test_session_caching(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        redis_client,
    ):
        """Test that sessions are properly cached."""
        # Make authenticated request
        await async_client.get(
            "/api/user/me",
            headers={"Authorization": f"Bearer {authenticated_user['access_token']}"}
        )
        
        # Verify session is in Redis
        session_key = f"session:{authenticated_user['id']}"
        session_data = await redis_client.get(session_key)
        
        assert session_data is not None
    
    async def test_cache_invalidation(
        self,
        async_client: AsyncClient,
        authenticated_user: Dict[str, Any],
        test_project_in_db: Dict[str, Any],
        redis_client,
    ):
        """Test that cache is invalidated on updates."""
        headers = {"Authorization": f"Bearer {authenticated_user['access_token']}"}
        
        # Get project (should cache)
        await async_client.get(
            f"/api/projects/{test_project_in_db['id']}",
            headers=headers
        )
        
        # Update project
        await async_client.put(
            f"/api/projects/{test_project_in_db['id']}",
            json={"description": "Updated"},
            headers=headers
        )
        
        # Get again - should have new data
        response = await async_client.get(
            f"/api/projects/{test_project_in_db['id']}",
            headers=headers
        )
        
        assert response.json()["description"] == "Updated"
