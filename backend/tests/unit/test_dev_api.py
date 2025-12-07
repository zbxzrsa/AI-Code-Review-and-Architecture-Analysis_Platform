"""
Unit tests for the modular dev_api package.

Tests the refactored development API server structure.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Test imports work correctly
def test_dev_api_imports():
    """Test that all dev_api modules can be imported."""
    from dev_api import app, create_app
    from dev_api.config import (
        MOCK_MODE,
        ENVIRONMENT,
        IS_PRODUCTION,
        CORS_ORIGINS,
        Literals,
        Constants,
        logger,
    )
    from dev_api.models import (
        Project,
        ProjectSettings,
        DashboardMetrics,
        Activity,
    )
    from dev_api.mock_data import (
        mock_projects,
        mock_activities,
    )
    from dev_api.middleware import RequestSizeLimitMiddleware
    
    # Verify app is a FastAPI instance
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)


def test_create_app_factory():
    """Test that create_app returns a configured FastAPI instance."""
    from dev_api import create_app
    from fastapi import FastAPI
    
    app = create_app()
    assert isinstance(app, FastAPI)
    assert app.title == "Dev API Server"


def test_config_literals():
    """Test that Literals class contains expected constants."""
    from dev_api.config import Literals
    
    assert Literals.BACKEND_SERVICES == "Backend Services"
    assert Literals.PROJECT_NOT_FOUND == "Project not found"
    assert Literals.DEMO_EMAIL == "demo@example.com"
    assert Literals.GPT4_TURBO == "GPT-4 Turbo"


def test_config_constants_alias():
    """Test backward compatibility - Constants is alias for Literals."""
    from dev_api.config import Literals, Constants
    
    assert Constants is Literals


def test_mock_mode_config():
    """Test MOCK_MODE configuration."""
    from dev_api.config import MOCK_MODE
    
    # MOCK_MODE should be a boolean
    assert isinstance(MOCK_MODE, bool)


def test_cors_origins_config():
    """Test CORS_ORIGINS configuration."""
    from dev_api.config import CORS_ORIGINS, IS_PRODUCTION
    
    # In development, CORS_ORIGINS should be a list
    assert isinstance(CORS_ORIGINS, list)
    
    if not IS_PRODUCTION:
        # Should include localhost origins in development
        assert any("localhost" in origin for origin in CORS_ORIGINS)


class TestProjectModel:
    """Tests for Project model."""
    
    def test_project_creation(self):
        """Test creating a Project instance."""
        from dev_api.models import Project, ProjectSettings
        
        project = Project(
            id="test_1",
            name="Test Project",
            language="Python",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert project.id == "test_1"
        assert project.name == "Test Project"
        assert project.status == "active"  # default
        assert project.issues_count == 0  # default
    
    def test_project_with_settings(self):
        """Test Project with custom settings."""
        from dev_api.models import Project, ProjectSettings
        
        settings = ProjectSettings(
            auto_review=True,
            review_on_push=True,
            severity_threshold="error",
        )
        
        project = Project(
            id="test_2",
            name="Test Project 2",
            language="TypeScript",
            settings=settings,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert project.settings.auto_review is True
        assert project.settings.severity_threshold == "error"


class TestDashboardMetrics:
    """Tests for DashboardMetrics model."""
    
    def test_dashboard_metrics(self):
        """Test DashboardMetrics model."""
        from dev_api.models import DashboardMetrics
        
        metrics = DashboardMetrics(
            total_projects=10,
            total_analyses=50,
            issues_found=100,
            issues_resolved=80,
            resolution_rate=0.8,
        )
        
        assert metrics.total_projects == 10
        assert metrics.resolution_rate == 0.8


class TestMockData:
    """Tests for mock data."""
    
    def test_mock_projects_exist(self):
        """Test that mock projects are available."""
        from dev_api.mock_data import mock_projects
        
        assert len(mock_projects) > 0
        assert all(hasattr(p, 'id') for p in mock_projects)
    
    def test_mock_activities_exist(self):
        """Test that mock activities are available."""
        from dev_api.mock_data import mock_activities
        
        assert len(mock_activities) > 0


class TestRoutes:
    """Tests for route modules."""
    
    def test_routes_import(self):
        """Test that all route modules can be imported."""
        from dev_api.routes import (
            dashboard_router,
            projects_router,
            analysis_router,
            oauth_router,
            users_router,
            admin_router,
            three_version_router,
            security_router,
            reports_router,
        )
        
        from fastapi import APIRouter
        
        assert isinstance(dashboard_router, APIRouter)
        assert isinstance(projects_router, APIRouter)
        assert isinstance(admin_router, APIRouter)


class TestMiddleware:
    """Tests for middleware."""
    
    def test_request_size_limit_middleware(self):
        """Test RequestSizeLimitMiddleware exists."""
        from dev_api.middleware import RequestSizeLimitMiddleware
        from starlette.middleware.base import BaseHTTPMiddleware
        
        assert issubclass(RequestSizeLimitMiddleware, BaseHTTPMiddleware)


# Async tests using pytest-asyncio
@pytest.mark.asyncio
async def test_app_health_endpoint():
    """Test the /health endpoint."""
    from dev_api import app
    from httpx import AsyncClient, ASGITransport
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_app_root_endpoint():
    """Test the root endpoint."""
    from dev_api import app
    from httpx import AsyncClient, ASGITransport
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Dev API Server"
        assert data["status"] == "running"


@pytest.mark.asyncio
async def test_dashboard_metrics_endpoint():
    """Test dashboard metrics endpoint."""
    from dev_api import app
    from httpx import AsyncClient, ASGITransport
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/metrics/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "total_projects" in data
        assert "resolution_rate" in data


@pytest.mark.asyncio
async def test_projects_list_endpoint():
    """Test projects list endpoint."""
    from dev_api import app
    from httpx import AsyncClient, ASGITransport
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data


class TestBackwardCompatibility:
    """Tests for backward compatibility."""
    
    def test_entry_point_import(self):
        """Test that dev-api-server.py entry point works."""
        import sys
        import os
        
        # Add backend to path
        backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        # This should work without errors
        from dev_api import app
        assert app is not None
    
    def test_constants_backward_compat(self):
        """Test Constants alias for backward compatibility."""
        from dev_api.config import Constants, Literals
        
        # Constants should be the same as Literals
        assert Constants.DEMO_EMAIL == Literals.DEMO_EMAIL
        assert Constants.GPT4_TURBO == Literals.GPT4_TURBO
