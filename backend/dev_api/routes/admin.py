"""
Admin Routes

Administrative endpoints for user, project, and system management.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException
from ..mock_data import mock_users, mock_projects, mock_providers, mock_experiments
from ..config import Literals

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ============================================
# User Management
# ============================================

@router.get("/users")
async def list_users(page: int = 1, limit: int = 10, search: Optional[str] = None):
    """List all users."""
    users = mock_users
    if search:
        users = [u for u in users if search.lower() in u["name"].lower() or search.lower() in u["email"].lower()]
    
    return {
        "items": users,
        "total": len(users),
        "page": page,
        "limit": limit
    }


@router.get("/users/stats")
async def get_user_stats():
    """Get user statistics."""
    return {
        "total_users": len(mock_users),
        "active_users": sum(1 for u in mock_users if u["status"] == "active"),
        "new_users_today": 3,
        "new_users_week": 12,
    }


@router.post("/users")
async def create_user():
    """Create new user."""
    return {
        "id": f"user_{secrets.token_hex(4)}",
        "message": "User created"
    }


@router.put("/users/{user_id}")
async def update_user(user_id: str):
    """Update user."""
    return {"message": f"User {user_id} updated"}


@router.post("/users/{user_id}/suspend")
async def suspend_user(user_id: str):
    """Suspend user."""
    return {"message": f"User {user_id} suspended"}


@router.post("/users/{user_id}/reactivate")
async def reactivate_user(user_id: str):
    """Reactivate user."""
    return {"message": f"User {user_id} reactivated"}


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(user_id: str):
    """Reset user password."""
    return {"message": f"Password reset email sent to user {user_id}"}


@router.delete("/users/{user_id}")
async def deactivate_user(user_id: str):
    """Deactivate user."""
    return {"message": f"User {user_id} deactivated"}


# ============================================
# Project Management
# ============================================

@router.get("/projects")
async def admin_list_projects(page: int = 1, limit: int = 10, search: Optional[str] = None):
    """List all projects (admin)."""
    projects = [p.dict() for p in mock_projects]
    return {
        "items": projects,
        "total": len(projects),
        "page": page,
        "limit": limit
    }


@router.delete("/projects/{project_id}")
async def admin_delete_project(project_id: str):
    """Delete project (admin)."""
    return {"message": f"Project {project_id} deleted"}


# ============================================
# Audit Logs
# ============================================

@router.get("/audit")
async def get_audit_logs(page: int = 1, limit: int = 20):
    """Get audit logs."""
    # Cap limit to prevent resource exhaustion from user-controlled input
    MAX_LIMIT = 100
    safe_limit = min(max(1, limit), MAX_LIMIT)
    logs = [
        {
            "id": f"log_{i}",
            "action": ["user.login", "project.create", "analysis.start", "settings.update"][i % 4],
            "user_id": "user_1",
            "user_email": Literals.JOHN_EMAIL,
            "ip_address": "192.168.1.1",
            "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
            "details": {"browser": "Chrome", "os": "Windows"},
        }
        for i in range(safe_limit)
    ]
    return {"items": logs, "total": 100, "page": page, "limit": safe_limit}


@router.get("/audit/analytics")
async def get_audit_analytics():
    """Get audit analytics."""
    return {
        "total_events": 1234,
        "events_today": 45,
        "top_actions": [
            {"action": "user.login", "count": 234},
            {"action": "analysis.start", "count": 189},
            {"action": "project.update", "count": 123},
        ],
        "by_user": [
            {"user_id": "user_1", "count": 456},
            {"user_id": "user_2", "count": 234},
        ]
    }


@router.get("/audit/security-alerts")
async def get_security_alerts():
    """Get security alerts."""
    return {
        "items": [
            {
                "id": "alert_1",
                "type": "failed_login",
                "severity": "medium",
                "message": "Multiple failed login attempts",
                "timestamp": datetime.now().isoformat(),
            }
        ],
        "total": 1
    }


# ============================================
# AI Providers
# ============================================

@router.get("/providers")
async def list_providers():
    """List AI providers."""
    return {"items": mock_providers}


@router.get("/providers/{provider_id}")
async def get_provider(provider_id: str):
    """Get provider details."""
    for provider in mock_providers:
        if provider["id"] == provider_id:
            return provider
    raise HTTPException(status_code=404, detail="Provider not found")


@router.get("/providers/{provider_id}/models")
async def get_provider_models(provider_id: str):
    """Get provider models."""
    for provider in mock_providers:
        if provider["id"] == provider_id:
            return {"models": provider["models"]}
    raise HTTPException(status_code=404, detail="Provider not found")


@router.get("/providers/{provider_id}/metrics")
async def get_provider_metrics(provider_id: str):
    """Get provider metrics."""
    return {
        "requests_total": 1234,
        "requests_today": 45,
        "avg_latency_ms": 250,
        "error_rate": 0.02,
        "cost_today": 12.50,
    }


@router.post("/providers/{provider_id}/test")
async def test_provider(provider_id: str):
    """Test provider connection."""
    return {
        "status": "success",
        "latency_ms": 234,
        "message": "Connection successful"
    }


@router.put("/providers/{provider_id}")
async def update_provider(provider_id: str):
    """Update provider configuration."""
    return {"message": f"Provider {provider_id} updated"}


# ============================================
# Experiments
# ============================================

@router.get("/experiments")
async def list_experiments():
    """List experiments."""
    return {"items": mock_experiments}


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details."""
    for exp in mock_experiments:
        if exp["id"] == experiment_id:
            return exp
    raise HTTPException(status_code=404, detail="Experiment not found")


@router.post("/experiments")
async def create_experiment():
    """Create new experiment."""
    return {
        "id": f"exp_{secrets.token_hex(4)}",
        "message": "Experiment created"
    }


@router.put("/experiments/{experiment_id}")
async def update_experiment(experiment_id: str):
    """Update experiment."""
    return {"message": f"Experiment {experiment_id} updated"}


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete experiment."""
    return {"message": f"Experiment {experiment_id} deleted"}
