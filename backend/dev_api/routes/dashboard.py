"""
Dashboard Routes

Dashboard metrics and activity endpoints.
"""

import random
from fastapi import APIRouter, Query
from ..models import DashboardMetrics
from ..mock_data import mock_projects, mock_activities

router = APIRouter(prefix="/api", tags=["Dashboard"])


@router.get("/metrics/dashboard")
async def get_dashboard_metrics():
    """Get dashboard metrics."""
    return DashboardMetrics(
        total_projects=len(mock_projects),
        total_analyses=47,
        issues_found=156,
        issues_resolved=131,
        resolution_rate=0.84
    )


@router.get("/metrics/system")
async def get_system_metrics():
    """Get system metrics."""
    return {
        "cpu_usage": random.uniform(20, 60),
        "memory_usage": random.uniform(40, 70),
        "disk_usage": random.uniform(30, 50),
        "active_users": random.randint(5, 20)
    }


@router.get("/activity")
async def get_activity(limit: int = Query(10, ge=1, le=50)):
    """Get recent activity."""
    return {
        "items": [a.dict() for a in mock_activities[:limit]],
        "total": len(mock_activities)
    }
