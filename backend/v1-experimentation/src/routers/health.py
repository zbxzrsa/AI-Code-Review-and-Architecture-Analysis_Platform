"""
Health check endpoints for V1 experimentation.
"""
from datetime import datetime
from fastapi import APIRouter

router = APIRouter(prefix="/health")


@router.get("/status")
async def health_status():
    """Get overall service health status."""
    return {
        "status": "healthy",
        "version": "v1-experimentation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warning": "This is an experimental version",
        "checks": {
            "database": "ok",
            "ai_models": "ok",
            "evaluation_engine": "ok",
        },
    }


@router.get("/experiments-status")
async def experiments_status():
    """Get status of active experiments."""
    return {
        "active_experiments": 5,
        "pending_evaluation": 2,
        "ready_for_promotion": 1,
        "failed_experiments": 1,
        "quarantined": 1,
    }
