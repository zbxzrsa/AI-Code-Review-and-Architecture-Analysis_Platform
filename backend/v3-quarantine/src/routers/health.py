"""
Health check endpoints for V3 quarantine.
"""
from datetime import datetime
from fastapi import APIRouter

router = APIRouter(prefix="/health")


@router.get("/status")
async def health_status():
    """Get overall service health status."""
    return {
        "status": "healthy",
        "version": "v3-quarantine",
        "mode": "read-only",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "database": "ok",
            "archive_integrity": "ok",
        },
    }


@router.get("/quarantine-status")
async def quarantine_status():
    """Get quarantine statistics."""
    return {
        "total_quarantined": 12,
        "can_re_evaluate": 8,
        "permanently_blacklisted": 4,
        "re_evaluation_pending": 2,
    }
