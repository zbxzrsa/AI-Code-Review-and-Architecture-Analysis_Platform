"""
Health Check Routes / 健康检查路由
"""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/")
async def root():
    """Root endpoint / 根端点"""
    return {"message": "AI Code Review Platform API"}


@router.get("/health")
async def health_check():
    """Health check endpoint / 健康检查端点"""
    return {"status": "healthy"}


@router.get("/api/health")
async def api_health_check():
    """API health check endpoint / API 健康检查端点"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "healthy",
            "database": "healthy",
            "cache": "healthy",
        }
    }
