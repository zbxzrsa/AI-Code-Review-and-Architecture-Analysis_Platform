"""
API Routes / API 路由

This module exports all route handlers for the API.
此模块导出所有 API 路由处理器。
"""

from .health import router as health_router
from .projects import router as projects_router
from .admin import router as admin_router
from .oauth import router as oauth_router
from .analysis import router as analysis_router
from .user import router as user_router

__all__ = [
    "health_router",
    "projects_router", 
    "admin_router",
    "oauth_router",
    "analysis_router",
    "user_router",
]
