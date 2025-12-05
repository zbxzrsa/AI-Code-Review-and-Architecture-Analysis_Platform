"""
AI Code Review Platform - Backend Application
AI 代码审查平台 - 后端应用

Enterprise-grade modular FastAPI application.
企业级模块化 FastAPI 应用。
"""

from .config import (
    MOCK_MODE, 
    ENVIRONMENT, 
    IS_PRODUCTION,
    IS_DEVELOPMENT,
    CORS_ORIGINS,
)

__version__ = "1.0.0"
__all__ = [
    "MOCK_MODE", 
    "ENVIRONMENT", 
    "IS_PRODUCTION",
    "IS_DEVELOPMENT",
    "CORS_ORIGINS",
]
