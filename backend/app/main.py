"""
Main Application Entry Point / 主应用入口
AI Code Review Platform - FastAPI Application

This is the modular version of the API server.
这是模块化版本的 API 服务器。

Usage:
    uvicorn app.main:app --reload --port 8000
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import (
    MOCK_MODE, ENVIRONMENT, HOST, PORT, 
    CORS_ORIGINS, CORS_METHODS, CORS_HEADERS,
    IS_PRODUCTION, MAX_REQUEST_SIZE_BYTES, MAX_REQUEST_SIZE_MB,
    LOG_LEVEL, LOG_FORMAT
)
from .routes import (
    health_router,
    projects_router,
    admin_router,
    oauth_router,
    analysis_router,
    user_router,
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ============================================
# Request Size Limiting Middleware
# ============================================
class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""
    
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        
        if content_length and int(content_length) > MAX_REQUEST_SIZE_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large. Maximum size is {MAX_REQUEST_SIZE_MB}MB"}
            )
        
        return await call_next(request)


# ============================================
# Create FastAPI Application
# ============================================
app = FastAPI(
    title="AI Code Review Platform API",
    description="Enterprise-grade AI-powered code review platform",
    version="1.0.0",
    docs_url="/docs" if not IS_PRODUCTION else None,  # Disable docs in production
    redoc_url="/redoc" if not IS_PRODUCTION else None,
    openapi_url="/openapi.json" if not IS_PRODUCTION else None,
)

# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add request size limiting
app.add_middleware(RequestSizeLimitMiddleware)

# Configure CORS with secure settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# Include routers
app.include_router(health_router)
app.include_router(projects_router)
app.include_router(admin_router)
app.include_router(oauth_router)
app.include_router(analysis_router)
app.include_router(user_router)


@app.on_event("startup")
async def startup_event():
    """Application startup / 应用启动"""
    logger.info("=" * 60)
    logger.info("🚀 AI Code Review Platform - Modular API Server")
    logger.info("=" * 60)
    logger.info(f"🔧 Environment: {ENVIRONMENT}")
    logger.info(f"🎭 Mock Mode: {'ENABLED' if MOCK_MODE else 'DISABLED'}")
    logger.info(f"🔒 Production Mode: {'YES' if IS_PRODUCTION else 'NO'}")
    logger.info(f"🌐 CORS Origins: {len(CORS_ORIGINS)} configured")
    logger.info("=" * 60)
    
    if not IS_PRODUCTION:
        logger.info(f"📖 API Docs:  http://{HOST}:{PORT}/docs")
    
    logger.info(f"❤️  Health:    http://{HOST}:{PORT}/health")
    logger.info("=" * 60)
    
    if IS_PRODUCTION:
        logger.warning("⚠️  Running in PRODUCTION mode - API docs disabled")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
