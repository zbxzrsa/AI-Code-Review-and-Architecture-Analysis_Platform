"""
V2 Production API - Stable code review service for end users.
"""
import logging
from contextlib import asynccontextmanager
from datetime, timezone import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge
import structlog

from config.settings import settings
from database.connection import init_db, get_db_session
from routers import code_review, health, metrics
from middleware.monitoring import MonitoringMiddleware
from middleware.slo import SLOMiddleware

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
request_count = Counter(
    "v2_requests_total",
    "Total requests to V2 API",
    ["method", "endpoint", "status"],
)

request_duration = Histogram(
    "v2_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
)

active_requests = Gauge(
    "v2_active_requests",
    "Number of active requests",
)

error_count = Counter(
    "v2_errors_total",
    "Total errors in V2 API",
    ["error_type"],
)

slo_violations = Counter(
    "v2_slo_violations_total",
    "Total SLO violations",
    ["metric"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting V2 Production API", version="v2", environment=settings.environment.value)
    await init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down V2 Production API")


# Create FastAPI app
app = FastAPI(
    title="AI Code Review Platform - V2 Production",
    description="Stable production API for code review and architecture analysis",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(MonitoringMiddleware)
app.add_middleware(SLOMiddleware)

# Include routers
app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(code_review.router, prefix=settings.api_prefix, tags=["code-review"])
app.include_router(metrics.router, prefix=settings.api_prefix, tags=["metrics"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AI Code Review Platform",
        "version": "v2-production",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe."""
    try:
        session = await get_db_session()
        await session.execute("SELECT 1")
        return {"status": "ready"}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    error_type = type(exc).__name__
    error_count.labels(error_type=error_type).inc()

    logger.error(
        "Unhandled exception",
        error_type=error_type,
        error_message=str(exc),
        path=request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_id": str(datetime.now(timezone.utc).timestamp()),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
