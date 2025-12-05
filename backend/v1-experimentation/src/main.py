"""
V1 Experimentation API - Testing ground for new AI models and techniques.
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge
import structlog

from config.settings import settings
from database.connection import init_db, get_db_session
from routers import experiments, health, evaluation
from middleware.monitoring import MonitoringMiddleware

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
experiment_count = Counter(
    "v1_experiments_total",
    "Total experiments created",
    ["status"],
)

experiment_duration = Histogram(
    "v1_experiment_duration_seconds",
    "Experiment execution duration",
)

promotion_count = Counter(
    "v1_promotions_total",
    "Total promotions to V2",
    ["result"],
)

quarantine_count = Counter(
    "v1_quarantines_total",
    "Total experiments quarantined to V3",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting V1 Experimentation API", version="v1", environment=settings.environment.value)
    await init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down V1 Experimentation API")


# Create FastAPI app
app = FastAPI(
    title="AI Code Review Platform - V1 Experimentation",
    description="Experimentation zone for testing new AI models and techniques",
    version="1.0.0",
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

# Include routers
app.include_router(health.router, prefix=settings.api_prefix, tags=["health"])
app.include_router(experiments.router, prefix=settings.api_prefix, tags=["experiments"])
app.include_router(evaluation.router, prefix=settings.api_prefix, tags=["evaluation"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AI Code Review Platform",
        "version": "v1-experimentation",
        "status": "experimental",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warning": "This is an experimental version. Use V2 for production.",
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
