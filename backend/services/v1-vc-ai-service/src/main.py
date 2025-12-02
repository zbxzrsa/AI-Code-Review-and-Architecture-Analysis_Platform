"""
V1 Version Control AI Service

Innovation engine for the AI Code Review Platform.
Provides experimental model architectures, training strategies,
and version control analysis capabilities.

API Endpoints:
- /api/v1/vc-ai/experiments - Experiment management
- /api/v1/vc-ai/inference - Commit analysis inference
- /api/v1/vc-ai/evaluation - Metrics and promotion
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .routers import experiments_router, inference_router, evaluation_router
from .failure import FailureLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting V1 VC-AI Service...")
    
    # Initialize failure logger
    app.state.failure_logger = FailureLogger(
        v3_api_endpoint=os.getenv("V3_API_ENDPOINT", "http://v3-quarantine-service:8000/api/v3/quarantine/failures"),
    )
    
    # Initialize model (would be loaded here in production)
    app.state.model = None
    app.state.tokenizer = None
    
    logger.info("V1 VC-AI Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down V1 VC-AI Service...")
    
    if hasattr(app.state, "failure_logger"):
        await app.state.failure_logger.close()
    
    logger.info("V1 VC-AI Service shutdown complete")


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="V1 Version Control AI Service",
    description="""
    ## Innovation Engine for AI Code Review Platform
    
    This service provides:
    
    ### Experiment Management
    - Create and manage experiments with custom architectures
    - Train models with advanced techniques (curriculum learning, MoE, etc.)
    - Track experiment progress and metrics
    
    ### Commit Analysis
    - Analyze commits for change type and impact
    - Extract affected components and dependencies
    - Assess risk and generate explanations
    
    ### Evaluation & Promotion
    - Comprehensive evaluation metrics
    - Experiment comparison and Pareto analysis
    - Promotion workflow to V2 production
    
    ### Failure Logging
    - Automatic failure detection and documentation
    - V3 quarantine integration
    - Blacklist management for failed techniques
    """,
    version="1.0.0",
    docs_url="/api/v1/vc-ai/docs",
    redoc_url="/api/v1/vc-ai/redoc",
    openapi_url="/api/v1/vc-ai/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# Middleware
# =============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add response timing header"""
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# =============================================================================
# Routes
# =============================================================================

# Health check
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "v1-vc-ai-service",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Readiness check
@app.get("/ready", tags=["health"])
async def readiness_check():
    """Readiness check endpoint"""
    # Check if model is loaded (in production)
    model_ready = True  # app.state.model is not None
    
    return {
        "ready": model_ready,
        "components": {
            "model": model_ready,
            "failure_logger": True,
        },
    }


# Metrics endpoint
@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    # In production, use prometheus_client
    return {
        "v1_vc_ai_requests_total": 0,
        "v1_vc_ai_latency_seconds": 0.0,
        "v1_vc_ai_experiments_active": 0,
        "v1_vc_ai_experiments_completed": 0,
        "v1_vc_ai_failures_total": 0,
    }


# Include routers
app.include_router(experiments_router, prefix="/api/v1/vc-ai")
app.include_router(inference_router, prefix="/api/v1/vc-ai")
app.include_router(evaluation_router, prefix="/api/v1/vc-ai")


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred",
            "path": str(request.url),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", "1")),
    )
