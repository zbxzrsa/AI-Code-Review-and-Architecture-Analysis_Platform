"""
V1 Code Review AI Service

Experimental code review module with advanced analysis techniques:
- Multi-dimensional review (correctness, security, performance, etc.)
- Multiple review strategies (CoT, few-shot, ensemble)
- Hallucination detection and mitigation
- Comprehensive evaluation metrics

API Endpoints:
- /api/v1/cr-ai/review - Code review
- /api/v1/cr-ai/analysis - Advanced analysis
- /api/v1/cr-ai/metrics - Performance metrics
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .routers import review_router, analysis_router, metrics_router

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
    logger.info("Starting V1 Code Review AI Service...")
    
    # Initialize review engine
    from .review import ReviewEngine
    from .hallucination import HallucinationDetector
    
    app.state.review_engine = ReviewEngine()
    app.state.hallucination_detector = HallucinationDetector()
    
    logger.info("V1 CR-AI Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down V1 CR-AI Service...")
    logger.info("V1 CR-AI Service shutdown complete")


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="V1 Code Review AI Service",
    description="""
    ## Experimental Code Review AI
    
    Advanced code analysis with novel LLM techniques:
    
    ### Multi-Dimensional Review
    - **Correctness**: Logic errors, boundary conditions, null safety
    - **Security**: SQL injection, XSS, authentication flaws (OWASP Top 10)
    - **Performance**: Algorithmic complexity, memory efficiency
    - **Maintainability**: Code complexity, naming, documentation
    - **Architecture**: Design patterns, SOLID principles
    - **Testing**: Coverage, test quality
    
    ### Review Strategies
    - **Baseline**: Direct instruction-tuned review
    - **Chain-of-Thought**: Step-by-step reasoning
    - **Few-Shot**: In-context learning with examples
    - **Contrastive**: Compare correct vs buggy versions
    - **Ensemble**: Multiple strategies with voting
    
    ### Hallucination Detection
    - Consistency checking across multiple runs
    - Fact verification against actual code
    - Confidence scoring and filtering
    
    ### Evaluation Metrics
    - Precision, recall, F1-score
    - Per-dimension accuracy
    - Latency and throughput
    - Hallucination rate
    """,
    version="1.0.0",
    docs_url="/api/v1/cr-ai/docs",
    redoc_url="/api/v1/cr-ai/redoc",
    openapi_url="/api/v1/cr-ai/openapi.json",
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
# Health Endpoints
# =============================================================================

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "v1-cr-ai-service",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Readiness check endpoint"""
    return {
        "ready": True,
        "components": {
            "review_engine": True,
            "hallucination_detector": True,
        },
    }


@app.get("/metrics", tags=["monitoring"])
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint"""
    return {
        "v1_cr_ai_requests_total": 0,
        "v1_cr_ai_latency_seconds": 0.0,
        "v1_cr_ai_findings_total": 0,
        "v1_cr_ai_hallucinations_detected": 0,
        "v1_cr_ai_cache_hit_rate": 0.0,
    }


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(review_router, prefix="/api/v1/cr-ai")
app.include_router(analysis_router, prefix="/api/v1/cr-ai")
app.include_router(metrics_router, prefix="/api/v1/cr-ai")


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
