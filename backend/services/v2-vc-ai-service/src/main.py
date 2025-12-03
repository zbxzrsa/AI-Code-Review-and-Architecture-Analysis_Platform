"""
V2 Version Control AI Service

Enterprise-grade production service for version control AI analysis.
Prioritizes reliability, regulatory compliance, and user trust.

Key Features:
- 99.99% availability SLO
- Deterministic outputs
- Multi-stage update gate
- Comprehensive audit logging
- Circuit breaker protection

API Endpoints:
- /api/v2/vc-ai/versions - Version management
- /api/v2/vc-ai/analysis - Commit analysis
- /api/v2/vc-ai/compliance - Audit and compliance
- /api/v2/vc-ai/slo - SLO monitoring
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .routers import version_router, analysis_router, compliance_router, slo_router
from .config.settings import settings

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
    logger.info("Starting V2 Version Control AI Service...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"SLO Targets: availability={settings.slo_availability}, p99_latency={settings.slo_p99_latency_ms}ms")
    
    # Initialize SLO Monitor
    from .core.slo_monitor import SLOMonitor
    app.state.slo_monitor = SLOMonitor(
        service_name=settings.service_name,
    )
    logger.info("SLO Monitor initialized")
    
    # Initialize Circuit Breaker
    from .core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    
    def on_circuit_state_change(old_state, new_state):
        logger.warning(f"Circuit breaker state changed: {old_state} -> {new_state}")
    
    app.state.circuit_breaker = CircuitBreaker(
        name="primary_model",
        config=CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout_seconds=60,
            call_timeout_seconds=settings.primary_timeout,
        ),
        on_state_change=on_circuit_state_change,
    )
    logger.info("Circuit Breaker initialized")
    
    # Initialize Model Client (if API keys available)
    from .core.model_client import ModelClient, ModelConfig, ModelProvider
    
    if settings.primary_api_key:
        primary_config = ModelConfig(
            model=settings.primary_model,
            provider=ModelProvider.OPENAI,
            api_key=settings.primary_api_key,
            api_base=settings.primary_api_base,
            timeout=settings.primary_timeout,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
        )
        
        backup_config = None
        if settings.backup_api_key:
            backup_config = ModelConfig(
                model=settings.backup_model,
                provider=ModelProvider.ANTHROPIC,
                api_key=settings.backup_api_key,
                api_base=settings.backup_api_base,
                timeout=settings.backup_timeout,
                temperature=settings.temperature,
                top_p=settings.top_p,
                top_k=settings.top_k,
            )
        
        app.state.model_client = ModelClient(
            primary_config=primary_config,
            backup_config=backup_config,
        )
        logger.info(f"Model Client initialized with {settings.primary_model}")
        
        # Initialize Analysis Engine
        from .core.analysis_engine import AnalysisEngine
        app.state.analysis_engine = AnalysisEngine(
            model_client=app.state.model_client,
            circuit_breaker=app.state.circuit_breaker,
        )
        logger.info("Analysis Engine initialized")
    else:
        logger.warning("No API keys configured, running in mock mode")
        app.state.model_client = None
        app.state.analysis_engine = None
    
    # Initialize Update Gate
    from .core.update_gate import UpdateGate
    app.state.update_gate = UpdateGate()
    logger.info("Update Gate initialized")
    
    logger.info("V2 VC-AI Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down V2 VC-AI Service...")
    
    if app.state.model_client:
        await app.state.model_client.__aexit__(None, None, None)
    
    logger.info("V2 VC-AI Service shutdown complete")


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="V2 Version Control AI Service",
    description="""
    ## Enterprise-Grade Version Control AI
    
    Production-ready AI service for version control analysis with:
    
    ### Core Features
    - **99.99% Availability SLO** - Enterprise-grade reliability
    - **Deterministic Outputs** - Same input always produces same output
    - **Multi-Model Failover** - Primary (GPT-4) + Backup (Claude 3)
    - **Circuit Breaker** - Automatic failure handling
    
    ### Version Management
    - Semantic versioning with auto-increment
    - Release note generation
    - Version comparison and diff
    - Timeline visualization
    
    ### Commit Analysis
    - Change type classification
    - Impact assessment (LOW/MEDIUM/HIGH/CRITICAL)
    - Risk evaluation
    - Breaking change detection
    - Rollback planning
    
    ### Compliance
    - SOC 2 Type II aligned
    - GDPR compliant
    - Immutable audit logging
    - Data retention policies
    
    ### SLO Monitoring
    - Real-time metrics tracking
    - Error budget calculation
    - Alert management
    - Incident reporting
    """,
    version="2.0.0",
    docs_url="/api/v2/vc-ai/docs",
    redoc_url="/api/v2/vc-ai/redoc",
    openapi_url="/api/v2/vc-ai/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# Middleware
# =============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing and SLO tracking middleware
@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    """Track request metrics for SLO monitoring"""
    import time
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    latency_ms = process_time * 1000
    
    # Add timing header
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    response.headers["X-SLO-Compliant"] = "true" if latency_ms < settings.slo_p99_latency_ms else "false"
    
    # Record metrics
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    if slo_monitor:
        success = response.status_code < 500
        await slo_monitor.record_request(
            latency_ms=latency_ms,
            success=success,
        )
    
    return response


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.version,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/ready", tags=["health"])
async def readiness_check(request: Request):
    """Readiness check endpoint"""
    circuit_breaker = getattr(request.app.state, "circuit_breaker", None)
    cb_ready = True
    if circuit_breaker:
        cb_ready = not circuit_breaker.is_open()
    
    return {
        "ready": cb_ready,
        "components": {
            "circuit_breaker": cb_ready,
            "model_client": request.app.state.model_client is not None,
            "slo_monitor": request.app.state.slo_monitor is not None,
        },
    }


@app.get("/metrics", tags=["monitoring"])
async def prometheus_metrics(request: Request):
    """Prometheus-compatible metrics endpoint"""
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    model_client = getattr(request.app.state, "model_client", None)
    analysis_engine = getattr(request.app.state, "analysis_engine", None)
    
    metrics = {
        "v2_vc_ai_requests_total": 0,
        "v2_vc_ai_latency_seconds": 0.0,
        "v2_vc_ai_availability": 0.9999,
        "v2_vc_ai_error_rate": 0.0,
        "v2_vc_ai_slo_compliant": 1,
    }
    
    if slo_monitor:
        slo_metrics = slo_monitor.get_current_metrics()
        metrics["v2_vc_ai_availability"] = slo_metrics.availability
        metrics["v2_vc_ai_error_rate"] = slo_metrics.error_rate
        metrics["v2_vc_ai_latency_p99_seconds"] = slo_metrics.latency_p99_ms / 1000
        metrics["v2_vc_ai_slo_compliant"] = 1 if slo_metrics.overall_state.value == "compliant" else 0
    
    if model_client:
        client_metrics = model_client.get_metrics()
        metrics["v2_vc_ai_requests_total"] = client_metrics["total_requests"]
        metrics["v2_vc_ai_cache_hit_rate"] = client_metrics["cache_hit_rate"]
    
    if analysis_engine:
        engine_metrics = analysis_engine.get_metrics()
        metrics["v2_vc_ai_analyses_total"] = engine_metrics["total_analyses"]
        metrics["v2_vc_ai_analysis_success_rate"] = engine_metrics["success_rate"]
    
    return metrics


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(version_router, prefix="/api/v2/vc-ai")
app.include_router(analysis_router, prefix="/api/v2/vc-ai")
app.include_router(compliance_router, prefix="/api/v2/vc-ai")
app.include_router(slo_router, prefix="/api/v2/vc-ai")


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Record error in SLO monitor
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    if slo_monitor:
        await slo_monitor.record_request(latency_ms=0, success=False)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
            "path": str(request.url),
            "timestamp": datetime.utcnow().isoformat(),
            "service": settings.service_name,
        },
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers,
    )
