"""
Three-Version Evolution Service - Main Application

FastAPI application for the three-version self-evolution cycle.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8010 --reload
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from api import router, v1_router, v2_router, v3_router, get_evolution_cycle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("=" * 60)
    logger.info("THREE-VERSION EVOLUTION SERVICE STARTING")
    logger.info("=" * 60)
    logger.info("V1 (New): Experimentation, trial and error")
    logger.info("V2 (Stable): Production, fixes V1 errors, user-facing")
    logger.info("V3 (Old): Quarantine, comparison, exclusion")
    logger.info("=" * 60)
    
    # Initialize evolution cycle (but don't start automatically)
    cycle = get_evolution_cycle()
    logger.info("Evolution cycle initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down evolution service...")
    if cycle._running:
        await cycle.stop()
    logger.info("Evolution service stopped")


# Create FastAPI application
app = FastAPI(
    title="Three-Version Evolution Service",
    description="""
    ## Three-Version Self-Evolution Cycle API
    
    Manages concurrent development across three versions:
    
    ### V1 (New/Experimentation)
    - Tests new technologies with trial and error
    - Shadow traffic for testing
    - Admin access only
    
    ### V2 (Stable/Production)
    - User-facing Code Review AI
    - Fixes V1 errors and optimizes compatibility
    - Strict SLO enforcement
    
    ### V3 (Old/Quarantine)
    - Archive for failed experiments
    - Comparison baseline for V1
    - Technology exclusion decisions
    
    ### Spiral Evolution Cycle
    V1 experiments → V2 validates & fixes → promote to V2 → degrade to V3 → re-evaluate → V1
    """,
    version="1.1.0",
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

# Include API routers
app.include_router(router)
app.include_router(v1_router)
app.include_router(v2_router)
app.include_router(v3_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Three-Version Evolution Service",
        "version": "1.2.0",
        "description": "Manages V1/V2/V3 self-evolution cycle with AI endpoints",
        "docs": "/docs",
        "endpoints": {
            "evolution": {
                "status": "/api/v1/evolution/status",
                "start": "/api/v1/evolution/start",
                "stop": "/api/v1/evolution/stop",
                "v1_errors": "/api/v1/evolution/v1/errors",
                "promote": "/api/v1/evolution/promote",
                "degrade": "/api/v1/evolution/degrade",
                "reeval": "/api/v1/evolution/reeval",
                "ai_status": "/api/v1/evolution/ai/status",
                "health": "/api/v1/evolution/health",
            },
            "v1_ai": {
                "chat": "/api/v1/ai/chat",
                "analyze": "/api/v1/ai/analyze",
            },
            "v2_ai": {
                "chat": "/api/v2/ai/chat",
                "analyze": "/api/v2/ai/analyze",
                "fix": "/api/v2/ai/fix",
                "feedback": "/api/v2/ai/feedback",
            },
            "v3_ai": {
                "chat": "/api/v3/ai/chat",
                "analyze": "/api/v3/ai/analyze",
            },
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
