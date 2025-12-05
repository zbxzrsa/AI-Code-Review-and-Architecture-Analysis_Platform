"""
V3 Version Control AI Service (Quarantine/Archive)

Read-only archive of deprecated version control AI decisions.
Used for auditing past decisions and learning from failures.

Key Features:
- Read-only mode
- Decision audit trail
- Failure analysis
- Re-evaluation capability

Access Control:
- Admin/System only
- Used for quarterly reviews
"""

import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class Settings:
    service_name: str = "v3-vc-ai-service"
    version: str = "3.0.0-archived"
    environment: str = os.getenv("ENVIRONMENT", "quarantine")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8014"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    cors_origins: list = ["*"]
    read_only: bool = True
    retention_days: int = 730  # 2 years

settings = Settings()


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting V3 Version Control AI Service (Quarantine)...")
    logger.info("Mode: READ-ONLY (Archive)")
    logger.info(f"Retention: {settings.retention_days} days")
    
    # Load quarantine data
    app.state.quarantine_records = load_quarantine_records()
    app.state.failed_promotions = load_failed_promotions()
    app.state.re_evaluation_queue = []
    
    logger.info(f"Loaded {len(app.state.quarantine_records)} quarantine records")
    
    yield
    
    logger.info("Shutting down V3 VC-AI Service...")


def load_quarantine_records() -> list:
    """Load quarantine records."""
    return [
        {
            "id": "qr-001",
            "experiment_id": "exp-gpt35-security",
            "quarantined_at": "2024-01-15T10:00:00Z",
            "failure_type": "quality",
            "root_cause": "Model accuracy below 85% threshold",
            "metrics_at_failure": {
                "accuracy": 0.72,
                "error_rate": 0.08,
                "false_negatives": 0.15,
            },
            "review_scheduled_at": "2024-04-15T10:00:00Z",
            "status": "archived",
        },
        {
            "id": "qr-002",
            "experiment_id": "exp-fast-analysis",
            "quarantined_at": "2024-02-20T14:00:00Z",
            "failure_type": "operational",
            "root_cause": "Cost 3x budget with minimal accuracy improvement",
            "metrics_at_failure": {
                "accuracy": 0.88,
                "cost_per_request": 0.45,
                "latency_p95_ms": 1200,
            },
            "review_scheduled_at": "2024-05-20T14:00:00Z",
            "status": "pending_review",
        },
    ]


def load_failed_promotions() -> list:
    """Load failed promotion attempts."""
    return [
        {
            "id": "fp-001",
            "experiment_id": "exp-gpt35-security",
            "attempted_at": "2024-01-10T10:00:00Z",
            "phase_failed": "phase_1_10_percent",
            "failure_reason": "Error rate spiked to 8% during canary",
            "rollback_duration_seconds": 45,
        },
    ]


# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="V3 Version Control AI Service (Quarantine)",
    description="""
    ## Version Control Decision Archive
    
    Read-only archive of quarantined experiments and failed promotions.
    
    ### Purpose
    - **Audit Trail**: All version control decisions logged
    - **Failure Analysis**: Deep dive into why experiments failed
    - **Re-evaluation**: Queue items for V1 retry when context changes
    - **Learning**: Extract patterns from failures
    
    ### Retention
    - All records retained for 2 years
    - Quarterly reviews for potential re-evaluation
    - No automatic deletion (compliance requirement)
    
    ### Access Control
    - **Admin/System Only**: No user access
    - **Read Only**: Except for re-evaluation requests
    """,
    version=settings.version,
    docs_url="/api/v3/vc-ai/docs",
    redoc_url="/api/v3/vc-ai/redoc",
    openapi_url="/api/v3/vc-ai/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # POST only for re-evaluation requests
    allow_headers=["*"],
)


@app.middleware("http")
async def require_admin_access(request: Request, call_next):
    """Require admin access for V3."""
    if request.url.path in ["/health", "/ready"]:
        return await call_next(request)
    
    user_role = request.headers.get("X-User-Role", "user")
    
    if user_role not in ["admin", "system"]:
        return JSONResponse(
            status_code=403,
            content={
                "error": "Forbidden",
                "message": "V3 VC-AI is admin-only",
            },
        )
    
    return await call_next(request)


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.version,
        "mode": "quarantine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready", tags=["health"])
async def readiness_check(request: Request):
    """Readiness check endpoint."""
    return {
        "ready": True,
        "quarantine_records": len(getattr(request.app.state, "quarantine_records", [])),
        "pending_reviews": len([
            r for r in getattr(request.app.state, "quarantine_records", [])
            if r.get("status") == "pending_review"
        ]),
    }


# =============================================================================
# Quarantine Endpoints
# =============================================================================

@app.get("/api/v3/vc-ai/quarantine", tags=["quarantine"])
async def list_quarantine_records(
    request: Request,
    status: str = None,
    failure_type: str = None,
):
    """List quarantine records with optional filters."""
    records = request.app.state.quarantine_records
    
    if status:
        records = [r for r in records if r.get("status") == status]
    
    if failure_type:
        records = [r for r in records if r.get("failure_type") == failure_type]
    
    return {
        "records": records,
        "total": len(records),
        "filters_applied": {"status": status, "failure_type": failure_type},
    }


@app.get("/api/v3/vc-ai/quarantine/{record_id}", tags=["quarantine"])
async def get_quarantine_record(record_id: str, request: Request):
    """Get detailed quarantine record."""
    for record in request.app.state.quarantine_records:
        if record["id"] == record_id:
            return {
                "record": record,
                "age_days": (datetime.now(timezone.utc) - datetime.fromisoformat(
                    record["quarantined_at"].replace("Z", "+00:00")
                ).replace(tzinfo=None)).days,
                "retention_remaining_days": settings.retention_days - (
                    datetime.now(timezone.utc) - datetime.fromisoformat(
                        record["quarantined_at"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                ).days,
            }
    
    raise HTTPException(status_code=404, detail=f"Record {record_id} not found")


@app.get("/api/v3/vc-ai/failed-promotions", tags=["promotions"])
async def list_failed_promotions(request: Request):
    """List failed promotion attempts."""
    return {
        "promotions": request.app.state.failed_promotions,
        "total": len(request.app.state.failed_promotions),
        "note": "These promotions failed during canary deployment",
    }


@app.get("/api/v3/vc-ai/pending-reviews", tags=["reviews"])
async def list_pending_reviews(request: Request):
    """List items pending quarterly review."""
    pending = [
        r for r in request.app.state.quarantine_records
        if r.get("status") == "pending_review"
    ]
    
    return {
        "pending": pending,
        "count": len(pending),
        "next_review": "Quarterly review scheduled",
    }


@app.post("/api/v3/vc-ai/request-reevaluation", tags=["reviews"])
async def request_reevaluation(
    request: Request,
    record_id: str,
    reason: str,
):
    """Request re-evaluation of a quarantined item."""
    # Find the record
    record = None
    for r in request.app.state.quarantine_records:
        if r["id"] == record_id:
            record = r
            break
    
    if not record:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
    
    # Add to re-evaluation queue
    re_eval_request = {
        "record_id": record_id,
        "experiment_id": record["experiment_id"],
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_by": request.headers.get("X-User-Id", "admin"),
        "reason": reason,
        "status": "pending_approval",
    }
    
    request.app.state.re_evaluation_queue.append(re_eval_request)
    
    return {
        "status": "submitted",
        "request": re_eval_request,
        "note": "Request will be reviewed in next quarterly review cycle",
    }


@app.get("/api/v3/vc-ai/statistics", tags=["analytics"])
async def get_quarantine_statistics(request: Request):
    """Get quarantine statistics for analysis."""
    records = request.app.state.quarantine_records
    
    by_failure_type = {}
    for r in records:
        ft = r.get("failure_type", "unknown")
        by_failure_type[ft] = by_failure_type.get(ft, 0) + 1
    
    return {
        "total_quarantined": len(records),
        "by_failure_type": by_failure_type,
        "by_status": {
            "archived": len([r for r in records if r.get("status") == "archived"]),
            "pending_review": len([r for r in records if r.get("status") == "pending_review"]),
        },
        "failed_promotions": len(request.app.state.failed_promotions),
        "re_evaluation_requests": len(request.app.state.re_evaluation_queue),
    }


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"V3 VC-AI Error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
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
    )
