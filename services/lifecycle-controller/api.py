"""
Lifecycle Controller API

REST API for managing the three-version self-evolution cycle.
Provides endpoints for:
- Cycle status and monitoring
- Version registration and management
- Recovery operations
- Manual interventions
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .controller import LifecycleController, VersionState, PromotionThresholds
from .recovery_manager import RecoveryManager, RecoveryConfig
from .cycle_orchestrator import CycleOrchestrator
from .metrics_collector import CycleMetricsCollector
from .event_publisher import EventPublisher, EventType, InMemoryEventBackend

logger = logging.getLogger(__name__)

# ==================== Constants ====================
ORCHESTRATOR_NOT_INITIALIZED = "Orchestrator not initialized"

# ==================== Request/Response Models ====================

class VersionRegistration(BaseModel):
    version_id: str
    model_version: str
    prompt_version: str
    metadata: Dict[str, Any] = {}


class QuarantineRequest(BaseModel):
    version_id: str
    reason: str
    metrics: Dict[str, Any] = {}


class ThresholdUpdate(BaseModel):
    p95_latency_ms: Optional[float] = None
    error_rate: Optional[float] = None
    accuracy_delta: Optional[float] = None
    security_pass_rate: Optional[float] = None
    cost_increase_max: Optional[float] = None


class RecoveryConfigUpdate(BaseModel):
    initial_cooldown_hours: Optional[int] = None
    max_recovery_attempts: Optional[int] = None
    gold_set_accuracy_threshold: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    cycle_active: bool
    components: Dict[str, str]
    timestamp: str


class CycleStatusResponse(BaseModel):
    cycle_active: bool
    versions: Dict[str, int]
    recovery_stats: Dict[str, Any]
    recent_events: List[Dict[str, Any]]
    cycle_health: Dict[str, bool]


# ==================== Application Setup ====================

# Global instances
orchestrator: Optional[CycleOrchestrator] = None
metrics_collector: Optional[CycleMetricsCollector] = None
event_publisher: Optional[EventPublisher] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global orchestrator, metrics_collector, event_publisher
    
    # Initialize metrics collector
    metrics_collector = CycleMetricsCollector()
    
    # Initialize event publisher with in-memory backend
    event_publisher = EventPublisher()
    event_publisher.add_backend(InMemoryEventBackend())
    await event_publisher.start()
    
    # Initialize components
    lifecycle_controller = LifecycleController()
    recovery_manager = RecoveryManager()
    orchestrator = CycleOrchestrator(lifecycle_controller, recovery_manager)
    
    # Start the cycle
    await orchestrator.start()
    logger.info("Self-evolution cycle started")
    
    yield
    
    # Shutdown
    await orchestrator.stop()
    await event_publisher.stop()
    logger.info("Self-evolution cycle stopped")


app = FastAPI(
    title="Lifecycle Controller API",
    description="API for managing the three-version self-evolution cycle",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if orchestrator and orchestrator._running else "unhealthy",
        cycle_active=orchestrator._running if orchestrator else False,
        components={
            "lifecycle_controller": "healthy",
            "recovery_manager": "healthy",
            "cycle_orchestrator": "healthy",
        },
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if not orchestrator or not orchestrator._running:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"ready": True}


# ==================== Cycle Status Endpoints ====================

@app.get("/cycle/status", response_model=CycleStatusResponse)
async def get_cycle_status():
    """Get current status of the self-evolution cycle"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    status = orchestrator.get_cycle_status()
    return CycleStatusResponse(**status)


@app.get("/cycle/events")
async def get_cycle_events(
    version_id: Optional[str] = None,
    limit: int = 50
):
    """Get recent cycle events"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    events = orchestrator.get_cycle_events(version_id=version_id, limit=limit)
    return {"events": [e.__dict__ for e in events]}


@app.get("/cycle/diagram")
async def get_cycle_diagram():
    """Get ASCII diagram of the current cycle state"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    status = orchestrator.get_cycle_status()
    v = status["versions"]
    
    diagram = f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║               THREE-VERSION SELF-EVOLUTION CYCLE                   ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║   ┌──────────────┐         Shadow        ┌──────────────┐         ║
    ║   │     V1       │ ────────Traffic────►  │     V2       │         ║
    ║   │  EXPERIMENT  │                       │  PRODUCTION  │         ║
    ║   │   ({v1_exp:^3})      │                       │   ({v2_prod:^3})      │         ║
    ║   └──────▲───────┘                       └──────┬───────┘         ║
    ║          │                                      │                 ║
    ║          │ Recovery                   Demotion  │                 ║
    ║          │ (Gold-set)                 (SLO fail)│                 ║
    ║          │                                      │                 ║
    ║   ┌──────┴───────┐                       ┌──────▼───────┐         ║
    ║   │     V3       │ ◄──────────────────── │   ROLLBACK   │         ║
    ║   │  QUARANTINE  │                       │              │         ║
    ║   │   ({v3_quar:^3})      │                       └──────────────┘         ║
    ║   └──────────────┘                                                ║
    ║                                                                    ║
    ║   Gray-Scale: {v2_gray:^3} | Recovering: {v3_recov:^3}                           ║
    ║                                                                    ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """.format(
        v1_exp=v["v1_experiments"],
        v2_prod=v["v2_production"],
        v2_gray=v["v2_gray_scale"],
        v3_quar=v["v3_quarantined"],
        v3_recov=v["v3_recovering"]
    )
    
    return {"diagram": diagram, "versions": v}


# ==================== Version Management Endpoints ====================

@app.get("/versions")
async def list_versions():
    """List all versions in the cycle"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    versions = []
    for vid, config in orchestrator.lifecycle.active_versions.items():
        versions.append({
            "version_id": vid,
            "model_version": config.model_version,
            "prompt_version": config.prompt_version,
            "state": config.current_state.value,
            "created_at": config.created_at.isoformat(),
            "last_evaluation": config.last_evaluation.isoformat() if config.last_evaluation else None,
            "consecutive_failures": config.consecutive_failures,
            "metadata": config.metadata
        })
    
    return {"versions": versions, "total": len(versions)}


@app.post("/versions/register")
async def register_version(registration: VersionRegistration):
    """Register a new experiment (enters V1)"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    config = await orchestrator.register_new_experiment(
        version_id=registration.version_id,
        model_version=registration.model_version,
        prompt_version=registration.prompt_version,
        metadata=registration.metadata
    )
    
    return {
        "success": True,
        "message": f"Version {registration.version_id} registered in V1 (Experiment)",
        "version": {
            "version_id": config.version_id,
            "state": config.current_state.value
        }
    }


@app.post("/versions/{version_id}/start-shadow")
async def start_shadow_evaluation(version_id: str):
    """Start shadow traffic evaluation for a version"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    try:
        await orchestrator.start_shadow_evaluation(version_id)
        return {
            "success": True,
            "message": f"Shadow evaluation started for {version_id}"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/versions/{version_id}/quarantine")
async def quarantine_version(version_id: str, request: QuarantineRequest):
    """Manually quarantine a version (sends to V3)"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    await orchestrator.trigger_quarantine(
        version_id=version_id,
        reason=request.reason,
        metrics=request.metrics
    )
    
    return {
        "success": True,
        "message": f"Version {version_id} quarantined: {request.reason}"
    }


# ==================== Recovery Endpoints ====================

@app.get("/recovery/status")
async def get_recovery_status():
    """Get recovery manager status"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    stats = orchestrator.recovery.get_recovery_statistics()
    return {"recovery_statistics": stats}


@app.get("/recovery/{version_id}")
async def get_version_recovery_status(version_id: str):
    """Get recovery status for a specific version"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    record = orchestrator.recovery.get_recovery_status(version_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No recovery record for {version_id}")
    
    return {
        "version_id": record.version_id,
        "status": record.recovery_status.value,
        "quarantine_time": record.quarantine_time.isoformat(),
        "quarantine_reason": record.quarantine_reason,
        "recovery_attempts": record.recovery_attempts,
        "last_attempt_score": record.last_attempt_score,
        "best_score": record.best_score,
        "next_eligible_time": record.next_eligible_time.isoformat() if record.next_eligible_time else None
    }


@app.post("/recovery/{version_id}/force-evaluate")
async def force_recovery_evaluation(version_id: str, background_tasks: BackgroundTasks):
    """Force immediate recovery evaluation (bypass cooldown)"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    record = orchestrator.recovery.get_recovery_status(version_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"No recovery record for {version_id}")
    
    # Set eligible immediately
    from .recovery_manager import RecoveryStatus
    record.recovery_status = RecoveryStatus.ELIGIBLE
    
    return {
        "success": True,
        "message": f"Recovery evaluation queued for {version_id}"
    }


# ==================== Configuration Endpoints ====================

@app.get("/config/thresholds")
async def get_thresholds():
    """Get current promotion thresholds"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    t = orchestrator.lifecycle.thresholds
    return {
        "thresholds": {
            "p95_latency_ms": t.p95_latency_ms,
            "error_rate": t.error_rate,
            "accuracy_delta": t.accuracy_delta,
            "security_pass_rate": t.security_pass_rate,
            "cost_increase_max": t.cost_increase_max,
            "min_shadow_requests": t.min_shadow_requests,
            "min_shadow_duration_hours": t.min_shadow_duration_hours,
            "statistical_significance_p": t.statistical_significance_p,
            "consecutive_failures_for_downgrade": t.consecutive_failures_for_downgrade
        }
    }


@app.put("/config/thresholds")
async def update_thresholds(update: ThresholdUpdate):
    """Update promotion thresholds"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    t = orchestrator.lifecycle.thresholds
    
    if update.p95_latency_ms is not None:
        t.p95_latency_ms = update.p95_latency_ms
    if update.error_rate is not None:
        t.error_rate = update.error_rate
    if update.accuracy_delta is not None:
        t.accuracy_delta = update.accuracy_delta
    if update.security_pass_rate is not None:
        t.security_pass_rate = update.security_pass_rate
    if update.cost_increase_max is not None:
        t.cost_increase_max = update.cost_increase_max
    
    return {"success": True, "message": "Thresholds updated"}


# ==================== Metrics Endpoints ====================

@app.get("/metrics")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics not initialized")
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        content=metrics_collector.get_prometheus_metrics(),
        media_type="text/plain"
    )


@app.get("/metrics/json")
async def get_metrics_json():
    """Get metrics as JSON"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics not initialized")
    
    return metrics_collector.get_metrics_dict()


# ==================== Events Endpoints ====================

@app.get("/events")
async def get_events(
    event_type: Optional[str] = None,
    version_id: Optional[str] = None,
    limit: int = 100
):
    """Get lifecycle events"""
    if not event_publisher:
        raise HTTPException(status_code=503, detail="Events not initialized")
    
    # Get from in-memory backend
    for backend in event_publisher.backends:
        if isinstance(backend, InMemoryEventBackend):
            events = backend.get_events(
                event_type=event_type,
                version_id=version_id,
                limit=limit
            )
            return {
                "events": [e.to_dict() for e in events],
                "total": len(events)
            }
    
    return {"events": [], "total": 0}


@app.get("/events/types")
async def get_event_types():
    """Get available event types"""
    return {
        "event_types": [e.value for e in EventType]
    }


# ==================== Comparison & Analytics Endpoints ====================

@app.get("/analytics/detailed/{version_id}")
async def get_detailed_analytics(version_id: str, time_window_hours: int = 24):
    """
    Get detailed analytics for a version including:
    - Issue breakdown by severity and type
    - Latency distribution statistics
    - Hourly trends
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    # Get version config
    if version_id not in orchestrator.lifecycle.active_versions:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    
    config = orchestrator.lifecycle.active_versions[version_id]
    
    # Build analytics response
    analytics = {
        "version_id": version_id,
        "current_state": config.current_state.value,
        "time_window_hours": time_window_hours,
        "basic_info": {
            "model_version": config.model_version,
            "prompt_version": config.prompt_version,
            "created_at": config.created_at.isoformat(),
            "consecutive_failures": config.consecutive_failures,
        },
        "evaluation_summary": {
            "last_evaluation": config.last_evaluation.isoformat() if config.last_evaluation else None,
            "total_evaluations": len([
                e for e in orchestrator.lifecycle.evaluation_history
                if e.get("version_id") == version_id
            ]),
        },
    }
    
    return analytics


@app.get("/analytics/comparison")
async def get_version_comparison(
    version_a: str,
    version_b: str
):
    """Compare two versions side-by-side"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    versions = orchestrator.lifecycle.active_versions
    
    if version_a not in versions:
        raise HTTPException(status_code=404, detail=f"Version {version_a} not found")
    if version_b not in versions:
        raise HTTPException(status_code=404, detail=f"Version {version_b} not found")
    
    config_a = versions[version_a]
    config_b = versions[version_b]
    
    return {
        "comparison": {
            "version_a": {
                "version_id": version_a,
                "state": config_a.current_state.value,
                "model_version": config_a.model_version,
                "consecutive_failures": config_a.consecutive_failures,
            },
            "version_b": {
                "version_id": version_b,
                "state": config_b.current_state.value,
                "model_version": config_b.model_version,
                "consecutive_failures": config_b.consecutive_failures,
            },
        }
    }


@app.get("/analytics/promotion-history")
async def get_promotion_history(limit: int = 20):
    """Get history of promotion decisions"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    # Filter promotion events from history
    promotion_events = [
        e for e in orchestrator.lifecycle.evaluation_history
        if e.get("event_type") in ["promoted", "gray_scale_started", "promoted_to_stable"]
    ][-limit:]
    
    return {
        "promotion_history": promotion_events,
        "total_promotions": len(promotion_events),
    }


@app.get("/analytics/quarantine-history")
async def get_quarantine_history(limit: int = 20):
    """Get history of quarantine decisions"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    # Filter quarantine events from history
    quarantine_events = [
        e for e in orchestrator.lifecycle.evaluation_history
        if e.get("event_type") in ["downgraded_to_v3", "quarantined", "rollback"]
    ][-limit:]
    
    return {
        "quarantine_history": quarantine_events,
        "total_quarantines": len(quarantine_events),
    }


# ==================== Debug/Admin Endpoints ====================

@app.get("/debug/state")
async def get_debug_state():
    """Get detailed internal state (for debugging)"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail=ORCHESTRATOR_NOT_INITIALIZED)
    
    return {
        "lifecycle": {
            "active_versions": len(orchestrator.lifecycle.active_versions),
            "evaluation_history": len(orchestrator.lifecycle.evaluation_history),
        },
        "recovery": orchestrator.recovery.get_recovery_statistics(),
        "cycle": {
            "events_count": len(orchestrator.cycle_events),
            "running": orchestrator._running,
        },
        "metrics": metrics_collector.get_metrics_dict() if metrics_collector else None,
    }


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
