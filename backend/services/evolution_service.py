"""
Evolution Cycle Service API

FastAPI endpoints for managing the three-version self-evolution cycle:
- V1 Experimentation management
- V2 Production monitoring
- V3 Quarantine management
- Promotion and degradation workflows
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/evolution", tags=["evolution"])


# =============================================================================
# Constants
# =============================================================================

TECHNOLOGY_NOT_FOUND = "Technology not found"
EXPERIMENT_NOT_FOUND = "Experiment not found"


# =============================================================================
# Models
# =============================================================================

class Version(str, Enum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class TechnologyStatus(str, Enum):
    EXPERIMENTAL = "experimental"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"
    RE_EVALUATING = "re-evaluating"


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"


class TechnologyMetrics(BaseModel):
    accuracy: float = Field(default=0.0, ge=0, le=1)
    error_rate: float = Field(default=0.0, ge=0, le=1)
    latency_p95_ms: float = Field(default=0.0, ge=0)
    sample_count: int = Field(default=0, ge=0)


class Technology(BaseModel):
    tech_id: str
    name: str
    category: str
    description: str
    source: str
    version: Version
    status: TechnologyStatus
    metrics: TechnologyMetrics
    created_at: datetime


class Experiment(BaseModel):
    experiment_id: str
    name: str
    technology_type: str
    category: str
    status: ExperimentStatus
    samples_collected: int
    min_samples: int
    accuracy: float
    error_rate: float
    latency_p95: float
    started_at: datetime
    recommendation: Optional[str] = None


class VersionMetrics(BaseModel):
    request_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    technology_count: int = 0


class PromotionRecord(BaseModel):
    record_id: str
    from_version: Version
    to_version: Version
    technology_name: str
    reason: str
    timestamp: datetime


class CycleStatus(BaseModel):
    running: bool
    cycle_count: int
    last_cycle_at: Optional[datetime]
    promotions: int
    degradations: int
    v1_metrics: VersionMetrics
    v2_metrics: VersionMetrics
    v3_metrics: VersionMetrics


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateExperimentRequest(BaseModel):
    technology_type: str
    name: Optional[str] = None
    custom_config: Optional[Dict[str, Any]] = None


class StartExperimentResponse(BaseModel):
    success: bool
    experiment_id: str
    message: str


class PromoteTechnologyRequest(BaseModel):
    tech_id: str
    reason: Optional[str] = "Passed all evaluation criteria"


class DegradeTechnologyRequest(BaseModel):
    tech_id: str
    reason: str


class TechnologiesResponse(BaseModel):
    technologies: List[Technology]
    total: int


class ExperimentsResponse(BaseModel):
    experiments: List[Experiment]
    total: int


class PromotionHistoryResponse(BaseModel):
    records: List[PromotionRecord]
    total: int


# =============================================================================
# Mock Data Store (replace with database in production)
# =============================================================================

_cycle_status = CycleStatus(
    running=True,
    cycle_count=24,
    last_cycle_at=datetime.now(timezone.utc),
    promotions=5,
    degradations=2,
    v1_metrics=VersionMetrics(
        request_count=2500,
        error_count=75,
        error_rate=0.03,
        avg_latency_ms=2500,
        technology_count=3,
    ),
    v2_metrics=VersionMetrics(
        request_count=15000,
        error_count=150,
        error_rate=0.01,
        avg_latency_ms=1800,
        technology_count=5,
    ),
    v3_metrics=VersionMetrics(
        request_count=500,
        error_count=100,
        error_rate=0.20,
        avg_latency_ms=4000,
        technology_count=2,
    ),
)

_technologies: Dict[str, Technology] = {}
_experiments: Dict[str, Experiment] = {}
_promotion_history: List[PromotionRecord] = []


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status", response_model=CycleStatus)
async def get_cycle_status():
    """Get current evolution cycle status."""
    return _cycle_status


@router.post("/start")
async def start_cycle(background_tasks: BackgroundTasks):
    """Start the evolution cycle."""
    global _cycle_status
    if _cycle_status.running:
        return {"success": False, "message": "Cycle already running"}
    
    _cycle_status.running = True
    return {"success": True, "message": "Evolution cycle started"}


@router.post("/stop")
async def stop_cycle():
    """Stop the evolution cycle."""
    global _cycle_status
    if not _cycle_status.running:
        return {"success": False, "message": "Cycle not running"}
    
    _cycle_status.running = False
    return {"success": True, "message": "Evolution cycle stopped"}


@router.get("/technologies", response_model=TechnologiesResponse)
async def list_technologies(
    version: Optional[Version] = None,
    status: Optional[TechnologyStatus] = None,
):
    """List all technologies with optional filters."""
    techs = list(_technologies.values())
    
    if version:
        techs = [t for t in techs if t.version == version]
    if status:
        techs = [t for t in techs if t.status == status]
    
    return TechnologiesResponse(technologies=techs, total=len(techs))


@router.get("/technologies/{tech_id}", response_model=Technology)
async def get_technology(tech_id: str):
    """Get a specific technology by ID."""
    if tech_id not in _technologies:
        raise HTTPException(status_code=404, detail=TECHNOLOGY_NOT_FOUND)
    return _technologies[tech_id]


@router.get("/experiments", response_model=ExperimentsResponse)
async def list_experiments(
    status: Optional[ExperimentStatus] = None,
):
    """List all experiments with optional status filter."""
    exps = list(_experiments.values())
    
    if status:
        exps = [e for e in exps if e.status == status]
    
    return ExperimentsResponse(experiments=exps, total=len(exps))


@router.post("/experiments", response_model=StartExperimentResponse)
async def create_experiment(request: CreateExperimentRequest):
    """Create a new experiment."""
    import uuid
    
    exp_id = str(uuid.uuid4())
    experiment = Experiment(
        experiment_id=exp_id,
        name=request.name or request.technology_type,
        technology_type=request.technology_type,
        category="attention",  # Would be determined by technology_type
        status=ExperimentStatus.PENDING,
        samples_collected=0,
        min_samples=1000,
        accuracy=0.0,
        error_rate=0.0,
        latency_p95=0.0,
        started_at=datetime.now(timezone.utc),
    )
    
    _experiments[exp_id] = experiment
    
    return StartExperimentResponse(
        success=True,
        experiment_id=exp_id,
        message=f"Experiment '{experiment.name}' created",
    )


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start an experiment."""
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail=EXPERIMENT_NOT_FOUND)
    
    _experiments[experiment_id].status = ExperimentStatus.RUNNING
    _experiments[experiment_id].started_at = datetime.now(timezone.utc)
    
    return {"success": True, "message": "Experiment started"}


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop an experiment."""
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail=EXPERIMENT_NOT_FOUND)
    
    _experiments[experiment_id].status = ExperimentStatus.COMPLETED
    
    return {"success": True, "message": "Experiment stopped"}


@router.post("/experiments/{experiment_id}/evaluate")
async def evaluate_experiment(experiment_id: str):
    """Evaluate an experiment and get recommendation."""
    if experiment_id not in _experiments:
        raise HTTPException(status_code=404, detail=EXPERIMENT_NOT_FOUND)
    
    exp = _experiments[experiment_id]
    exp.status = ExperimentStatus.EVALUATING
    
    # Evaluation logic (simplified)
    if exp.accuracy >= 0.85 and exp.error_rate <= 0.05:
        exp.recommendation = "PROMOTE: All thresholds met"
    elif exp.error_rate > 0.20 or exp.accuracy < 0.70:
        exp.recommendation = "QUARANTINE: Critical threshold violations"
    else:
        exp.recommendation = "CONTINUE: Needs more data or tuning"
    
    exp.status = ExperimentStatus.COMPLETED
    
    return {
        "success": True,
        "recommendation": exp.recommendation,
        "metrics": {
            "accuracy": exp.accuracy,
            "error_rate": exp.error_rate,
            "latency_p95": exp.latency_p95,
            "samples": exp.samples_collected,
        },
    }


@router.post("/promote")
async def promote_technology(request: PromoteTechnologyRequest):
    """Promote a technology from V1 to V2."""
    import uuid
    
    if request.tech_id not in _technologies:
        raise HTTPException(status_code=404, detail=TECHNOLOGY_NOT_FOUND)
    
    tech = _technologies[request.tech_id]
    if tech.version != Version.V1:
        raise HTTPException(status_code=400, detail="Technology not in V1")
    
    # Update technology
    tech.version = Version.V2
    tech.status = TechnologyStatus.PROMOTED
    
    # Record promotion
    record = PromotionRecord(
        record_id=str(uuid.uuid4()),
        from_version=Version.V1,
        to_version=Version.V2,
        technology_name=tech.name,
        reason=request.reason,
        timestamp=datetime.now(timezone.utc),
    )
    _promotion_history.append(record)
    
    global _cycle_status
    _cycle_status.promotions += 1
    
    return {
        "success": True,
        "message": f"Technology '{tech.name}' promoted to V2",
        "record_id": record.record_id,
    }


@router.post("/degrade")
async def degrade_technology(request: DegradeTechnologyRequest):
    """Degrade a technology to V3 quarantine."""
    import uuid
    
    if request.tech_id not in _technologies:
        raise HTTPException(status_code=404, detail=TECHNOLOGY_NOT_FOUND)
    
    tech = _technologies[request.tech_id]
    from_version = tech.version
    
    # Update technology
    tech.version = Version.V3
    tech.status = TechnologyStatus.QUARANTINED
    
    # Record degradation
    record = PromotionRecord(
        record_id=str(uuid.uuid4()),
        from_version=from_version,
        to_version=Version.V3,
        technology_name=tech.name,
        reason=request.reason,
        timestamp=datetime.now(timezone.utc),
    )
    _promotion_history.append(record)
    
    global _cycle_status
    _cycle_status.degradations += 1
    
    return {
        "success": True,
        "message": f"Technology '{tech.name}' degraded to V3",
        "record_id": record.record_id,
    }


@router.post("/re-evaluate/{tech_id}")
async def request_re_evaluation(tech_id: str):
    """Request re-evaluation of a quarantined technology."""
    if tech_id not in _technologies:
        raise HTTPException(status_code=404, detail=TECHNOLOGY_NOT_FOUND)
    
    tech = _technologies[tech_id]
    if tech.version != Version.V3:
        raise HTTPException(status_code=400, detail="Technology not in V3")
    
    # Move back to V1
    tech.version = Version.V1
    tech.status = TechnologyStatus.RE_EVALUATING
    tech.metrics = TechnologyMetrics()  # Reset metrics
    
    return {
        "success": True,
        "message": f"Technology '{tech.name}' moved to V1 for re-evaluation",
    }


@router.get("/history", response_model=PromotionHistoryResponse)
async def get_promotion_history(
    limit: int = 20,
    offset: int = 0,
):
    """Get promotion/degradation history."""
    sorted_history = sorted(
        _promotion_history,
        key=lambda r: r.timestamp,
        reverse=True,
    )
    
    paginated = sorted_history[offset : offset + limit]
    
    return PromotionHistoryResponse(
        records=paginated,
        total=len(_promotion_history),
    )


@router.get("/metrics/{version}")
async def get_version_metrics(version: Version):
    """Get metrics for a specific version."""
    if version == Version.V1:
        return _cycle_status.v1_metrics
    elif version == Version.V2:
        return _cycle_status.v2_metrics
    else:
        return _cycle_status.v3_metrics


@router.get("/available-technologies")
async def get_available_technologies():
    """Get list of predefined technologies available for experimentation."""
    return {
        "technologies": [
            {
                "type": "multi_head_attention",
                "category": "attention",
                "description": "Standard Multi-Head Attention",
                "source": "LLMs-from-scratch Ch03",
            },
            {
                "type": "grouped_query_attention",
                "category": "attention",
                "description": "Grouped-Query Attention (GQA)",
                "source": "LLMs-from-scratch Ch04/04_gqa",
            },
            {
                "type": "sliding_window_attention",
                "category": "attention",
                "description": "Sliding Window Attention (SWA)",
                "source": "LLMs-from-scratch Ch04/06_swa",
            },
            {
                "type": "llama_architecture",
                "category": "architecture",
                "description": "Llama 3.2 with RoPE and RMSNorm",
                "source": "LLMs-from-scratch Ch05/07_gpt_to_llama",
            },
            {
                "type": "mixture_of_experts",
                "category": "architecture",
                "description": "Mixture of Experts (MoE)",
                "source": "LLMs-from-scratch Ch04/07_moe",
            },
            {
                "type": "direct_preference_optimization",
                "category": "training",
                "description": "DPO for LLM alignment",
                "source": "LLMs-from-scratch Ch07/04",
            },
            {
                "type": "kv_cache_optimization",
                "category": "optimization",
                "description": "KV Cache for efficient generation",
                "source": "LLMs-from-scratch Ch04/03_kv-cache",
            },
            {
                "type": "flash_attention",
                "category": "optimization",
                "description": "Flash Attention for memory efficiency",
                "source": "LLMs-from-scratch Ch03/02_bonus",
            },
        ]
    }
