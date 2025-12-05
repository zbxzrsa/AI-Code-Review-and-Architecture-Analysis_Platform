"""
Experiment management endpoints for V1.
"""
import logging
from typing import Optional, List
from datetime, timezone import datetime, timezone
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

from config.settings import settings
from models.experiment import Experiment, ExperimentStatus, ExperimentMetrics, PromotionStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments")


class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment."""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    primary_model: str = Field(..., description="Primary AI model to test")
    secondary_model: Optional[str] = Field(None, description="Secondary AI model")
    prompt_template: str = Field(..., description="Prompt template for analysis")
    routing_strategy: str = Field(
        default="primary",
        description="Routing strategy: primary, secondary, ensemble, adaptive",
    )
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")


class ExperimentResponse(BaseModel):
    """Response model for experiment."""
    id: str
    name: str
    description: Optional[str]
    status: str
    primary_model: str
    secondary_model: Optional[str]
    routing_strategy: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    promotion_status: str
    metrics: Optional[dict]


class RunExperimentRequest(BaseModel):
    """Request to run an experiment."""
    code_samples: List[str] = Field(..., description="Code samples to analyze")
    language: str = Field(..., description="Programming language")


# In-memory storage for demo (replace with database)
experiments_store = {}


@router.post("/create", response_model=ExperimentResponse)
async def create_experiment(request: CreateExperimentRequest) -> ExperimentResponse:
    """Create a new experiment."""
    experiment = Experiment(
        name=request.name,
        description=request.description,
        primary_model=request.primary_model,
        secondary_model=request.secondary_model,
        prompt_template=request.prompt_template,
        routing_strategy=request.routing_strategy,
        tags=request.tags or [],
        created_by="system",
    )

    experiments_store[experiment.id] = experiment

    logger.info(
        "Experiment created",
        experiment_id=experiment.id,
        name=request.name,
        primary_model=request.primary_model,
    )

    return ExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        status=experiment.status.value,
        primary_model=experiment.primary_model,
        secondary_model=experiment.secondary_model,
        routing_strategy=experiment.routing_strategy,
        created_at=experiment.created_at,
        started_at=experiment.started_at,
        completed_at=experiment.completed_at,
        promotion_status=experiment.promotion_status.value,
        metrics=None,
    )


@router.post("/run/{experiment_id}")
async def run_experiment(
    experiment_id: str,
    request: RunExperimentRequest,
):
    """Run an experiment with provided code samples."""
    if experiment_id not in experiments_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )

    experiment = experiments_store[experiment_id]
    experiment.status = ExperimentStatus.RUNNING
    experiment.started_at = datetime.now(timezone.utc)

    try:
        # Simulate analysis
        start_time = time.time()

        # Mock metrics
        metrics = ExperimentMetrics(
            accuracy=0.94,
            latency_ms=2800,
            cost=15.50,
            error_rate=0.025,
            throughput=100,
            user_satisfaction=4.2,
            false_positives=2,
            false_negatives=3,
        )

        experiment.metrics = metrics
        experiment.completed_at = datetime.now(timezone.utc)
        experiment.status = ExperimentStatus.COMPLETED

        # Auto-evaluate for promotion
        if metrics.meets_v2_threshold():
            experiment.promotion_status = PromotionStatus.PASSED
            logger.info(
                "Experiment passed evaluation",
                experiment_id=experiment_id,
                metrics=metrics.to_dict(),
            )
        else:
            experiment.promotion_status = PromotionStatus.FAILED
            logger.warning(
                "Experiment failed evaluation",
                experiment_id=experiment_id,
                metrics=metrics.to_dict(),
            )

        return {
            "experiment_id": experiment_id,
            "status": experiment.status.value,
            "promotion_status": experiment.promotion_status.value,
            "metrics": metrics.to_dict(),
            "duration_ms": (time.time() - start_time) * 1000,
        }

    except Exception as e:
        experiment.status = ExperimentStatus.FAILED
        logger.error(
            "Experiment execution failed",
            experiment_id=experiment_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Experiment execution failed",
        )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str) -> ExperimentResponse:
    """Get experiment details."""
    if experiment_id not in experiments_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )

    exp = experiments_store[experiment_id]

    return ExperimentResponse(
        id=exp.id,
        name=exp.name,
        description=exp.description,
        status=exp.status.value,
        primary_model=exp.primary_model,
        secondary_model=exp.secondary_model,
        routing_strategy=exp.routing_strategy,
        created_at=exp.created_at,
        started_at=exp.started_at,
        completed_at=exp.completed_at,
        promotion_status=exp.promotion_status.value,
        metrics=exp.metrics.to_dict() if exp.metrics else None,
    )


@router.get("/")
async def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List experiments."""
    experiments = list(experiments_store.values())

    if status:
        experiments = [e for e in experiments if e.status.value == status]

    total = len(experiments)
    experiments = experiments[offset : offset + limit]

    return {
        "experiments": [
            {
                "id": e.id,
                "name": e.name,
                "status": e.status.value,
                "promotion_status": e.promotion_status.value,
                "created_at": e.created_at,
            }
            for e in experiments
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
