"""
Version Control Service - Experiment and version management.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Version Control Service",
    description="Experiment lifecycle and version management",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"


class ExperimentCreate(BaseModel):
    name: str
    config: Dict[str, Any]
    dataset_id: str


class ExperimentResponse(BaseModel):
    id: str
    name: str
    status: ExperimentStatus
    config: Dict[str, Any]
    dataset_id: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Optional[Dict[str, float]] = None


class ExperimentListResponse(BaseModel):
    items: List[ExperimentResponse]
    total: int


# Health endpoints
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/health/live")
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    return {"status": "ready"}


# Experiment endpoints
@app.get("/api/experiments", response_model=ExperimentListResponse)
async def list_experiments(status: Optional[str] = None, page: int = 1, limit: int = 20):
    """List all experiments."""
    return ExperimentListResponse(
        items=[
            ExperimentResponse(
                id="exp_1",
                name="GPT-4 Turbo Test",
                status=ExperimentStatus.COMPLETED,
                config={"model": "gpt-4-turbo", "temperature": 0.7},
                dataset_id="dataset_1",
                created_at=datetime.utcnow(),
                metrics={"accuracy": 0.92, "latency_p95": 2500, "error_rate": 0.01},
            )
        ],
        total=1,
    )


@app.post("/api/experiments", response_model=ExperimentResponse)
async def create_experiment(experiment: ExperimentCreate):
    """Create new experiment."""
    return ExperimentResponse(
        id="exp_new",
        name=experiment.name,
        status=ExperimentStatus.PENDING,
        config=experiment.config,
        dataset_id=experiment.dataset_id,
        created_at=datetime.utcnow(),
    )


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment details."""
    return ExperimentResponse(
        id=experiment_id,
        name="Test Experiment",
        status=ExperimentStatus.COMPLETED,
        config={"model": "gpt-4", "temperature": 0.7},
        dataset_id="dataset_1",
        created_at=datetime.utcnow(),
        metrics={"accuracy": 0.91, "latency_p95": 2800},
    )


@app.post("/api/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start experiment execution."""
    return {"message": "Experiment started", "id": experiment_id}


@app.post("/api/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop experiment execution."""
    return {"message": "Experiment stopped", "id": experiment_id}


@app.post("/api/experiments/{experiment_id}/evaluate")
async def evaluate_experiment(experiment_id: str):
    """Evaluate experiment results."""
    return {
        "id": experiment_id,
        "metrics": {
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.93,
            "f1_score": 0.92,
            "latency_p95": 2500,
            "error_rate": 0.01,
        },
        "passed_gate": True,
    }


@app.post("/api/experiments/{experiment_id}/promote")
async def promote_experiment(experiment_id: str):
    """Promote experiment to v2 production."""
    return {"message": "Experiment promoted to v2", "id": experiment_id}


@app.post("/api/experiments/{experiment_id}/quarantine")
async def quarantine_experiment(experiment_id: str, reason: str = "Failed gate check"):
    """Move experiment to v3 quarantine."""
    return {"message": "Experiment quarantined", "id": experiment_id, "reason": reason}


@app.get("/api/experiments/{experiment_id}/metrics")
async def get_experiment_metrics(experiment_id: str):
    """Get detailed experiment metrics."""
    return {
        "id": experiment_id,
        "metrics": {
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.93,
            "f1_score": 0.92,
            "latency_p50": 1200,
            "latency_p95": 2500,
            "latency_p99": 3500,
            "error_rate": 0.01,
            "cost_per_analysis": 0.025,
        },
    }


# Version endpoints
@app.get("/api/versions")
async def list_versions():
    """List all versions."""
    return {
        "versions": [
            {"version": "v1", "status": "experimentation", "experiments": 5},
            {"version": "v2", "status": "production", "model": "gpt-4"},
            {"version": "v3", "status": "quarantine", "count": 12},
        ]
    }


@app.get("/api/versions/current")
async def get_current_version():
    """Get current production version."""
    return {
        "version": "v2",
        "model": "gpt-4",
        "deployed_at": datetime.utcnow().isoformat(),
        "metrics": {"accuracy": 0.94, "latency_p95": 2200},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
