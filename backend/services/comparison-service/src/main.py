"""
Comparison Service - A/B testing and statistical comparison.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Comparison Service",
    description="A/B testing and statistical analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ComparisonRequest(BaseModel):
    experiment_a: str
    experiment_b: str
    metrics: List[str] = ["accuracy", "latency", "cost"]


class ComparisonResult(BaseModel):
    id: str
    experiment_a: str
    experiment_b: str
    created_at: datetime
    results: Dict[str, Any]
    winner: Optional[str] = None
    confidence: float


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


# Comparison endpoints
@app.get("/api/comparisons")
async def list_comparisons(page: int = 1, limit: int = 20):
    """List all comparisons."""
    return {
        "items": [
            {
                "id": "comp_1",
                "experiment_a": "exp_1",
                "experiment_b": "exp_2",
                "winner": "exp_1",
                "confidence": 0.95,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "total": 1,
    }


@app.post("/api/comparisons", response_model=ComparisonResult)
async def create_comparison(request: ComparisonRequest):
    """Compare two experiments."""
    return ComparisonResult(
        id="comp_new",
        experiment_a=request.experiment_a,
        experiment_b=request.experiment_b,
        created_at=datetime.now(timezone.utc),
        results={
            "accuracy": {
                "a": 0.92,
                "b": 0.89,
                "difference": 0.03,
                "p_value": 0.023,
                "significant": True,
            },
            "latency_p95": {
                "a": 2500,
                "b": 2800,
                "difference": -300,
                "p_value": 0.041,
                "significant": True,
            },
            "cost": {
                "a": 0.025,
                "b": 0.022,
                "difference": 0.003,
                "p_value": 0.12,
                "significant": False,
            },
        },
        winner=request.experiment_a,
        confidence=0.95,
    )


@app.get("/api/comparisons/{comparison_id}", response_model=ComparisonResult)
async def get_comparison(comparison_id: str):
    """Get comparison details."""
    return ComparisonResult(
        id=comparison_id,
        experiment_a="exp_1",
        experiment_b="exp_2",
        created_at=datetime.now(timezone.utc),
        results={
            "accuracy": {"a": 0.92, "b": 0.89, "p_value": 0.023},
            "latency_p95": {"a": 2500, "b": 2800, "p_value": 0.041},
        },
        winner="exp_1",
        confidence=0.95,
    )


@app.get("/api/experiments/compare")
async def compare_experiments(experiment_a: str, experiment_b: str):
    """Quick comparison between two experiments."""
    return {
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "metrics": {
            "accuracy": {"a": 0.92, "b": 0.89, "winner": "a"},
            "latency_p95": {"a": 2500, "b": 2800, "winner": "a"},
            "error_rate": {"a": 0.01, "b": 0.02, "winner": "a"},
            "cost": {"a": 0.025, "b": 0.022, "winner": "b"},
        },
        "overall_winner": experiment_a,
        "confidence": 0.92,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
