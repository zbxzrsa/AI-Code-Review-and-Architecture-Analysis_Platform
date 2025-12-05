"""
Evaluation and Promotion API Endpoints

Handles:
- Experiment metrics retrieval
- Experiment comparison
- Promotion to V2
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import MetricThresholds, PromotionConfig

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ExperimentMetrics(BaseModel):
    """Detailed metrics for an experiment"""
    experiment_id: str
    
    # Performance metrics
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1_score: float = Field(ge=0, le=1)
    latency_p50_ms: int
    latency_p95_ms: int
    latency_p99_ms: int
    throughput_rps: float
    
    # Efficiency metrics
    cost_per_1000_requests: float
    model_size_gb: float
    memory_usage_gb: float
    gpu_utilization: float
    
    # Innovation metrics
    improvement_vs_baseline: float
    techniques_tested: int
    
    # Quality metrics
    error_rate: float
    hallucination_rate: float


class CompareExperimentsRequest(BaseModel):
    """Request to compare experiments"""
    experiment_ids: List[str] = Field(..., min_items=2, max_items=10)


class MetricComparison(BaseModel):
    """Comparison of a single metric across experiments"""
    metric_name: str
    values: Dict[str, float]  # experiment_id -> value
    best_experiment: str
    improvement_percent: float


class CompareExperimentsResponse(BaseModel):
    """Response with experiment comparison"""
    experiments: List[str]
    comparisons: List[MetricComparison]
    pareto_frontier: List[str]  # Experiments on the Pareto frontier
    recommendation: str


class PromotionRequest(BaseModel):
    """Request to promote experiment to V2"""
    justification: str = Field(..., min_length=10, max_length=1000)
    skip_validation: bool = Field(default=False)


class PromotionResponse(BaseModel):
    """Response for promotion request"""
    promotion_request_id: str
    experiment_id: str
    status: str  # approved, conditional, rejected, blocked
    message: str
    validation_results: Dict[str, Any]
    required_actions: List[str]


# =============================================================================
# In-Memory Storage
# =============================================================================

metrics_db: Dict[str, Dict[str, Any]] = {}
promotions_db: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/metrics/{experiment_id}", response_model=ExperimentMetrics)
async def get_experiment_metrics(experiment_id: str):
    """
    Get detailed metrics for an experiment.
    
    Returns comprehensive evaluation metrics including performance,
    efficiency, innovation, and quality metrics.
    """
    # Check if metrics exist, otherwise generate mock data
    if experiment_id not in metrics_db:
        # Generate mock metrics for demo
        import random
        metrics_db[experiment_id] = {
            "experiment_id": experiment_id,
            "accuracy": round(random.uniform(0.85, 0.95), 3),
            "precision": round(random.uniform(0.85, 0.95), 3),
            "recall": round(random.uniform(0.82, 0.92), 3),
            "f1_score": round(random.uniform(0.84, 0.93), 3),
            "latency_p50_ms": random.randint(100, 300),
            "latency_p95_ms": random.randint(300, 450),
            "latency_p99_ms": random.randint(400, 500),
            "throughput_rps": round(random.uniform(80, 150), 1),
            "cost_per_1000_requests": round(random.uniform(0.05, 0.12), 3),
            "model_size_gb": round(random.uniform(10, 15), 1),
            "memory_usage_gb": round(random.uniform(18, 23), 1),
            "gpu_utilization": round(random.uniform(0.75, 0.95), 2),
            "improvement_vs_baseline": round(random.uniform(-0.02, 0.08), 3),
            "techniques_tested": random.randint(3, 8),
            "error_rate": round(random.uniform(0.005, 0.02), 4),
            "hallucination_rate": round(random.uniform(0.001, 0.01), 4),
        }
    
    return ExperimentMetrics(**metrics_db[experiment_id])


@router.post("/compare", response_model=CompareExperimentsResponse)
async def compare_experiments(request: CompareExperimentsRequest):
    """
    Compare multiple experiment versions.
    
    Returns comparative metrics, identifies Pareto-optimal experiments,
    and provides a recommendation.
    """
    experiment_ids = request.experiment_ids
    
    # Get metrics for all experiments
    all_metrics = {}
    for exp_id in experiment_ids:
        if exp_id not in metrics_db:
            # Generate mock data
            await get_experiment_metrics(exp_id)
        all_metrics[exp_id] = metrics_db[exp_id]
    
    # Compare key metrics
    comparisons = []
    
    # Accuracy comparison
    accuracy_values = {exp_id: m["accuracy"] for exp_id, m in all_metrics.items()}
    best_accuracy_exp = max(accuracy_values, key=accuracy_values.get)
    comparisons.append(MetricComparison(
        metric_name="accuracy",
        values=accuracy_values,
        best_experiment=best_accuracy_exp,
        improvement_percent=round(
            (max(accuracy_values.values()) - min(accuracy_values.values())) * 100, 2
        ),
    ))
    
    # Latency comparison (lower is better)
    latency_values = {exp_id: m["latency_p99_ms"] for exp_id, m in all_metrics.items()}
    best_latency_exp = min(latency_values, key=latency_values.get)
    comparisons.append(MetricComparison(
        metric_name="latency_p99_ms",
        values={k: float(v) for k, v in latency_values.items()},
        best_experiment=best_latency_exp,
        improvement_percent=round(
            (max(latency_values.values()) - min(latency_values.values())) / max(latency_values.values()) * 100, 2
        ),
    ))
    
    # Cost comparison (lower is better)
    cost_values = {exp_id: m["cost_per_1000_requests"] for exp_id, m in all_metrics.items()}
    best_cost_exp = min(cost_values, key=cost_values.get)
    comparisons.append(MetricComparison(
        metric_name="cost_per_1000_requests",
        values=cost_values,
        best_experiment=best_cost_exp,
        improvement_percent=round(
            (max(cost_values.values()) - min(cost_values.values())) / max(cost_values.values()) * 100, 2
        ),
    ))
    
    # Throughput comparison
    throughput_values = {exp_id: m["throughput_rps"] for exp_id, m in all_metrics.items()}
    best_throughput_exp = max(throughput_values, key=throughput_values.get)
    comparisons.append(MetricComparison(
        metric_name="throughput_rps",
        values=throughput_values,
        best_experiment=best_throughput_exp,
        improvement_percent=round(
            (max(throughput_values.values()) - min(throughput_values.values())) / min(throughput_values.values()) * 100, 2
        ),
    ))
    
    # Simple Pareto frontier calculation
    # An experiment is Pareto-optimal if no other experiment is better in all metrics
    pareto_frontier = []
    for exp_id in experiment_ids:
        is_dominated = False
        for other_id in experiment_ids:
            if other_id == exp_id:
                continue
            
            # Check if other dominates this
            better_in_all = (
                all_metrics[other_id]["accuracy"] >= all_metrics[exp_id]["accuracy"] and
                all_metrics[other_id]["latency_p99_ms"] <= all_metrics[exp_id]["latency_p99_ms"] and
                all_metrics[other_id]["cost_per_1000_requests"] <= all_metrics[exp_id]["cost_per_1000_requests"]
            )
            strictly_better_in_one = (
                all_metrics[other_id]["accuracy"] > all_metrics[exp_id]["accuracy"] or
                all_metrics[other_id]["latency_p99_ms"] < all_metrics[exp_id]["latency_p99_ms"] or
                all_metrics[other_id]["cost_per_1000_requests"] < all_metrics[exp_id]["cost_per_1000_requests"]
            )
            
            if better_in_all and strictly_better_in_one:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_frontier.append(exp_id)
    
    # Generate recommendation
    if len(pareto_frontier) == 1:
        recommendation = f"Experiment {pareto_frontier[0]} is the clear winner across all metrics."
    elif best_accuracy_exp == best_latency_exp == best_cost_exp:
        recommendation = f"Experiment {best_accuracy_exp} excels in all key metrics - recommended for promotion."
    else:
        recommendation = f"Trade-offs exist. For accuracy, choose {best_accuracy_exp}. For cost efficiency, choose {best_cost_exp}."
    
    return CompareExperimentsResponse(
        experiments=experiment_ids,
        comparisons=comparisons,
        pareto_frontier=pareto_frontier,
        recommendation=recommendation,
    )


@router.post("/promote/{experiment_id}", response_model=PromotionResponse)
async def promote_experiment(experiment_id: str, request: PromotionRequest):
    """
    Submit experiment for V2 validation.
    
    Validates the experiment against promotion criteria and
    either approves, conditionally approves, or rejects the promotion.
    """
    # Get experiment metrics
    if experiment_id not in metrics_db:
        await get_experiment_metrics(experiment_id)
    
    metrics = metrics_db[experiment_id]
    thresholds = MetricThresholds()
    
    # Validation results
    validation_results = {}
    required_actions = []
    
    # Check must-pass criteria
    accuracy_pass = metrics["accuracy"] >= thresholds.min_accuracy
    validation_results["accuracy >= 0.92"] = accuracy_pass
    if not accuracy_pass:
        required_actions.append(f"Improve accuracy from {metrics['accuracy']:.3f} to >= {thresholds.min_accuracy}")
    
    latency_pass = metrics["latency_p99_ms"] <= thresholds.max_latency_p99_ms
    validation_results["latency_p99 <= 500ms"] = latency_pass
    if not latency_pass:
        required_actions.append(f"Reduce latency from {metrics['latency_p99_ms']}ms to <= {thresholds.max_latency_p99_ms}ms")
    
    error_rate_pass = metrics["error_rate"] <= thresholds.max_error_rate
    validation_results["error_rate <= 2%"] = error_rate_pass
    if not error_rate_pass:
        required_actions.append(f"Reduce error rate from {metrics['error_rate']*100:.2f}% to <= 2%")
    
    # Check improvement threshold
    improvement_pass = metrics["improvement_vs_baseline"] >= thresholds.min_accuracy_improvement
    validation_results["improvement >= 5%"] = improvement_pass
    
    # Determine status
    all_must_pass = accuracy_pass and latency_pass and error_rate_pass
    
    if all_must_pass and improvement_pass:
        status = "approved"
        message = "Experiment meets all promotion criteria. Approved for V2 validation."
    elif all_must_pass and not improvement_pass:
        status = "conditional"
        message = "Experiment meets basic criteria but improvement threshold not met. Requires human review."
        required_actions.append("Demonstrate unique capability or cost benefit to justify promotion")
    elif not all_must_pass and request.skip_validation:
        status = "conditional"
        message = "Validation skipped by request. Requires human review."
    else:
        status = "rejected"
        message = "Experiment does not meet promotion criteria. See required actions."
    
    # Create promotion record
    promotion_id = str(uuid4())
    promotions_db[promotion_id] = {
        "promotion_request_id": promotion_id,
        "experiment_id": experiment_id,
        "status": status,
        "justification": request.justification,
        "validation_results": validation_results,
        "created_at": datetime.now(timezone.utc),
    }
    
    return PromotionResponse(
        promotion_request_id=promotion_id,
        experiment_id=experiment_id,
        status=status,
        message=message,
        validation_results=validation_results,
        required_actions=required_actions,
    )


@router.get("/promotions")
async def list_promotions(
    status: Optional[str] = None,
    limit: int = 50,
):
    """
    List promotion requests.
    """
    promotions = list(promotions_db.values())
    
    if status:
        promotions = [p for p in promotions if p["status"] == status]
    
    return promotions[:limit]


@router.get("/promotions/{promotion_id}")
async def get_promotion(promotion_id: str):
    """
    Get promotion request details.
    """
    if promotion_id not in promotions_db:
        raise HTTPException(status_code=404, detail="Promotion request not found")
    
    return promotions_db[promotion_id]
