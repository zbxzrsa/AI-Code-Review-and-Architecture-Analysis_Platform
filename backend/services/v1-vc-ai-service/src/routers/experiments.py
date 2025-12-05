"""
Experiment Management API Endpoints

Handles experiment lifecycle:
- Create experiments
- Run experiments
- Get experiment status
- List experiments
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

router = APIRouter(prefix="/experiments", tags=["experiments"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ArchitectureConfig(BaseModel):
    """Architecture configuration for an experiment"""
    model_type: str = Field(default="mistral_7b", description="Base model type")
    quantization_bits: int = Field(default=4, ge=2, le=16)
    lora_rank: int = Field(default=128, ge=1, le=512)
    lora_alpha: int = Field(default=256)
    use_moe: bool = Field(default=False)
    num_experts: int = Field(default=8, ge=1, le=16)
    attention_type: str = Field(default="flash_attention_2")
    use_sparse_attention: bool = Field(default=False)


class TrainingConfig(BaseModel):
    """Training configuration for an experiment"""
    batch_size: int = Field(default=256, ge=1, le=1024)
    learning_rate: float = Field(default=2e-4, gt=0, lt=1)
    num_epochs: int = Field(default=3, ge=1, le=100)
    warmup_steps: int = Field(default=500, ge=0)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    use_curriculum_learning: bool = Field(default=True)
    use_multi_task: bool = Field(default=True)
    use_contrastive_learning: bool = Field(default=True)


class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment"""
    experiment_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    architecture_config: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    data_sources: List[str] = Field(default_factory=lambda: [
        "tensorflow/tensorflow",
        "pytorch/pytorch",
    ])
    tags: List[str] = Field(default_factory=list)


class ExecutionParameters(BaseModel):
    """Parameters for experiment execution"""
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$")
    max_runtime_hours: int = Field(default=24, ge=1, le=168)
    gpu_count: int = Field(default=1, ge=1, le=8)
    checkpoint_interval_steps: int = Field(default=200, ge=10)


class RunExperimentRequest(BaseModel):
    """Request to run an experiment"""
    execution_parameters: ExecutionParameters = Field(default_factory=ExecutionParameters)
    resume_from_checkpoint: Optional[str] = None


class ExperimentMetrics(BaseModel):
    """Metrics for an experiment"""
    accuracy: Optional[float] = None
    latency_p99_ms: Optional[int] = None
    throughput_rps: Optional[float] = None
    cost_per_1000_requests: Optional[float] = None
    memory_gb: Optional[float] = None
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None


class ExperimentStatus(BaseModel):
    """Current status of an experiment"""
    experiment_id: str
    experiment_name: str
    status: str  # pending, running, completed, failed, cancelled
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percent: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    metrics: ExperimentMetrics = Field(default_factory=ExperimentMetrics)
    error_message: Optional[str] = None


class ExperimentResponse(BaseModel):
    """Response for experiment operations"""
    experiment_id: str
    status: str
    message: str
    initial_metrics: Optional[ExperimentMetrics] = None


class ExecutionResponse(BaseModel):
    """Response for experiment execution"""
    execution_id: str
    experiment_id: str
    status: str
    estimated_duration_hours: float
    message: str


# =============================================================================
# In-Memory Storage (Replace with database in production)
# =============================================================================

experiments_db: Dict[str, Dict[str, Any]] = {}
executions_db: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=ExperimentResponse)
async def create_experiment(request: CreateExperimentRequest):
    """
    Create a new version control experiment.
    
    Creates an experiment with the specified architecture and training configuration.
    The experiment will be in 'pending' status until execution is triggered.
    """
    experiment_id = str(uuid4())
    
    experiment = {
        "experiment_id": experiment_id,
        "experiment_name": request.experiment_name,
        "description": request.description,
        "architecture_config": request.architecture_config.dict(),
        "training_config": request.training_config.dict(),
        "data_sources": request.data_sources,
        "tags": request.tags,
        "status": "pending",
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "completed_at": None,
        "metrics": {},
        "error_message": None,
    }
    
    experiments_db[experiment_id] = experiment
    
    return ExperimentResponse(
        experiment_id=experiment_id,
        status="pending",
        message=f"Experiment '{request.experiment_name}' created successfully",
        initial_metrics=None,
    )


@router.get("/{experiment_id}", response_model=ExperimentStatus)
async def get_experiment(experiment_id: str):
    """
    Fetch experiment status and metrics.
    
    Returns the current status, progress, and metrics for the specified experiment.
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    exp = experiments_db[experiment_id]
    
    return ExperimentStatus(
        experiment_id=exp["experiment_id"],
        experiment_name=exp["experiment_name"],
        status=exp["status"],
        created_at=exp["created_at"],
        started_at=exp.get("started_at"),
        completed_at=exp.get("completed_at"),
        progress_percent=exp.get("progress_percent", 0.0),
        current_step=exp.get("current_step", 0),
        total_steps=exp.get("total_steps", 0),
        metrics=ExperimentMetrics(**exp.get("metrics", {})),
        error_message=exp.get("error_message"),
    )


@router.post("/{experiment_id}/run", response_model=ExecutionResponse)
async def run_experiment(
    experiment_id: str,
    request: RunExperimentRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger experiment execution.
    
    Starts the experiment with the specified execution parameters.
    The experiment will run in the background.
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    exp = experiments_db[experiment_id]
    
    if exp["status"] == "running":
        raise HTTPException(status_code=400, detail="Experiment is already running")
    
    # Create execution record
    execution_id = str(uuid4())
    execution = {
        "execution_id": execution_id,
        "experiment_id": experiment_id,
        "parameters": request.execution_parameters.dict(),
        "resume_from": request.resume_from_checkpoint,
        "status": "starting",
        "started_at": datetime.now(timezone.utc),
    }
    
    executions_db[execution_id] = execution
    
    # Update experiment status
    exp["status"] = "running"
    exp["started_at"] = datetime.now(timezone.utc)
    
    # Estimate duration based on config
    training_config = exp["training_config"]
    estimated_hours = (
        training_config["num_epochs"] * 2.0  # Base time per epoch
        * (256 / training_config["batch_size"])  # Batch size adjustment
    )
    
    # Add background task for actual execution
    background_tasks.add_task(execute_experiment_task, experiment_id, execution_id)
    
    return ExecutionResponse(
        execution_id=execution_id,
        experiment_id=experiment_id,
        status="starting",
        estimated_duration_hours=estimated_hours,
        message="Experiment execution started",
    )


@router.get("", response_model=List[ExperimentStatus])
async def list_experiments(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List all experiments with optional filtering.
    """
    experiments = list(experiments_db.values())
    
    # Filter by status
    if status:
        experiments = [e for e in experiments if e["status"] == status]
    
    # Sort by created_at descending
    experiments.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Paginate
    experiments = experiments[offset:offset + limit]
    
    return [
        ExperimentStatus(
            experiment_id=exp["experiment_id"],
            experiment_name=exp["experiment_name"],
            status=exp["status"],
            created_at=exp["created_at"],
            started_at=exp.get("started_at"),
            completed_at=exp.get("completed_at"),
            progress_percent=exp.get("progress_percent", 0.0),
            current_step=exp.get("current_step", 0),
            total_steps=exp.get("total_steps", 0),
            metrics=ExperimentMetrics(**exp.get("metrics", {})),
            error_message=exp.get("error_message"),
        )
        for exp in experiments
    ]


@router.delete("/{experiment_id}")
async def cancel_experiment(experiment_id: str):
    """
    Cancel a running experiment.
    """
    if experiment_id not in experiments_db:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    exp = experiments_db[experiment_id]
    
    if exp["status"] != "running":
        raise HTTPException(status_code=400, detail="Experiment is not running")
    
    exp["status"] = "cancelled"
    exp["completed_at"] = datetime.now(timezone.utc)
    
    return {"message": f"Experiment {experiment_id} cancelled"}


# =============================================================================
# Background Tasks
# =============================================================================

async def execute_experiment_task(experiment_id: str, execution_id: str):
    """
    Background task for experiment execution.
    
    In production, this would:
    1. Load the model with specified configuration
    2. Prepare training data
    3. Run training loop
    4. Evaluate and save checkpoints
    5. Update metrics periodically
    """
    import asyncio
    
    try:
        exp = experiments_db[experiment_id]
        execution = executions_db[execution_id]
        
        # Simulate training progress
        total_steps = 1000
        exp["total_steps"] = total_steps
        
        for step in range(total_steps):
            if exp["status"] == "cancelled":
                break
            
            # Simulate step
            await asyncio.sleep(0.01)
            
            # Update progress
            exp["current_step"] = step + 1
            exp["progress_percent"] = (step + 1) / total_steps * 100
            
            # Update metrics periodically
            if step % 100 == 0:
                exp["metrics"] = {
                    "accuracy": 0.85 + (step / total_steps) * 0.07,
                    "latency_p99_ms": 450 - (step / total_steps) * 50,
                    "training_loss": 1.5 - (step / total_steps) * 1.2,
                }
        
        if exp["status"] != "cancelled":
            exp["status"] = "completed"
            exp["completed_at"] = datetime.now(timezone.utc)
            exp["metrics"]["accuracy"] = 0.92
            exp["metrics"]["latency_p99_ms"] = 400
        
        execution["status"] = "completed"
        
    except Exception as e:
        exp["status"] = "failed"
        exp["error_message"] = str(e)
        exp["completed_at"] = datetime.now(timezone.utc)
        execution["status"] = "failed"
