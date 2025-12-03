"""
Experiment Framework

Framework for testing new AI technologies in V1 Experimentation zone.
Integrates knowledge from LLMs-from-scratch repository.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Experiment status."""
    PENDING = "pending"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"


class TechnologyCategory(str, Enum):
    """Categories of technologies."""
    ATTENTION = "attention"
    ARCHITECTURE = "architecture"
    TRAINING = "training"
    OPTIMIZATION = "optimization"
    TOKENIZATION = "tokenization"
    INFERENCE = "inference"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    name: str
    category: TechnologyCategory
    description: str
    technology_type: str
    technology_config: Dict[str, Any]
    max_duration_hours: int = 24
    min_samples: int = 1000
    accuracy_threshold: float = 0.85
    error_rate_threshold: float = 0.05
    latency_threshold_ms: float = 3000
    traffic_percentage: float = 0.10


@dataclass
class ExperimentResult:
    """Result of an experiment."""
    experiment_id: str
    status: ExperimentStatus
    total_samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    total_cost: float = 0.0
    cost_per_request: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    accuracy_delta: float = 0.0
    latency_delta: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    should_promote: bool = False
    should_quarantine: bool = False
    recommendation: str = ""


# Pre-defined technologies from LLMs-from-scratch
PREDEFINED_TECHNOLOGIES = {
    # Attention Mechanisms
    "multi_head_attention": {
        "category": TechnologyCategory.ATTENTION,
        "description": "Standard Multi-Head Attention",
        "source": "LLMs-from-scratch Ch03",
        "default_config": {"attention_type": "multi_head", "num_heads": 8},
    },
    "grouped_query_attention": {
        "category": TechnologyCategory.ATTENTION,
        "description": "Grouped-Query Attention (GQA)",
        "source": "LLMs-from-scratch Ch04/04_gqa",
        "default_config": {"attention_type": "gqa", "num_heads": 8, "num_kv_heads": 2},
    },
    "sliding_window_attention": {
        "category": TechnologyCategory.ATTENTION,
        "description": "Sliding Window Attention (SWA)",
        "source": "LLMs-from-scratch Ch04/06_swa",
        "default_config": {"attention_type": "swa", "window_size": 4096},
    },
    # Architectures
    "llama_architecture": {
        "category": TechnologyCategory.ARCHITECTURE,
        "description": "Llama 3.2 with RoPE and RMSNorm",
        "source": "LLMs-from-scratch Ch05/07_gpt_to_llama",
        "default_config": {"architecture_type": "llama", "use_rope": True},
    },
    "mixture_of_experts": {
        "category": TechnologyCategory.ARCHITECTURE,
        "description": "Mixture of Experts (MoE)",
        "source": "LLMs-from-scratch Ch04/07_moe",
        "default_config": {"num_experts": 8, "num_experts_per_tok": 2},
    },
    # Training
    "direct_preference_optimization": {
        "category": TechnologyCategory.TRAINING,
        "description": "DPO for LLM alignment",
        "source": "LLMs-from-scratch Ch07/04_preference-tuning-with-dpo",
        "default_config": {"training_type": "dpo", "dpo_beta": 0.1},
    },
    # Optimizations
    "kv_cache_optimization": {
        "category": TechnologyCategory.OPTIMIZATION,
        "description": "KV Cache for efficient generation",
        "source": "LLMs-from-scratch Ch04/03_kv-cache",
        "default_config": {"use_kv_cache": True},
    },
    "flash_attention": {
        "category": TechnologyCategory.OPTIMIZATION,
        "description": "Flash Attention for memory efficiency",
        "source": "LLMs-from-scratch Ch03/02_bonus",
        "default_config": {"use_flash_attention": True},
    },
}


@dataclass
class TechnologyExperiment:
    """Active technology experiment."""
    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.PENDING
    result: Optional[ExperimentResult] = None
    samples: List[Dict[str, Any]] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    
    def record_sample(self, success: bool, latency_ms: float, accuracy: float = 0):
        """Record a sample result."""
        self.samples.append({
            "success": success,
            "latency_ms": latency_ms,
            "accuracy": accuracy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.latencies.append(latency_ms)
        if not success:
            self.errors.append(f"Error at sample {len(self.samples)}")


class ExperimentFramework:
    """
    Framework for managing technology experiments.
    
    Handles the V1 experimentation cycle:
    1. Register new technologies
    2. Run experiments with traffic splitting
    3. Evaluate results against thresholds
    4. Recommend promotion or quarantine
    """
    
    def __init__(self, version_manager=None, event_bus=None):
        self.version_manager = version_manager
        self.event_bus = event_bus
        self._experiments: Dict[str, TechnologyExperiment] = {}
        self._active_experiments: List[str] = []
        self._lock = asyncio.Lock()
    
    async def create_experiment(
        self,
        technology_type: str,
        name: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> TechnologyExperiment:
        """Create a new experiment from predefined or custom technology."""
        tech_info = PREDEFINED_TECHNOLOGIES.get(technology_type, {})
        
        config = ExperimentConfig(
            experiment_id=str(uuid.uuid4()),
            name=name or technology_type,
            category=tech_info.get("category", TechnologyCategory.OPTIMIZATION),
            description=tech_info.get("description", "Custom experiment"),
            technology_type=technology_type,
            technology_config={**tech_info.get("default_config", {}), **(custom_config or {})},
        )
        
        experiment = TechnologyExperiment(
            experiment_id=config.experiment_id,
            config=config,
        )
        
        self._experiments[config.experiment_id] = experiment
        logger.info(f"Created experiment: {config.name} ({config.experiment_id})")
        
        return experiment
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment."""
        async with self._lock:
            exp = self._experiments.get(experiment_id)
            if not exp:
                return False
            
            exp.status = ExperimentStatus.RUNNING
            exp.started_at = datetime.now(timezone.utc)
            self._active_experiments.append(experiment_id)
            
            logger.info(f"Started experiment: {exp.config.name}")
            return True
    
    async def record_result(
        self,
        experiment_id: str,
        success: bool,
        latency_ms: float,
        accuracy: float = 0,
    ):
        """Record a result for an experiment."""
        exp = self._experiments.get(experiment_id)
        if exp and exp.status == ExperimentStatus.RUNNING:
            exp.record_sample(success, latency_ms, accuracy)
    
    async def evaluate_experiment(self, experiment_id: str) -> ExperimentResult:
        """Evaluate experiment results and generate recommendation."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp.status = ExperimentStatus.EVALUATING
        config = exp.config
        
        # Calculate metrics
        total = len(exp.samples)
        successes = sum(1 for s in exp.samples if s["success"])
        accuracies = [s["accuracy"] for s in exp.samples if s["accuracy"] > 0]
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.COMPLETED,
            total_samples=total,
            accuracy=sum(accuracies) / len(accuracies) if accuracies else 0,
            error_count=len(exp.errors),
            error_rate=(total - successes) / total if total > 0 else 0,
            started_at=exp.started_at,
            completed_at=datetime.now(timezone.utc),
        )
        
        # Calculate latency percentiles
        if exp.latencies:
            sorted_lat = sorted(exp.latencies)
            result.latency_p50 = sorted_lat[int(len(sorted_lat) * 0.50)]
            result.latency_p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
            result.latency_p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
        
        # Determine recommendation
        meets_accuracy = result.accuracy >= config.accuracy_threshold
        meets_error = result.error_rate <= config.error_rate_threshold
        meets_latency = result.latency_p95 <= config.latency_threshold_ms
        meets_samples = result.total_samples >= config.min_samples
        
        if meets_accuracy and meets_error and meets_latency and meets_samples:
            result.should_promote = True
            result.recommendation = "PROMOTE: All thresholds met"
        elif result.error_rate > 0.20 or result.accuracy < 0.70:
            result.should_quarantine = True
            result.recommendation = "QUARANTINE: Critical threshold violations"
        else:
            result.recommendation = "CONTINUE: Needs more data or tuning"
        
        exp.result = result
        exp.status = result.status
        
        logger.info(f"Evaluated {exp.config.name}: {result.recommendation}")
        return result
    
    async def get_experiment(self, experiment_id: str) -> Optional[TechnologyExperiment]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)
    
    async def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[TechnologyExperiment]:
        """List all experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return experiments
    
    async def get_active_experiments(self) -> List[TechnologyExperiment]:
        """Get currently running experiments."""
        return [
            self._experiments[eid]
            for eid in self._active_experiments
            if eid in self._experiments
        ]
    
    def get_available_technologies(self) -> Dict[str, Any]:
        """Get list of predefined technologies available for experimentation."""
        return {
            name: {
                "category": info["category"].value,
                "description": info["description"],
                "source": info["source"],
            }
            for name, info in PREDEFINED_TECHNOLOGIES.items()
        }
