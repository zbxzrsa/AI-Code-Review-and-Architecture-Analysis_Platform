"""
Configuration classes for practical deployment.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class QuantizationType(str, Enum):
    """Quantization types for efficient inference."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    GPTQ = "gptq"
    AWQ = "awq"


class RetrainingFrequency(str, Enum):
    """Adapter retraining frequency."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_DEMAND = "on_demand"


@dataclass
class PracticalDeploymentConfig:
    """Configuration for practical lightweight deployment."""
    # Base model (frozen)
    base_model_path: str = "models/base"
    freeze_base_model: bool = True
    
    # LoRA configuration
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # RAG configuration
    enable_rag: bool = True
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    rag_index_path: str = "indices/rag"
    rag_use_faiss: bool = True  # Use FAISS for production
    
    # Retraining schedule
    retraining_frequency: RetrainingFrequency = RetrainingFrequency.WEEKLY
    retraining_data_path: str = "data/retraining"
    min_samples_for_retraining: int = 1000
    
    # Quantization
    quantization_type: QuantizationType = QuantizationType.INT8
    int4_compute_dtype: str = "float16"  # Compute dtype for INT4
    int4_quant_type: str = "nf4"  # nf4 or fp4
    int4_double_quant: bool = True  # Double quantization for memory savings
    
    # Distillation
    enable_distillation: bool = False
    student_model_size: str = "small"  # small, medium, large
    
    # Active learning
    enable_active_learning: bool = True
    uncertainty_threshold: float = 0.3
    
    # Cost control
    max_daily_tokens: int = 100_000_000  # 100M tokens/day
    max_monthly_cost_usd: float = 100_000
    
    # Fault tolerance
    checkpoint_interval_minutes: int = 30
    max_retries: int = 3
    health_check_interval_seconds: int = 60
