"""
Configuration module for V1 VC-AI Service.

Contains all configuration classes for:
- Model architecture
- Training parameters
- Inference settings
- Evaluation metrics
"""

from .model_config import (
    ModelConfig,
    LoRAConfig,
    QuantizationConfig,
    AttentionConfig,
)
from .training_config import (
    TrainingConfig,
    DataConfig,
    CurriculumConfig,
    MultiTaskConfig,
    ContrastiveLearningConfig,
)
from .inference_config import (
    InferenceConfig,
    GenerationConfig,
    BatchingConfig,
    CachingConfig,
)
from .evaluation_config import (
    EvaluationConfig,
    PromotionConfig,
    MetricThresholds,
)

__all__ = [
    "ModelConfig",
    "LoRAConfig", 
    "QuantizationConfig",
    "AttentionConfig",
    "TrainingConfig",
    "DataConfig",
    "CurriculumConfig",
    "MultiTaskConfig",
    "ContrastiveLearningConfig",
    "InferenceConfig",
    "GenerationConfig",
    "BatchingConfig",
    "CachingConfig",
    "EvaluationConfig",
    "PromotionConfig",
    "MetricThresholds",
]
