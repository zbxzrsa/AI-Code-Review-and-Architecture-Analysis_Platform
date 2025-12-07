"""
Practical Deployment Module

Plan A: Lightweight Continuous Learning (Production-Ready)

A cost-effective, production-ready approach:
- Frozen base model with LoRA adapters
- RAG system for real-time information retrieval
- Periodic retraining scheduler
- Quantization (INT8/INT4) for efficiency
- Model distillation for deployment

Cost: Controllable (~$1M-5M/year for enterprise)

Modules:
- lora: LoRA adapter management
- rag: Retrieval-Augmented Generation
- quantization: INT8/INT4/GPTQ/AWQ quantization
- retraining: Periodic retraining scheduler
- fault_tolerance: Checkpointing and recovery
- cost_control: Usage and cost monitoring
"""

from .config import (
    QuantizationType,
    RetrainingFrequency,
    PracticalDeploymentConfig,
)
from .lora import LoRALayer, LoRAAdapterManager
from .rag import RAGDocument, RAGIndex, RAGSystem
from .quantization import ModelQuantizer, INT4Quantizer
from .retraining import RetrainingDataCollector, RetrainingScheduler
from .distillation import (
    ModelDistiller,
    StudentModelConfig,
    StudentModelBuilder,
    StudentSizePreset,
    GenericTransformer,
)
from .fault_tolerance import HealthChecker, FaultToleranceManager
from .cost_control import CostController
from .system import (
    PracticalDeploymentSystem,
    SystemState,
    ContextManagerState,
    StartupError,
    ShutdownError,
)

__all__ = [
    # Config
    "QuantizationType",
    "RetrainingFrequency", 
    "PracticalDeploymentConfig",
    # LoRA
    "LoRALayer",
    "LoRAAdapterManager",
    # RAG
    "RAGDocument",
    "RAGIndex",
    "RAGSystem",
    # Quantization
    "ModelQuantizer",
    "INT4Quantizer",
    # Retraining
    "RetrainingDataCollector",
    "RetrainingScheduler",
    # Distillation
    "ModelDistiller",
    "StudentModelConfig",
    "StudentModelBuilder",
    "StudentSizePreset",
    "GenericTransformer",
    # Fault Tolerance
    "HealthChecker",
    "FaultToleranceManager",
    # Cost Control
    "CostController",
    # Main System
    "PracticalDeploymentSystem",
    # Context Manager Types
    "SystemState",
    "ContextManagerState",
    "StartupError",
    "ShutdownError",
]
