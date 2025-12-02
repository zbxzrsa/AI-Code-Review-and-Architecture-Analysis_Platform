"""
API Routers for V1 VC-AI Service.

Provides endpoints for:
- Experiment management
- Inference
- Evaluation and metrics
- Promotion
"""

from .experiments import router as experiments_router
from .inference import router as inference_router
from .evaluation import router as evaluation_router

__all__ = [
    "experiments_router",
    "inference_router",
    "evaluation_router",
]
