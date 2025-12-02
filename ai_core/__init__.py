"""
AI Core Module - Self-Evolving AI Training System

This module provides:
- AI model version control with Git-based tracking
- Continuous learning framework with incremental/online learning
- Data cleaning pipeline with automated quality assessment
- Modular AI architecture for general problem-solving

Author: AI Code Review Platform
Version: 1.0.0
"""

from .version_control import ModelVersionControl, ModelRegistry
from .continuous_learning import ContinuousLearner, KnowledgeDistillation
from .data_pipeline import DataCleaningPipeline, QualityAssessor
from .model_architecture import ModularAIArchitecture, ReasoningEngine

__all__ = [
    'ModelVersionControl',
    'ModelRegistry',
    'ContinuousLearner',
    'KnowledgeDistillation',
    'DataCleaningPipeline',
    'QualityAssessor',
    'ModularAIArchitecture',
    'ReasoningEngine'
]

__version__ = '1.0.0'
