"""
AI Core Module - Self-Evolving AI Training System

This module provides:
- AI model version control with Git-based tracking
- Continuous learning framework with incremental/online learning
- Data cleaning pipeline with automated quality assessment
- Modular AI architecture for general problem-solving
- Three-version self-evolution cycle (V1/V2/V3)
- Automated bug detection and fixing

Author: AI Code Review Platform
Version: 2.0.0
"""

from .version_control import ModelVersionControl, ModelRegistry
from .continuous_learning import ContinuousLearner, KnowledgeDistillation
from .data_pipeline import DataCleaningPipeline, QualityAssessor
from .model_architecture import ModularAIArchitecture, ReasoningEngine

# Self-Evolution modules
from .self_evolution import (
    BugFixerEngine,
    AutoFixCycle,
    FixVerifier,
    FixCycleConfig,
    FixCyclePhase,
    FixStrategy,
    create_bug_fixer,
    create_auto_fix_cycle,
)

# Three-Version Cycle modules
from .three_version_cycle import (
    VersionManager,
    SelfEvolutionCycle,
    ExperimentFramework,
    V1ExperimentalAI,
    V2ProductionAI,
    V3QuarantineAI,
)

__all__ = [
    # Version Control
    'ModelVersionControl',
    'ModelRegistry',
    # Continuous Learning
    'ContinuousLearner',
    'KnowledgeDistillation',
    # Data Pipeline
    'DataCleaningPipeline',
    'QualityAssessor',
    # Model Architecture
    'ModularAIArchitecture',
    'ReasoningEngine',
    # Self-Evolution
    'BugFixerEngine',
    'AutoFixCycle',
    'FixVerifier',
    'FixCycleConfig',
    'FixCyclePhase',
    'FixStrategy',
    'create_bug_fixer',
    'create_auto_fix_cycle',
    # Three-Version Cycle
    'VersionManager',
    'SelfEvolutionCycle',
    'ExperimentFramework',
    'V1ExperimentalAI',
    'V2ProductionAI',
    'V3QuarantineAI',
]

__version__ = '2.0.0'
