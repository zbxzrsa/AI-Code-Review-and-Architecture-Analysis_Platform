"""
Three-Version Self-Evolution Cycle

Implements the concurrent three-version AI system:
- V1 Experimentation: New technology testing with trial-and-error
- V2 Production: Stable user-facing AI with proven technologies  
- V3 Quarantine: Archive for failed experiments and deprecation

Each version has its own Version Control AI and Code Review AI.
"""

from .version_manager import (
    VersionManager,
    VersionState,
    Version,
)

from .experiment_framework import (
    TechnologyExperiment,
    ExperimentResult,
    ExperimentStatus,
    ExperimentFramework,
)

from .version_ai_engine import (
    VersionAIEngine,
    V1ExperimentalAI,
    V2ProductionAI,
    V3QuarantineAI,
)

from .self_evolution_cycle import (
    SelfEvolutionCycle,
    EvolutionMetrics,
    PromotionCriteria,
)

__all__ = [
    # Version Management
    "VersionManager",
    "VersionState",
    "Version",
    # Experiment Framework
    "TechnologyExperiment",
    "ExperimentResult",
    "ExperimentStatus",
    "ExperimentFramework",
    # AI Engines
    "VersionAIEngine",
    "V1ExperimentalAI",
    "V2ProductionAI",
    "V3QuarantineAI",
    # Evolution Cycle
    "SelfEvolutionCycle",
    "EvolutionMetrics",
    "PromotionCriteria",
]
