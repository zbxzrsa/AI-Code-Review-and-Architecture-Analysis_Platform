"""
Three-Version Self-Evolution Cycle

Implements the concurrent three-version AI system:
- V1 Experimentation: New technology testing with trial-and-error
- V2 Production: Stable user-facing AI with proven technologies  
- V3 Quarantine: Archive for failed experiments and deprecation

Each version has its own Version Control AI (VC-AI) and Code Review AI (CR-AI).

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                    THREE-VERSION SPIRAL EVOLUTION                       │
├─────────────────────────────────────────────────────────────────────────┤
│  V1 (New/Experiment)     V2 (Stable/Production)    V3 (Old/Quarantine) │
│  ┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐ │
│  │ V1-VCAI (Admin) │     │ V2-VCAI (Admin) │       │ V3-VCAI (Admin) │ │
│  │ - Experiments   │────▶│ - Fixes V1 bugs │──────▶│ - Compares      │ │
│  │ - Trial/Error   │     │ - Optimizes     │       │ - Excludes      │ │
│  └─────────────────┘     └─────────────────┘       └─────────────────┘ │
│  ┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐ │
│  │ V1-CRAI (Test)  │     │ V2-CRAI (Users) │       │ V3-CRAI (Ref)   │ │
│  │ - Shadow tests  │     │ - Production    │       │ - Baseline      │ │
│  └─────────────────┘     └─────────────────┘       └─────────────────┘ │
│                                                                         │
│  SPIRAL: V1 → V2 (promote) → V3 (degrade) → V1 (re-eval) → ...        │
└─────────────────────────────────────────────────────────────────────────┘
"""

from .version_manager import (
    VersionManager,
    VersionState,
    Version,
    VersionConfig,
    VersionMetrics,
    Technology,
    PromotionRecord,
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
    create_version_ai,
)

from .self_evolution_cycle import (
    SelfEvolutionCycle,
    EvolutionMetrics,
    PromotionCriteria,
    EnhancedSelfEvolutionCycle,
)

from .cross_version_feedback import (
    CrossVersionFeedbackSystem,
    V1Error,
    V2Fix,
    FeedbackRecord,
    ErrorType,
    FixStatus,
)

from .v3_comparison_engine import (
    V3ComparisonEngine,
    TechnologyProfile,
    ComparisonResult,
    ExclusionReason,
    ExclusionDecision,
)

from .dual_ai_coordinator import (
    DualAICoordinator,
    AIType,
    AIStatus,
    AccessLevel,
    AIInstance,
    VersionAIPair,
    DualAIRequestHandler,
)

from .spiral_evolution_manager import (
    SpiralEvolutionManager,
    CyclePhase,
    EvolutionEvent,
    EvolutionCycleState,
    SpiralCycleConfig,
)

__all__ = [
    # Version Management
    "VersionManager",
    "VersionState",
    "Version",
    "VersionConfig",
    "VersionMetrics",
    "Technology",
    "PromotionRecord",
    
    # Experiment Framework
    "TechnologyExperiment",
    "ExperimentResult",
    "ExperimentStatus",
    "ExperimentFramework",
    
    # AI Engines (per version)
    "VersionAIEngine",
    "V1ExperimentalAI",
    "V2ProductionAI",
    "V3QuarantineAI",
    "create_version_ai",
    
    # Self Evolution Cycle
    "SelfEvolutionCycle",
    "EvolutionMetrics",
    "PromotionCriteria",
    "EnhancedSelfEvolutionCycle",
    
    # Cross-Version Feedback (V2 fixes V1)
    "CrossVersionFeedbackSystem",
    "V1Error",
    "V2Fix",
    "FeedbackRecord",
    "ErrorType",
    "FixStatus",
    
    # V3 Comparison & Exclusion
    "V3ComparisonEngine",
    "TechnologyProfile",
    "ComparisonResult",
    "ExclusionReason",
    "ExclusionDecision",
    
    # Dual-AI Coordinator (VCAI + CRAI per version)
    "DualAICoordinator",
    "AIType",
    "AIStatus",
    "AccessLevel",
    "AIInstance",
    "VersionAIPair",
    "DualAIRequestHandler",
    
    # Spiral Evolution Manager (full cycle orchestration)
    "SpiralEvolutionManager",
    "CyclePhase",
    "EvolutionEvent",
    "EvolutionCycleState",
    "SpiralCycleConfig",
]
