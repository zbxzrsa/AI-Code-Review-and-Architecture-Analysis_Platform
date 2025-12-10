"""
三版本自演化循环 (Three-Version Self-Evolution Cycle)

模块功能描述:
    实现并发的三版本 AI 系统，支持持续演化和改进。

版本说明:
    - V1 实验版: 新技术测试，允许试错
    - V2 生产版: 面向用户的稳定 AI，使用经过验证的技术
    - V3 隔离版: 失败实验和废弃技术的归档

每个版本都有自己的版本控制 AI (VC-AI) 和代码审查 AI (CR-AI)。

架构图:
┌─────────────────────────────────────────────────────────────────────────┐
│                       三版本螺旋演化架构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  V1 (新/实验)           V2 (稳定/生产)          V3 (旧/隔离)           │
│  ┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐ │
│  │ V1-VCAI (管理)  │     │ V2-VCAI (管理)  │       │ V3-VCAI (管理)  │ │
│  │ - 实验          │────▶│ - 修复V1错误    │──────▶│ - 比较          │ │
│  │ - 试错          │     │ - 优化          │       │ - 排除          │ │
│  └─────────────────┘     └─────────────────┘       └─────────────────┘ │
│  ┌─────────────────┐     ┌─────────────────┐       ┌─────────────────┐ │
│  │ V1-CRAI (测试)  │     │ V2-CRAI (用户)  │       │ V3-CRAI (参考)  │ │
│  │ - 影子测试      │     │ - 生产          │       │ - 基准          │ │
│  └─────────────────┘     └─────────────────┘       └─────────────────┘ │
│                                                                         │
│  螺旋: V1 → V2 (升级) → V3 (降级) → V1 (重评估) → ...                 │
└─────────────────────────────────────────────────────────────────────────┘

主要组件:
    - VersionManager: 版本管理器
    - ExperimentFramework: 实验框架
    - SelfEvolutionCycle: 自演化循环
    - CrossVersionFeedbackSystem: 跨版本反馈系统
    - DualAICoordinator: 双AI协调器
    - SpiralEvolutionManager: 螺旋演化管理器

最后修改日期: 2024-12-07
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
    # Technology Elimination
    TechEliminationConfig,
    TechEliminationManager,
    EliminationRecord,
)

from .parallel_orchestrator import (
    ThreeVersionOrchestrator,
    ParallelVersionStatus,
    create_three_version_system,
    run_three_version_demo,
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
    
    # Technology Elimination
    "TechEliminationConfig",
    "TechEliminationManager",
    "EliminationRecord",
    
    # Three-Version Parallel Orchestrator (MAIN ENTRY POINT)
    "ThreeVersionOrchestrator",
    "ParallelVersionStatus",
    "create_three_version_system",
    "run_three_version_demo",
]

