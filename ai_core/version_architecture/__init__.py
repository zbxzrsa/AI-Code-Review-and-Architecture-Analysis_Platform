"""
版本架构核心模块 (Version Architecture Core Module)

实现三版本螺旋演化架构：
- V2 稳定版：100%稳定可靠，面向用户的生产版本
- V1 开发版：实验性技术集成，完整沙箱环境和错误隔离
- V3 基准版：技术对比参考，保留完整历史版本数据和性能参数

每个版本独立配备：
- Version Control AI (VC-AI)：技术更新、错误修复、兼容性优化、功能增强
- User Code AI (UC-AI)：实时代码审查、错误诊断、修正建议、修复教程

最后修改日期: 2024-12-20
"""

from .version_config import (
    VersionType,
    VersionRole,
    VersionConfig,
    VersionIsolationConfig,
    SandboxConfig,
    ErrorIsolationConfig,
)

from .version_ai_system import (
    VersionAISystem,
    VersionControlAI,
    UserCodeAI,
    AISecurityChannel,
    AICoordinationProtocol,
)

from .version_collaboration import (
    VersionCollaborationEngine,
    TechnologyComparisonEngine,
    ExperimentFramework,
    TripleAIDiagnosisSystem,
    TechnologyPromotionPipeline,
)

from .update_standards import (
    TechnologyUpdateValidator,
    DevelopmentCycleTracker,
    PerformanceImprovementValidator,
    TripleAIVerificationSystem,
    StressTestFramework,
    UserScenarioSimulator,
)

from .monitoring_rollback import (
    VersionMonitoringSystem,
    RealTimeMetricsCollector,
    RollbackManager,
    MinuteLevelRollbackPlan,
    HealthCheckSystem,
)

from .resource_scheduler import (
    DynamicResourceScheduler,
    ResourceAllocationPolicy,
    ComputeResourcePool,
    PriorityBasedScheduler,
)

from .api_compatibility import (
    APIVersionManager,
    CompatibilityLayer,
    VersionTransitionManager,
    BackwardCompatibilityValidator,
)

from .documentation_system import (
    DocumentationSystem,
    TechnologyUpdateLog,
    EvaluationParameter,
    DecisionRecord,
    VersionSwitchManual,
    VersionSwitchStep,
)

__all__ = [
    # Version Configuration
    "VersionType",
    "VersionRole",
    "VersionConfig",
    "VersionIsolationConfig",
    "SandboxConfig",
    "ErrorIsolationConfig",
    
    # Version AI System
    "VersionAISystem",
    "VersionControlAI",
    "UserCodeAI",
    "AISecurityChannel",
    "AICoordinationProtocol",
    
    # Version Collaboration
    "VersionCollaborationEngine",
    "TechnologyComparisonEngine",
    "ExperimentFramework",
    "TripleAIDiagnosisSystem",
    "TechnologyPromotionPipeline",
    
    # Update Standards
    "TechnologyUpdateValidator",
    "DevelopmentCycleTracker",
    "PerformanceImprovementValidator",
    "TripleAIVerificationSystem",
    "StressTestFramework",
    "UserScenarioSimulator",
    
    # Monitoring & Rollback
    "VersionMonitoringSystem",
    "RealTimeMetricsCollector",
    "RollbackManager",
    "MinuteLevelRollbackPlan",
    "HealthCheckSystem",
    
    # Resource Scheduler
    "DynamicResourceScheduler",
    "ResourceAllocationPolicy",
    "ComputeResourcePool",
    "PriorityBasedScheduler",
    
    # API Compatibility
    "APIVersionManager",
    "CompatibilityLayer",
    "VersionTransitionManager",
    "BackwardCompatibilityValidator",
    
    # Documentation System
    "DocumentationSystem",
    "TechnologyUpdateLog",
    "EvaluationParameter",
    "DecisionRecord",
    "VersionSwitchManual",
    "VersionSwitchStep",
]

