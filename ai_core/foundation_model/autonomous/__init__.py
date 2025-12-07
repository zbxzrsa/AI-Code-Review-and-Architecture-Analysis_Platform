"""
自主学习模块 (Autonomous Learning Module)

模块功能描述:
    具有自演化 AI 能力的模块化自主学习系统。

模块结构:
    - config.py: 配置类和枚举
    - online_learning.py: 在线学习缓冲区和模块
    - memory/
        - episodic.py: 情节记忆
        - semantic.py: 语义记忆
        - working.py: 工作记忆
        - manager.py: 记忆管理
    - evaluation.py: 自评估系统
    - safety.py: 安全监控器

主要组件:
    - AutonomousConfig: 自主学习配置
    - OnlineLearningModule: 在线学习模块
    - MemoryManagement: 记忆管理
    - SelfEvaluationSystem: 自评估系统
    - SafetyMonitor: 安全监控器

使用示例:
    from ai_core.foundation_model.autonomous import (
        AutonomousConfig,
        OnlineLearningModule,
        MemoryManagement,
        SelfEvaluationSystem,
        SafetyMonitor,
    )

状态: ✅ 已重构 - 模块化结构完成
最后修改日期: 2024-12-07
"""

# Import from modular components
from .config import (
    # Enums
    LearningMode,
    MemoryType,
    SafetyLevel,
    ExceptionSeverity,
    LearningErrorCode,
    # Config
    AutonomousConfig,
    LearningEvent,
    KnowledgeGap,
    LearningException,
    BenchmarkResult,
    Episode,
)

from .online_learning import (
    OnlineLearningBuffer,
    OnlineLearningModule,
    LearningStats,
)

from .memory import (
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
    MemoryManagement,
)

from .evaluation import (
    SelfEvaluationSystem,
    EvaluationConfig,
)

from .safety import (
    SafetyMonitor,
    SafetyConfig,
    SafetyViolation,
)

# Backward compatibility: Also import from original file if available
try:
    from ..autonomous_learning import (
        AutonomousLearningAgent,
        RAGSystem,
        ToolUseSystem,
        KnowledgeIntegration,
    )
    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False
    # Placeholders for type hints
    AutonomousLearningAgent = None
    RAGSystem = None
    ToolUseSystem = None
    KnowledgeIntegration = None


__all__ = [
    # Enums
    "LearningMode",
    "MemoryType",
    "SafetyLevel",
    "ExceptionSeverity",
    "LearningErrorCode",
    # Config
    "AutonomousConfig",
    "LearningEvent",
    "KnowledgeGap",
    "LearningException",
    "BenchmarkResult",
    "Episode",
    # Online Learning
    "OnlineLearningBuffer",
    "OnlineLearningModule",
    "LearningStats",
    # Memory
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
    "MemoryManagement",
    # Evaluation
    "SelfEvaluationSystem",
    "EvaluationConfig",
    # Safety
    "SafetyMonitor",
    "SafetyConfig",
    "SafetyViolation",
    # Legacy (from original file)
    "AutonomousLearningAgent",
    "RAGSystem",
    "ToolUseSystem",
    "KnowledgeIntegration",
]
