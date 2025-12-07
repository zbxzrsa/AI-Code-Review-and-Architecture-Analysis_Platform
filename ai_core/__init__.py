"""
AI 核心模块 - 自演化 AI 训练系统 (AI Core Module - Self-Evolving AI Training System)

模块功能描述:
    提供完整的自演化 AI 训练和管理功能。

主要功能:
    - 基于 Git 的 AI 模型版本控制
    - 支持增量/在线学习的持续学习框架
    - 带自动质量评估的数据清洗管道
    - 通用问题解决的模块化 AI 架构
    - 三版本自演化循环（V1/V2/V3）
    - 自动化错误检测和修复

主要子模块:
    - version_control: 模型版本控制
    - continuous_learning: 持续学习框架
    - data_pipeline: 数据清洗管道
    - model_architecture: 模块化 AI 架构
    - self_evolution: 自演化模块
    - three_version_cycle: 三版本循环
    - distributed_vc: 分布式版本控制
    - foundation_model: 基础模型训练

作者: AI Code Review Platform
版本: 2.0.0
最后修改日期: 2024-12-07
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
