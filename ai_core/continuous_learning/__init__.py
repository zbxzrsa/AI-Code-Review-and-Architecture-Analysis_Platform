"""
持续学习模块 (Continuous Learning Module)

模块功能描述:
    提供 AI 模型的持续学习和知识保留能力。

主要功能:
    - 增量学习和在线学习
    - 知识蒸馏和模型融合
    - 长期记忆和经验回放
    - 弹性权重合并（EWC）防止灾难性遗忘

主要组件:
    - ContinuousLearner: 持续学习器
    - OnlineLearner: 在线学习器
    - KnowledgeDistillation: 知识蒸馏
    - LongTermMemory: 长期记忆系统
    - IncrementalLearner: 增量学习器

最后修改日期: 2024-12-07
"""

from .continuous_learner import ContinuousLearner, OnlineLearner
from .knowledge_distillation import KnowledgeDistillation, ModelFusion
from .memory_system import LongTermMemory, ExperienceReplay
from .incremental_learning import IncrementalLearner, EWC

__all__ = [
    'ContinuousLearner',
    'OnlineLearner',
    'KnowledgeDistillation',
    'ModelFusion',
    'LongTermMemory',
    'ExperienceReplay',
    'IncrementalLearner',
    'EWC'
]
