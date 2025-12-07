"""
模型架构模块 (Model Architecture Module)

模块功能描述:
    提供模块化 AI 架构和分布式训练功能。

主要功能:
    - 模块化架构和插件管理
    - 动态推理引擎
    - 多任务学习和迁移学习
    - 分布式训练和模型并行

主要组件:
    - ModularAIArchitecture: 模块化AI架构
    - PluginManager: 插件管理器
    - ReasoningEngine: 推理引擎
    - MultiTaskLearner: 多任务学习器
    - DistributedTrainer: 分布式训练器

最后修改日期: 2024-12-07
"""

from .modular_architecture import ModularAIArchitecture, PluginManager
from .reasoning_engine import ReasoningEngine, DynamicRouter
from .multi_task import MultiTaskLearner, TransferLearning
from .distributed_training import DistributedTrainer, ModelParallel

__all__ = [
    'ModularAIArchitecture',
    'PluginManager',
    'ReasoningEngine',
    'DynamicRouter',
    'MultiTaskLearner',
    'TransferLearning',
    'DistributedTrainer',
    'ModelParallel'
]
