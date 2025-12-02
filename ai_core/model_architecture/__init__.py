"""Model Architecture Module"""

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
