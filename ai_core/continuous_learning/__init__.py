"""Continuous Learning Module"""

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
