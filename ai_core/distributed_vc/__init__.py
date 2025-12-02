"""
Distributed Version Control AI System

A microservice-based AI system with:
- Real-time network learning
- Dual-loop update mechanism (Project + AI self-iteration)
- Continuous learning (7×24)
- Automatic version comparison and merging
- Safe rollback and high availability

Performance Standards:
- Learning delay: < 5 minutes
- Version iteration cycle: ≤ 24 hours
- System availability: > 99.9%
- Auto merge success rate: > 95%
"""

from .core_module import DistributedVCAI, VCAIConfig
from .learning_engine import OnlineLearningEngine, LearningChannel
from .dual_loop import DualLoopUpdater, ProjectLoop, AIIterationLoop
from .version_engine import VersionComparisonEngine, AutoMerger
from .monitoring import PerformanceMonitor, LearningMetrics
from .rollback import SafeRollbackManager
from .protocol import BidirectionalProtocol

__all__ = [
    'DistributedVCAI',
    'VCAIConfig',
    'OnlineLearningEngine',
    'LearningChannel',
    'DualLoopUpdater',
    'ProjectLoop',
    'AIIterationLoop',
    'VersionComparisonEngine',
    'AutoMerger',
    'PerformanceMonitor',
    'LearningMetrics',
    'SafeRollbackManager',
    'BidirectionalProtocol'
]
