"""
分布式版本控制 AI 系统 (Distributed Version Control AI System)

模块功能描述:
    本模块实现了基于微服务架构的分布式版本控制 AI 系统，提供完整的代码版本管理、
    自动化学习和智能合并功能。系统采用事件驱动架构，支持水平扩展和高可用性部署。

主要子模块:
    - core_module: 核心微服务架构，包含服务发现、负载均衡和熔断器
    - learning_engine: 在线学习引擎，支持 7×24 小时持续学习
    - auto_network_learning: V1/V3 自动网络学习系统
    - data_cleansing_pipeline: 数据清洗流水线
    - infinite_learning_manager: 无限学习管理器，支持分层存储
    - data_lifecycle_manager: 数据生命周期管理器
    - dual_loop: 双循环更新器（项目循环 + AI 自迭代循环）
    - version_engine: 版本比较引擎和自动合并器
    - monitoring: 性能监控和学习指标
    - rollback: 安全回滚管理器
    - protocol: 双向通信协议

性能指标:
    - 学习延迟: < 5 分钟
    - 版本迭代周期: ≤ 24 小时
    - 系统可用性: > 99.9%
    - 自动合并成功率: > 95%

最后修改日期: 2024-12-07
"""

from .core_module import DistributedVCAI, VCAIConfig
from .learning_engine import (
    OnlineLearningEngine,
    LearningChannel,
    EnhancedLearningEngine,
    AutoNetworkLearningConfig,
    DataQualityFilter,
    AsyncRateLimiter,
)
from .auto_network_learning import (
    V1V3AutoLearningSystem,
    NetworkLearningConfig,
    DataSource,
    LearningData,
    LearningStatus,
    QualityAssessor,
    DataCleaner,
    create_learning_system,
)
from .data_cleansing_pipeline import (
    DataCleansingPipeline,
    CleansingConfig,
    CleansingResult,
    CleansingStage,
    RejectionReason,
    DeduplicationCache,
    ContentNormalizer,
    ValidationRule,
    create_integrated_pipeline,
)
from .infinite_learning_manager import (
    InfiniteLearningManager,
    MemoryConfig,
    LearningCheckpoint,
    StorageTier,
    MemoryPressureLevel,
    TieredStorageManager,
    MemoryMonitor,
)
from .data_lifecycle_manager import (
    DataLifecycleManager,
    DataLifecycleConfig,
    DataEntry,
    DataState,
    DeletionReason,
    DeletionRecord,
    ArchiveManager,
)
from .dual_loop import DualLoopUpdater, ProjectLoop, AIIterationLoop
from .version_engine import VersionComparisonEngine, AutoMerger
from .monitoring import PerformanceMonitor, LearningMetrics
from .rollback import SafeRollbackManager
from .protocol import BidirectionalProtocol

__all__ = [
    # Core
    'DistributedVCAI',
    'VCAIConfig',
    # Learning Engine
    'OnlineLearningEngine',
    'LearningChannel',
    # V1/V3 Enhanced Learning
    'EnhancedLearningEngine',
    'AutoNetworkLearningConfig',
    'DataQualityFilter',
    'AsyncRateLimiter',
    # V1/V3 Auto Network Learning
    'V1V3AutoLearningSystem',
    'NetworkLearningConfig',
    'DataSource',
    'LearningData',
    'LearningStatus',
    'QualityAssessor',
    'DataCleaner',
    'create_learning_system',
    # Data Cleansing Pipeline
    'DataCleansingPipeline',
    'CleansingConfig',
    'CleansingResult',
    'CleansingStage',
    'RejectionReason',
    'DeduplicationCache',
    'ContentNormalizer',
    'ValidationRule',
    'create_integrated_pipeline',
    # Infinite Learning Manager
    'InfiniteLearningManager',
    'MemoryConfig',
    'LearningCheckpoint',
    'StorageTier',
    'MemoryPressureLevel',
    'TieredStorageManager',
    'MemoryMonitor',
    # Data Lifecycle Manager
    'DataLifecycleManager',
    'DataLifecycleConfig',
    'DataEntry',
    'DataState',
    'DeletionReason',
    'DeletionRecord',
    'ArchiveManager',
    # Dual Loop
    'DualLoopUpdater',
    'ProjectLoop',
    'AIIterationLoop',
    # Version Engine
    'VersionComparisonEngine',
    'AutoMerger',
    # Monitoring
    'PerformanceMonitor',
    'LearningMetrics',
    # Rollback
    'SafeRollbackManager',
    # Protocol
    'BidirectionalProtocol',
]
