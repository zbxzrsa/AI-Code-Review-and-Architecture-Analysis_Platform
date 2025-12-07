"""
自动化网络学习系统 (Automatic Networked Learning System)

模块功能描述:
    为 V1（实验版）和 V3（隔离版）提供自主学习能力。

主要功能:
    - 多源数据收集（GitHub、ArXiv、技术博客）
    - 带质量评分的数据清洗管道
    - 带内存管理的无限学习
    - 技术废弃机制
    - 过期数据处理
    - 用户审核工作流

主要组件:
    - NetworkedLearningSystem: 网络学习系统主类
    - Collectors: 数据收集器（GitHub、ArXiv、博客）
    - DataCleaningPipeline: 数据清洗管道
    - StorageManager: 存储管理器
    - SystemMonitor: 系统监控

使用示例:
    from ai_core.networked_learning import NetworkedLearningSystem
    
    system = NetworkedLearningSystem(config)
    await system.start()
    
    # 或使用上下文管理器
    async with NetworkedLearningSystem(config) as system:
        await system.run_collection_cycle()

最后修改日期: 2024-12-07
"""

from .config import (
    NetworkedLearningConfig,
    DataSourcePriority,
    CollectionSchedule,
    QualityThresholds,
    RetentionPolicy,
    DeprecationCriteria,
)
from .system import NetworkedLearningSystem
from .collectors import (
    BaseCollector,
    GitHubCollector,
    ArXivCollector,
    TechBlogCollector,
)
from .pipeline import (
    DataCleaningPipeline,
    QualityAssessor,
    DuplicateDetector,
    FormatNormalizer,
)
from .storage import (
    StorageManager,
    LRUCache,
    ShardedStorage,
)
from .monitoring import (
    SystemMonitor,
    MetricsCollector,
    AlertManager,
)

__all__ = [
    # Config
    "NetworkedLearningConfig",
    "DataSourcePriority",
    "CollectionSchedule",
    "QualityThresholds",
    "RetentionPolicy",
    "DeprecationCriteria",
    # System
    "NetworkedLearningSystem",
    # Collectors
    "BaseCollector",
    "GitHubCollector",
    "ArXivCollector",
    "TechBlogCollector",
    # Pipeline
    "DataCleaningPipeline",
    "QualityAssessor",
    "DuplicateDetector",
    "FormatNormalizer",
    # Storage
    "StorageManager",
    "LRUCache",
    "ShardedStorage",
    # Monitoring
    "SystemMonitor",
    "MetricsCollector",
    "AlertManager",
]
