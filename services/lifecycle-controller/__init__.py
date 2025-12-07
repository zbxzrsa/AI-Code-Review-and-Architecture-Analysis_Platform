"""
生命周期控制器服务 (Lifecycle Controller Service)

模块功能描述:
    管理三版本自演化循环。

版本流转:
    - V1 (实验版) → V2 (生产版) 升级
    - V2 (生产版) → V3 (隔离版) 降级
    - V3 (隔离版) → V1 (实验版) 恢复

该循环自主运行，确保在无需人工干预的情况下持续改进。

主要组件:
    - CycleOrchestrator: 循环编排器
    - LifecycleController: 生命周期控制器
    - RecoveryManager: 恢复管理器
    - CycleMetricsCollector: 循环指标收集器
    - EventPublisher: 事件发布器

使用示例:
    from services.lifecycle_controller import (
        CycleOrchestrator,
        LifecycleController,
        RecoveryManager,
    )
    
    # 初始化
    lifecycle = LifecycleController()
    recovery = RecoveryManager()
    orchestrator = CycleOrchestrator(lifecycle, recovery)
    
    # 启动自演化循环
    await orchestrator.start()
    
    # 注册新实验
    await orchestrator.register_new_experiment(
        version_id="v1-exp-001",
        model_version="gpt-4o",
        prompt_version="code-review-v4"
    )

循环流程图:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                       自演化循环                                  │
    │                                                                 │
    │   ┌──────────┐    影子流量   ┌──────────┐    灰度发布          │
    │   │   V1     │ ──────────►   │   V2     │ ◄────────────┐       │
    │   │  实验版   │              │  生产版   │              │       │
    │   └────▲─────┘               └─────┬────┘              │       │
    │        │                           │                    │       │
    │        │ 恢复                      │ SLO 违规          │       │
    │        │ (金标集)                  │ 回滚              │       │
    │        │                           ▼                    │       │
    │   ┌────┴─────┐              ┌──────────┐               │       │
    │   │   V3     │ ◄────────────│   降级    │───────────────┘       │
    │   │  隔离版   │              └──────────┘                        │
    │   └──────────┘                                                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

最后修改日期: 2024-12-07
"""

from .controller import (
    LifecycleController,
    VersionState,
    VersionConfig,
    PromotionThresholds,
    EvaluationMetrics,
    EvaluationResult,
)

from .recovery_manager import (
    RecoveryManager,
    RecoveryConfig,
    RecoveryRecord,
    RecoveryStatus,
)

from .cycle_orchestrator import (
    CycleOrchestrator,
    CyclePhase,
    CycleEvent,
    CycleHealth,
)

from .metrics_collector import (
    CycleMetricsCollector,
    MetricType,
)

from .event_publisher import (
    EventPublisher,
    EventType,
    LifecycleEvent,
    RedisEventBackend,
    WebhookEventBackend,
    InMemoryEventBackend,
)

__all__ = [
    # Main orchestrator
    "CycleOrchestrator",
    "CyclePhase",
    "CycleEvent", 
    "CycleHealth",
    
    # Lifecycle controller
    "LifecycleController",
    "VersionState",
    "VersionConfig",
    "PromotionThresholds",
    "EvaluationMetrics",
    "EvaluationResult",
    
    # Recovery manager
    "RecoveryManager",
    "RecoveryConfig",
    "RecoveryRecord",
    "RecoveryStatus",
    
    # Metrics
    "CycleMetricsCollector",
    "MetricType",
    
    # Events
    "EventPublisher",
    "EventType",
    "LifecycleEvent",
    "RedisEventBackend",
    "WebhookEventBackend",
    "InMemoryEventBackend",
]

__version__ = "1.0.0"
