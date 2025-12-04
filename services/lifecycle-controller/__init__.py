"""
Lifecycle Controller Service

Manages the three-version self-evolution cycle:
- V1 (Experiment) → V2 (Production) promotion
- V2 (Production) → V3 (Quarantine) demotion  
- V3 (Quarantine) → V1 (Experiment) recovery

The cycle operates autonomously, ensuring continuous improvement
without manual intervention.

Usage:
    from services.lifecycle_controller import (
        CycleOrchestrator,
        LifecycleController,
        RecoveryManager,
    )
    
    # Initialize
    lifecycle = LifecycleController()
    recovery = RecoveryManager()
    orchestrator = CycleOrchestrator(lifecycle, recovery)
    
    # Start the self-evolution cycle
    await orchestrator.start()
    
    # Register a new experiment
    await orchestrator.register_new_experiment(
        version_id="v1-exp-001",
        model_version="gpt-4o",
        prompt_version="code-review-v4"
    )

Cycle Flow:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    SELF-EVOLUTION CYCLE                         │
    │                                                                 │
    │   ┌──────────┐    Shadow     ┌──────────┐    Gray-Scale        │
    │   │   V1     │ ──────────►   │   V2     │ ◄────────────┐       │
    │   │ Experiment│   Traffic    │ Production│              │       │
    │   └────▲─────┘               └─────┬────┘              │       │
    │        │                           │                    │       │
    │        │ Recovery                  │ SLO Breach        │       │
    │        │ (Gold-set)                │ Rollback          │       │
    │        │                           ▼                    │       │
    │   ┌────┴─────┐              ┌──────────┐               │       │
    │   │   V3     │ ◄────────────│ Demotion │───────────────┘       │
    │   │Quarantine│              └──────────┘                        │
    │   └──────────┘                                                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
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
