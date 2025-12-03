# Three-Version Self-Evolving Cycle Coordination Protocol

## Overview

This document describes the coordination protocol for the three-version self-evolving AI code review platform.

```
┌─────────────────────────────────────────────────────────────┐
│                   VERSION LIFECYCLE FLOW                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  V1 (Experimentation)                                        │
│  ├─ New AI models, prompts, routing strategies              │
│  ├─ Test on shadow traffic (5% of V2 load)                  │
│  ├─ Evaluation period: 7-14 days                            │
│  └─ Decision gate: Version Control AI evaluates             │
│      ↓                                                       │
│      ├─ PASS → Promote to V2 (Canary Deployment)            │
│      ├─ HOLD → Continue testing with modifications          │
│      └─ FAIL → Quarantine to V3                             │
│                                                              │
│  V2 (Production)                                             │
│  ├─ Stable, battle-tested version                           │
│  ├─ Serves 100% of user traffic                             │
│  ├─ SLO enforcement: <3s p95 latency, <2% error rate        │
│  └─ Continuous monitoring for degradation                   │
│                                                              │
│  V3 (Quarantine)                                             │
│  ├─ Archive of failed experiments                           │
│  ├─ Read-only, minimal resources                            │
│  └─ Quarterly review for re-evaluation                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Lifecycle Coordinator (`lifecycle_coordinator.py`)

Central orchestration for the entire cycle:

- Coordinates experiment creation and approval
- Routes evaluation decisions to appropriate managers
- Handles emergency rollbacks
- Provides system status overview

```python
from backend.shared.coordination import LifecycleCoordinator

coordinator = LifecycleCoordinator(
    event_bus=event_bus,
    db_connection=db,
    metrics_client=prometheus,
)

# Start coordinator
await coordinator.start()

# Create experiment
proposal = await coordinator.create_experiment(
    title="Test GPT-4.5 for security analysis",
    hypothesis="GPT-4.5 will improve vulnerability detection by 20%",
    success_criteria={
        "accuracy": "> 90%",
        "latency_p95": "< 3s",
        "cost_per_review": "< $0.15",
    },
)

# Evaluate and get decision
decision = await coordinator.evaluate_experiment(
    experiment_id=proposal.experiment_id,
    metrics={"accuracy": 0.92, "latency_p95": 2.5, "cost_per_review": 0.12},
)
# Returns: "PROMOTE", "HOLD", or "QUARANTINE"
```

### 2. Promotion Manager (`promotion_manager.py`)

Handles V1 → V2 promotion with canary deployment:

**Phases:**

1. **Validation** - Pre-promotion checks (metrics, load test, security)
2. **Phase 1** - 10% traffic for 24 hours
3. **Phase 2** - 50% traffic for 48 hours
4. **Phase 3** - 100% traffic for 7 days
5. **Finalization** - Archive old V2, update docs

```python
from backend.shared.coordination import PromotionManager

promotion = PromotionManager(event_bus=event_bus)

# Request promotion
request = await promotion.request_promotion(
    experiment_id="exp-123",
    evaluation_metrics={"accuracy": 0.95},
    confidence_score=0.92,
)

# Admin approves (triggers canary deployment)
await promotion.approve_promotion(request.request_id, "admin@example.com")
```

### 3. Quarantine Manager (`quarantine_manager.py`)

Handles V1 → V3 quarantine:

**Steps:**

1. **Evidence Capture** - Code, configs, logs, metrics
2. **Root Cause Analysis** - Categorize and identify cause
3. **Store in V3** - Read-only archive
4. **Blacklist Update** - Prevent similar experiments

```python
from backend.shared.coordination import QuarantineManager

quarantine = QuarantineManager(event_bus=event_bus)

# Quarantine failed experiment
record = await quarantine.quarantine_experiment(
    experiment_id="exp-456",
    failure_type="accuracy_degradation",
    failure_evidence={"accuracy": 0.72},
    error_logs="Model hallucination detected...",
    metrics_at_failure={"accuracy": 0.72, "error_rate": 0.15},
)

# Review for potential retry (quarterly)
await quarantine.review_quarantined(
    record_id=record.record_id,
    reviewer="admin@example.com",
    retry_approved=True,
    notes="New model available, worth retrying",
)
```

### 4. Health Monitor (`health_monitor.py`)

Continuous V2 production monitoring:

**Thresholds:**
| Metric | Warning | Critical | Auto-Remediate |
|--------|---------|----------|----------------|
| error_rate | 2% | 5% | Rollback |
| latency_p95 | 3000ms | 10000ms | Scale up |
| accuracy_rate | 90% | 85% | - |
| cpu_utilization | 70% | 90% | Scale up |

```python
from backend.shared.coordination import HealthMonitor

monitor = HealthMonitor(event_bus=event_bus)

# Start monitoring
await monitor.start()

# Check SLO compliance
slo = monitor.calculate_slo_compliance()
# {"compliant": True, "details": {...}}

# Get active alerts
alerts = monitor.get_active_alerts()
```

### 5. Experiment Generator (`experiment_generator.py`)

Proactive experiment suggestion:

**Sources:**

- Industry research (new models, papers)
- Internal signals (bottlenecks, high costs)
- User feedback (complaints, requests)
- V3 re-evaluation (context changed)

```python
from backend.shared.coordination import ExperimentGenerator

generator = ExperimentGenerator(
    quarantine_manager=quarantine,
    health_monitor=monitor,
)

# Generate proposals from all sources
proposals = await generator.generate_proposals()
```

## Event Types

```python
# Experiment Events
EXPERIMENT_CREATED = "experiment.created"
EXPERIMENT_STARTED = "experiment.started"
EXPERIMENT_EVALUATION_STARTED = "experiment.evaluation.started"
EXPERIMENT_EVALUATION_COMPLETED = "experiment.evaluation.completed"

# Promotion Events
PROMOTION_REQUESTED = "promotion.requested"
PROMOTION_APPROVED = "promotion.approved"
PROMOTION_PHASE_CHANGED = "promotion.phase.changed"
PROMOTION_COMPLETED = "promotion.completed"
PROMOTION_ROLLBACK = "promotion.rollback"

# Quarantine Events
QUARANTINE_REQUESTED = "quarantine.requested"
QUARANTINE_COMPLETED = "quarantine.completed"
QUARANTINE_REVIEWED = "quarantine.reviewed"

# Monitoring Events
MONITORING_ALERT = "monitoring.alert"
MONITORING_RECOVERY = "monitoring.recovery"
AUTO_REMEDIATION = "system.auto_remediation"
```

## RBAC (Role-Based Access Control)

| Role           | V1  | V2      | V3  | Actions                       |
| -------------- | --- | ------- | --- | ----------------------------- |
| Admin          | RW  | RW      | RW  | Promote, Quarantine, Rollback |
| User           | -   | R (API) | -   | Submit code, View results     |
| System (VC-AI) | R   | R       | RW  | Evaluate, Recommend           |

## Network Policies

```
V1 ← X → V2: No direct communication (event bus only)
V2 ← X → V3: No communication (V3 is archived)
V1 → Internet: Allowed (AI API calls)
V2 → Internet: Allowed (AI API calls)
V3 → Internet: Denied (read-only archive)
```

## Implementation Checklist

- [x] Event types and data models (`event_types.py`)
- [x] Promotion manager with canary deployment (`promotion_manager.py`)
- [x] Quarantine manager with RCA (`quarantine_manager.py`)
- [x] Health monitor with auto-remediation (`health_monitor.py`)
- [x] Experiment generator (`experiment_generator.py`)
- [x] Lifecycle coordinator (`lifecycle_coordinator.py`)
- [x] Database schemas (experiments_v1, production, quarantine)
- [x] Kubernetes namespaces (platform-v1-exp, platform-v2-stable)
- [x] Network policies (version isolation)
- [x] RBAC rules (Admin vs User access)
- [x] Monitoring stack (Prometheus + Grafana)
- [x] Feature flags (gradual rollout)
- [x] Rollback automation
- [x] Cost tracking
- [x] Audit logging

## Files Created

```
backend/shared/coordination/
├── __init__.py                 (25 lines)
├── event_types.py              (200 lines)
├── promotion_manager.py        (350 lines)
├── quarantine_manager.py       (400 lines)
├── health_monitor.py           (450 lines)
├── experiment_generator.py     (300 lines)
└── lifecycle_coordinator.py    (400 lines)

Total: ~2,125 lines
```

## Usage Example

```python
import asyncio
from backend.shared.coordination import LifecycleCoordinator

async def main():
    # Initialize
    coordinator = LifecycleCoordinator()
    await coordinator.start()

    # Create experiment
    proposal = await coordinator.create_experiment(
        title="Test new prompt strategy",
        hypothesis="Shorter prompts reduce cost 25%",
        success_criteria={
            "cost_reduction": "> 20%",
            "accuracy": "> 95%",
        },
    )

    # Start experiment
    await coordinator.start_experiment(
        proposal.experiment_id,
        approver="admin@example.com",
    )

    # ... run experiment for evaluation period ...

    # Evaluate
    decision = await coordinator.evaluate_experiment(
        proposal.experiment_id,
        metrics={"cost_reduction": 0.28, "accuracy": 0.96},
    )

    if decision == "PROMOTE":
        # Approve promotion (triggers canary)
        promotions = coordinator.promotion.get_active_promotions()
        if promotions:
            await coordinator.approve_promotion(
                promotions[0].request_id,
                approver="admin@example.com",
            )

    # Get system status
    status = coordinator.get_system_status()
    print(f"System: {status}")

asyncio.run(main())
```
