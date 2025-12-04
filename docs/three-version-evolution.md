# Three-Version Self-Evolution System

> **Status**: ✅ Production Ready  
> **Version**: 1.1.0  
> **Last Updated**: December 2024

## Overview

The Three-Version Self-Evolution System enables concurrent development across three versions with dedicated AI instances per version. This creates a spiral cycle where V1 experiments, V2 stabilizes and fixes errors, and V3 provides comparison baselines.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THREE-VERSION SPIRAL EVOLUTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   V1 (New/Experiment)      V2 (Stable/Production)     V3 (Old/Quarantine)  │
│   ┌─────────────────┐      ┌─────────────────┐        ┌─────────────────┐  │
│   │ V1-VCAI (Admin) │      │ V2-VCAI (Admin) │        │ V3-VCAI (Admin) │  │
│   │ • Experiments   │─────▶│ • Fixes V1 bugs │───────▶│ • Compares      │  │
│   │ • Trial/Error   │      │ • Optimizes     │        │ • Excludes      │  │
│   └─────────────────┘      └─────────────────┘        └─────────────────┘  │
│   ┌─────────────────┐      ┌─────────────────┐        ┌─────────────────┐  │
│   │ V1-CRAI (Test)  │      │ V2-CRAI (Users) │        │ V3-CRAI (Ref)   │  │
│   │ • Shadow tests  │      │ • Production    │        │ • Baseline      │  │
│   └─────────────────┘      └─────────────────┘        └─────────────────┘  │
│           │                                                    │           │
│           │                                                    │           │
│           └────────────── Re-evaluation (30+ days) ───────────┘           │
│                                                                             │
│   SPIRAL PHASES:                                                            │
│   1. Experimentation → 2. Error Remediation → 3. Evaluation → 4. Promotion │
│   5. Stabilization → 6. Degradation → 7. Comparison → 8. Re-evaluation     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Version Roles

| Version         | Purpose                      | VC-AI Access | CR-AI Access          |
| --------------- | ---------------------------- | ------------ | --------------------- |
| **V1** (New)    | Experimentation, trial/error | Admin only   | Admin only (shadow)   |
| **V2** (Stable) | Production, user-facing      | Admin only   | **All users**         |
| **V3** (Old)    | Quarantine, comparison       | Admin only   | Admin only (baseline) |

## Dual-AI Architecture

Each version has two dedicated AI instances:

### Version Control AI (VC-AI)

- **Access**: Admin only
- **Purpose**: Manage version transitions, evaluate experiments, make promotion/degradation decisions
- **Responsibilities**:
  - Evaluate experiment metrics
  - Decide promotions (V1→V2)
  - Trigger degradations (V2→V3)
  - Approve re-evaluations (V3→V1)

### Code Review AI (CR-AI)

- **Access**: V2 is user-facing; V1/V3 are admin only
- **Purpose**: Perform code analysis and review
- **Responsibilities**:
  - V1: Shadow testing of new technologies
  - V2: Production code review for users
  - V3: Baseline comparison analysis

## Spiral Evolution Phases

### Phase 1: Experimentation (V1)

```python
# V1 tests new technologies
await cycle.start_experiment(
    technology="multi_query_attention",
    config={"num_kv_heads": 4}
)
```

### Phase 2: Error Remediation (V2 fixes V1)

```python
# V1 reports error
await cycle.report_v1_error(
    tech_id="mqa_123",
    tech_name="Multi-Query Attention",
    error_type="compatibility",
    description="Incompatible with existing transformer layers"
)

# V2 automatically analyzes and generates fix
# Fix is applied back to V1
```

### Phase 3: Evaluation

```python
# Check if technology meets promotion criteria
# - Accuracy >= 85%
# - Error rate <= 5%
# - Latency p95 <= 3000ms
# - Minimum 1000 samples
```

### Phase 4: Promotion (V1 → V2)

```python
await cycle.trigger_promotion("mqa_123")
```

### Phase 5: Stabilization (V2)

- V2 stabilizes newly promoted technology
- Monitors SLOs in production
- Rollback if issues detected

### Phase 6: Degradation (V2 → V3)

```python
await cycle.trigger_degradation(
    "old_tech_456",
    reason="High error rate in production"
)
```

### Phase 7: Comparison (V3)

- V3 provides baseline for V1 experiments
- Tracks failure patterns
- Makes exclusion decisions

### Phase 8: Re-evaluation (V3 → V1)

```python
# After 30+ days, technology can be re-evaluated
await cycle.request_reevaluation("old_tech_456")
```

## API Reference

### Base URL

```
/api/v1/evolution
```

### Endpoints

| Method | Endpoint          | Description           |
| ------ | ----------------- | --------------------- |
| GET    | `/status`         | Get cycle status      |
| POST   | `/start`          | Start evolution cycle |
| POST   | `/stop`           | Stop evolution cycle  |
| POST   | `/v1/errors`      | Report V1 error       |
| GET    | `/v1/experiments` | List V1 experiments   |
| GET    | `/v2/status`      | Get V2 status         |
| GET    | `/v2/fixes`       | List V2 fixes         |
| GET    | `/v3/quarantine`  | Get quarantine status |
| GET    | `/v3/exclusions`  | Get exclusion list    |
| POST   | `/promote`        | Trigger promotion     |
| POST   | `/degrade`        | Trigger degradation   |
| POST   | `/reeval`         | Request re-evaluation |
| GET    | `/ai/status`      | Get all AI status     |
| GET    | `/ai/user`        | Get user AI status    |
| GET    | `/history`        | Get cycle history     |
| GET    | `/metrics`        | Get metrics           |
| GET    | `/health`         | Health check          |
| GET    | `/prometheus`     | Prometheus metrics    |

### Example: Report V1 Error

```bash
curl -X POST http://localhost:8010/api/v1/evolution/v1/errors \
  -H "Content-Type: application/json" \
  -d '{
    "tech_id": "mqa_123",
    "tech_name": "Multi-Query Attention",
    "error_type": "compatibility",
    "description": "Incompatible with existing layers"
  }'
```

### Example: Trigger Promotion

```bash
curl -X POST http://localhost:8010/api/v1/evolution/promote \
  -H "Content-Type: application/json" \
  -d '{
    "tech_id": "mqa_123",
    "reason": "Passed all evaluation criteria"
  }'
```

## Python SDK

### Basic Usage

```python
from ai_core.three_version_cycle import EnhancedSelfEvolutionCycle

# Create and start cycle
cycle = EnhancedSelfEvolutionCycle()
await cycle.start()

# Report V1 error
await cycle.report_v1_error(
    tech_id="new_tech_123",
    tech_name="New Attention Mechanism",
    error_type="compatibility",
    description="Incompatible with existing system"
)

# Trigger promotion
await cycle.trigger_promotion("new_tech_123")

# Get status
status = cycle.get_full_status()
print(f"Running: {status['running']}")
print(f"Phase: {status['spiral_status']['current_cycle']['phase']}")

# Stop cycle
await cycle.stop()
```

### Access Dual-AI

```python
from ai_core.three_version_cycle import DualAICoordinator, AIType

coordinator = DualAICoordinator()

# Get V2 CR-AI (user-accessible)
user_ai = coordinator.get_user_accessible_ai()

# Route request based on role
result = await coordinator.route_request(
    user_role="user",  # or "admin"
    request_type="code_review",
    request_data={"code": "print('hello')"}
)
```

## Monitoring

### Grafana Dashboard

Import the dashboard from:

```
monitoring/grafana/provisioning/dashboards/three-version-evolution.json
```

### Key Metrics

| Metric                          | Description                   |
| ------------------------------- | ----------------------------- |
| `evolution_cycle_running`       | Cycle status (1=running)      |
| `evolution_cycles_total`        | Total completed cycles        |
| `v1_experiments_total`          | V1 experiment count by status |
| `v1_errors_total`               | V1 errors by type             |
| `v2_fixes_total`                | V2 fixes by status            |
| `v2_fix_success_rate`           | V2 fix success rate           |
| `quarantine_technologies_total` | Total quarantined             |
| `permanent_exclusions_total`    | Permanently excluded          |
| `ai_instance_status`            | AI instance status by version |
| `ai_instance_error_rate`        | AI error rate                 |
| `pending_promotions`            | Pending promotions count      |
| `pending_degradations`          | Pending degradations count    |

### Alerting Rules

Alerts are defined in:

```
monitoring/prometheus/rules/three-version-alerts.yml
```

Key alerts:

- `EvolutionCycleStopped` - Cycle not running
- `V2CRAIDown` - User-facing AI offline (CRITICAL)
- `V2FixSuccessRateLow` - Fix quality degraded
- `QuarantineSecurityIssues` - Security-related quarantine

## Deployment

### Docker Compose

```bash
docker-compose up -d three-version-service
```

### Kubernetes

```bash
kubectl apply -f kubernetes/deployments/three-version-service.yaml
```

### Environment Variables

| Variable                         | Default | Description           |
| -------------------------------- | ------- | --------------------- |
| `DATABASE_URL`                   | -       | PostgreSQL connection |
| `REDIS_URL`                      | -       | Redis connection      |
| `KAFKA_BOOTSTRAP_SERVERS`        | -       | Kafka servers         |
| `EVOLUTION_CYCLE_INTERVAL_HOURS` | 6       | Cycle interval        |
| `LOG_LEVEL`                      | INFO    | Logging level         |

## File Structure

```
ai_core/three_version_cycle/
├── __init__.py
├── version_manager.py          # V1/V2/V3 state management
├── version_ai_engine.py        # V1/V2/V3 AI engines
├── experiment_framework.py     # V1 experiments
├── self_evolution_cycle.py     # Basic + Enhanced cycle
├── cross_version_feedback.py   # V2 fixes V1 errors
├── v3_comparison_engine.py     # Comparison & exclusion
├── dual_ai_coordinator.py      # VCAI + CRAI management
└── spiral_evolution_manager.py # 8-phase orchestration

backend/services/three-version-service/
├── api.py                      # REST API
├── main.py                     # FastAPI app
├── metrics.py                  # Prometheus metrics
├── requirements.txt
└── Dockerfile

frontend/src/
├── pages/admin/ThreeVersionControl.tsx  # Admin UI
└── services/threeVersionService.ts      # API client
```

## Troubleshooting

### Cycle Not Starting

```python
# Check status
status = cycle.get_full_status()
print(status)

# Check AI instances
ai_status = cycle.get_dual_ai_status()
for version, status in ai_status.items():
    print(f"{version}: VC-AI={status['vc_ai']['status']}, CR-AI={status['cr_ai']['status']}")
```

### V2 Fixes Failing

```python
# Get feedback statistics
stats = cycle.spiral_manager.feedback_system.get_feedback_statistics()
print(f"Success rate: {stats['fix_success_rate']}")
print(f"Templates learned: {stats['fix_templates_learned']}")
```

### High Quarantine Rate

```python
# Get quarantine insights
insights = cycle.spiral_manager.comparison_engine.get_failure_insights()
for insight in insights:
    print(f"{insight['category']}: {insight['insight']}")
```

## Best Practices

1. **Start with V1 experiments before expecting promotions**
2. **Monitor V2 fix success rate** - Low rate indicates learning issues
3. **Review quarantine reasons** - Patterns indicate systemic problems
4. **Set appropriate thresholds** - Default 85% accuracy may need adjustment
5. **Use re-evaluation** - Quarantined tech may improve with fixes

## Related Documentation

- [THREE_VERSION_CYCLE_SUMMARY.md](../THREE_VERSION_CYCLE_SUMMARY.md)
- [THREE_VERSION_IMPLEMENTATION_TRACKER.md](THREE_VERSION_IMPLEMENTATION_TRACKER.md)
- [Self-Evolution Cycle](../ai_core/three_version_cycle/README.md)
