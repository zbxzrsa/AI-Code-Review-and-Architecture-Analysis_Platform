# SelfHealing_V2 API Reference

## Overview

Production self-healing with predictive capabilities and runbook automation.

## Classes

### HealthMonitor

SLO-based health monitoring.

```python
from modules.SelfHealing_V2.src.health_monitor import HealthMonitor

monitor = HealthMonitor()

# Register service
async def check_db():
    return await db.ping()

monitor.register_service(
    "database",
    check_db,
    slo_latency_ms=100,
    unhealthy_threshold=3
)

# Check health
result = await monitor.check_service("database")
overall = monitor.get_overall_status()
```

### RecoveryManager

Runbook-based automated recovery.

```python
from modules.SelfHealing_V2.src.recovery_manager import (
    RecoveryManager, Runbook, RecoveryStep, RecoveryAction
)

manager = RecoveryManager()

# Define runbook
runbook = Runbook(
    name="api-recovery",
    description="API service recovery",
    steps=[
        RecoveryStep(RecoveryAction.CLEAR_CACHE, clear_cache_handler),
        RecoveryStep(RecoveryAction.RESTART, restart_handler),
    ],
    cooldown_seconds=300
)

manager.register_runbook(runbook)
manager.assign_runbook("api-service", "api-recovery")

# Execute recovery
execution = await manager.execute_recovery("api-service")
```

### PredictiveHealer

ML-based failure prediction.

```python
from modules.SelfHealing_V2.src.predictive_healer import PredictiveHealer

healer = PredictiveHealer()

# Record metrics
await healer.record_metric("api", "cpu_usage", 75)
await healer.record_metric("api", "memory_usage", 80)

# Get predictions
prediction = await healer.predict_failures("api")
print(f"Risk: {prediction.risk_level}")
print(f"Actions: {prediction.recommended_actions}")

# Get proactive actions
actions = await healer.get_proactive_actions()
```

## Configuration

See `config/self_healing_config.yaml`
