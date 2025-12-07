# Module Integration Guide

## Overview

This guide documents the integration between versioned modules (`modules/`) and backend implementations (`backend/shared/`).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Module_V1   │  │ Module_V2   │  │ Module_V3   │        │
│  │ (Experiment)│  │ (Production)│  │ (Quarantine)│        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                  │
│              ┌───────────▼───────────┐                     │
│              │  backend_integration  │                     │
│              │    (Bridge Layer)     │                     │
│              └───────────┬───────────┘                     │
│                          │                                  │
├──────────────────────────┼──────────────────────────────────┤
│                          │                                  │
│              ┌───────────▼───────────┐                     │
│              │   backend/shared/     │                     │
│              │ (Core Implementation) │                     │
│              └───────────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Integration Bridges

Each module version includes a `backend_integration.py` file that:

1. **Auto-detects backend availability** - Falls back to local implementation if backend unavailable
2. **Provides unified interface** - Same API regardless of backend status
3. **Adds version-specific features** - V2 adds SLO tracking, V1 is experimental

### V1 Integration (Experimental)

```python
from modules.SelfHealing_V1.src.backend_integration import (
    get_health_monitor,
    get_auto_repair,
    BACKEND_AVAILABLE,
)

# Use backend if available, otherwise local implementation
monitor = get_health_monitor(use_backend=True)
```

### V2 Integration (Production)

```python
from modules.SelfHealing_V2.src.backend_integration import (
    get_health_monitor,
    get_recovery_manager,
    SLOStatus,
)

# Production features with SLO tracking
monitor = get_health_monitor(
    slo_latency_ms=1000,
    slo_availability=99.9,
)

result = await monitor.check_health("api-service")
print(f"SLO Status: {result.slo_status}")
print(f"Error Budget: {result.error_budget_remaining}%")
```

## Module Mapping

| Module              | Backend Source                                     | Integration Features                    |
| ------------------- | -------------------------------------------------- | --------------------------------------- |
| **SelfHealing**     | `backend/shared/self_healing/`                     | Health monitoring, auto-repair, alerts  |
| **Monitoring**      | `backend/shared/monitoring/`                       | Prometheus metrics, tracing, SLO alerts |
| **Caching**         | `backend/shared/cache/`                            | Multi-tier cache, semantic cache        |
| **Authentication**  | `backend/shared/auth/`, `backend/shared/security/` | OAuth, JWT, rate limiting               |
| **AIOrchestration** | `backend/services/ai-orchestrator/`                | Load balancing, circuit breaker         |

## Version Differences

### V1 (Experimental)

- Basic integration with backend
- Simple fallback to local implementation
- Minimal overhead

### V2 (Production)

- SLO tracking and enforcement
- Error budget calculation
- Advanced features (circuit breakers, load balancing)
- Comprehensive metrics

### V3 (Quarantine)

- Read-only access
- Used for comparison baseline
- No active development

## Testing

### Run All Integration Tests

```bash
# From project root
pytest modules/tests/test_integration.py -v

# Run specific module tests
pytest modules/SelfHealing_V2/tests/ -v
pytest modules/Monitoring_V2/tests/ -v
```

### Test Coverage

```bash
pytest modules/ --cov=modules --cov-report=html
```

## Usage Examples

### Health Monitoring with SLO

```python
from modules.SelfHealing_V2.src.backend_integration import get_health_monitor

monitor = get_health_monitor(slo_latency_ms=1000)

# Register service
async def check_api():
    # Your health check logic
    return True

monitor._local.register_service("api", check_api)

# Check health
result = await monitor.check_health("api")

if result.slo_status == SLOStatus.VIOLATED:
    # Trigger recovery
    pass
```

### Metrics with SLO Tracking

```python
from modules.Monitoring_V2.src.backend_integration import get_metrics_collector

collector = get_metrics_collector(prefix="myapp")

# Define SLO
collector.define_slo("latency_p99", 99.0, "request_duration", "<=", window_minutes=60)

# Record metrics
collector.record_metric("request_duration", 150.0)

# Check SLO
status = collector.get_slo_status()
```

### Production Cache with Warming

```python
from modules.Caching_V2.src.backend_integration import get_cache_manager, get_cache_warmer

cache = get_cache_manager(slo_hit_rate=0.85)
warmer = get_cache_warmer(cache)

# Register warming tasks
async def load_user_stats():
    return {"users": 1000}

warmer.register_task("user:stats", load_user_stats, ttl=3600)

# Start background warming
await warmer.start_background_warming(interval=60)
```

## Best Practices

1. **Always check `BACKEND_AVAILABLE`** before using backend-specific features
2. **Use factory functions** (`get_*`) rather than direct class instantiation
3. **Configure SLOs** appropriate to your service requirements
4. **Monitor error budgets** and trigger alerts when low
5. **Test with `use_backend=False`** to ensure fallback works

## Troubleshooting

### Backend Not Available

If `BACKEND_AVAILABLE` is `False`:

- Check that backend path is correct
- Verify backend dependencies are installed
- Check for import errors in backend modules

### SLO Violations

If experiencing SLO violations:

1. Check current metrics with `get_slo_status()`
2. Review error budget consumption
3. Adjust thresholds if too aggressive
4. Add more capacity or optimize code

## Migration Guide

### V1 to V2

```python
# V1
from modules.SelfHealing_V1.src.backend_integration import get_health_monitor
monitor = get_health_monitor()

# V2 (adds SLO)
from modules.SelfHealing_V2.src.backend_integration import get_health_monitor
monitor = get_health_monitor(slo_latency_ms=1000, slo_availability=99.9)
```

### V2 to V3 (Deprecation)

V3 modules are read-only. Before deprecating:

1. Export relevant data/configurations
2. Update dependent services to use V2
3. Keep V3 for comparison baseline only
