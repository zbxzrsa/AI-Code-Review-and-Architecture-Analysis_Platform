# Backend Integration Mapping

Complete mapping between versioned modules and backend shared implementations.

## Backend Sources

### `backend/shared/self_healing/`

| File                   | Classes                                          | Used By                                                      |
| ---------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| `health_monitor.py`    | `HealthMonitor`, `HealthStatus`, `HealthMetrics` | SelfHealing_V1, SelfHealing_V2                               |
| `auto_repair.py`       | `AutoRepair`, `RepairAction`, `RepairResult`     | SelfHealing_V1, SelfHealing_V2                               |
| `alert_manager.py`     | `AlertManager`, `Alert`, `AlertSeverity`         | SelfHealing_V1, SelfHealing_V2                               |
| `metrics_collector.py` | `MetricsCollector`, `Metric`, `MetricSeries`     | SelfHealing_V1, SelfHealing_V2, Monitoring_V1, Monitoring_V2 |
| `orchestrator.py`      | `SelfHealingOrchestrator`                        | SelfHealing_V2                                               |

### `backend/shared/monitoring/`

| File                     | Classes                                           | Used By                      |
| ------------------------ | ------------------------------------------------- | ---------------------------- |
| `metrics.py`             | `MetricsManager`, `Counter`, `Gauge`, `Histogram` | Monitoring_V1, Monitoring_V2 |
| `slo_alerts.py`          | `SLOAlertManager`, `SLODefinition`                | Monitoring_V2                |
| `distributed_tracing.py` | `TracingService`, `Span`, `TraceContext`          | Monitoring_V2                |

### `backend/shared/cache/`

| File                  | Classes                        | Used By                |
| --------------------- | ------------------------------ | ---------------------- |
| `ai_result_cache.py`  | `AIResultCache`, `CacheEntry`  | Caching_V1, Caching_V2 |
| `redis_client.py`     | `RedisClient`, `RedisConfig`   | Caching_V2             |
| `cache_strategies.py` | `CacheStrategy`, `LRUStrategy` | Caching_V2             |
| `analysis_cache.py`   | `AnalysisCache`                | Caching_V1             |

### `backend/shared/auth/`

| File                 | Classes                                       | Used By                              |
| -------------------- | --------------------------------------------- | ------------------------------------ |
| `oauth_providers.py` | `OAuthProvider`, `GoogleOAuth`, `GitHubOAuth` | Authentication_V1, Authentication_V2 |

### `backend/shared/security/`

| File                 | Classes                        | Used By                              |
| -------------------- | ------------------------------ | ------------------------------------ |
| `auth.py`            | `JWTManager`, `PasswordHasher` | Authentication_V1, Authentication_V2 |
| `provider_health.py` | `ProviderHealthTracker`        | AIOrchestration_V2                   |
| `opa_client.py`      | `OPAClient`, `PolicyEngine`    | Authentication_V2                    |
| `audit_logger.py`    | `AuditLogger`                  | Authentication_V2                    |

### `backend/services/ai-orchestrator/`

| Component        | Used By                                |
| ---------------- | -------------------------------------- |
| `AIOrchestrator` | AIOrchestration_V1, AIOrchestration_V2 |
| `ProviderRouter` | AIOrchestration_V2                     |
| `FallbackChain`  | AIOrchestration_V1, AIOrchestration_V2 |

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Versioned Modules Layer                       │
├─────────────────────────────────────────────────────────────────┤
│  V1 (Experimental)  │  V2 (Production)  │  V3 (Quarantine)      │
│  - Basic features   │  - SLO tracking   │  - Read-only          │
│  - Relaxed SLOs     │  - Circuit break  │  - Deprecation warn   │
│  - Shadow traffic   │  - Load balancing │  - Legacy bridge      │
├─────────────────────────────────────────────────────────────────┤
│                   Backend Integration Bridges                    │
│  backend_integration.py - Wraps backend with version features   │
├─────────────────────────────────────────────────────────────────┤
│                    Backend Shared Layer                          │
│  self_healing/ │ monitoring/ │ cache/ │ auth/ │ security/       │
└─────────────────────────────────────────────────────────────────┘
```

## Version-Specific Features

### V1 Experimental

- Direct backend integration with fallback
- Feature flags for A/B testing
- Relaxed error handling

### V2 Production

- **SLO Enforcement**: p95 latency < 3s, error rate < 2%
- **Circuit Breakers**: Automatic failure isolation
- **Load Balancing**: Health-aware provider selection
- **Error Budgets**: Automatic degradation

### V3 Quarantine

- Read-only access to legacy implementations
- Deprecation warnings on all methods
- Comparison utilities for migration

## Import Examples

```python
# V1 - Experimental
from modules.SelfHealing_V1.src.backend_integration import get_health_monitor
monitor = get_health_monitor(use_backend=True)

# V2 - Production
from modules.Monitoring_V2.src.backend_integration import get_slo_tracker
tracker = get_slo_tracker(latency_target_ms=3000)

# V3 - Legacy comparison
from modules.Caching_V3.src.legacy_bridge import get_legacy_cache_manager
legacy = get_legacy_cache_manager()  # Issues deprecation warning
```

## Test Verification

```bash
# Run all integration tests
cd modules
python -m pytest tests/ -v

# Test specific version
python -m pytest tests/test_integration.py::TestSelfHealingV2Integration -v

# Test lifecycle transitions
python -m pytest tests/test_version_lifecycle.py -v
```
