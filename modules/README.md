# Function Modules Organization

This directory contains all functional modules organized by version (V1-V3) following the three-version self-evolving architecture.

> **For detailed documentation, see [MODULE_INDEX.md](MODULE_INDEX.md)**

## Quick Reference

### Naming Convention

- Format: `FunctionName_V{Version}`
- Example: `Authentication_V1`, `CodeReviewAI_V2`

### Version Definitions

| Version | Status       | Access    | Purpose               |
| ------- | ------------ | --------- | --------------------- |
| **V1**  | Experimental | Admin     | Testing new features  |
| **V2**  | Production   | All users | Stable production use |
| **V3**  | Quarantine   | Admin     | Comparison baseline   |

## Module Structure

```
FunctionName_V{X}/
├── __init__.py       # Module initialization
├── README.md         # Module documentation
├── src/              # Source code
├── tests/            # Unit and integration tests
├── config/           # Configuration files
└── docs/             # API documentation
```

## Available Modules

| Module              | Description             | V1  | V2  | V3  | Status   |
| ------------------- | ----------------------- | :-: | :-: | :-: | -------- |
| **CodeReviewAI**    | AI-powered code review  | ✅  | ✅  | ✅  | Complete |
| **Authentication**  | User auth & sessions    | ✅  | ✅  | ✅  | Complete |
| **SelfHealing**     | System self-healing     | ✅  | ✅  | ✅  | Complete |
| **AIOrchestration** | AI model orchestration  | ✅  | ✅  | ✅  | Complete |
| **Caching**         | Multi-level caching     | ✅  | ✅  | ✅  | Complete |
| **Monitoring**      | Metrics & observability | ✅  | ✅  | ✅  | Complete |

## Backend Integration

All modules integrate with `backend/shared/` implementations via bridge layers:

| Module          | Backend Source              | Key Features                    |
| --------------- | --------------------------- | ------------------------------- |
| SelfHealing     | `self_healing/`             | Health monitoring, auto-repair  |
| Monitoring      | `monitoring/`               | Metrics, SLO alerts, tracing    |
| Caching         | `cache/`                    | Multi-tier, semantic cache      |
| Authentication  | `auth/`, `security/`        | JWT, OAuth, MFA                 |
| AIOrchestration | `services/ai-orchestrator/` | Load balancing, circuit breaker |

> See [BACKEND_MAPPING.md](BACKEND_MAPPING.md) for detailed integration docs

## Quick Start

### Use Production Module (V2)

```python
from modules import get_production_module

# Get any production module
monitoring = get_production_module("Monitoring")
self_healing = get_production_module("SelfHealing")

# Or import directly with backend integration
from modules.SelfHealing_V2.src.backend_integration import get_health_monitor

monitor = get_health_monitor()
result = await monitor.check_health("api-service")
```

### Use Experimental Module (V1)

```python
from modules.Authentication_V1.src.backend_integration import get_auth_manager

auth = get_auth_manager(use_backend=True)
result = await auth.authenticate(email, password)
```

### Use Legacy Module (V3) for Comparison

```python
from modules.Caching_V3.src.legacy_bridge import get_legacy_cache_manager

# Issues deprecation warning
legacy = get_legacy_cache_manager()
comparison = legacy.compare_with_current(v2_cache)
```

## Version Lifecycle

```
┌─────────┐    Promotion    ┌─────────┐    Degradation    ┌─────────┐
│   V1    │ ──────────────► │   V2    │ ────────────────► │   V3    │
│  (Exp)  │  Quality Gates  │ (Prod)  │   New V1→V2       │(Legacy) │
└─────────┘                 └─────────┘                   └─────────┘
     │                           │                             │
     │ Develop & Test           │ Serve Production            │ Compare
     │ (Admin only)             │ (All users)                 │ (Admin)
```

## Quality Gates (V1 → V2)

- [ ] 100% test pass rate
- [ ] Documentation complete
- [ ] Code review approved
- [ ] Performance benchmarks met
- [ ] Security scan passed

## Run Tests

```bash
# All modules
pytest modules/ -v

# Specific module
pytest modules/CodeReviewAI_V2/tests/ -v

# With coverage
pytest modules/ --cov=modules --cov-report=html
```

---

> See [MODULE_INDEX.md](MODULE_INDEX.md) for complete documentation
