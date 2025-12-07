# Module Index - Version-Organized Function Modules

## Overview

This document provides a comprehensive index of all functional modules organized by version according to the three-version self-evolving architecture.

## Backend Integration

All modules integrate with `backend/shared/` implementations via bridge layers:

| Module          | Backend Source                      | Integration File         |
| --------------- | ----------------------------------- | ------------------------ |
| SelfHealing     | `backend/shared/self_healing/`      | `backend_integration.py` |
| Monitoring      | `backend/shared/monitoring/`        | `backend_integration.py` |
| Caching         | `backend/shared/cache/`             | `backend_integration.py` |
| Authentication  | `backend/shared/auth/`, `security/` | `backend_integration.py` |
| AIOrchestration | `backend/services/ai-orchestrator/` | `backend_integration.py` |

See `INTEGRATION_GUIDE.md` for detailed integration documentation.

## Version Definitions

| Version | Status       | Purpose                                    | Access     |
| ------- | ------------ | ------------------------------------------ | ---------- |
| **V1**  | Experimental | New features under development and testing | Admin only |
| **V2**  | Production   | Stable, production-ready implementations   | All users  |
| **V3**  | Quarantine   | Deprecated implementations for comparison  | Admin only |

## Module Summary

### Total Statistics

- **Functional Groups**: 8
- **Total Modules**: 24 (8 × 3 versions)
- **Production Modules**: 8 (V2)
- **Experimental Modules**: 8 (V1)
- **Quarantine Modules**: 8 (V3)

---

## Functional Modules

### 1. CodeReviewAI

AI-powered code review with multi-dimensional analysis.

| Version             | Status       | Features                                                |
| ------------------- | ------------ | ------------------------------------------------------- |
| **CodeReviewAI_V1** | Experimental | Basic review, issue detection, fix suggestions          |
| **CodeReviewAI_V2** | Production   | + Hallucination detection, SLO enforcement, OWASP rules |
| **CodeReviewAI_V3** | Quarantine   | Legacy baseline for comparison                          |

**Key Components:**

- `CodeReviewer` - Main review orchestrator
- `IssueDetector` - Pattern-based detection
- `FixSuggester` - Auto-fix generation
- `QualityScorer` - Quality scoring
- `HallucinationDetector` (V2) - Verification
- `ComparisonEngine` (V3) - Baseline comparison

---

### 2. Authentication

JWT-based authentication with session management.

| Version               | Status       | Features                               |
| --------------------- | ------------ | -------------------------------------- |
| **Authentication_V1** | Experimental | JWT tokens, sessions, RBAC             |
| **Authentication_V2** | Production   | + MFA (TOTP), OAuth 2.0, rate limiting |
| **Authentication_V3** | Quarantine   | Legacy auth baseline                   |

**Key Components:**

- `AuthManager` - Core authentication
- `SessionManager` - Session lifecycle
- `TokenService` - JWT operations
- `MFAService` (V2) - Multi-factor auth
- `OAuthProvider` (V2) - OAuth integration

---

### 3. SelfHealing

System self-healing and automatic recovery.

| Version            | Status       | Features                                   |
| ------------------ | ------------ | ------------------------------------------ |
| **SelfHealing_V1** | Experimental | Health monitoring, recovery, incidents     |
| **SelfHealing_V2** | Production   | + Predictive healing, ML anomaly detection |
| **SelfHealing_V3** | Quarantine   | Legacy healing baseline                    |

**Key Components:**

- `HealthMonitor` - Service health tracking
- `RecoveryManager` - Automated recovery
- `IncidentDetector` - Incident classification
- `PredictiveHealer` (V2) - ML-based prediction

---

### 4. AIOrchestration

AI model routing and load balancing.

| Version                | Status       | Features                               |
| ---------------------- | ------------ | -------------------------------------- |
| **AIOrchestration_V1** | Experimental | Multi-provider, routing, fallback      |
| **AIOrchestration_V2** | Production   | + Load balancing, circuit breaker, SLO |
| **AIOrchestration_V3** | Quarantine   | Legacy orchestration baseline          |

**Key Components:**

- `Orchestrator` - Request orchestration
- `ProviderRouter` - Provider selection
- `FallbackChain` - Failover management
- `LoadBalancer` (V2) - Request distribution
- `CircuitBreaker` (V2) - Fault tolerance

---

### 5. Caching

Multi-level caching with Redis support.

| Version        | Status       | Features                                |
| -------------- | ------------ | --------------------------------------- |
| **Caching_V1** | Experimental | Redis caching, basic patterns           |
| **Caching_V2** | Production   | + Semantic deduplication, cache warming |
| **Caching_V3** | Quarantine   | Legacy caching baseline                 |

**Key Components:**

- `CacheManager` - Cache operations
- `RedisClient` - Redis integration
- `SemanticCache` - Code hash deduplication
- `CacheWarmer` (V2) - Proactive caching

---

### 6. Monitoring

Metrics collection and observability.

| Version           | Status       | Features                            |
| ----------------- | ------------ | ----------------------------------- |
| **Monitoring_V1** | Experimental | Metrics, alerts, dashboards         |
| **Monitoring_V2** | Production   | + SLO tracking, distributed tracing |
| **Monitoring_V3** | Quarantine   | Legacy monitoring baseline          |

**Key Components:**

- `MetricsCollector` - Prometheus metrics
- `AlertManager` - Alert rules
- `DashboardService` - Grafana dashboards
- `SLOTracker` (V2) - SLO compliance
- `TracingService` (V2) - Distributed tracing

---

### 7. Security (Planned)

Security layer with OPA integration.

| Version         | Status  | Features                      |
| --------------- | ------- | ----------------------------- |
| **Security_V1** | Planned | Basic auth, RBAC              |
| **Security_V2** | Planned | + OPA policies, audit logging |
| **Security_V3** | Planned | Legacy security baseline      |

---

### 8. DataPipeline (Planned)

Data processing and quality assurance.

| Version             | Status  | Features                             |
| ------------------- | ------- | ------------------------------------ |
| **DataPipeline_V1** | Planned | Data cleaning, validation            |
| **DataPipeline_V2** | Planned | + Anomaly detection, quality scoring |
| **DataPipeline_V3** | Planned | Legacy pipeline baseline             |

---

## Directory Structure

```
modules/
├── README.md                    # This index
├── MODULE_INDEX.md             # Detailed documentation
│
├── CodeReviewAI_V1/            # Experimental
│   ├── __init__.py
│   ├── README.md
│   ├── src/
│   │   ├── code_reviewer.py
│   │   ├── issue_detector.py
│   │   ├── fix_suggester.py
│   │   └── quality_scorer.py
│   ├── tests/
│   ├── config/
│   └── docs/
│
├── CodeReviewAI_V2/            # Production
│   ├── __init__.py
│   ├── README.md
│   ├── src/
│   │   ├── code_reviewer.py
│   │   ├── issue_detector.py
│   │   ├── fix_suggester.py
│   │   ├── quality_scorer.py
│   │   └── hallucination_detector.py  # V2 addition
│   ├── tests/
│   ├── config/
│   └── docs/
│
├── CodeReviewAI_V3/            # Quarantine
│   └── ...
│
├── Authentication_V1/
├── Authentication_V2/
├── Authentication_V3/
│
├── SelfHealing_V1/
├── SelfHealing_V2/
├── SelfHealing_V3/
│
├── AIOrchestration_V1/
├── AIOrchestration_V2/
├── AIOrchestration_V3/
│
├── Caching_V1/
├── Caching_V2/
├── Caching_V3/
│
├── Monitoring_V1/
├── Monitoring_V2/
└── Monitoring_V3/
```

---

## Version Lifecycle

### V1 → V2 Promotion

Requirements for promotion:

- [ ] 100% test pass rate
- [ ] Documentation complete
- [ ] Code review approved
- [ ] Performance benchmarks met
- [ ] Security scan passed
- [ ] SLO targets defined

### V2 → V3 Degradation

Triggers for degradation:

- New V1 promoted to V2
- Critical bugs in V2
- SLO violations > threshold
- Security vulnerabilities

### V3 → Delete/Re-evaluate

- Re-evaluation possible after fixes
- Deletion after retention period (90 days)

---

## Usage Examples

### Import V2 Production Module

```python
from modules.CodeReviewAI_V2 import CodeReviewer

reviewer = CodeReviewer(
    enable_hallucination_check=True,
    slo_timeout_ms=3000
)
result = await reviewer.review(code, language="python")
```

### Import V1 Experimental Module

```python
from modules.Authentication_V1 import AuthManager

auth = AuthManager()
result = await auth.login(email, password)
```

### Compare V2 vs V3 (Admin)

```python
from modules.CodeReviewAI_V2 import CodeReviewer as V2Reviewer
from modules.CodeReviewAI_V3 import ComparisonEngine

v2_result = await V2Reviewer().review(code)
comparison = await ComparisonEngine().compare(v3_baseline, v2_result)
```

---

## Test Commands

```bash
# Test all modules
pytest modules/ -v

# Test specific version
pytest modules/CodeReviewAI_V2/tests/ -v

# Test with coverage
pytest modules/ --cov=modules --cov-report=html
```

---

## Quality Gates

### Test Requirements

- Unit test coverage: > 80%
- Integration tests: Pass
- Performance tests: Meet SLO

### Documentation Requirements

- README.md complete
- API documentation current
- Migration guide (for V2)

### Code Quality

- Linting: Pass (ruff, black)
- Type hints: Required
- Security scan: Pass

---

## Maintenance

### Adding New Module

1. Create `FunctionName_V1/` folder
2. Add `__init__.py` with version info
3. Add `README.md` with documentation
4. Add `src/` with implementation
5. Add `tests/` with test cases
6. Add `config/` with configuration
7. Update this MODULE_INDEX.md

### Promoting Module

1. Run all quality gates
2. Create `FunctionName_V2/` from V1
3. Add production enhancements
4. Update documentation
5. Move V2 → V3 if exists
6. Update this MODULE_INDEX.md

---

## Configuration Files

Each V2 production module includes a YAML configuration file:

| Module             | Config File                        |
| ------------------ | ---------------------------------- |
| AIOrchestration_V2 | `config/orchestration_config.yaml` |
| Authentication_V2  | `config/auth_config.yaml`          |
| Caching_V2         | `config/caching_config.yaml`       |
| Monitoring_V2      | `config/monitoring_config.yaml`    |
| SelfHealing_V2     | `config/self_healing_config.yaml`  |
| CodeReviewAI_V2    | `config/review_config.yaml`        |

---

## Test Coverage

| Module          | V1 Tests | V2 Tests | V3 Tests |
| --------------- | :------: | :------: | :------: |
| CodeReviewAI    |    ✅    |    ✅    |    ✅    |
| Authentication  |    ✅    |    ✅    |    -     |
| SelfHealing     |    ✅    |    ✅    |    -     |
| AIOrchestration |    ✅    |    ✅    |    -     |
| Caching         |    ✅    |    -     |    -     |
| Monitoring      |    ✅    |    ✅    |    -     |

---

## Version History

| Date    | Change                            |
| ------- | --------------------------------- |
| 2024-12 | Initial module organization       |
| 2024-12 | CodeReviewAI V1/V2/V3 complete    |
| 2024-12 | Authentication V1/V2/V3 created   |
| 2024-12 | SelfHealing V1/V2/V3 created      |
| 2024-12 | AIOrchestration V1/V2/V3 complete |
| 2024-12 | Caching V1/V2/V3 complete         |
| 2024-12 | Monitoring V1/V2/V3 complete      |
| 2024-12 | All V2 config files added         |
| 2024-12 | Test coverage expanded            |
