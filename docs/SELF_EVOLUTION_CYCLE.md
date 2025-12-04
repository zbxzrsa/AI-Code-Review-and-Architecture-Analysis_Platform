# Three-Version Self-Evolution Cycle

## Overview

The AI Code Review Platform implements a **fully autonomous self-evolution cycle** where AI model versions automatically progress through experimentation, production, and recovery phases without manual intervention.

## Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     THREE-VERSION SELF-EVOLUTION CYCLE                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║                              ┌─────────────────┐                              ║
║                              │     GATEWAY     │                              ║
║                              │  Traffic Router │                              ║
║                              └────────┬────────┘                              ║
║                                       │                                       ║
║                    ┌──────────────────┼──────────────────┐                    ║
║                    │ 100% Production  │ Mirror (Shadow)  │                    ║
║                    ▼                  │                  ▼                    ║
║     ┌──────────────────────┐          │     ┌──────────────────────┐         ║
║     │         V2           │          │     │         V1           │         ║
║     │     PRODUCTION       │          │     │     EXPERIMENT       │         ║
║     ├──────────────────────┤          │     ├──────────────────────┤         ║
║     │ • User-facing        │          │     │ • Shadow traffic     │         ║
║     │ • Strict SLOs        │          │     │ • New models/prompts │         ║
║     │ • P95 < 3000ms       │          │     │ • Scale to zero      │         ║
║     │ • Error rate < 2%    │          │     │ • GPU nodes          │         ║
║     │ • Argo Rollouts      │          │     │ • Comparison eval    │         ║
║     └───────────┬──────────┘          │     └──────────┬───────────┘         ║
║                 │                     │                │                     ║
║                 │ SLO                 │                │ Passes              ║
║                 │ Breach              │                │ Evaluation          ║
║                 │                     │                │                     ║
║                 ▼                     │                ▼                     ║
║     ┌──────────────────────┐          │     ┌──────────────────────┐         ║
║     │      ROLLBACK        │          │     │    GRAY-SCALE        │         ║
║     │    + DEMOTION        │          │     │    PROMOTION         │         ║
║     └───────────┬──────────┘          │     ├──────────────────────┤         ║
║                 │                     │     │ 1% → 5% → 25% →      │         ║
║                 │                     │     │ 50% → 100%           │         ║
║                 │                     │     │ (Argo Rollouts)      │         ║
║                 │                     │     └──────────┬───────────┘         ║
║                 │                     │                │                     ║
║                 │                     │                │ Reaches             ║
║                 │                     │                │ Stable              ║
║                 │                     │                │                     ║
║                 │                     │                └────────────────┐    ║
║                 │                     │                                 │    ║
║                 ▼                     │                                 │    ║
║     ┌──────────────────────┐          │                                 │    ║
║     │         V3           │          │                                 │    ║
║     │     QUARANTINE       │◄─────────┘                                 │    ║
║     ├──────────────────────┤                                            │    ║
║     │ • Failed versions    │                                            │    ║
║     │ • Recovery queue     │                                            │    ║
║     │ • Minimal resources  │                                            │    ║
║     │ • Re-evaluation      │                                            │    ║
║     └───────────┬──────────┘                                            │    ║
║                 │                                                       │    ║
║                 │ Gold-set                              Becomes new     │    ║
║                 │ Passes                                V2 Stable       │    ║
║                 │ (≥95%)                                                │    ║
║                 │                                                       │    ║
║                 ▼                                                       │    ║
║     ┌──────────────────────┐                                            │    ║
║     │      RECOVERY        │                                            │    ║
║     │    (Back to V1)      │────────────────────────────────────────────┘    ║
║     └──────────────────────┘                                                 ║
║                                                                               ║
║                           🔄 CYCLE COMPLETES 🔄                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

## Cycle Components

### 1. V1 Experiment (Shadow Traffic)

**Purpose**: Test new AI models and prompts without impacting users

**Entry Points**:

- New experiment registration
- Recovery from V3 quarantine

**Exit Points**:

- Promotion to V2 (gray-scale)
- Demotion to V3 (failure)

**Evaluation Criteria**:

```yaml
Shadow Evaluation Thresholds:
  min_requests: 1000
  min_duration_hours: 24
  accuracy_delta: ≥ +2% vs baseline
  p95_latency: ≤ 3000ms
  error_rate: ≤ 2%
  security_pass_rate: ≥ 99%
  cost_increase: ≤ 10%
  statistical_significance: p < 0.05
```

### 2. V2 Production (Gray-Scale Rollout)

**Purpose**: Serve user traffic with strict SLO compliance

**Rollout Phases**:

```
1% ────► 5% ────► 25% ────► 50% ────► 100%
 │        │        │         │         │
 ▼        ▼        ▼         ▼         ▼
Eval    Eval     Eval      Eval    STABLE
```

**SLO Requirements**:

```yaml
Production SLOs:
  p95_latency: < 3000ms
  error_rate: < 2%
  availability: > 99.9%
  security_pass_rate: ≥ 99%
```

**Rollback Triggers**:

- 3 consecutive SLO breaches
- Error rate > 10%
- P95 latency > 9000ms (3x threshold)
- Security pass rate < 95%

### 3. V3 Quarantine (Recovery Queue)

**Purpose**: Isolate failed versions and attempt recovery

**Recovery Process**:

```
┌───────────────────────────────────────────────────────┐
│                  RECOVERY TIMELINE                    │
├───────────────────────────────────────────────────────┤
│                                                       │
│   Quarantine ──► 24h cooldown ──► Gold-set Eval      │
│        │                              │               │
│        │                         Pass │ Fail         │
│        │                              │   │          │
│        │                              ▼   ▼          │
│        │                        ┌─────────────┐      │
│        │                        │ Retry with  │      │
│        │                        │ exponential │      │
│        │                        │ backoff     │      │
│        │                        │ (12h→24h→   │      │
│        │                        │  48h→96h)   │      │
│        │                        └──────┬──────┘      │
│        │                               │             │
│        │                    Max 5 attempts           │
│        │                               │             │
│        ▼                               ▼             │
│   ┌─────────┐                   ┌─────────────┐      │
│   │ Archive │ ◄─────────────────│ Abandoned   │      │
│   └─────────┘                   └─────────────┘      │
│                                                       │
└───────────────────────────────────────────────────────┘
```

**Gold-Set Recovery Thresholds** (stricter than promotion):

```yaml
Recovery Thresholds:
  accuracy: ≥ 95% (vs 90% for promotion)
  security_pass_rate: ≥ 99%
  false_positive_rate: ≤ 2%
```

## Data Flow

### Shadow Traffic Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHADOW TRAFFIC FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Request                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │ Gateway │                                                   │
│   └────┬────┘                                                   │
│        │                                                        │
│   ┌────┴────┬───────────────────────────────────┐              │
│   │         │ nginx.mirror-uri annotation        │              │
│   │         │                                    │              │
│   ▼         ▼                                    │              │
│ ┌───────┐ ┌───────┐                              │              │
│ │  V2   │ │  V1   │ (Shadow - no response)      │              │
│ └───┬───┘ └───┬───┘                              │              │
│     │         │                                  │              │
│     │         │                                  │              │
│     ▼         ▼                                  │              │
│ ┌─────────────────────────────────┐              │              │
│ │     SHADOW COMPARATOR           │              │              │
│ │  • Record V1 output             │              │              │
│ │  • Record V2 output             │              │              │
│ │  • Pair by code_hash            │              │              │
│ │  • Compare issues/latency/cost  │              │              │
│ │  • Statistical significance     │              │              │
│ │  • Promotion recommendation     │              │              │
│ └─────────────────────────────────┘              │              │
│                                                  │              │
│   Response to User ◄─────────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Lifecycle Controller

```
GET  /health                    # Health check
GET  /cycle/status              # Cycle status with counts
GET  /cycle/diagram             # ASCII diagram
GET  /cycle/events              # Recent events

POST /versions/register         # Register new experiment
POST /versions/{id}/start-shadow
POST /versions/{id}/quarantine

GET  /recovery/status           # Recovery statistics
GET  /recovery/{id}             # Version recovery status
POST /recovery/{id}/force-evaluate
```

### Evaluation Pipeline

```
POST /shadow/record/v1          # Record V1 output
POST /shadow/record/v2          # Record V2 output
GET  /shadow/status             # Comparator status
GET  /shadow/recommendation/{id}

POST /evaluate/gold-set         # Run gold-set evaluation
GET  /evaluate/gold-set/categories
```

## Monitoring & Alerts

### Prometheus Metrics

```yaml
Cycle Health:
  - lifecycle_versions_total{state="experiment|shadow|gray|stable|quarantine"}
  - lifecycle_promotions_total
  - lifecycle_demotions_total
  - lifecycle_recoveries_total
  - lifecycle_recovery_attempts_total

Shadow Comparison:
  - shadow_pairs_complete_total
  - shadow_pairs_pending
  - shadow_accuracy_delta
  - shadow_latency_improvement_pct

Gold-Set:
  - goldset_evaluations_total{result="pass|fail"}
  - goldset_score
  - goldset_security_score
```

### Alert Rules

```yaml
Alerts:
  - CycleStalled: No promotions in 7 days
  - HighQuarantineRate: > 50% of experiments fail
  - RecoveryBacklog: > 10 versions stuck in quarantine
  - ShadowTrafficDown: No shadow pairs in 1 hour
```

## Quick Start

### Deploy the Cycle

```bash
# Install with Helm
helm install coderev ./charts/coderev-platform \
  -f values-production.yaml

# Verify cycle is running
curl http://localhost:8080/cycle/status
```

### Register an Experiment

```bash
# Register new V1 experiment
curl -X POST http://localhost:8080/versions/register \
  -H "Content-Type: application/json" \
  -d '{
    "version_id": "v1-exp-001",
    "model_version": "gpt-4o",
    "prompt_version": "code-review-v5"
  }'

# Start shadow evaluation
curl -X POST http://localhost:8080/versions/v1-exp-001/start-shadow
```

### Monitor Progress

```bash
# Get cycle status
curl http://localhost:8080/cycle/diagram

# Get promotion recommendation
curl http://localhost:8080/shadow/recommendation/v1-exp-001

# Check recovery status
curl http://localhost:8080/recovery/status
```

## Conclusion

The three-version self-evolution cycle ensures:

✅ **Continuous Improvement** - New models automatically evaluated  
✅ **Safe Deployments** - Gray-scale rollout with automatic rollback  
✅ **No Dead Ends** - Every version has a path forward  
✅ **Autonomous Operation** - No manual intervention required  
✅ **Data-Driven Decisions** - OPA policies + statistical tests
