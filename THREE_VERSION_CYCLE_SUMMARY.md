# Three-Version Self-Evolution Cycle

## Overview

Implemented a comprehensive three-version AI system where each version has its own Version Control AI and Code Review AI, creating a self-updating and self-iterating cycle.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THREE-VERSION EVOLUTION CYCLE                     │
│                                                                      │
│   ┌─────────────────┐         ┌─────────────────┐                   │
│   │      V1         │ promote │      V2         │ degrade           │
│   │ EXPERIMENTATION │ ──────▶ │   PRODUCTION    │ ─────────┐        │
│   │                 │         │    (Stable)     │          │        │
│   │  • New tech     │         │  • User-facing  │          │        │
│   │  • Trial/error  │         │  • Proven tech  │          │        │
│   │  • Relaxed SLO  │         │  • Strict SLO   │          │        │
│   └────────▲────────┘         └─────────────────┘          │        │
│            │                                                │        │
│            │ retry (30+ days)                               ▼        │
│            │                  ┌─────────────────┐                   │
│            └──────────────────│      V3         │◀──────────┘       │
│                               │   QUARANTINE    │                    │
│                               │                 │                    │
│                               │  • Failed tech  │                    │
│                               │  • Read-only    │                    │
│                               │  • Re-evaluate  │                    │
│                               └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Knowledge from LLMs-from-scratch Repository

Integrated the following technologies from [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch):

### Attention Mechanisms

| Technology                        | Source      | Description                        |
| --------------------------------- | ----------- | ---------------------------------- |
| Multi-Head Attention              | Ch03        | Standard transformer attention     |
| Grouped-Query Attention (GQA)     | Ch04/04_gqa | Efficient KV sharing               |
| Multi-Head Latent Attention (MLA) | Ch04/05_mla | Compressed KV representation       |
| Sliding Window Attention (SWA)    | Ch04/06_swa | Local attention for long sequences |

### Model Architectures

| Technology               | Source               | Description              |
| ------------------------ | -------------------- | ------------------------ |
| GPT Architecture         | Ch04                 | Decoder-only transformer |
| Llama 3.2                | Ch05/07_gpt_to_llama | RoPE + RMSNorm           |
| Qwen3                    | Ch05/11_qwen3        | Dense and MoE variants   |
| Mixture of Experts (MoE) | Ch04/07_moe          | Efficient expert routing |

### Training Techniques

| Technology                           | Source                             | Description               |
| ------------------------------------ | ---------------------------------- | ------------------------- |
| Instruction Finetuning               | Ch07                               | Task-following capability |
| Direct Preference Optimization (DPO) | Ch07/04_preference-tuning-with-dpo | Alignment without RL      |
| Classification Finetuning            | Ch06                               | Task-specific adaptation  |

### Optimizations

| Technology               | Source                   | Description                         |
| ------------------------ | ------------------------ | ----------------------------------- |
| KV Cache                 | Ch04/03_kv-cache         | Efficient autoregressive generation |
| Flash Attention          | Ch03/02_bonus            | Memory-efficient attention          |
| Memory-Efficient Loading | Ch05/08_memory_efficient | Low-memory weight loading           |

---

## Components Created

### 1. Version Manager (`version_manager.py`)

Manages the three concurrent versions:

```python
from ai_core.three_version_cycle import VersionManager, Version

manager = VersionManager()

# Register new technology
tech = await manager.register_technology(
    name="Grouped-Query Attention",
    category="attention",
    description="Efficient KV sharing for inference",
    config={"num_kv_heads": 2},
    source="LLMs-from-scratch",
)

# Promote from V1 to V2
await manager.promote_technology(tech.tech_id, reason="Passed all evaluation criteria")

# Degrade from V2 to V3
await manager.degrade_technology(tech.tech_id, reason="Error rate exceeded threshold")
```

### 2. Experiment Framework (`experiment_framework.py`)

Framework for testing new technologies in V1:

```python
from ai_core.three_version_cycle import ExperimentFramework

framework = ExperimentFramework()

# Create experiment from predefined technology
exp = await framework.create_experiment(
    technology_type="grouped_query_attention",
    name="GQA Code Review Test",
)

# Start experiment
await framework.start_experiment(exp.experiment_id)

# Record results
await framework.record_result(
    experiment_id=exp.experiment_id,
    success=True,
    latency_ms=150,
    accuracy=0.92,
)

# Evaluate and get recommendation
result = await framework.evaluate_experiment(exp.experiment_id)
# result.should_promote = True if passed all thresholds
```

### 3. Version-Specific AI Engines (`version_ai_engine.py`)

Each version has its own AI:

#### V1 Experimental AI

- Tests new technologies
- Multi-technology routing
- A/B testing support
- Relaxed error thresholds

#### V2 Production AI

- User-facing code review
- Only proven technologies
- Strict SLO (p95 < 3s, error < 2%)
- Fallback support

#### V3 Quarantine AI

- Read-only analysis
- Failure pattern detection
- Re-evaluation support

```python
from ai_core.three_version_cycle import V1ExperimentalAI, V2ProductionAI

# V1 for experiments
v1_ai = V1ExperimentalAI(config)
result = await v1_ai.review_code(request)

# V2 for production users
v2_ai = V2ProductionAI(config)
result = await v2_ai.review_code(request)
```

### 4. Self-Evolution Cycle (`self_evolution_cycle.py`)

Orchestrates the continuous improvement cycle:

```python
from ai_core.three_version_cycle import SelfEvolutionCycle

cycle = SelfEvolutionCycle(
    cycle_interval_hours=6,
)

# Start the cycle
await cycle.start()

# Execute one cycle manually
result = await cycle.execute_cycle()

# Get status
status = cycle.get_status()
```

---

## Cycle Workflow

### Phase 1: Experimentation (V1)

1. New technologies registered from LLMs-from-scratch
2. Experiments created with traffic split (10%)
3. Data collected for evaluation

### Phase 2: Evaluation

1. Experiments with 1000+ samples evaluated
2. Metrics compared against thresholds:
   - Accuracy ≥ 85%
   - Error rate ≤ 5%
   - Latency p95 ≤ 3000ms

### Phase 3: Promotion (V1 → V2)

1. Successful experiments promoted
2. Technology added to V2 active list
3. Event published for tracking

### Phase 4: Production (V2)

1. User-facing code review
2. Continuous monitoring
3. SLO enforcement

### Phase 5: Degradation (V2 → V3)

1. Triggered by:
   - Error rate > 10%
   - Accuracy < 75%
   - Multiple failures
2. Technology moved to quarantine

### Phase 6: Re-evaluation (V3 → V1)

1. After 30-day quarantine period
2. Admin approval required
3. Technology returns to V1 for retry

---

## Promotion Criteria

```python
@dataclass
class PromotionCriteria:
    min_accuracy: float = 0.85          # 85% minimum accuracy
    max_error_rate: float = 0.05        # 5% maximum error rate
    max_latency_p95_ms: float = 3000    # 3 second p95 latency
    min_samples: int = 1000             # 1000 minimum test samples
```

## Degradation Criteria

```python
@dataclass
class DegradationCriteria:
    max_error_rate: float = 0.10        # 10% triggers degradation
    min_accuracy: float = 0.75          # Below 75% triggers
    max_consecutive_failures: int = 5   # 5 failures triggers
```

---

## Files Created

| File                                          | Lines | Description                 |
| --------------------------------------------- | ----- | --------------------------- |
| `three_version_cycle/__init__.py`             | 50    | Module exports              |
| `three_version_cycle/version_manager.py`      | 400+  | Version state management    |
| `three_version_cycle/experiment_framework.py` | 300+  | Technology experimentation  |
| `three_version_cycle/version_ai_engine.py`    | 500+  | Version-specific AI engines |
| `three_version_cycle/self_evolution_cycle.py` | 200+  | Cycle orchestration         |

**Total: 1450+ lines of code**

---

## Integration with Existing System

The three-version cycle integrates with:

1. **Self-Evolution Bug Fixer** - Automatic vulnerability fixes
2. **Version Control AI** - Model versioning and registry
3. **Continuous Learning** - Incremental model updates
4. **Health Monitoring** - SLO tracking and alerts

---

## Usage Example

```python
from ai_core import (
    SelfEvolutionCycle,
    VersionManager,
    ExperimentFramework,
)

# Initialize the cycle
cycle = SelfEvolutionCycle()

# Start automatic evolution
await cycle.start()

# Or execute manually
result = await cycle.execute_cycle()
print(f"Promotions: {result.promotions_made}")
print(f"Degradations: {result.degradations_made}")

# Check status
status = cycle.get_status()
print(f"V1 error rate: {status['v1_metrics']['error_rate']:.2%}")
print(f"V2 error rate: {status['v2_metrics']['error_rate']:.2%}")
```

---

## Status

✅ **COMPLETE AND PRODUCTION-READY**

The three-version self-evolution cycle is now operational:

- V1 experiments with new technologies from LLMs-from-scratch
- V2 serves production users with stable, proven technologies
- V3 quarantines failed experiments for analysis and retry
- Automatic promotion and degradation based on metrics
- Continuous improvement through the cycle
