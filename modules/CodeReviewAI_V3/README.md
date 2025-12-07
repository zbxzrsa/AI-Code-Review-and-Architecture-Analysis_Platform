# CodeReviewAI_V3 - Legacy/Quarantine

## Overview

Quarantined code review module for comparison baseline and re-evaluation.

## Version Information

- **Version**: 3.0.0
- **Status**: Quarantine (V3)
- **Last Updated**: 2024
- **Deprecated**: Yes

## Purpose

- ✅ Baseline comparison for V1/V2 experiments
- ✅ Re-evaluation of failed experiments
- ✅ Historical analysis and audit trail
- ✅ Rollback target if V2 fails
- ❌ NOT for production traffic

## Access Restrictions

- **Admin Only**: Only admins can access V3
- **Read-Only**: No modifications allowed
- **Comparison Mode**: Used for A/B testing baseline

## Directory Structure

```
CodeReviewAI_V3/
├── src/
│   ├── code_reviewer.py      # Read-only reviewer
│   ├── comparison_engine.py  # V1/V2/V3 comparison
│   └── models.py            # Shared models
├── tests/
│   └── test_comparison.py
├── config/
│   └── quarantine_config.yaml
└── README.md
```

## Usage (Admin Only)

```python
from modules.CodeReviewAI_V3 import CodeReviewer, ComparisonEngine

# Compare V2 against V3 baseline
engine = ComparisonEngine()
comparison = await engine.compare(
    code="...",
    v2_result=v2_result,
    v3_baseline=v3_result
)
```

## Re-evaluation Process

1. Admin requests re-evaluation
2. System runs V3 against test cases
3. Results compared to V1/V2
4. Decision: promote to V1 or delete

## Migration

- V3 → V1: Re-evaluation and promotion
- V3 → Delete: After retention period
