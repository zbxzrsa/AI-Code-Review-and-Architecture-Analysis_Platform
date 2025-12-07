# Migration Guide: V1 to V2

## Overview

This guide covers migrating from CodeReviewAI_V1 (Experimental) to CodeReviewAI_V2 (Production).

## Breaking Changes

### 1. Configuration Changes

```python
# V1
config = ReviewConfig(
    dimensions=[Dimension.SECURITY],
    max_findings=50
)

# V2 - Additional required fields
config = ReviewConfig(
    dimensions=[Dimension.SECURITY],
    max_findings=50,
    enable_hallucination_check=True,  # NEW
    slo_timeout_ms=3000,              # NEW
    enable_caching=True,              # NEW
)
```

### 2. New Imports

```python
# V1
from modules.CodeReviewAI_V1 import CodeReviewer, IssueDetector

# V2 - New components
from modules.CodeReviewAI_V2 import (
    CodeReviewer,
    IssueDetector,
    HallucinationDetector,  # NEW
)
```

### 3. ReviewResult Changes

```python
# V2 adds new fields
result.hallucination_check_passed  # bool
result.verified_findings_count     # int
result.rejected_findings_count     # int
result.slo_met                     # bool
result.from_cache                  # bool
```

### 4. Finding Changes

```python
# V2 adds verification fields
finding.verification_status      # str
finding.verification_confidence  # float
finding.verified_at             # datetime
finding.verification_method     # str
```

## Migration Steps

### Step 1: Update Dependencies

```bash
# No new external dependencies required
```

### Step 2: Update Imports

```python
# Before
from modules.CodeReviewAI_V1 import CodeReviewer

# After
from modules.CodeReviewAI_V2 import CodeReviewer
```

### Step 3: Update Configuration

```python
# Add V2 specific config
config = ReviewConfig(
    enable_hallucination_check=True,
    slo_timeout_ms=3000,
)
```

### Step 4: Handle New Response Fields

```python
result = await reviewer.review(code)

# Check SLO compliance
if not result.slo_met:
    logger.warning("SLO violated")

# Check hallucination results
if result.rejected_findings_count > 0:
    logger.info(f"Filtered {result.rejected_findings_count} hallucinations")
```

## Backward Compatibility

- V2 API is largely backward compatible with V1
- Default values ensure V1-style usage works
- New features are opt-in via configuration

## Rollback Procedure

If issues occur, rollback by:

1. Change imports back to V1
2. Remove V2-specific configuration
3. Handle any missing fields in response
