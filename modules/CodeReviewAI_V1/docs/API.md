# CodeReviewAI_V1 API Documentation

## Overview

The CodeReviewAI_V1 module provides AI-powered code review capabilities with multi-dimensional analysis and configurable review strategies.

## Quick Start

```python
from modules.CodeReviewAI_V1.src import CodeReviewer, ReviewConfig

# Create reviewer
reviewer = CodeReviewer(strategy="ensemble")

# Review code
result = await reviewer.review(
    code="def test(): pass",
    language="python"
)

print(f"Score: {result.overall_score}")
print(f"Findings: {len(result.findings)}")
```

## Classes

### CodeReviewer

Main code review orchestrator.

#### Constructor

```python
CodeReviewer(
    strategy: str = "ensemble",
    config: Optional[ReviewConfig] = None
)
```

**Parameters:**

- `strategy`: Review strategy to use
  - `"baseline"`: Direct instruction-tuned review
  - `"chain_of_thought"`: Step-by-step reasoning
  - `"ensemble"`: Multiple strategies with voting (default)
- `config`: Optional ReviewConfig instance

#### Methods

##### review()

```python
async def review(
    code: str,
    language: str = "python",
    config: Optional[ReviewConfig] = None
) -> ReviewResult
```

Execute code review on provided source code.

**Parameters:**

- `code`: Source code to review
- `language`: Programming language (default: "python")
- `config`: Optional config override

**Returns:** `ReviewResult` with findings and scores

**Example:**

```python
result = await reviewer.review(
    code="""
def vulnerable():
    password = "secret123"
    return eval(user_input)
""",
    language="python"
)
```

##### get_metrics()

```python
def get_metrics() -> Dict[str, Any]
```

Get reviewer performance metrics.

**Returns:** Dictionary with:

- `review_count`: Total reviews performed
- `total_findings`: Total findings detected
- `avg_findings_per_review`: Average findings per review
- `total_time_ms`: Total processing time
- `avg_time_ms`: Average processing time

### IssueDetector

Pattern-based issue detection.

#### Constructor

```python
IssueDetector()
```

#### Methods

##### detect()

```python
async def detect(
    code: str,
    language: str = "python",
    dimensions: Optional[List[Dimension]] = None
) -> List[Finding]
```

Detect issues in code.

**Parameters:**

- `code`: Source code to analyze
- `language`: Programming language
- `dimensions`: Dimensions to check (all if None)

**Returns:** List of `Finding` objects

##### add_rule()

```python
def add_rule(rule: DetectionRule)
```

Add a custom detection rule.

##### enable_rule() / disable_rule()

```python
def enable_rule(rule_id: str)
def disable_rule(rule_id: str)
```

Enable or disable a specific rule.

### FixSuggester

Generate fix suggestions for detected issues.

#### Methods

##### suggest_fixes()

```python
async def suggest_fixes(
    code: str,
    findings: List[Finding]
) -> List[FixSuggestion]
```

Generate fix suggestions.

##### apply_fix()

```python
async def apply_fix(
    code: str,
    suggestion: FixSuggestion
) -> str
```

Apply a fix suggestion to code.

##### validate_fix()

```python
async def validate_fix(
    original_code: str,
    fixed_code: str
) -> Dict[str, Any]
```

Validate that a fix doesn't introduce new issues.

### QualityScorer

Calculate code quality scores.

#### Methods

##### score()

```python
async def score(
    code: str,
    findings: List[Finding]
) -> QualityScore
```

Calculate quality score for code.

## Data Models

### ReviewConfig

```python
@dataclass
class ReviewConfig:
    dimensions: List[Dimension]  # Dimensions to check
    max_findings: int = 50       # Maximum findings
    min_confidence: float = 0.7  # Minimum confidence
    include_suggestions: bool = True
    include_explanations: bool = True
    language: Optional[str] = None
    context_lines: int = 3
```

### Finding

```python
@dataclass
class Finding:
    dimension: str        # Dimension (security, performance, etc.)
    issue: str           # Issue description
    line_numbers: List[int]
    severity: str        # critical, high, medium, low, info
    confidence: float    # 0.0 - 1.0
    suggestion: str      # Fix suggestion
    explanation: str     # Detailed explanation
    cwe_id: Optional[str]
    rule_id: Optional[str]
    code_snippet: Optional[str]
    fix_snippet: Optional[str]
    reasoning_steps: List[str]
```

### ReviewResult

```python
@dataclass
class ReviewResult:
    review_id: str
    code_hash: str
    status: ReviewStatus
    findings: List[Finding]
    overall_score: float       # 0-100
    dimension_scores: Dict[str, float]
    model_version: str
    strategy_used: str
    processing_time_ms: float
    timestamp: datetime
    avg_confidence: float
    min_confidence: float
```

### QualityScore

```python
@dataclass
class QualityScore:
    overall: float           # 0-100
    dimensions: Dict[str, float]
    grade: str              # A, B, C, D, F
    timestamp: datetime
    findings_count: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    code_lines: int
    issues_per_line: float
```

## Enumerations

### Dimension

```python
class Dimension(str, Enum):
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
```

### Severity

```python
class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
```

### ReviewStatus

```python
class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
```

## Detection Rules

### Security Rules

| Rule ID | Description                   | Severity | CWE     |
| ------- | ----------------------------- | -------- | ------- |
| SEC001  | eval() usage                  | Critical | CWE-95  |
| SEC002  | SQL injection                 | Critical | CWE-89  |
| SEC003  | Hardcoded credentials         | High     | CWE-798 |
| SEC004  | Shell injection (shell=True)  | High     | CWE-78  |
| SEC005  | Unsafe pickle deserialization | Medium   | CWE-502 |

### Performance Rules

| Rule ID | Description                  | Severity |
| ------- | ---------------------------- | -------- |
| PERF001 | Nested loop                  | Medium   |
| PERF002 | String concatenation in loop | Low      |
| PERF003 | Blocking sleep call          | Medium   |

### Maintainability Rules

| Rule ID  | Description        | Severity |
| -------- | ------------------ | -------- |
| MAINT001 | Missing docstring  | Low      |
| MAINT002 | TODO/FIXME comment | Medium   |
| MAINT003 | Bare except clause | High     |

### Correctness Rules

| Rule ID | Description                | Severity |
| ------- | -------------------------- | -------- |
| CORR001 | == None instead of is None | High     |
| CORR002 | Mutable default argument   | Medium   |

## Error Handling

```python
try:
    result = await reviewer.review(code)
except Exception as e:
    # Review failed
    print(f"Review error: {e}")
```

Failed reviews return a `ReviewResult` with:

- `status = ReviewStatus.FAILED`
- `findings = []`
- `overall_score = 0`

## Version History

| Version | Date | Changes                      |
| ------- | ---- | ---------------------------- |
| 1.0.0   | 2024 | Initial experimental release |
