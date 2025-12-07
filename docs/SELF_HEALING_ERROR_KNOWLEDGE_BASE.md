# Self-Healing Error Knowledge Base

## Overview

The Error Knowledge Base is a comprehensive system for storing, matching, and automatically repairing known errors in the codebase. It integrates with the self-healing layer to provide automated error detection and resolution.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Error Knowledge Base                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Error        │  │ Pattern      │  │ Repair       │          │
│  │ Records      │  │ Matching     │  │ Execution    │          │
│  │              │  │              │  │              │          │
│  │ • ID         │  │ • Regex      │  │ • Apply Fix  │          │
│  │ • Category   │  │ • File Path  │  │ • Verify     │          │
│  │ • Severity   │  │ • Line       │  │ • Rollback   │          │
│  │ • Fixes      │  │ • Context    │  │ • Log        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Storage Layer (JSON)                     │  │
│  │  data/error_knowledge/knowledge_base.json                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. ErrorRecord

Complete error information including:

| Field                 | Description                           |
| --------------------- | ------------------------------------- |
| `error_id`            | Unique identifier                     |
| `error_type`          | "code_smell", "bug", "vulnerability"  |
| `category`            | ErrorCategory enum                    |
| `severity`            | BLOCKER, CRITICAL, MAJOR, MINOR, INFO |
| `title`               | Short description                     |
| `description`         | Detailed description                  |
| `file_path`           | Affected file(s)                      |
| `line_number`         | Specific line (optional)              |
| `trigger_conditions`  | What causes the error                 |
| `reproduction_steps`  | How to reproduce                      |
| `patterns`            | Matching patterns                     |
| `fixes`               | Code modifications                    |
| `verification_tests`  | Tests to verify fix                   |
| `priority`            | 1-10 (1 = highest)                    |
| `auto_repair_enabled` | Allow automatic fixing                |

### 2. ErrorPattern

Pattern matching for error identification:

```python
@dataclass
class ErrorPattern:
    pattern_id: str
    regex_pattern: str      # Match error message
    file_pattern: str       # Glob pattern for files
    line_pattern: str       # Match specific line content
    context_patterns: list  # Additional context
```

### 3. CodeFix

Code modification to repair error:

```python
@dataclass
class CodeFix:
    file_path: str
    old_content: str
    new_content: str
    description: str
    line_number: int
    is_regex: bool
```

### 4. VerificationTest

Test to verify repair success:

```python
@dataclass
class VerificationTest:
    test_name: str
    test_type: str          # "unit", "integration", "lint"
    test_command: str
    expected_result: str
    timeout_seconds: int
```

## Stored Error Patterns

### Error Type 1: Method Always Returns Same Value

**SonarQube Rule**: python:S3516

| Property | Value                                                                         |
| -------- | ----------------------------------------------------------------------------- |
| Severity | BLOCKER                                                                       |
| Category | Code Smell                                                                    |
| Files    | `**/deployment/system.py`, `**/practical_deployment.py`, `**/redis_client.py` |

**Pattern**:

```regex
Refactor this method to not always return the same value
```

**Fix Example**:

```python
# Before
def is_healthy(self) -> bool:
    return True

# After
def is_healthy(self) -> bool:
    try:
        cpu_ok = self.metrics.get("cpu_usage", 0) < 90
        memory_ok = self.metrics.get("memory_usage", 0) < 90
        return cpu_ok and memory_ok
    except Exception:
        return False
```

### Error Type 2: Undefined Exports in **all**

**SonarQube Rule**: Undefined name in **all**

| Property | Value                   |
| -------- | ----------------------- |
| Severity | BLOCKER                 |
| Category | Bug                     |
| Files    | `modules/*/__init__.py` |

**Pattern**:

```regex
is not defined
```

**Fix**: Add placeholder implementations for exported names.

### Error Type 3: Unexpected Named Arguments

**SonarQube Rule**: CWE - Unexpected keyword argument

| Property | Value           |
| -------- | --------------- |
| Severity | BLOCKER         |
| Category | Bug             |
| Files    | `tests/**/*.py` |

**Fix**: Update parameter names to match function signature.

## Usage

### Python API

```python
from backend.shared.self_healing import (
    get_knowledge_base,
    ErrorRecord,
    ErrorCategory,
    ErrorSeverity
)

# Get knowledge base instance
kb = get_knowledge_base()

# Find matching errors
matches = kb.find_matching_errors(
    error_message="Refactor this method to not always return the same value",
    file_path="src/module.py",
    line_content="return True"
)

# Auto-repair an error
log = await kb.auto_repair("ERR_METHOD_ALWAYS_SAME_VALUE_001")
print(f"Repair status: {log.status.value}")

# Get statistics
stats = kb.get_statistics()
print(f"Total errors: {stats['total_errors_stored']}")
```

### Command Line

```bash
# List all known errors
python scripts/apply_self_healing_fixes.py --list

# Show statistics
python scripts/apply_self_healing_fixes.py --stats

# Dry run (preview changes)
python scripts/apply_self_healing_fixes.py --dry-run

# Apply all fixes
python scripts/apply_self_healing_fixes.py

# Apply specific error fix
python scripts/apply_self_healing_fixes.py --error-id ERR_METHOD_ALWAYS_SAME_VALUE_001
```

### Makefile

```bash
# Run self-healing analysis
make self-heal-report

# Apply fixes
make self-heal-apply
```

## Adding New Error Patterns

### 1. Define the Error Record

```python
from backend.shared.self_healing import (
    ErrorRecord, ErrorPattern, CodeFix, VerificationTest,
    ErrorCategory, ErrorSeverity, get_knowledge_base
)

new_error = ErrorRecord(
    error_id="ERR_NEW_PATTERN_001",
    error_type="bug",
    category=ErrorCategory.BUG,
    severity=ErrorSeverity.MAJOR,
    title="Description of the error",
    description="Detailed explanation",
    file_path="**/*.py",
    patterns=[
        ErrorPattern(
            pattern_id="PAT_NEW",
            regex_pattern=r"error pattern to match",
            file_pattern="**/*.py"
        )
    ],
    fixes=[
        CodeFix(
            file_path="path/to/file.py",
            old_content="buggy code",
            new_content="fixed code",
            description="What the fix does"
        )
    ],
    verification_tests=[
        VerificationTest(
            test_name="test_fix_works",
            test_type="unit",
            test_command="pytest tests/ -k test_name",
            expected_result="All tests pass"
        )
    ],
    priority=3,
    auto_repair_enabled=True
)

kb = get_knowledge_base()
kb.add_error(new_error)
```

### 2. Pattern Matching Best Practices

- Use specific regex patterns to avoid false positives
- Include file patterns to limit scope
- Add context patterns for disambiguation
- Test patterns against known instances

### 3. Fix Guidelines

- Make fixes as minimal as possible
- Ensure fixes are idempotent (can be applied multiple times)
- Support regex patterns for variable content
- Include rollback capability

## Auto-Repair Flow

```
┌──────────────┐
│ Error        │
│ Detected     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Pattern      │──No Match──▶ Manual Review
│ Matching     │
└──────┬───────┘
       │ Match Found
       ▼
┌──────────────┐
│ Auto-Repair  │──Disabled──▶ Manual Review
│ Enabled?     │
└──────┬───────┘
       │ Yes
       ▼
┌──────────────┐
│ Apply        │
│ Fixes        │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Run          │──Failed────▶ Rollback
│ Verification │
└──────┬───────┘
       │ Passed
       ▼
┌──────────────┐
│ Log &        │
│ Record       │
└──────────────┘
```

## Statistics & Monitoring

The knowledge base tracks:

| Metric                   | Description              |
| ------------------------ | ------------------------ |
| `total_errors_stored`    | Number of error patterns |
| `total_repairs_executed` | Total repair attempts    |
| `successful_repairs`     | Successful fixes         |
| `failed_repairs`         | Failed attempts          |
| `patterns_matched`       | Pattern match count      |

## Integration with SonarQube

The error patterns are designed to match SonarQube issue formats:

1. **Blocker Issues**: Priority 1-2, auto-repair enabled
2. **Critical Issues**: Priority 3-4, auto-repair with review
3. **Major Issues**: Priority 5-6, manual review recommended
4. **Minor Issues**: Priority 7-10, optional auto-repair

## Files

| File                                                  | Description         |
| ----------------------------------------------------- | ------------------- |
| `backend/shared/self_healing/error_knowledge_base.py` | Core implementation |
| `backend/shared/self_healing/__init__.py`             | Module exports      |
| `scripts/apply_self_healing_fixes.py`                 | CLI tool            |
| `data/error_knowledge/knowledge_base.json`            | Persistent storage  |

---

_Last updated: December 2024_
