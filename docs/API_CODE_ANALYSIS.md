# Code Analysis and Intelligent Correction API Documentation

## Overview

The Code Analysis and Intelligent Correction module provides comprehensive code scanning, error detection, and intelligent correction capabilities. It supports multiple programming languages, implements machine learning pattern recognition, and integrates with the three-version system for continuous improvement.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Analysis Engine API](#analysis-engine-api)
4. [Intelligent Correction System API](#intelligent-correction-system-api)
5. [ML Pattern Recognition API](#ml-pattern-recognition-api)
6. [Version Training API](#version-training-api)
7. [Error Types Reference](#error-types-reference)
8. [Best Practices](#best-practices)

---

## Quick Start

### Basic Code Analysis

```python
from ai_core.code_analysis import CodeAnalysisEngine

# Create engine
engine = CodeAnalysisEngine()

# Analyze a file
result = await engine.analyze_file("path/to/file.py")
print(f"Found {result.issue_count} issues")

# Analyze a directory
project = await engine.analyze_directory("./my_project")
print(f"Total issues: {project.total_issues}")
```

### Generate Corrections

```python
from ai_core.code_analysis import (
    IntelligentCorrectionSystem,
    CorrectionMode,
)

correction_system = IntelligentCorrectionSystem()

# Generate correction for an issue
suggestion = await correction_system.suggest_correction(
    issue=result.issues[0],
    code=source_code,
    mode=CorrectionMode.BASIC  # or ADVANCED, TEACHING
)

# Apply correction (requires authorization for ADVANCED mode)
if suggestion.fix_available:
    result = await correction_system.apply_correction(
        suggestion_id=suggestion.suggestion_id,
        file_path="path/to/file.py",
        authorized=True,
    )
```

### Train Models

```python
from ai_core.code_analysis import (
    ThreeVersionTrainingCoordinator,
    FeedbackType,
)

# Create training coordinator
trainer = ThreeVersionTrainingCoordinator()
await trainer.start()

# Train on a project
result = await trainer.train_on_project("./my_project")

# Train from user feedback
await trainer.train_from_feedback(
    suggestion_id="...",
    feedback_type=FeedbackType.HELPFUL,
    correct_code="fixed_code_here",
)

await trainer.stop()
```

---

## Core Components

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CODE ANALYSIS MODULE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │  Analysis Engine │  │ Correction System│  │  ML Recognition  │       │
│  │                  │  │                  │  │                  │       │
│  │ • Python         │  │ • Basic Mode     │  │ • Pattern Learn  │       │
│  │ • JavaScript     │  │ • Advanced Mode  │  │ • Similarity     │       │
│  │ • TypeScript     │  │ • Teaching Mode  │  │ • Clustering     │       │
│  │ • Java           │  │                  │  │                  │       │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘       │
│           │                     │                     │                  │
│           └─────────────────────┼─────────────────────┘                  │
│                                 ▼                                        │
│                    ┌──────────────────────┐                              │
│                    │  Version Training    │                              │
│                    │  Coordinator         │                              │
│                    └──────────────────────┘                              │
│                           │                                               │
│         ┌─────────────────┼─────────────────┐                            │
│         ▼                 ▼                 ▼                            │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐                       │
│   │    V1    │      │    V2    │      │    V3    │                       │
│   │  (Exp)   │────▶ │  (Prod)  │────▶ │(Quarant) │                       │
│   └──────────┘      └──────────┘      └──────────┘                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Supported Languages

| Language | Extension | Analyzer |
|----------|-----------|----------|
| Python | `.py`, `.pyw` | `PythonAnalyzer` |
| JavaScript | `.js`, `.jsx`, `.mjs` | `JavaScriptAnalyzer` |
| TypeScript | `.ts`, `.tsx` | `JavaScriptAnalyzer` |
| Java | `.java` | `JavaAnalyzer` |
| Go | `.go` | (In development) |
| Rust | `.rs` | (In development) |

---

## Analysis Engine API

### `CodeAnalysisEngine`

Main engine for code analysis.

#### Constructor

```python
engine = CodeAnalysisEngine()
```

#### Methods

##### `analyze_file(file_path: str) -> AnalysisResult`

Analyze a single file.

```python
result = await engine.analyze_file("src/main.py")

# Access results
print(f"Language: {result.language}")
print(f"Issues: {result.issue_count}")
print(f"Critical: {result.critical_count}")

for issue in result.issues:
    print(f"  [{issue.severity}] {issue.message}")
    print(f"    Line {issue.location.line_start}: {issue.code_snippet}")
```

##### `analyze_directory(directory: str, recursive: bool = True, file_limit: int = 1000) -> ProjectAnalysis`

Analyze all files in a directory.

```python
project = await engine.analyze_directory(
    "./my_project",
    recursive=True,
    file_limit=500,
)

print(f"Files analyzed: {len(project.file_results)}")
print(f"Files with issues: {project.files_with_issues}")
print(f"Total issues: {project.total_issues}")
print(f"Issues by type: {project.summary['issues_by_type']}")
```

##### `detect_language(file_path: str) -> Language`

Detect programming language from file extension.

```python
lang = engine.detect_language("test.py")  # Language.PYTHON
lang = engine.detect_language("app.tsx")  # Language.TYPESCRIPT
```

##### `get_statistics() -> Dict[str, Any]`

Get engine statistics.

```python
stats = engine.get_statistics()
# {
#     "analyses_performed": 10,
#     "files_analyzed": 150,
#     "issues_found": 45,
#     "supported_languages": ["python", "javascript", ...]
# }
```

### `AnalysisResult`

Result of analyzing a single file.

| Attribute | Type | Description |
|-----------|------|-------------|
| `analysis_id` | `str` | Unique analysis identifier |
| `file_path` | `str` | Path to analyzed file |
| `language` | `Language` | Detected language |
| `issues` | `List[CodeIssue]` | List of detected issues |
| `metrics` | `Dict` | File metrics (LOC, etc.) |
| `success` | `bool` | Whether analysis succeeded |
| `error` | `Optional[str]` | Error message if failed |

### `CodeIssue`

Represents a detected code issue.

| Attribute | Type | Description |
|-----------|------|-------------|
| `issue_id` | `str` | Unique issue identifier |
| `error_type` | `ErrorType` | Type of error |
| `severity` | `Severity` | Severity level |
| `message` | `str` | Human-readable message |
| `location` | `CodeLocation` | Location in file |
| `code_snippet` | `str` | Problematic code |
| `rule_id` | `str` | Rule that detected the issue |
| `suggestion` | `str` | Suggested fix |
| `fix_available` | `bool` | Whether auto-fix is available |
| `confidence` | `float` | Detection confidence (0-1) |

---

## Intelligent Correction System API

### `IntelligentCorrectionSystem`

Main system for generating and applying code corrections.

#### Constructor

```python
system = IntelligentCorrectionSystem(
    backup_dir=".code_corrections/backups",
    enable_ml=True,
)
```

#### Correction Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `BASIC` | Detailed explanations and step-by-step instructions | Learning, understanding |
| `ADVANCED` | Automatic corrections with authorization | Experienced developers |
| `TEACHING` | Interactive examples with hints and quizzes | Education, training |

#### Methods

##### `suggest_correction(issue, code, mode) -> CorrectionSuggestion`

Generate a correction suggestion for an issue.

```python
suggestion = await system.suggest_correction(
    issue=code_issue,
    code=source_code,
    mode=CorrectionMode.BASIC,
)

if suggestion:
    print(f"Original: {suggestion.original_code}")
    print(f"Fixed: {suggestion.corrected_code}")
    print(f"Explanation: {suggestion.explanation}")
    
    # Step-by-step instructions
    for step in suggestion.steps:
        print(f"Step {step.step_number}: {step.action}")
        print(f"  {step.description}")
        print(f"  Explanation: {step.explanation}")
```

##### `apply_correction(suggestion_id, file_path, authorized) -> CorrectionResult`

Apply a correction to a file.

```python
result = await system.apply_correction(
    suggestion_id=suggestion.suggestion_id,
    file_path="path/to/file.py",
    authorized=True,  # Required for ADVANCED mode
)

if result.success:
    print("Correction applied successfully")
    print(f"Lines changed: {result.lines_changed}")
else:
    print(f"Failed: {result.message}")
```

##### `rollback_correction(result_id) -> bool`

Rollback a previously applied correction.

```python
success = await system.rollback_correction(result.result_id)
if success:
    print("Rollback successful")
```

##### `submit_feedback(suggestion_id, feedback_type, rating, comment, user_id) -> UserFeedback`

Submit feedback on a correction.

```python
from ai_core.code_analysis import FeedbackType

feedback = await system.submit_feedback(
    suggestion_id=suggestion.suggestion_id,
    feedback_type=FeedbackType.HELPFUL,
    rating=5,
    comment="Great suggestion!",
    user_id="user-123",
)
```

##### `list_teaching_examples(error_type, language, difficulty) -> List[TeachingExample]`

List available teaching examples.

```python
examples = system.list_teaching_examples(
    error_type=ErrorType.SECURITY,
    language=Language.PYTHON,
    difficulty="intermediate",
)

for example in examples:
    print(f"{example.title}")
    print(f"Buggy: {example.buggy_code}")
    print(f"Fixed: {example.fixed_code}")
```

### `CorrectionSuggestion`

A suggested code correction.

| Attribute | Type | Description |
|-----------|------|-------------|
| `suggestion_id` | `str` | Unique identifier |
| `issue_id` | `str` | Related issue ID |
| `mode` | `CorrectionMode` | Correction mode |
| `original_code` | `str` | Original problematic code |
| `corrected_code` | `str` | Suggested fix |
| `diff` | `str` | Unified diff |
| `explanation` | `str` | Detailed explanation |
| `steps` | `List[CorrectionStep]` | Step-by-step instructions |
| `confidence` | `float` | Fix confidence (0-1) |
| `risk_level` | `str` | Risk level (low/medium/high) |
| `reversible` | `bool` | Whether fix can be rolled back |
| `status` | `CorrectionStatus` | Current status |

---

## ML Pattern Recognition API

### `MLPatternRecognition`

Unified ML-based pattern recognition system.

#### Constructor

```python
ml = MLPatternRecognition()
```

#### Methods

##### `analyze_code(code, categories) -> Dict[str, Any]`

Analyze code using ML pattern recognition.

```python
analysis = await ml.analyze_code(
    code=source_code,
    categories=["security", "performance"],
)

print(f"Patterns detected: {analysis['patterns_detected']}")
print(f"Risk score: {analysis['risk_score']:.2%}")
print(f"Recommendations: {analysis['recommendations']}")

for pattern in analysis['patterns']:
    print(f"  {pattern['name']} (confidence: {pattern['confidence']:.2%})")
```

##### `learn_from_fix(buggy_code, fixed_code, issue_type)`

Learn from a successful fix.

```python
ml.learn_from_fix(
    buggy_code='password = "secret"',
    fixed_code='password = os.environ.get("PASSWORD")',
    issue_type="hardcoded_secret",
)
```

### `PatternLearningEngine`

Engine for learning code patterns.

##### `learn_pattern(name, positive_examples, negative_examples, pattern_type, categories) -> CodePattern`

Learn a new pattern from examples.

```python
pattern = engine.learn_pattern(
    name="todo_comment",
    positive_examples=[
        "# TODO: fix this",
        "// TODO fix later",
        "# TODO: implement feature",
    ],
    negative_examples=[
        "# This is a comment",
        "// Regular comment",
    ],
    pattern_type="style",
    categories=["style", "documentation"],
)
```

##### `detect_patterns(code, categories) -> List[PatternMatch]`

Detect patterns in code.

```python
matches = engine.detect_patterns(
    code=source_code,
    categories=["security"],
)

for match in matches:
    print(f"Pattern: {match.pattern_name}")
    print(f"Line: {match.location['line']}")
    print(f"Confidence: {match.confidence:.2%}")
```

##### `promote_pattern(pattern_id) -> bool`

Promote pattern from V1 to V2 after validation.

```python
if engine.promote_pattern("pattern-123"):
    print("Pattern promoted to production")
```

##### `quarantine_pattern(pattern_id, reason) -> bool`

Move pattern to V3 quarantine.

```python
engine.quarantine_pattern("pattern-456", "Low accuracy")
```

### `CodeSimilarityEngine`

Engine for finding similar code.

##### `add_code(code, issue_type, fix_applied, metadata) -> str`

Add code to the similarity index.

```python
code_hash = engine.add_code(
    code='password = "secret"',
    issue_type="hardcoded_secret",
    fix_applied='password = os.environ.get("PASSWORD")',
)
```

##### `find_similar(code, threshold, limit) -> List[SimilarCode]`

Find similar code in the index.

```python
similar = engine.find_similar(
    code='api_key = "key123"',
    threshold=0.7,
    limit=10,
)

for s in similar:
    print(f"Similarity: {s.similarity_score:.2%}")
    print(f"Issue type: {s.issue_type}")
    if s.fix_applied:
        print(f"Known fix: {s.fix_applied}")
```

---

## Version Training API

### `ThreeVersionTrainingCoordinator`

Coordinates training across all three versions.

#### Constructor

```python
from ai_core.code_analysis import (
    ThreeVersionTrainingCoordinator,
    TrainingConfig,
)

config = TrainingConfig(
    mode=TrainingMode.INCREMENTAL,
    batch_size=100,
    promotion_threshold=0.85,
    demotion_threshold=0.6,
)

coordinator = ThreeVersionTrainingCoordinator(config=config)
```

#### Methods

##### `start()` / `stop()`

Start and stop the training coordinator.

```python
await coordinator.start()
# ... training operations ...
await coordinator.stop()
```

##### `train_on_project(project_path, version) -> Dict[str, Any]`

Train on a project codebase.

```python
result = await coordinator.train_on_project(
    project_path="./my_project",
    version=ModelVersion.V1_EXPERIMENTAL,
)

print(f"Files analyzed: {result['files_analyzed']}")
print(f"Samples created: {result['samples_created']}")
print(f"Total issues: {result['total_issues']}")
```

##### `train_from_feedback(suggestion_id, feedback_type, correct_code) -> Dict[str, Any]`

Train from user feedback.

```python
result = await coordinator.train_from_feedback(
    suggestion_id="suggestion-123",
    feedback_type=FeedbackType.HELPFUL,
    correct_code='password = os.environ.get("PASSWORD")',
)
```

##### `run_batch_training(version) -> Dict[str, TrainingMetrics]`

Run batch training on accumulated samples.

```python
metrics = await coordinator.run_batch_training(
    version=ModelVersion.V1_EXPERIMENTAL,
)

for version, m in metrics.items():
    print(f"{version}: accuracy={m.accuracy:.2%}")
```

##### `evaluate_patterns() -> Dict[str, Any]`

Evaluate patterns and handle promotion/demotion.

```python
result = await coordinator.evaluate_patterns()

print(f"Patterns promoted: {result['patterns_promoted']}")
print(f"Patterns demoted: {result['patterns_demoted']}")
```

##### `get_training_status() -> Dict[str, Any]`

Get comprehensive training status.

```python
status = coordinator.get_training_status()

print(f"Total samples: {status['total_samples_trained']}")
print(f"V1 accuracy: {status['v1_status']['accuracy']:.2%}")
print(f"V2 accuracy: {status['v2_status']['accuracy']:.2%}")
```

### `TrainingConfig`

Configuration for training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `TrainingMode` | `INCREMENTAL` | Training mode |
| `batch_size` | `int` | `100` | Batch size for training |
| `learning_rate` | `float` | `0.01` | Learning rate |
| `validation_split` | `float` | `0.2` | Validation data percentage |
| `min_samples_for_pattern` | `int` | `5` | Min samples before evaluation |
| `promotion_threshold` | `float` | `0.85` | Accuracy for V1→V2 promotion |
| `demotion_threshold` | `float` | `0.6` | Accuracy below this → V3 |

---

## Error Types Reference

### ErrorType Enum

| Value | Description | Severity Range |
|-------|-------------|----------------|
| `SYNTAX` | Syntax errors | Critical |
| `RUNTIME` | Runtime errors | High-Critical |
| `LOGICAL` | Logic errors | Low-High |
| `SECURITY` | Security vulnerabilities | Medium-Critical |
| `PERFORMANCE` | Performance issues | Low-Medium |
| `STYLE` | Code style violations | Low |
| `DEPRECATED` | Deprecated code usage | Low-Medium |
| `TYPE_ERROR` | Type mismatches | Medium-High |

### Severity Enum

| Value | Priority | Action Required |
|-------|----------|-----------------|
| `CRITICAL` | 1 | Immediate fix required |
| `HIGH` | 2 | Fix before deployment |
| `MEDIUM` | 3 | Should be addressed |
| `LOW` | 4 | Consider fixing |
| `INFO` | 5 | Informational only |

### Rule ID Format

Rules follow the format: `{LANG}-{TYPE}-{NUM}`

- **LANG**: `PY` (Python), `JS` (JavaScript), `JAVA` (Java)
- **TYPE**: `SYN` (Syntax), `SEC` (Security), `PERF` (Performance), `STY` (Style), `LOG` (Logical), `RUN` (Runtime)
- **NUM**: Sequential number

Examples:
- `PY-SEC-001`: Python hardcoded secret
- `JS-STY-001`: JavaScript var usage
- `JAVA-PERF-001`: Java string concatenation in loop

---

## Best Practices

### 1. Analyze Before Commit

```python
async def pre_commit_check():
    engine = CodeAnalysisEngine()
    result = await engine.analyze_directory("./src")
    
    critical_issues = [
        i for r in result.file_results 
        for i in r.issues 
        if i.severity == Severity.CRITICAL
    ]
    
    if critical_issues:
        print(f"❌ {len(critical_issues)} critical issues found")
        for issue in critical_issues:
            print(f"  {issue.location.file_path}:{issue.location.line_start}")
            print(f"    {issue.message}")
        return False
    
    print("✅ No critical issues")
    return True
```

### 2. Use Teaching Mode for Learning

```python
# Get teaching examples for a specific error type
examples = system.list_teaching_examples(error_type=ErrorType.SECURITY)

for example in examples:
    print(f"\n=== {example.title} ===")
    print(f"Difficulty: {example.difficulty}")
    print(f"\nBuggy code:\n{example.buggy_code}")
    
    # Interactive learning
    input("Press Enter to see the fix...")
    print(f"\nFixed code:\n{example.fixed_code}")
    
    for step in example.steps:
        print(f"\nStep {step.step_number}: {step.description}")
        print(f"Explanation: {step.explanation}")
```

### 3. Continuous Learning Pipeline

```python
async def continuous_learning():
    coordinator = ThreeVersionTrainingCoordinator()
    await coordinator.start()
    
    try:
        while True:
            # Train on new code
            await coordinator.train_on_project("./src")
            
            # Evaluate patterns
            eval_result = await coordinator.evaluate_patterns()
            print(f"Promoted: {len(eval_result['patterns_promoted'])}")
            print(f"Demoted: {len(eval_result['patterns_demoted'])}")
            
            # Wait before next iteration
            await asyncio.sleep(3600)  # 1 hour
            
    finally:
        await coordinator.stop()
```

### 4. Security-First Analysis

```python
async def security_audit():
    engine = CodeAnalysisEngine()
    ml = MLPatternRecognition()
    
    project = await engine.analyze_directory("./src")
    
    # Get all security issues
    security_issues = []
    for result in project.file_results:
        for issue in result.issues:
            if issue.error_type == ErrorType.SECURITY:
                security_issues.append(issue)
    
    # ML analysis for additional patterns
    for result in project.file_results:
        with open(result.file_path, 'r') as f:
            code = f.read()
        
        ml_analysis = await ml.analyze_code(code, categories=["security"])
        if ml_analysis['risk_score'] > 0.5:
            print(f"⚠️ High risk: {result.file_path}")
    
    print(f"\nTotal security issues: {len(security_issues)}")
    return security_issues
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12-10 | Initial release |

## License

MIT License - See LICENSE file for details.
