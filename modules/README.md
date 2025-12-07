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
| **CodeReviewAI**    | AI-powered code review  |     |     |     | Complete |
| **Authentication**  | User auth & sessions    |     |     |     | Complete |
| **SelfHealing**     | System self-healing     |     |     |     | Complete |
| **AIOrchestration** | AI model orchestration  |     |     |     | Skeleton |
| **Caching**         | Multi-level caching     |     |     |     | Skeleton |
| **Monitoring**      | Metrics & observability |     |     |     | Skeleton |

## Quick Start

### Use Production Module (V2)

```python
from modules.CodeReviewAI_V2 import CodeReviewer

reviewer = CodeReviewer()
result = await reviewer.review(code, language="python")
```

### Use Experimental Module (V1)

```python
from modules.Authentication_V1 import AuthManager

auth = AuthManager()
result = await auth.login(email, password)
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
