# Automated Testing Improvement Guide

## Overview

This document outlines the comprehensive automated testing infrastructure implemented for the AI Code Review Platform.

| Test Type              | Priority | Acceptance Criteria               | Status      |
| ---------------------- | -------- | --------------------------------- | ----------- |
| Unit Tests             | High     | 80% coverage                      | ✅ Complete |
| API Contract Tests     | Medium   | All API changes pass verification | ✅ Complete |
| Performance Regression | Medium   | Response time ±15% variance       | ✅ Complete |
| Visual Regression      | Low      | All UI changes pass verification  | ✅ Complete |

---

## 1. Unit Test Enhancement (High Priority)

### Objective

Increase unit test coverage to 80% for all core modules.

### Implementation

**Test Configuration (`pytest.ini`):**

```ini
[pytest]
testpaths = tests
asyncio_mode = auto

[coverage:report]
fail_under = 80  # Enforced 80% coverage
```

**Key Test Files:**

- `tests/unit/conftest.py` - Shared fixtures and mock data
- `tests/unit/test_*.py` - Unit test modules

### Coverage Requirements

| Category          | Minimum | Target |
| ----------------- | ------- | ------ |
| Backend Services  | 80%     | 90%    |
| Business Logic    | 85%     | 95%    |
| API Endpoints     | 80%     | 85%    |
| Utility Functions | 80%     | 90%    |

### Running Unit Tests

```bash
# Backend tests with coverage
pytest tests/unit/ --cov=backend --cov-fail-under=80 -v

# Frontend tests with coverage
cd frontend && npm test -- --coverage --coverageThreshold='{"global":{"lines":80}}'
```

### CI/CD Integration

Coverage is enforced in CI pipeline:

- Tests fail if coverage < 80%
- Coverage reports uploaded to Codecov
- PR comments show coverage changes

---

## 2. API Contract Testing (Medium Priority)

### Objective

Ensure API interface compatibility and consistency.

### Implementation

**Contract Testing Framework (`tests/contract/api_contract_tests.py`):**

```python
from tests.contract.api_contract_tests import (
    APIContract,
    ContractValidator,
    ConsumerContract,
)

# Define API contract
contract = APIContract(
    title="Code Review API",
    version="1.0.0",
    base_url="http://localhost:8000",
)

# Add endpoint contracts
contract.endpoints.append(EndpointContract(
    path="/api/v1/analyze",
    method=HttpMethod.POST,
    request_body={
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "language": {"type": "string"},
        },
        "required": ["code"],
    },
    responses={200: {"description": "Analysis result"}},
))

# Validate response
validator = ContractValidator(contract)
violations = validator.validate_response(
    path="/api/v1/analyze",
    method=HttpMethod.POST,
    status_code=200,
    response_body={"issues": [], "metrics": {}},
)
```

### Pact-Style Consumer Contracts

```python
contract = ConsumerContract("frontend", "api")

contract.given("a project exists").upon_receiving(
    "a request to analyze code"
).with_request(
    method="POST",
    path="/api/v1/analyze",
    body={"code": "print('test')"},
).will_respond_with(
    status=200,
    body={"issues": [], "metrics": {}},
)

# Save Pact contract
contract.save("pacts/frontend-api.json")
```

### Running Contract Tests

```bash
# Run contract tests
pytest tests/contract/ -v

# Validate OpenAPI spec
openapi-spec-validator docs/openapi.yaml
```

---

## 3. Performance Regression Testing (Medium Priority)

### Objective

Prevent performance degradation in core interfaces.

### Implementation

**Performance Testing Framework (`tests/performance/performance_regression.py`):**

```python
from tests.performance.performance_regression import (
    LoadTester,
    LoadTestConfig,
    TestScenario,
    PerformanceThreshold,
    PerformanceBaseline,
)

# Configure load test
config = LoadTestConfig(
    base_url="http://localhost:8000",
    duration_seconds=60,
    concurrent_users=50,
    ramp_up_seconds=10,
)

# Define scenario
scenario = TestScenario(
    name="Core API Load Test",
    endpoints=[
        {"url": "/api/v1/health", "method": "GET", "weight": 1},
        {"url": "/api/v1/projects", "method": "GET", "weight": 3},
        {"url": "/api/v1/analyze", "method": "POST", "body": {"code": "test"}, "weight": 2},
    ],
    think_time_ms=(100, 500),
)

# Run test
tester = LoadTester(config)
report = await tester.run_test(scenario)

# Check thresholds
thresholds = PerformanceThreshold(
    response_time_p95_ms=500,
    error_rate_percent=1.0,
    variance_percent=15.0,
)
```

### Performance Thresholds

| Metric            | Threshold | Action   |
| ----------------- | --------- | -------- |
| P95 Response Time | 500ms     | Warning  |
| P99 Response Time | 1000ms    | Critical |
| Error Rate        | 1%        | Critical |
| Throughput        | 100 RPS   | Warning  |

### Baseline Management

```python
baseline = PerformanceBaseline("performance_baseline.json")

# Set baseline from test results
baseline.set_baseline("/api/v1/analyze", metrics)

# Check for regression (±15% allowed)
alerts = baseline.check_regression(
    "/api/v1/analyze",
    current_metrics,
    variance_percent=15.0
)
```

### Running Performance Tests

```bash
# Run performance regression tests
pytest tests/performance/ -v

# Run load test
python -m tests.performance.performance_regression --scenario core_api --duration 300

# Run stress test
python -m tests.performance.performance_regression --type stress --max-users 500
```

---

## 4. Visual Regression Testing (Low Priority)

### Objective

Detect unexpected UI changes.

### Implementation

**Visual Testing Framework (`tests/visual/visual_regression.py`):**

```python
from tests.visual.visual_regression import (
    VisualRegressionTester,
    VisualTestConfig,
    PlaywrightVisualTest,
)

# Configure visual tests
config = VisualTestConfig(
    baseline_dir="tests/visual/baselines",
    diff_dir="tests/visual/diffs",
    threshold_percent=0.1,  # 0.1% difference allowed
    viewport_width=1920,
    viewport_height=1080,
)

# Run visual test with Playwright
async def test_dashboard(page):
    visual = PlaywrightVisualTest(page, config)

    # Check full page
    result = await visual.check("dashboard", "http://localhost:3000/dashboard")
    assert result.passed

    # Check specific component
    result = await visual.check_component("sidebar", ".sidebar")
    assert result.passed

    # Assert no visual changes
    visual.assert_no_visual_changes()
```

### Key Pages to Test

| Page        | URL          | Threshold |
| ----------- | ------------ | --------- |
| Login       | /login       | 0.1%      |
| Dashboard   | /dashboard   | 0.5%      |
| Code Review | /code-review | 0.5%      |
| Settings    | /settings    | 0.1%      |
| Admin       | /admin/\*    | 0.5%      |

### Running Visual Tests

```bash
# Run visual regression tests
cd frontend && npx playwright test tests/visual/

# Update baselines
npx playwright test tests/visual/ --update-snapshots
```

---

## 5. Test Monitoring Dashboard

### Implementation

**Dashboard (`tests/reporting/test_dashboard.py`):**

```python
from tests.reporting.test_dashboard import (
    TestDashboard,
    TestResult,
    TestCategory,
    CoverageReport,
)

dashboard = get_test_dashboard()

# Record test run
results = [
    TestResult("test_login", TestCategory.UNIT, TestStatus.PASSED, 50, datetime.now()),
    TestResult("test_api", TestCategory.UNIT, TestStatus.PASSED, 30, datetime.now()),
]

coverage = CoverageReport(
    total_lines=10000,
    covered_lines=8200,
    total_branches=2000,
    covered_branches=1600,
    total_functions=500,
    covered_functions=420,
)

summary = dashboard.record_test_run(TestCategory.UNIT, results, coverage)

# Get dashboard data
data = dashboard.get_dashboard_data()
print(f"Success rate: {data['summary']['success_rate_24h']}%")
print(f"Coverage: {data['coverage']['overall_coverage']}%")

# Generate HTML report
report = dashboard.generate_report("reports/test_report.json")
```

### Dashboard Features

- **Real-time Results**: Track test execution in real-time
- **Coverage Tracking**: Monitor code coverage trends
- **Performance Trends**: Track response time changes
- **Alert Management**: Automated alerts for failures

### Alerts

| Alert Type        | Condition     | Severity         |
| ----------------- | ------------- | ---------------- |
| High Failure Rate | >20% failures | Critical         |
| Elevated Failures | >10% failures | Warning          |
| Low Coverage      | <80% coverage | Warning/Critical |
| Slow Tests        | >5s average   | Warning          |

---

## CI/CD Pipeline

### Workflow (`.github/workflows/testing-pipeline.yml`)

```yaml
jobs:
  unit-tests: # 80% coverage required
  contract-tests: # API validation
  performance-tests: # Regression checks
  visual-tests: # UI verification
  e2e-tests: # Full integration
  test-report: # Consolidated report
```

### Pipeline Flow

```
Code Push
    ↓
Unit Tests (coverage ≥80%)
    ↓
Contract Tests (API validation)
    ↓
Performance Tests (±15% variance)
    ↓
Visual Tests (screenshot comparison)
    ↓
E2E Tests (full integration)
    ↓
Generate Report
```

---

## Files Created

| File                                          | Lines | Purpose                    |
| --------------------------------------------- | ----- | -------------------------- |
| `tests/unit/conftest.py`                      | ~350  | Test fixtures and mocks    |
| `tests/contract/api_contract_tests.py`        | ~700  | Contract testing framework |
| `tests/performance/performance_regression.py` | ~650  | Performance testing        |
| `tests/visual/visual_regression.py`           | ~600  | Visual regression testing  |
| `tests/reporting/test_dashboard.py`           | ~500  | Test monitoring dashboard  |
| `.github/workflows/testing-pipeline.yml`      | ~350  | CI/CD workflow             |
| `docs/automated-testing-guide.md`             | ~400  | This documentation         |

**Total: ~3,550 lines of testing infrastructure**

---

## Quick Start

### Running All Tests

```bash
# Backend tests
pytest tests/ -v --cov=backend --cov-fail-under=80

# Frontend tests
cd frontend && npm test -- --coverage

# E2E tests
cd frontend && npx playwright test

# Performance tests
pytest tests/performance/ -v
```

### Generating Reports

```bash
# Generate test report
python -c "from tests.reporting.test_dashboard import get_test_dashboard; d=get_test_dashboard(); print(d.generate_report())"

# Generate HTML coverage report
pytest --cov=backend --cov-report=html
```

---

## Recommendations

1. **Maintain 80% Coverage**: All new code must have tests
2. **Review Contract Changes**: Update contracts when APIs change
3. **Monitor Performance**: Check baseline after deployments
4. **Update Visual Baselines**: Review and approve UI changes
5. **Fix Flaky Tests**: Investigate and fix unstable tests promptly
