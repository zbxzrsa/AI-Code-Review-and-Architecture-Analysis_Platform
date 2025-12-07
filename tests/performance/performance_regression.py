"""
Performance Regression Testing System (Testing Improvement Plan #3)

Provides performance testing with:
- Key path performance benchmarks
- Automated load testing (JMeter-style)
- Performance threshold alerts
- Load and stress testing

Acceptance Criteria: Core interface response time fluctuates by no more than ±15%
"""
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import aiohttp

import pytest

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """Types of performance tests."""
    SMOKE = "smoke"           # Quick sanity check
    LOAD = "load"             # Normal load testing
    STRESS = "stress"         # Beyond normal capacity
    SPIKE = "spike"           # Sudden traffic spikes
    SOAK = "soak"             # Extended duration
    BASELINE = "baseline"     # Establish benchmarks


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceThreshold:
    """Defines acceptable performance thresholds."""
    response_time_p50_ms: float = 200
    response_time_p95_ms: float = 500
    response_time_p99_ms: float = 1000
    error_rate_percent: float = 1.0
    throughput_min_rps: float = 100
    variance_percent: float = 15.0  # ±15% allowed


@dataclass
class RequestResult:
    """Result of a single request."""
    url: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_size_bytes: int = 0


@dataclass
class EndpointMetrics:
    """Aggregated metrics for an endpoint."""
    endpoint: str
    method: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100

    @property
    def error_rate(self) -> float:
        return 100 - self.success_rate

    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def p50_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        return sorted_times[int(len(sorted_times) * 0.5)]

    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        return sorted_times[int(len(sorted_times) * 0.95)]

    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        return sorted_times[int(len(sorted_times) * 0.99)]

    @property
    def min_response_time(self) -> float:
        return min(self.response_times) if self.response_times else 0.0

    @property
    def max_response_time(self) -> float:
        return max(self.response_times) if self.response_times else 0.0

    @property
    def std_dev(self) -> float:
        if len(self.response_times) < 2:
            return 0.0
        return statistics.stdev(self.response_times)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 2),
            "error_rate": round(self.error_rate, 2),
            "response_time": {
                "avg": round(self.avg_response_time, 2),
                "min": round(self.min_response_time, 2),
                "max": round(self.max_response_time, 2),
                "p50": round(self.p50_response_time, 2),
                "p95": round(self.p95_response_time, 2),
                "p99": round(self.p99_response_time, 2),
                "std_dev": round(self.std_dev, 2),
            },
        }


@dataclass
class PerformanceAlert:
    """Performance threshold violation alert."""
    severity: AlertSeverity
    endpoint: str
    metric: str
    threshold: float
    actual: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "endpoint": self.endpoint,
            "metric": self.metric,
            "threshold": self.threshold,
            "actual": self.actual,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LoadTestConfig:
    """Configuration for load tests."""
    base_url: str
    duration_seconds: int = 60
    concurrent_users: int = 10
    ramp_up_seconds: int = 10
    requests_per_second: Optional[float] = None
    timeout_seconds: float = 30.0
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class TestScenario:
    """Defines a test scenario with multiple endpoints."""
    name: str
    endpoints: List[Dict[str, Any]]  # [{url, method, body, weight}]
    think_time_ms: Tuple[int, int] = (100, 500)  # Random delay range


class PerformanceBaseline:
    """Manages performance baselines for regression detection."""

    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baselines: Dict[str, Dict[str, float]] = {}
        self._load()

    def _load(self):
        """Load baselines from file."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                self.baselines = json.load(f)

    def save(self):
        """Save baselines to file."""
        with open(self.baseline_file, "w") as f:
            json.dump(self.baselines, f, indent=2)

    def set_baseline(self, endpoint: str, metrics: EndpointMetrics):
        """Set baseline for an endpoint."""
        self.baselines[endpoint] = {
            "p50": metrics.p50_response_time,
            "p95": metrics.p95_response_time,
            "p99": metrics.p99_response_time,
            "avg": metrics.avg_response_time,
            "error_rate": metrics.error_rate,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def get_baseline(self, endpoint: str) -> Optional[Dict[str, float]]:
        """Get baseline for an endpoint."""
        return self.baselines.get(endpoint)

    def check_regression(
        self,
        endpoint: str,
        metrics: EndpointMetrics,
        variance_percent: float = 15.0
    ) -> List[PerformanceAlert]:
        """Check for performance regression against baseline."""
        alerts = []
        baseline = self.get_baseline(endpoint)

        if not baseline:
            return alerts

        # Check P95 response time regression
        if metrics.p95_response_time > baseline["p95"] * (1 + variance_percent / 100):
            regression_percent = ((metrics.p95_response_time - baseline["p95"]) / baseline["p95"]) * 100
            alerts.append(PerformanceAlert(
                severity=AlertSeverity.WARNING if regression_percent < 30 else AlertSeverity.CRITICAL,
                endpoint=endpoint,
                metric="p95_response_time",
                threshold=baseline["p95"] * (1 + variance_percent / 100),
                actual=metrics.p95_response_time,
                message=f"P95 response time increased by {regression_percent:.1f}% (baseline: {baseline['p95']:.0f}ms, current: {metrics.p95_response_time:.0f}ms)",
            ))

        # Check error rate regression
        if metrics.error_rate > baseline.get("error_rate", 0) + 5:
            alerts.append(PerformanceAlert(
                severity=AlertSeverity.CRITICAL,
                endpoint=endpoint,
                metric="error_rate",
                threshold=baseline.get("error_rate", 0) + 5,
                actual=metrics.error_rate,
                message=f"Error rate increased significantly (baseline: {baseline.get('error_rate', 0):.1f}%, current: {metrics.error_rate:.1f}%)",
            ))

        return alerts


class LoadTester:
    """
    Load testing engine similar to JMeter/LoadRunner.

    Features:
    - Concurrent user simulation
    - Ramp-up support
    - Think time simulation
    - Multiple endpoint testing
    - Real-time metrics collection
    """

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics: Dict[str, EndpointMetrics] = {}
        self.results: List[RequestResult] = []
        self.alerts: List[PerformanceAlert] = []
        self._running = False
        self._start_time: Optional[float] = None

    async def run_test(
        self,
        scenario: TestScenario,
        test_type: TestType = TestType.LOAD,
        thresholds: Optional[PerformanceThreshold] = None
    ) -> Dict[str, Any]:
        """
        Run a load test scenario.

        Args:
            scenario: Test scenario to execute
            test_type: Type of test to run
            thresholds: Performance thresholds to check

        Returns:
            Test results and metrics
        """
        self._running = True
        self._start_time = time.time()
        thresholds = thresholds or PerformanceThreshold()

        logger.info(f"Starting {test_type.value} test: {scenario.name}")
        logger.info(f"Duration: {self.config.duration_seconds}s, Users: {self.config.concurrent_users}")

        # Initialize metrics for each endpoint
        for endpoint in scenario.endpoints:
            key = f"{endpoint['method']} {endpoint['url']}"
            self.metrics[key] = EndpointMetrics(
                endpoint=endpoint['url'],
                method=endpoint['method'],
            )

        # Create user tasks
        tasks = []
        for user_id in range(self.config.concurrent_users):
            # Ramp-up delay
            delay = (user_id / self.config.concurrent_users) * self.config.ramp_up_seconds
            task = asyncio.create_task(
                self._user_session(user_id, scenario, delay)
            )
            tasks.append(task)

        # Wait for test duration
        await asyncio.sleep(self.config.duration_seconds + self.config.ramp_up_seconds)
        self._running = False

        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Check thresholds
        self._check_thresholds(thresholds)

        return self._generate_report(scenario, test_type)

    async def _user_session(
        self,
        _user_id: int,  # Available for user-specific behavior
        scenario: TestScenario,
        start_delay: float
    ):
        """Simulate a user session."""
        await asyncio.sleep(start_delay)

        async with aiohttp.ClientSession(
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        ) as session:
            while self._running:
                # Select endpoint based on weights
                endpoint = self._select_endpoint(scenario.endpoints)

                # Execute request
                result = await self._execute_request(session, endpoint)
                self.results.append(result)

                # Update metrics
                key = f"{endpoint['method']} {endpoint['url']}"
                metrics = self.metrics[key]
                metrics.total_requests += 1
                metrics.response_times.append(result.response_time_ms)

                if result.success:
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1

                # Think time
                think_time = random.uniform(
                    scenario.think_time_ms[0] / 1000,
                    scenario.think_time_ms[1] / 1000
                )
                await asyncio.sleep(think_time)

    def _select_endpoint(self, endpoints: List[Dict]) -> Dict:
        """Select endpoint based on weights."""
        weights = [e.get("weight", 1) for e in endpoints]
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)

        cumulative = 0
        for endpoint, weight in zip(endpoints, weights):
            cumulative += weight
            if r <= cumulative:
                return endpoint

        return endpoints[-1]

    async def _execute_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: Dict
    ) -> RequestResult:
        """Execute a single request."""
        url = f"{self.config.base_url}{endpoint['url']}"
        method = endpoint.get("method", "GET").upper()
        body = endpoint.get("body")

        start_time = time.perf_counter()

        try:
            async with session.request(
                method,
                url,
                json=body if body else None,
            ) as response:
                response_time = (time.perf_counter() - start_time) * 1000
                content = await response.read()

                return RequestResult(
                    url=endpoint['url'],
                    method=method,
                    status_code=response.status,
                    response_time_ms=response_time,
                    success=200 <= response.status < 400,
                    response_size_bytes=len(content),
                )

        except asyncio.TimeoutError:
            return RequestResult(
                url=endpoint['url'],
                method=method,
                status_code=0,
                response_time_ms=self.config.timeout_seconds * 1000,
                success=False,
                error="Timeout",
            )
        except Exception as e:
            return RequestResult(
                url=endpoint['url'],
                method=method,
                status_code=0,
                response_time_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error=str(e),
            )

    def _check_thresholds(self, thresholds: PerformanceThreshold):
        """Check if metrics exceed thresholds."""
        for key, metrics in self.metrics.items():
            # P95 response time
            if metrics.p95_response_time > thresholds.response_time_p95_ms:
                self.alerts.append(PerformanceAlert(
                    severity=AlertSeverity.WARNING,
                    endpoint=key,
                    metric="p95_response_time",
                    threshold=thresholds.response_time_p95_ms,
                    actual=metrics.p95_response_time,
                    message=f"P95 response time ({metrics.p95_response_time:.0f}ms) exceeds threshold ({thresholds.response_time_p95_ms}ms)",
                ))

            # P99 response time
            if metrics.p99_response_time > thresholds.response_time_p99_ms:
                self.alerts.append(PerformanceAlert(
                    severity=AlertSeverity.CRITICAL,
                    endpoint=key,
                    metric="p99_response_time",
                    threshold=thresholds.response_time_p99_ms,
                    actual=metrics.p99_response_time,
                    message=f"P99 response time ({metrics.p99_response_time:.0f}ms) exceeds threshold ({thresholds.response_time_p99_ms}ms)",
                ))

            # Error rate
            if metrics.error_rate > thresholds.error_rate_percent:
                self.alerts.append(PerformanceAlert(
                    severity=AlertSeverity.CRITICAL,
                    endpoint=key,
                    metric="error_rate",
                    threshold=thresholds.error_rate_percent,
                    actual=metrics.error_rate,
                    message=f"Error rate ({metrics.error_rate:.1f}%) exceeds threshold ({thresholds.error_rate_percent}%)",
                ))

    def _generate_report(
        self,
        scenario: TestScenario,
        test_type: TestType
    ) -> Dict[str, Any]:
        """Generate test report."""
        total_duration = time.time() - self._start_time if self._start_time else 0
        total_requests = sum(m.total_requests for m in self.metrics.values())

        return {
            "summary": {
                "scenario": scenario.name,
                "test_type": test_type.value,
                "duration_seconds": round(total_duration, 2),
                "concurrent_users": self.config.concurrent_users,
                "total_requests": total_requests,
                "throughput_rps": round(total_requests / total_duration, 2) if total_duration > 0 else 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "endpoints": {key: m.to_dict() for key, m in self.metrics.items()},
            "alerts": [a.to_dict() for a in self.alerts],
            "passed": len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]) == 0,
        }


class StressTester(LoadTester):
    """Stress testing with gradually increasing load."""

    async def run_stress_test(
        self,
        scenario: TestScenario,
        initial_users: int = 10,
        max_users: int = 100,
        step_users: int = 10,
        step_duration_seconds: int = 60,
        thresholds: Optional[PerformanceThreshold] = None
    ) -> Dict[str, Any]:
        """Run stress test with increasing load."""
        thresholds = thresholds or PerformanceThreshold()
        all_results = []
        breaking_point = None

        current_users = initial_users

        while current_users <= max_users:
            logger.info(f"Stress test step: {current_users} users")

            self.config.concurrent_users = current_users
            self.config.duration_seconds = step_duration_seconds

            # Reset metrics
            self.metrics = {}
            self.results = []
            self.alerts = []

            result = await self.run_test(scenario, TestType.STRESS, thresholds)
            all_results.append({
                "users": current_users,
                "result": result,
            })

            # Check if system is breaking down
            critical_alerts = [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]
            if critical_alerts and breaking_point is None:
                breaking_point = current_users
                logger.warning(f"Breaking point detected at {current_users} users")

            current_users += step_users

        return {
            "type": "stress_test",
            "scenario": scenario.name,
            "initial_users": initial_users,
            "max_users": max_users,
            "breaking_point": breaking_point,
            "steps": all_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Performance Test Fixtures
# ============================================================

@pytest.fixture
def performance_thresholds() -> PerformanceThreshold:
    """Default performance thresholds."""
    return PerformanceThreshold(
        response_time_p50_ms=200,
        response_time_p95_ms=500,
        response_time_p99_ms=1000,
        error_rate_percent=1.0,
        variance_percent=15.0,
    )


@pytest.fixture
def load_test_config() -> LoadTestConfig:
    """Default load test configuration."""
    return LoadTestConfig(
        base_url="http://localhost:8000",
        duration_seconds=30,
        concurrent_users=10,
        ramp_up_seconds=5,
        headers={"Content-Type": "application/json"},
    )


@pytest.fixture
def test_scenario() -> TestScenario:
    """Sample test scenario."""
    return TestScenario(
        name="Core API Test",
        endpoints=[
            {"url": "/api/v1/health", "method": "GET", "weight": 1},
            {"url": "/api/v1/projects", "method": "GET", "weight": 3},
            {"url": "/api/v1/analyze", "method": "POST", "body": {"code": "print('test')"}, "weight": 2},
        ],
        think_time_ms=(100, 500),
    )


class TestPerformanceRegression:
    """Performance regression tests."""

    def test_baseline_creation(self):
        """Test creating performance baseline."""
        baseline = PerformanceBaseline("test_baseline.json")

        metrics = EndpointMetrics(
            endpoint="/api/v1/test",
            method="GET",
            total_requests=100,
            successful_requests=98,
            failed_requests=2,
            response_times=[100, 150, 200, 250, 300] * 20,
        )

        baseline.set_baseline("/api/v1/test", metrics)
        loaded = baseline.get_baseline("/api/v1/test")

        assert loaded is not None
        assert "p95" in loaded
        assert "avg" in loaded

    def test_regression_detection(self):
        """Test performance regression detection."""
        baseline = PerformanceBaseline()
        baseline.baselines["/api/v1/test"] = {
            "p50": 100,
            "p95": 200,
            "p99": 300,
            "avg": 150,
            "error_rate": 1.0,
        }

        # Create metrics with regression
        metrics = EndpointMetrics(
            endpoint="/api/v1/test",
            method="GET",
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            response_times=[250, 300, 350, 400, 450] * 20,  # Slower
        )

        alerts = baseline.check_regression("/api/v1/test", metrics, variance_percent=15.0)

        assert len(alerts) > 0
        assert any(a.metric == "p95_response_time" for a in alerts)

    def test_threshold_check(self, performance_thresholds):
        """Test threshold checking."""
        metrics = EndpointMetrics(
            endpoint="/api/v1/test",
            method="GET",
            total_requests=100,
            successful_requests=100,
            response_times=[100] * 100,
        )

        # Should pass
        assert metrics.p95_response_time <= performance_thresholds.response_time_p95_ms
        assert metrics.error_rate <= performance_thresholds.error_rate_percent


class TestLoadTester:
    """Load tester tests."""

    def test_endpoint_selection(self, load_test_config, test_scenario):
        """Test weighted endpoint selection."""
        tester = LoadTester(load_test_config)

        selections = {}
        for _ in range(1000):
            endpoint = tester._select_endpoint(test_scenario.endpoints)
            key = endpoint['url']
            selections[key] = selections.get(key, 0) + 1

        # Projects should be selected more often (weight 3)
        assert selections.get("/api/v1/projects", 0) > selections.get("/api/v1/health", 0)

    def test_metrics_calculation(self):
        """Test metrics calculation."""
        metrics = EndpointMetrics(
            endpoint="/test",
            method="GET",
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            response_times=list(range(100, 200)),  # 100-199ms
        )

        assert metrics.success_rate == 95.0
        assert metrics.error_rate == 5.0
        assert metrics.avg_response_time == 149.5
        assert metrics.p50_response_time == 149
        assert metrics.p95_response_time == 194


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
