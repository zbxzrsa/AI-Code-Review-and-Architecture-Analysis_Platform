"""
Test Monitoring and Reporting Dashboard (Testing Improvement Plan #5)

Provides centralized test monitoring with:
- Automated test execution tracking
- Result monitoring and alerting
- Test effectiveness evaluation
- Historical trend analysis

Features:
- Aggregate results from all test types
- Coverage tracking
- Performance trends
- Alert notifications
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import statistics

logger = logging.getLogger(__name__)


class TestCategory(str, Enum):
    """Categories of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    CONTRACT = "contract"
    PERFORMANCE = "performance"
    VISUAL = "visual"
    E2E = "e2e"


class TestStatus(str, Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    category: TestCategory
    status: TestStatus
    duration_ms: float
    timestamp: datetime
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "file_path": self.file_path,
        }


@dataclass
class CoverageReport:
    """Code coverage report."""
    total_lines: int
    covered_lines: int
    total_branches: int
    covered_branches: int
    total_functions: int
    covered_functions: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def line_coverage(self) -> float:
        return (self.covered_lines / self.total_lines * 100) if self.total_lines else 0
    
    @property
    def branch_coverage(self) -> float:
        return (self.covered_branches / self.total_branches * 100) if self.total_branches else 0
    
    @property
    def function_coverage(self) -> float:
        return (self.covered_functions / self.total_functions * 100) if self.total_functions else 0
    
    @property
    def overall_coverage(self) -> float:
        return (self.line_coverage + self.branch_coverage + self.function_coverage) / 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_coverage": round(self.line_coverage, 2),
            "branch_coverage": round(self.branch_coverage, 2),
            "function_coverage": round(self.function_coverage, 2),
            "overall_coverage": round(self.overall_coverage, 2),
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TestRunSummary:
    """Summary of a test run."""
    run_id: str
    category: TestCategory
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_seconds: float
    timestamp: datetime
    coverage: Optional[CoverageReport] = None
    
    @property
    def success_rate(self) -> float:
        executed = self.total_tests - self.skipped
        return (self.passed / executed * 100) if executed else 0
    
    @property
    def failure_rate(self) -> float:
        return 100 - self.success_rate
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "run_id": self.run_id,
            "category": self.category.value,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "success_rate": round(self.success_rate, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "timestamp": self.timestamp.isoformat(),
        }
        if self.coverage:
            result["coverage"] = self.coverage.to_dict()
        return result


@dataclass
class TestAlert:
    """Test alert notification."""
    alert_type: str
    severity: str  # info, warning, critical
    message: str
    category: TestCategory
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class TestMetricsCollector:
    """
    Collects and aggregates test metrics.
    """
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.summaries: List[TestRunSummary] = []
        self.alerts: List[TestAlert] = []
        self.coverage_history: List[CoverageReport] = []
    
    def add_result(self, result: TestResult):
        """Add individual test result."""
        self.results.append(result)
    
    def add_summary(self, summary: TestRunSummary):
        """Add test run summary."""
        self.summaries.append(summary)
        self._check_alerts(summary)
    
    def add_coverage(self, coverage: CoverageReport):
        """Add coverage report."""
        self.coverage_history.append(coverage)
        self._check_coverage_alerts(coverage)
    
    def _check_alerts(self, summary: TestRunSummary):
        """Check for alert conditions."""
        # High failure rate
        if summary.failure_rate > 20:
            self.alerts.append(TestAlert(
                alert_type="high_failure_rate",
                severity="critical",
                message=f"{summary.category.value} tests have {summary.failure_rate:.1f}% failure rate",
                category=summary.category,
                metadata={"failure_rate": summary.failure_rate},
            ))
        elif summary.failure_rate > 10:
            self.alerts.append(TestAlert(
                alert_type="elevated_failure_rate",
                severity="warning",
                message=f"{summary.category.value} tests have {summary.failure_rate:.1f}% failure rate",
                category=summary.category,
                metadata={"failure_rate": summary.failure_rate},
            ))
        
        # Slow tests
        avg_duration = summary.duration_seconds / summary.total_tests if summary.total_tests else 0
        if avg_duration > 5:  # 5 seconds per test
            self.alerts.append(TestAlert(
                alert_type="slow_tests",
                severity="warning",
                message=f"{summary.category.value} tests averaging {avg_duration:.1f}s per test",
                category=summary.category,
                metadata={"avg_duration": avg_duration},
            ))
    
    def _check_coverage_alerts(self, coverage: CoverageReport):
        """Check for coverage alerts."""
        if coverage.overall_coverage < 80:
            severity = "critical" if coverage.overall_coverage < 60 else "warning"
            self.alerts.append(TestAlert(
                alert_type="low_coverage",
                severity=severity,
                message=f"Code coverage at {coverage.overall_coverage:.1f}% (target: 80%)",
                category=TestCategory.UNIT,
                metadata={"coverage": coverage.overall_coverage},
            ))
    
    def get_summary_by_category(self, category: TestCategory) -> List[TestRunSummary]:
        """Get summaries for a category."""
        return [s for s in self.summaries if s.category == category]
    
    def get_recent_alerts(self, hours: int = 24) -> List[TestAlert]:
        """Get recent alerts."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [a for a in self.alerts if a.timestamp > cutoff]
    
    def get_coverage_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get coverage trend over time."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [c for c in self.coverage_history if c.timestamp > cutoff]
        return [c.to_dict() for c in sorted(recent, key=lambda x: x.timestamp)]


class TestDashboard:
    """
    Test monitoring dashboard.
    
    Features:
    - Real-time test results
    - Coverage tracking
    - Performance trends
    - Alert management
    """
    
    def __init__(self, data_dir: str = "tests/reports"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collector = TestMetricsCollector()
        self._load_history()
    
    def _load_history(self):
        """Load historical data."""
        history_file = self.data_dir / "test_history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    # Load summaries
                    for s in data.get("summaries", []):
                        summary = TestRunSummary(
                            run_id=s["run_id"],
                            category=TestCategory(s["category"]),
                            total_tests=s["total_tests"],
                            passed=s["passed"],
                            failed=s["failed"],
                            skipped=s["skipped"],
                            errors=s.get("errors", 0),
                            duration_seconds=s["duration_seconds"],
                            timestamp=datetime.fromisoformat(s["timestamp"]),
                        )
                        self.collector.summaries.append(summary)
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
    
    def _save_history(self):
        """Save historical data."""
        history_file = self.data_dir / "test_history.json"
        data = {
            "summaries": [s.to_dict() for s in self.collector.summaries[-1000:]],  # Keep last 1000
            "coverage": [c.to_dict() for c in self.collector.coverage_history[-100:]],
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def record_test_run(
        self,
        category: TestCategory,
        results: List[TestResult],
        coverage: Optional[CoverageReport] = None
    ) -> TestRunSummary:
        """
        Record a test run.
        
        Args:
            category: Test category
            results: List of test results
            coverage: Optional coverage report
            
        Returns:
            TestRunSummary
        """
        import uuid
        
        # Calculate summary
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        total_duration = sum(r.duration_ms for r in results) / 1000
        
        summary = TestRunSummary(
            run_id=str(uuid.uuid4())[:8],
            category=category,
            total_tests=len(results),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration_seconds=total_duration,
            timestamp=datetime.now(timezone.utc),
            coverage=coverage,
        )
        
        # Record results
        for result in results:
            self.collector.add_result(result)
        
        self.collector.add_summary(summary)
        
        if coverage:
            self.collector.add_coverage(coverage)
        
        self._save_history()
        
        return summary
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data.
        
        Returns:
            Dashboard data dictionary
        """
        now = datetime.now(timezone.utc)
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        # Recent summaries
        recent_summaries = [
            s for s in self.collector.summaries
            if s.timestamp > last_24h
        ]
        
        # Calculate overall stats
        total_tests = sum(s.total_tests for s in recent_summaries)
        total_passed = sum(s.passed for s in recent_summaries)
        total_failed = sum(s.failed for s in recent_summaries)
        
        # Get latest coverage
        latest_coverage = None
        if self.collector.coverage_history:
            latest_coverage = self.collector.coverage_history[-1].to_dict()
        
        # Category breakdown
        category_stats = {}
        for category in TestCategory:
            cat_summaries = [s for s in recent_summaries if s.category == category]
            if cat_summaries:
                category_stats[category.value] = {
                    "runs": len(cat_summaries),
                    "total_tests": sum(s.total_tests for s in cat_summaries),
                    "passed": sum(s.passed for s in cat_summaries),
                    "failed": sum(s.failed for s in cat_summaries),
                    "avg_duration": statistics.mean(s.duration_seconds for s in cat_summaries),
                }
        
        return {
            "summary": {
                "total_tests_24h": total_tests,
                "passed_24h": total_passed,
                "failed_24h": total_failed,
                "success_rate_24h": round(total_passed / total_tests * 100, 2) if total_tests else 0,
                "test_runs_24h": len(recent_summaries),
            },
            "coverage": latest_coverage,
            "categories": category_stats,
            "recent_runs": [s.to_dict() for s in sorted(recent_summaries, key=lambda x: x.timestamp, reverse=True)[:10]],
            "alerts": [a.to_dict() for a in self.collector.get_recent_alerts()],
            "trends": {
                "coverage": self.collector.get_coverage_trend(30),
            },
            "timestamp": now.isoformat(),
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        data = self.get_dashboard_data()
        
        # Add recommendations
        recommendations = []
        
        # Coverage recommendation
        if data["coverage"] and data["coverage"]["overall_coverage"] < 80:
            recommendations.append({
                "type": "coverage",
                "priority": "high",
                "message": f"Increase code coverage from {data['coverage']['overall_coverage']:.1f}% to 80%",
            })
        
        # Failure rate recommendations
        for category, stats in data["categories"].items():
            if stats["failed"] > 0:
                failure_rate = stats["failed"] / stats["total_tests"] * 100
                if failure_rate > 10:
                    recommendations.append({
                        "type": "failures",
                        "priority": "high",
                        "message": f"Investigate {category} test failures ({failure_rate:.1f}% failure rate)",
                    })
        
        data["recommendations"] = recommendations
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        
        return data
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall test health status."""
        data = self.get_dashboard_data()
        
        # Determine health
        issues = []
        
        if data["coverage"] and data["coverage"]["overall_coverage"] < 80:
            issues.append("low_coverage")
        
        if data["summary"]["success_rate_24h"] < 90:
            issues.append("high_failure_rate")
        
        critical_alerts = [a for a in data["alerts"] if a["severity"] == "critical"]
        if critical_alerts:
            issues.append("critical_alerts")
        
        if not issues:
            status = "healthy"
            color = "green"
        elif "critical_alerts" in issues or "high_failure_rate" in issues:
            status = "unhealthy"
            color = "red"
        else:
            status = "degraded"
            color = "yellow"
        
        return {
            "status": status,
            "color": color,
            "issues": issues,
            "summary": data["summary"],
        }


# ============================================================
# Report Generators
# ============================================================

class HTMLReportGenerator:
    """Generates HTML test reports."""
    
    @staticmethod
    def generate(data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Report - {data['timestamp'][:10]}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
        .card.passed {{ border-left: 4px solid #4caf50; }}
        .card.failed {{ border-left: 4px solid #f44336; }}
        .card.warning {{ border-left: 4px solid #ff9800; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        .status-passed {{ color: #4caf50; }}
        .status-failed {{ color: #f44336; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
        .alert-critical {{ background: #ffebee; border-left: 4px solid #f44336; }}
        .alert-warning {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
    </style>
</head>
<body>
    <h1>Test Report</h1>
    <p>Generated: {data['timestamp']}</p>
    
    <h2>Summary</h2>
    <div class="summary">
        <div class="card passed">
            <h3>{data['summary']['passed_24h']}</h3>
            <p>Tests Passed</p>
        </div>
        <div class="card failed">
            <h3>{data['summary']['failed_24h']}</h3>
            <p>Tests Failed</p>
        </div>
        <div class="card">
            <h3>{data['summary']['success_rate_24h']}%</h3>
            <p>Success Rate</p>
        </div>
        <div class="card">
            <h3>{data['coverage']['overall_coverage'] if data['coverage'] else 'N/A'}%</h3>
            <p>Code Coverage</p>
        </div>
    </div>
    
    <h2>Alerts</h2>
    {''.join(f'<div class="alert alert-{a["severity"]}">{a["message"]}</div>' for a in data.get('alerts', []))}
    
    <h2>Category Breakdown</h2>
    <table>
        <tr><th>Category</th><th>Tests</th><th>Passed</th><th>Failed</th><th>Duration</th></tr>
        {''.join(f'<tr><td>{cat}</td><td>{stats["total_tests"]}</td><td class="status-passed">{stats["passed"]}</td><td class="status-failed">{stats["failed"]}</td><td>{stats["avg_duration"]:.1f}s</td></tr>' for cat, stats in data.get('categories', {}).items())}
    </table>
    
    <h2>Recent Runs</h2>
    <table>
        <tr><th>Run ID</th><th>Category</th><th>Tests</th><th>Result</th><th>Time</th></tr>
        {''.join(f'<tr><td>{run["run_id"]}</td><td>{run["category"]}</td><td>{run["total_tests"]}</td><td class="status-{"passed" if run["success_rate"] >= 90 else "failed"}">{run["success_rate"]}%</td><td>{run["timestamp"][:19]}</td></tr>' for run in data.get('recent_runs', []))}
    </table>
</body>
</html>"""


# Global dashboard instance
_dashboard: Optional[TestDashboard] = None


def get_test_dashboard() -> TestDashboard:
    """Get or create global dashboard."""
    global _dashboard
    if _dashboard is None:
        _dashboard = TestDashboard()
    return _dashboard


if __name__ == "__main__":
    # Demo usage
    dashboard = get_test_dashboard()
    
    # Simulate test results
    results = [
        TestResult("test_login", TestCategory.UNIT, TestStatus.PASSED, 50, datetime.now(timezone.utc)),
        TestResult("test_logout", TestCategory.UNIT, TestStatus.PASSED, 30, datetime.now(timezone.utc)),
        TestResult("test_invalid_login", TestCategory.UNIT, TestStatus.FAILED, 45, datetime.now(timezone.utc), "Assertion error"),
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
    print(f"Test run recorded: {summary.run_id}")
    
    report = dashboard.generate_report("tests/reports/latest_report.json")
    print(f"Report generated with {len(report['recommendations'])} recommendations")
