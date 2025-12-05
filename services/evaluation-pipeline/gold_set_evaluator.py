"""
Gold-Set Evaluator

Evaluates AI model versions against curated gold-set test cases.
This is the gate for both:
1. V1 → V2 promotion (shadow → gray-scale)
2. V3 → V1 recovery (quarantine → experiment)

Gold-sets contain known code samples with expected analysis results,
enabling objective measurement of model accuracy and regression detection.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime, timezone import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class TestCategory(str, Enum):
    """Categories of gold-set tests"""
    SECURITY = "security"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    FALSE_POSITIVE = "false_positive"


class IssueSeverity(str, Enum):
    """Severity levels for detected issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class GoldSetTestCase:
    """A single gold-set test case"""
    id: str
    name: str
    category: TestCategory
    language: str
    code: str
    expected_issues: List[Dict[str, Any]]
    timeout_ms: int = 30000
    weight: float = 1.0  # Importance weight for scoring


@dataclass
class TestResult:
    """Result of a single test case"""
    test_id: str
    passed: bool
    expected_issues: List[Dict[str, Any]]
    detected_issues: List[Dict[str, Any]]
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    version_id: str
    evaluation_type: str
    timestamp: str
    success: bool
    score: float
    
    # Category scores
    security_score: float = 0.0
    quality_score: float = 0.0
    performance_score: float = 0.0
    false_positive_rate: float = 0.0
    
    # Aggregate metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Individual results
    test_results: List[TestResult] = field(default_factory=list)
    
    # Timing
    total_duration_ms: float = 0.0
    
    # Recommendations
    passed: bool = False
    recommendations: List[str] = field(default_factory=list)


class GoldSetEvaluator:
    """
    Evaluates model versions against gold-set test suites.
    
    This is the objective judge for the self-evolution cycle,
    ensuring only quality versions progress through the pipeline.
    """
    
    def __init__(
        self,
        vcai_v1_url: str = "http://vcai-v1.platform-v1-exp.svc:8000",
        vcai_v2_url: str = "http://vcai-v2.platform-v2-stable.svc:8000",
        vcai_v3_url: str = "http://vcai-v3.platform-v3-legacy.svc:8000",
    ):
        self.vcai_urls = {
            "v1": vcai_v1_url,
            "v2": vcai_v2_url,
            "v3": vcai_v3_url,
        }
        
        self._http_client: Optional[httpx.AsyncClient] = None
        self._gold_sets: Dict[TestCategory, List[GoldSetTestCase]] = {}
        
        # Load gold-sets
        self._load_gold_sets()
    
    def _load_gold_sets(self):
        """Load gold-set test cases"""
        # Security tests
        self._gold_sets[TestCategory.SECURITY] = [
            GoldSetTestCase(
                id="sec-001",
                name="SQL Injection Detection",
                category=TestCategory.SECURITY,
                language="python",
                code='''
def get_user(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
''',
                expected_issues=[
                    {"type": "security", "severity": "critical", "pattern": "sql_injection"}
                ],
                weight=2.0  # Critical issues weighted higher
            ),
            GoldSetTestCase(
                id="sec-002",
                name="XSS Detection",
                category=TestCategory.SECURITY,
                language="javascript",
                code='''
function displayInput(userInput) {
    document.getElementById('output').innerHTML = userInput;
}
''',
                expected_issues=[
                    {"type": "security", "severity": "high", "pattern": "xss"}
                ],
                weight=1.5
            ),
            GoldSetTestCase(
                id="sec-003",
                name="Command Injection",
                category=TestCategory.SECURITY,
                language="python",
                code='''
import os
def run_command(user_input):
    os.system(f"echo {user_input}")
''',
                expected_issues=[
                    {"type": "security", "severity": "critical", "pattern": "command_injection"}
                ],
                weight=2.0
            ),
            GoldSetTestCase(
                id="sec-004",
                name="Hardcoded Secrets",
                category=TestCategory.SECURITY,
                language="python",
                code='''
API_KEY = "sk-1234567890abcdef"
DATABASE_PASSWORD = "super_secret_123"
''',
                expected_issues=[
                    {"type": "security", "severity": "high", "pattern": "hardcoded_secret"}
                ],
                weight=1.5
            ),
        ]
        
        # Quality tests
        self._gold_sets[TestCategory.QUALITY] = [
            GoldSetTestCase(
                id="qual-001",
                name="Unused Variables",
                category=TestCategory.QUALITY,
                language="python",
                code='''
def process_data(x, y, z):
    result = x + y
    unused = z * 2
    return result
''',
                expected_issues=[
                    {"type": "quality", "severity": "low", "pattern": "unused_variable"}
                ],
                weight=0.5
            ),
            GoldSetTestCase(
                id="qual-002",
                name="Empty Exception Handler",
                category=TestCategory.QUALITY,
                language="python",
                code='''
try:
    risky_operation()
except:
    pass
''',
                expected_issues=[
                    {"type": "quality", "severity": "medium", "pattern": "empty_except"}
                ],
                weight=1.0
            ),
        ]
        
        # False positive tests (should NOT detect issues)
        self._gold_sets[TestCategory.FALSE_POSITIVE] = [
            GoldSetTestCase(
                id="fp-001",
                name="Safe Parameterized Query",
                category=TestCategory.FALSE_POSITIVE,
                language="python",
                code='''
def get_user(user_id: int):
    cursor.execute(
        "SELECT * FROM users WHERE id = %s",
        (user_id,)
    )
    return cursor.fetchone()
''',
                expected_issues=[],  # Should detect NO issues
                weight=1.0
            ),
            GoldSetTestCase(
                id="fp-002",
                name="Safe DOM Manipulation",
                category=TestCategory.FALSE_POSITIVE,
                language="javascript",
                code='''
function displayText(text) {
    document.getElementById('output').textContent = text;
}
''',
                expected_issues=[],
                weight=1.0
            ),
            GoldSetTestCase(
                id="fp-003",
                name="Environment Variable Usage",
                category=TestCategory.FALSE_POSITIVE,
                language="python",
                code='''
import os
API_KEY = os.environ.get('API_KEY')
DATABASE_URL = os.environ['DATABASE_URL']
''',
                expected_issues=[],
                weight=1.0
            ),
        ]
    
    async def start(self):
        """Start the evaluator"""
        self._http_client = httpx.AsyncClient(timeout=60.0)
        logger.info("Gold-Set Evaluator started")
    
    async def stop(self):
        """Stop the evaluator"""
        if self._http_client:
            await self._http_client.aclose()
    
    # ==================== Main Evaluation ====================
    
    async def evaluate(
        self,
        version_id: str,
        evaluation_type: str = "promotion",
        include_categories: Optional[List[str]] = None
    ) -> EvaluationReport:
        """
        Run full gold-set evaluation for a version.
        
        Args:
            version_id: The version to evaluate (e.g., "v1-exp-001")
            evaluation_type: "promotion" or "recovery"
            include_categories: Categories to include (default: all)
        
        Returns:
            Complete evaluation report
        """
        start_time = datetime.now(timezone.utc)
        
        # Determine target URL based on version
        version_prefix = version_id.split("-")[0] if "-" in version_id else "v1"
        vcai_url = self.vcai_urls.get(version_prefix, self.vcai_urls["v1"])
        
        # Determine categories to test
        categories = include_categories or [c.value for c in TestCategory]
        
        # Run all tests
        all_results: List[TestResult] = []
        
        for category_name in categories:
            try:
                category = TestCategory(category_name)
                test_cases = self._gold_sets.get(category, [])
                
                for test_case in test_cases:
                    result = await self._run_test(vcai_url, version_id, test_case)
                    all_results.append(result)
                    
            except ValueError:
                logger.warning(f"Unknown category: {category_name}")
        
        # Calculate scores
        report = self._calculate_report(version_id, evaluation_type, all_results)
        report.total_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # Determine pass/fail
        report.passed = self._check_passed(report, evaluation_type)
        report.recommendations = self._generate_recommendations(report)
        
        logger.info(
            f"Evaluation complete for {version_id}: "
            f"score={report.score:.2%}, passed={report.passed}"
        )
        
        return report
    
    async def _run_test(
        self,
        vcai_url: str,
        version_id: str,
        test_case: GoldSetTestCase
    ) -> TestResult:
        """Run a single test case"""
        try:
            start = datetime.now(timezone.utc)
            
            # Call VCAI for analysis
            response = await self._http_client.post(
                f"{vcai_url}/analyze",
                json={
                    "code": test_case.code,
                    "language": test_case.language,
                    "version_id": version_id
                },
                timeout=test_case.timeout_ms / 1000
            )
            
            latency = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            
            if response.status_code != 200:
                return TestResult(
                    test_id=test_case.id,
                    passed=False,
                    expected_issues=test_case.expected_issues,
                    detected_issues=[],
                    latency_ms=latency,
                    error=f"HTTP {response.status_code}"
                )
            
            data = response.json()
            detected_issues = data.get("issues", [])
            
            # Compare expected vs detected
            result = self._compare_issues(test_case.expected_issues, detected_issues)
            result.test_id = test_case.id
            result.expected_issues = test_case.expected_issues
            result.detected_issues = detected_issues
            result.latency_ms = latency
            
            return result
            
        except Exception as e:
            logger.error(f"Test {test_case.id} failed: {e}")
            return TestResult(
                test_id=test_case.id,
                passed=False,
                expected_issues=test_case.expected_issues,
                detected_issues=[],
                error=str(e)
            )
    
    def _compare_issues(
        self,
        expected: List[Dict[str, Any]],
        detected: List[Dict[str, Any]]
    ) -> TestResult:
        """Compare expected vs detected issues"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Match expected issues
        detected_patterns = {
            d.get("pattern", d.get("type", "unknown"))
            for d in detected
        }
        expected_patterns = {
            e.get("pattern", e.get("type", "unknown"))
            for e in expected
        }
        
        true_positives = len(detected_patterns & expected_patterns)
        false_positives = len(detected_patterns - expected_patterns)
        false_negatives = len(expected_patterns - detected_patterns)
        
        # For false positive tests, any detection is a failure
        if not expected:
            passed = len(detected) == 0
        else:
            # Pass if all expected issues detected with acceptable false positives
            passed = (
                false_negatives == 0 and
                false_positives <= len(expected)  # Allow some false positives
            )
        
        return TestResult(
            test_id="",
            passed=passed,
            expected_issues=expected,
            detected_issues=detected,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
    
    def _calculate_report(
        self,
        version_id: str,
        evaluation_type: str,
        results: List[TestResult]
    ) -> EvaluationReport:
        """Calculate evaluation report from test results"""
        # Separate by category
        security_results = [r for r in results if r.test_id.startswith("sec-")]
        quality_results = [r for r in results if r.test_id.startswith("qual-")]
        fp_results = [r for r in results if r.test_id.startswith("fp-")]
        
        # Calculate category scores
        security_score = self._calculate_category_score(security_results)
        quality_score = self._calculate_category_score(quality_results)
        
        # False positive rate (lower is better)
        fp_failed = sum(1 for r in fp_results if not r.passed)
        false_positive_rate = fp_failed / len(fp_results) if fp_results else 0
        
        # Overall score (weighted)
        overall_score = (
            security_score * 0.4 +
            quality_score * 0.3 +
            (1 - false_positive_rate) * 0.3
        )
        
        # Aggregate metrics
        total_tp = sum(r.true_positives for r in results)
        total_fp = sum(r.false_positives for r in results)
        total_fn = sum(r.false_negatives for r in results)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
        
        return EvaluationReport(
            version_id=version_id,
            evaluation_type=evaluation_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=True,
            score=overall_score,
            security_score=security_score,
            quality_score=quality_score,
            false_positive_rate=false_positive_rate,
            metrics={
                "accuracy": overall_score,
                "security_pass_rate": security_score,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "false_positive_rate": false_positive_rate,
                "avg_latency_ms": avg_latency,
                "total_tests": len(results),
                "tests_passed": sum(1 for r in results if r.passed),
            },
            test_results=results
        )
    
    def _calculate_category_score(self, results: List[TestResult]) -> float:
        """Calculate score for a category"""
        if not results:
            return 0.0
        return sum(1 for r in results if r.passed) / len(results)
    
    def _check_passed(self, report: EvaluationReport, evaluation_type: str) -> bool:
        """Determine if evaluation passed"""
        if evaluation_type == "promotion":
            # V1 → V2: Standard thresholds
            return (
                report.security_score >= 0.90 and
                report.quality_score >= 0.80 and
                report.false_positive_rate <= 0.05
            )
        elif evaluation_type == "recovery":
            # V3 → V1: Stricter thresholds for recovery
            return (
                report.security_score >= 0.95 and
                report.quality_score >= 0.85 and
                report.false_positive_rate <= 0.02
            )
        return False
    
    def _generate_recommendations(self, report: EvaluationReport) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        if report.security_score < 0.90:
            recommendations.append(
                f"Security detection needs improvement ({report.security_score:.0%}). "
                "Focus on SQL injection and XSS detection."
            )
        
        if report.quality_score < 0.80:
            recommendations.append(
                f"Quality detection below threshold ({report.quality_score:.0%}). "
                "Review code quality rule coverage."
            )
        
        if report.false_positive_rate > 0.05:
            recommendations.append(
                f"False positive rate too high ({report.false_positive_rate:.0%}). "
                "Tune detection rules to reduce noise."
            )
        
        if report.passed:
            recommendations.append("All thresholds met - eligible for promotion!")
        
        return recommendations
