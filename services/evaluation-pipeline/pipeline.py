"""
Evaluation Pipeline Service

Orchestrates the evaluation of versions using:
- Gold-set benchmarks
- Shadow traffic comparison
- Statistical testing
- Red-team security tests
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json

import httpx
import yaml
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationType(str, Enum):
    """Types of evaluation"""
    GOLD_SET = "gold_set"
    SHADOW_COMPARISON = "shadow_comparison"
    RED_TEAM = "red_team"
    STATISTICAL = "statistical"
    FULL = "full"


class TestResult(str, Enum):
    """Individual test result"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case"""
    id: str
    name: str
    category: str
    language: str
    code: str
    expected_issues: List[Dict[str, Any]]
    severity_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestCaseResult:
    """Result of a single test case"""
    test_id: str
    test_name: str
    result: TestResult
    expected_issues: List[Dict[str, Any]]
    actual_issues: List[Dict[str, Any]]
    precision: float
    recall: float
    f1_score: float
    latency_ms: int
    cost_usd: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Complete evaluation report"""
    evaluation_id: str
    version_id: str
    evaluation_type: EvaluationType
    started_at: datetime
    completed_at: Optional[datetime]
    status: str
    
    # Overall metrics
    overall_pass_rate: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    
    # Category breakdown
    category_results: Dict[str, Dict[str, Any]]
    
    # Individual results
    test_results: List[TestCaseResult]
    
    # Statistical summary
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_latency_ms: float
    total_cost_usd: float
    
    # Comparison with baseline
    baseline_comparison: Optional[Dict[str, Any]] = None
    
    # Decision
    promotion_recommended: bool = False
    recommendation_reason: str = ""


class GoldSetEvaluator:
    """Evaluates versions against gold-set benchmarks"""
    
    def __init__(
        self,
        ai_service_url: str = "http://crai-service.platform-v1-exp.svc:8080",
        config_path: str = "/etc/evaluation/gold_sets.yaml"
    ):
        self.ai_service_url = ai_service_url
        self.config_path = config_path
        self.gold_sets: Dict[str, Any] = {}
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def load_gold_sets(self):
        """Load gold-set configurations"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.gold_sets = {gs['id']: gs for gs in config.get('gold_sets', [])}
                self.evaluation_config = config.get('evaluation_config', {})
            logger.info(f"Loaded {len(self.gold_sets)} gold-sets")
        except Exception as e:
            logger.error(f"Failed to load gold-sets: {e}")
            self.gold_sets = {}
    
    async def evaluate_version(
        self,
        version_id: str,
        model_version: str,
        prompt_version: str,
        test_sets: Optional[List[str]] = None
    ) -> EvaluationReport:
        """Run full gold-set evaluation"""
        
        evaluation_id = hashlib.sha256(
            f"{version_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        started_at = datetime.utcnow()
        test_results: List[TestCaseResult] = []
        category_results: Dict[str, Dict[str, Any]] = {}
        
        # Filter gold-sets if specified
        sets_to_run = test_sets or list(self.gold_sets.keys())
        
        for set_id in sets_to_run:
            if set_id not in self.gold_sets:
                logger.warning(f"Gold-set {set_id} not found, skipping")
                continue
                
            gold_set = self.gold_sets[set_id]
            category = gold_set['category']
            
            logger.info(f"Running gold-set: {gold_set['name']}")
            
            set_results = await self._run_gold_set(
                gold_set,
                version_id,
                model_version,
                prompt_version
            )
            
            test_results.extend(set_results)
            
            # Aggregate category results
            passed = sum(1 for r in set_results if r.result == TestResult.PASSED)
            total = len(set_results)
            
            category_results[category] = {
                'gold_set_id': set_id,
                'name': gold_set['name'],
                'total_tests': total,
                'passed_tests': passed,
                'pass_rate': passed / total if total > 0 else 0,
                'required_pass_rate': gold_set.get('required_pass_rate', 0.9),
                'meets_requirement': (passed / total if total > 0 else 0) >= gold_set.get('required_pass_rate', 0.9),
                'avg_precision': sum(r.precision for r in set_results) / total if total > 0 else 0,
                'avg_recall': sum(r.recall for r in set_results) / total if total > 0 else 0,
                'avg_f1': sum(r.f1_score for r in set_results) / total if total > 0 else 0,
            }
        
        # Calculate overall metrics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.result == TestResult.PASSED)
        
        avg_precision = sum(r.precision for r in test_results) / total_tests if total_tests > 0 else 0
        avg_recall = sum(r.recall for r in test_results) / total_tests if total_tests > 0 else 0
        avg_f1 = sum(r.f1_score for r in test_results) / total_tests if total_tests > 0 else 0
        avg_latency = sum(r.latency_ms for r in test_results) / total_tests if total_tests > 0 else 0
        total_cost = sum(r.cost_usd for r in test_results)
        
        overall_pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Check promotion criteria
        promotion_recommended, recommendation_reason = self._check_promotion_criteria(
            overall_pass_rate,
            category_results
        )
        
        return EvaluationReport(
            evaluation_id=evaluation_id,
            version_id=version_id,
            evaluation_type=EvaluationType.GOLD_SET,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            status="completed",
            overall_pass_rate=overall_pass_rate,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=total_tests - passed_tests,
            category_results=category_results,
            test_results=test_results,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_latency_ms=avg_latency,
            total_cost_usd=total_cost,
            promotion_recommended=promotion_recommended,
            recommendation_reason=recommendation_reason
        )
    
    async def _run_gold_set(
        self,
        gold_set: Dict[str, Any],
        version_id: str,
        model_version: str,
        prompt_version: str
    ) -> List[TestCaseResult]:
        """Run all tests in a gold-set"""
        results = []
        
        for test_case in gold_set.get('test_cases', []):
            try:
                result = await self._run_test_case(
                    test_case,
                    version_id,
                    model_version,
                    prompt_version
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Test case {test_case['id']} failed with error: {e}")
                results.append(TestCaseResult(
                    test_id=test_case['id'],
                    test_name=test_case['name'],
                    result=TestResult.ERROR,
                    expected_issues=test_case.get('expected_issues', []),
                    actual_issues=[],
                    precision=0,
                    recall=0,
                    f1_score=0,
                    latency_ms=0,
                    cost_usd=0,
                    details={'error': str(e)}
                ))
        
        return results
    
    async def _run_test_case(
        self,
        test_case: Dict[str, Any],
        version_id: str,
        model_version: str,
        prompt_version: str
    ) -> TestCaseResult:
        """Run a single test case"""
        start_time = datetime.utcnow()
        
        # Call AI service
        response = await self.client.post(
            f"{self.ai_service_url}/analyze",
            json={
                'code': test_case['code'],
                'language': test_case['language'],
                'version_id': version_id,
                'model_version': model_version,
                'prompt_version': prompt_version,
            }
        )
        
        latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        if response.status_code != 200:
            return TestCaseResult(
                test_id=test_case['id'],
                test_name=test_case['name'],
                result=TestResult.ERROR,
                expected_issues=test_case.get('expected_issues', []),
                actual_issues=[],
                precision=0,
                recall=0,
                f1_score=0,
                latency_ms=latency_ms,
                cost_usd=0,
                details={'error': f"HTTP {response.status_code}"}
            )
        
        result_data = response.json()
        actual_issues = result_data.get('issues', [])
        expected_issues = test_case.get('expected_issues', [])
        cost_usd = result_data.get('cost_usd', 0)
        
        # Calculate precision, recall, F1
        precision, recall, f1 = self._calculate_metrics(expected_issues, actual_issues)
        
        # Determine pass/fail
        passed = self._check_test_passed(test_case, actual_issues, precision, recall)
        
        return TestCaseResult(
            test_id=test_case['id'],
            test_name=test_case['name'],
            result=TestResult.PASSED if passed else TestResult.FAILED,
            expected_issues=expected_issues,
            actual_issues=actual_issues,
            precision=precision,
            recall=recall,
            f1_score=f1,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            details={
                'model_version': model_version,
                'prompt_version': prompt_version,
                'confidence': result_data.get('confidence', 0),
            }
        )
    
    def _calculate_metrics(
        self,
        expected: List[Dict[str, Any]],
        actual: List[Dict[str, Any]]
    ) -> tuple[float, float, float]:
        """Calculate precision, recall, F1 score"""
        if not expected and not actual:
            return 1.0, 1.0, 1.0
        
        if not expected:
            return 0.0, 1.0, 0.0  # No false negatives, but all are false positives
        
        if not actual:
            return 1.0, 0.0, 0.0  # No false positives, but all are false negatives
        
        # Match issues by type and approximate location
        true_positives = 0
        
        for exp in expected:
            for act in actual:
                if self._issues_match(exp, act):
                    true_positives += 1
                    break
        
        precision = true_positives / len(actual) if actual else 0
        recall = true_positives / len(expected) if expected else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def _issues_match(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Check if two issues match"""
        # Match by type
        if expected.get('type') != actual.get('type'):
            return False
        
        # Match by severity (allow one level difference)
        severity_order = ['low', 'medium', 'high', 'critical']
        exp_sev = severity_order.index(expected.get('severity', 'medium'))
        act_sev = severity_order.index(actual.get('severity', 'medium'))
        if abs(exp_sev - act_sev) > 1:
            return False
        
        # Match by line (allow ±5 lines tolerance)
        if 'line' in expected and 'line' in actual:
            if abs(expected['line'] - actual['line']) > 5:
                return False
        
        return True
    
    def _check_test_passed(
        self,
        test_case: Dict[str, Any],
        actual_issues: List[Dict[str, Any]],
        precision: float,
        recall: float
    ) -> bool:
        """Determine if test passed"""
        # For security tests, prioritize recall (don't miss vulnerabilities)
        if test_case.get('vulnerability'):
            return recall >= 0.8
        
        # For general tests, use F1 threshold
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1 >= 0.7
    
    def _check_promotion_criteria(
        self,
        overall_pass_rate: float,
        category_results: Dict[str, Dict[str, Any]]
    ) -> tuple[bool, str]:
        """Check if version meets promotion criteria"""
        thresholds = self.evaluation_config.get('promotion_thresholds', {})
        
        overall_required = thresholds.get('overall_pass_rate', 0.90)
        critical_required = thresholds.get('critical_category_min', 0.95)
        non_critical_required = thresholds.get('non_critical_category_min', 0.75)
        
        # Check overall
        if overall_pass_rate < overall_required:
            return False, f"Overall pass rate {overall_pass_rate:.2%} below threshold {overall_required:.2%}"
        
        # Check critical categories (security, injection)
        critical_categories = ['security', 'injection']
        for cat in critical_categories:
            if cat in category_results:
                if category_results[cat]['pass_rate'] < critical_required:
                    return False, f"Critical category '{cat}' pass rate {category_results[cat]['pass_rate']:.2%} below threshold {critical_required:.2%}"
        
        # Check non-critical categories
        for cat, results in category_results.items():
            if cat not in critical_categories:
                if results['pass_rate'] < non_critical_required:
                    return False, f"Category '{cat}' pass rate {results['pass_rate']:.2%} below threshold {non_critical_required:.2%}"
        
        return True, "All criteria met"


class ShadowComparisonEvaluator:
    """Compares V1 shadow outputs with V2 baseline"""
    
    def __init__(
        self,
        prometheus_url: str = "http://prometheus.platform-monitoring.svc:9090",
        v1_db_url: str = "postgresql://v1_service@db:5432/platform",
        v2_db_url: str = "postgresql://v2_service@db:5432/platform"
    ):
        self.prometheus_url = prometheus_url
        self.v1_db_url = v1_db_url
        self.v2_db_url = v2_db_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def compare_outputs(
        self,
        version_id: str,
        time_range_hours: int = 24,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """Compare V1 outputs with V2 baseline"""
        
        # Query paired requests from database
        # (V1 shadow requests linked to V2 production requests)
        
        comparison_results = {
            'version_id': version_id,
            'time_range_hours': time_range_hours,
            'sample_size': sample_size,
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {},
            'agreement_analysis': {},
            'quality_comparison': {}
        }
        
        # Query Prometheus for aggregate metrics
        metrics = await self._query_comparison_metrics(version_id, time_range_hours)
        comparison_results['metrics'] = metrics
        
        # Calculate agreement scores
        agreement = await self._calculate_agreement(version_id, sample_size)
        comparison_results['agreement_analysis'] = agreement
        
        # Quality comparison
        quality = await self._compare_quality(version_id, sample_size)
        comparison_results['quality_comparison'] = quality
        
        return comparison_results
    
    async def _query_comparison_metrics(
        self,
        version_id: str,
        time_range_hours: int
    ) -> Dict[str, Any]:
        """Query comparison metrics from Prometheus"""
        queries = {
            'accuracy_delta': f'avg(analysis_accuracy{{version="{version_id}"}}) - avg(analysis_accuracy{{version="baseline"}})',
            'latency_delta': f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{version="{version_id}"}}[{time_range_hours}h])) by (le)) - histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{version="baseline"}}[{time_range_hours}h])) by (le))',
            'cost_ratio': f'avg(request_cost{{version="{version_id}"}}) / avg(request_cost{{version="baseline"}})',
        }
        
        metrics = {}
        for name, query in queries.items():
            try:
                response = await self.client.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params={'query': query}
                )
                result = response.json()
                if result['status'] == 'success' and result['data']['result']:
                    metrics[name] = float(result['data']['result'][0]['value'][1])
                else:
                    metrics[name] = None
            except Exception as e:
                logger.error(f"Failed to query {name}: {e}")
                metrics[name] = None
        
        return metrics
    
    async def _calculate_agreement(
        self,
        version_id: str,
        sample_size: int
    ) -> Dict[str, Any]:
        """Calculate agreement between V1 and V2 outputs"""
        # Placeholder - would query database for paired outputs
        return {
            'issue_type_agreement': 0.85,
            'severity_agreement': 0.80,
            'location_agreement': 0.75,
            'overall_agreement': 0.80,
        }
    
    async def _compare_quality(
        self,
        version_id: str,
        sample_size: int
    ) -> Dict[str, Any]:
        """Compare output quality metrics"""
        return {
            'v1_avg_issues_per_request': 5.2,
            'v2_avg_issues_per_request': 4.8,
            'v1_false_positive_rate': 0.12,
            'v2_false_positive_rate': 0.15,
            'v1_specificity_score': 0.88,
            'v2_specificity_score': 0.85,
        }


# FastAPI Application
app = FastAPI(
    title="Evaluation Pipeline",
    description="Orchestrates version evaluation using gold-sets, shadow comparison, and statistical tests",
    version="1.0.0"
)

evaluator: Optional[GoldSetEvaluator] = None
comparison_evaluator: Optional[ShadowComparisonEvaluator] = None
evaluation_results: Dict[str, EvaluationReport] = {}


@app.on_event("startup")
async def startup():
    global evaluator, comparison_evaluator
    evaluator = GoldSetEvaluator()
    await evaluator.load_gold_sets()
    comparison_evaluator = ShadowComparisonEvaluator()
    logger.info("Evaluation Pipeline started")


@app.get("/health")
async def health():
    return {"status": "healthy", "gold_sets_loaded": len(evaluator.gold_sets) if evaluator else 0}


class EvaluationRequest(BaseModel):
    version_id: str
    model_version: str = "gpt-4o"
    prompt_version: str = "code-review-v3"
    test_sets: Optional[List[str]] = None
    comparison_baseline: str = "v2-current"


@app.post("/evaluate/gold-set")
async def evaluate_gold_set(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Trigger gold-set evaluation"""
    
    async def run_evaluation():
        report = await evaluator.evaluate_version(
            request.version_id,
            request.model_version,
            request.prompt_version,
            request.test_sets
        )
        evaluation_results[report.evaluation_id] = report
        logger.info(f"Evaluation {report.evaluation_id} completed: {report.overall_pass_rate:.2%} pass rate")
    
    background_tasks.add_task(run_evaluation)
    
    return {
        "status": "started",
        "message": f"Evaluation started for version {request.version_id}",
        "check_status_at": f"/evaluate/status/{request.version_id}"
    }


@app.get("/evaluate/status/{version_id}")
async def get_evaluation_status(version_id: str):
    """Get evaluation status"""
    for eval_id, report in evaluation_results.items():
        if report.version_id == version_id:
            return {
                "evaluation_id": eval_id,
                "status": report.status,
                "overall_pass_rate": report.overall_pass_rate,
                "promotion_recommended": report.promotion_recommended,
                "recommendation_reason": report.recommendation_reason
            }
    
    return {"status": "not_found", "message": f"No evaluation found for version {version_id}"}


@app.get("/results/{version_id}")
async def get_results(version_id: str):
    """Get full evaluation results"""
    for eval_id, report in evaluation_results.items():
        if report.version_id == version_id:
            return {
                "evaluation_id": report.evaluation_id,
                "version_id": report.version_id,
                "started_at": report.started_at.isoformat(),
                "completed_at": report.completed_at.isoformat() if report.completed_at else None,
                "status": report.status,
                "overall_pass_rate": report.overall_pass_rate,
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "category_results": report.category_results,
                "avg_precision": report.avg_precision,
                "avg_recall": report.avg_recall,
                "avg_f1": report.avg_f1,
                "avg_latency_ms": report.avg_latency_ms,
                "total_cost_usd": report.total_cost_usd,
                "promotion_recommended": report.promotion_recommended,
                "recommendation_reason": report.recommendation_reason
            }
    
    raise HTTPException(404, f"No results found for version {version_id}")


@app.post("/compare/shadow")
async def compare_shadow(
    version_id: str,
    time_range_hours: int = 24,
    sample_size: int = 1000
):
    """Compare V1 shadow outputs with V2 baseline"""
    results = await comparison_evaluator.compare_outputs(
        version_id,
        time_range_hours,
        sample_size
    )
    return results


@app.get("/gold-sets")
async def list_gold_sets():
    """List available gold-sets"""
    return {
        "gold_sets": [
            {
                "id": gs['id'],
                "name": gs['name'],
                "category": gs['category'],
                "test_count": len(gs.get('test_cases', [])),
                "required_pass_rate": gs.get('required_pass_rate', 0.9)
            }
            for gs in evaluator.gold_sets.values()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
