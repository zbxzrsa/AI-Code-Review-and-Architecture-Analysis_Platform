#!/usr/bin/env python3
"""
Statistical Tests for Version Comparison

Performs t-tests, Mann-Whitney U tests, and chi-square tests
to determine if differences between versions are statistically significant.

Usage:
    python statistical_tests.py --version v1-abc123 --baseline v2-current --output results.json
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
from scipy import stats
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Result of a statistical test"""
    test_name: str
    metric: str
    version_value: float
    baseline_value: float
    difference: float
    p_value: float
    statistic: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size_version: int
    sample_size_baseline: int
    interpretation: str


@dataclass
class ComparisonReport:
    """Full comparison report"""
    version_id: str
    baseline_id: str
    timestamp: str
    overall_passed: bool
    tests: List[StatisticalTestResult]
    summary: Dict[str, any]


class StatisticalAnalyzer:
    """Performs statistical analysis for version comparison"""
    
    def __init__(
        self,
        prometheus_url: str = "http://prometheus.platform-monitoring.svc:9090",
        significance_level: float = 0.05
    ):
        self.prometheus_url = prometheus_url
        self.significance_level = significance_level
        self.client = httpx.Client(timeout=30.0)
    
    def query_prometheus(self, query: str, time_range: str = "1h") -> List[float]:
        """Query Prometheus for metric values"""
        try:
            # Query for range data
            response = self.client.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z",
                    "end": datetime.utcnow().isoformat() + "Z",
                    "step": "60s"
                }
            )
            result = response.json()
            
            if result["status"] == "success" and result["data"]["result"]:
                values = result["data"]["result"][0]["values"]
                return [float(v[1]) for v in values if v[1] != "NaN"]
            return []
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return []
    
    def get_metric_samples(
        self, 
        metric_name: str, 
        version_id: str,
        namespace: str
    ) -> List[float]:
        """Get metric samples for a specific version"""
        queries = {
            "accuracy": f'analysis_accuracy{{version="{version_id}",namespace="{namespace}"}}',
            "latency": f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{version="{version_id}",namespace="{namespace}"}}[1m])) by (le)) * 1000',
            "error_rate": f'sum(rate(http_requests_total{{version="{version_id}",namespace="{namespace}",status=~"5.."}}[1m])) / sum(rate(http_requests_total{{version="{version_id}",namespace="{namespace}"}}[1m]))',
            "cost": f'avg(request_cost{{version="{version_id}",namespace="{namespace}"}})',
            "security_pass_rate": f'sum(rate(security_checks_passed{{version="{version_id}",namespace="{namespace}"}}[1m])) / sum(rate(security_checks_total{{version="{version_id}",namespace="{namespace}"}}[1m]))',
        }
        
        if metric_name in queries:
            return self.query_prometheus(queries[metric_name])
        return []
    
    def t_test(
        self,
        version_samples: List[float],
        baseline_samples: List[float],
        metric_name: str,
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """Perform independent samples t-test"""
        if len(version_samples) < 2 or len(baseline_samples) < 2:
            return self._insufficient_data_result("t-test", metric_name)
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(
            version_samples, 
            baseline_samples,
            alternative=alternative
        )
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(version_samples) - 1) * np.var(version_samples, ddof=1) +
             (len(baseline_samples) - 1) * np.var(baseline_samples, ddof=1)) /
            (len(version_samples) + len(baseline_samples) - 2)
        )
        effect_size = (np.mean(version_samples) - np.mean(baseline_samples)) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference of means
        version_mean = np.mean(version_samples)
        baseline_mean = np.mean(baseline_samples)
        difference = version_mean - baseline_mean
        
        se = np.sqrt(
            np.var(version_samples, ddof=1) / len(version_samples) +
            np.var(baseline_samples, ddof=1) / len(baseline_samples)
        )
        ci = stats.t.interval(
            0.95,
            len(version_samples) + len(baseline_samples) - 2,
            loc=difference,
            scale=se
        )
        
        significant = p_value < self.significance_level
        
        return StatisticalTestResult(
            test_name="Independent Samples t-test",
            metric=metric_name,
            version_value=version_mean,
            baseline_value=baseline_mean,
            difference=difference,
            p_value=p_value,
            statistic=statistic,
            significant=significant,
            effect_size=effect_size,
            confidence_interval=ci,
            sample_size_version=len(version_samples),
            sample_size_baseline=len(baseline_samples),
            interpretation=self._interpret_effect_size(effect_size, significant)
        )
    
    def mann_whitney_test(
        self,
        version_samples: List[float],
        baseline_samples: List[float],
        metric_name: str
    ) -> StatisticalTestResult:
        """Perform Mann-Whitney U test (non-parametric alternative to t-test)"""
        if len(version_samples) < 2 or len(baseline_samples) < 2:
            return self._insufficient_data_result("Mann-Whitney U", metric_name)
        
        statistic, p_value = stats.mannwhitneyu(
            version_samples,
            baseline_samples,
            alternative="two-sided"
        )
        
        # Calculate rank-biserial correlation as effect size
        n1, n2 = len(version_samples), len(baseline_samples)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        version_median = np.median(version_samples)
        baseline_median = np.median(baseline_samples)
        
        significant = p_value < self.significance_level
        
        return StatisticalTestResult(
            test_name="Mann-Whitney U test",
            metric=metric_name,
            version_value=version_median,
            baseline_value=baseline_median,
            difference=version_median - baseline_median,
            p_value=p_value,
            statistic=statistic,
            significant=significant,
            effect_size=effect_size,
            confidence_interval=(np.nan, np.nan),  # Not easily computed for Mann-Whitney
            sample_size_version=n1,
            sample_size_baseline=n2,
            interpretation=self._interpret_effect_size(effect_size, significant)
        )
    
    def chi_square_test(
        self,
        version_passed: int,
        version_total: int,
        baseline_passed: int,
        baseline_total: int,
        metric_name: str
    ) -> StatisticalTestResult:
        """Perform chi-square test for proportions (e.g., security pass rate)"""
        # Create contingency table
        observed = np.array([
            [version_passed, version_total - version_passed],
            [baseline_passed, baseline_total - baseline_passed]
        ])
        
        if observed.min() < 5:
            # Use Fisher's exact test for small samples
            _, p_value = stats.fisher_exact(observed)
            test_name = "Fisher's Exact Test"
            statistic = np.nan
        else:
            statistic, p_value, _, _ = stats.chi2_contingency(observed)
            test_name = "Chi-Square Test"
        
        version_rate = version_passed / version_total if version_total > 0 else 0
        baseline_rate = baseline_passed / baseline_total if baseline_total > 0 else 0
        
        # Effect size: Phi coefficient
        n = version_total + baseline_total
        effect_size = np.sqrt(statistic / n) if not np.isnan(statistic) else abs(version_rate - baseline_rate)
        
        significant = p_value < self.significance_level
        
        # Wilson score interval for difference
        ci = self._proportion_ci(version_rate, version_total, baseline_rate, baseline_total)
        
        return StatisticalTestResult(
            test_name=test_name,
            metric=metric_name,
            version_value=version_rate,
            baseline_value=baseline_rate,
            difference=version_rate - baseline_rate,
            p_value=p_value,
            statistic=statistic if not np.isnan(statistic) else 0,
            significant=significant,
            effect_size=effect_size,
            confidence_interval=ci,
            sample_size_version=version_total,
            sample_size_baseline=baseline_total,
            interpretation=f"{'Significant' if significant else 'No significant'} difference in {metric_name}"
        )
    
    def _proportion_ci(
        self,
        p1: float, n1: int,
        p2: float, n2: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for difference of proportions"""
        z = stats.norm.ppf((1 + confidence) / 2)
        diff = p1 - p2
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 > 0 and n2 > 0 else 0
        return (diff - z * se, diff + z * se)
    
    def _interpret_effect_size(self, effect_size: float, significant: bool) -> str:
        """Interpret effect size using Cohen's conventions"""
        abs_effect = abs(effect_size)
        
        if not significant:
            return "No statistically significant difference"
        
        if abs_effect < 0.2:
            magnitude = "negligible"
        elif abs_effect < 0.5:
            magnitude = "small"
        elif abs_effect < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        direction = "improvement" if effect_size > 0 else "regression"
        return f"Statistically significant {magnitude} {direction}"
    
    def _insufficient_data_result(self, test_name: str, metric: str) -> StatisticalTestResult:
        """Return result for insufficient data"""
        return StatisticalTestResult(
            test_name=test_name,
            metric=metric,
            version_value=np.nan,
            baseline_value=np.nan,
            difference=np.nan,
            p_value=1.0,
            statistic=np.nan,
            significant=False,
            effect_size=0,
            confidence_interval=(np.nan, np.nan),
            sample_size_version=0,
            sample_size_baseline=0,
            interpretation="Insufficient data for statistical analysis"
        )
    
    def run_full_comparison(
        self,
        version_id: str,
        baseline_id: str
    ) -> ComparisonReport:
        """Run full statistical comparison between versions"""
        tests = []
        
        # Determine namespaces
        version_ns = "platform-v1-exp"
        baseline_ns = "platform-v2-stable"
        
        # Accuracy comparison (t-test, higher is better)
        logger.info("Running accuracy comparison...")
        version_accuracy = self.get_metric_samples("accuracy", version_id, version_ns)
        baseline_accuracy = self.get_metric_samples("accuracy", baseline_id, baseline_ns)
        
        accuracy_test = self.t_test(
            version_accuracy, baseline_accuracy, 
            "accuracy", alternative="greater"
        )
        tests.append(accuracy_test)
        
        # Latency comparison (Mann-Whitney, lower is better)
        logger.info("Running latency comparison...")
        version_latency = self.get_metric_samples("latency", version_id, version_ns)
        baseline_latency = self.get_metric_samples("latency", baseline_id, baseline_ns)
        
        latency_test = self.mann_whitney_test(
            version_latency, baseline_latency, "latency_p95"
        )
        tests.append(latency_test)
        
        # Cost comparison (t-test, lower or similar is better)
        logger.info("Running cost comparison...")
        version_cost = self.get_metric_samples("cost", version_id, version_ns)
        baseline_cost = self.get_metric_samples("cost", baseline_id, baseline_ns)
        
        cost_test = self.t_test(
            version_cost, baseline_cost, "cost_per_request"
        )
        tests.append(cost_test)
        
        # Security pass rate (chi-square)
        logger.info("Running security comparison...")
        # Query for pass/fail counts
        version_security = self.query_prometheus(
            f'sum(increase(security_checks_passed{{version="{version_id}"}}[1h]))'
        )
        version_security_total = self.query_prometheus(
            f'sum(increase(security_checks_total{{version="{version_id}"}}[1h]))'
        )
        baseline_security = self.query_prometheus(
            f'sum(increase(security_checks_passed{{version="{baseline_id}"}}[1h]))'
        )
        baseline_security_total = self.query_prometheus(
            f'sum(increase(security_checks_total{{version="{baseline_id}"}}[1h]))'
        )
        
        security_test = self.chi_square_test(
            int(version_security[-1]) if version_security else 0,
            int(version_security_total[-1]) if version_security_total else 0,
            int(baseline_security[-1]) if baseline_security else 0,
            int(baseline_security_total[-1]) if baseline_security_total else 0,
            "security_pass_rate"
        )
        tests.append(security_test)
        
        # Determine overall pass/fail
        # Criteria: accuracy significantly better, latency not significantly worse,
        # cost increase <= 10%, security >= 99%
        accuracy_ok = accuracy_test.significant and accuracy_test.difference > 0
        latency_ok = not (latency_test.significant and latency_test.difference > 0)
        cost_ok = cost_test.difference <= baseline_cost[0] * 0.1 if baseline_cost else True
        security_ok = security_test.version_value >= 0.99 if not np.isnan(security_test.version_value) else False
        
        overall_passed = accuracy_ok and latency_ok and cost_ok and security_ok
        
        # Summary
        summary = {
            "accuracy_delta": accuracy_test.difference,
            "accuracy_significant": accuracy_test.significant,
            "latency_delta_ms": latency_test.difference,
            "latency_significant": latency_test.significant,
            "cost_delta": cost_test.difference,
            "cost_delta_percent": (cost_test.difference / cost_test.baseline_value * 100) if cost_test.baseline_value else 0,
            "security_pass_rate": security_test.version_value,
            "criteria": {
                "accuracy_ok": accuracy_ok,
                "latency_ok": latency_ok,
                "cost_ok": cost_ok,
                "security_ok": security_ok
            }
        }
        
        return ComparisonReport(
            version_id=version_id,
            baseline_id=baseline_id,
            timestamp=datetime.utcnow().isoformat(),
            overall_passed=overall_passed,
            tests=tests,
            summary=summary
        )


def main():
    parser = argparse.ArgumentParser(description="Run statistical tests for version comparison")
    parser.add_argument("--version", required=True, help="Version ID to test")
    parser.add_argument("--baseline", required=True, help="Baseline version ID")
    parser.add_argument("--output", default="stats-results.json", help="Output file path")
    parser.add_argument("--prometheus-url", default="http://prometheus.platform-monitoring.svc:9090")
    parser.add_argument("--significance", type=float, default=0.05, help="Significance level")
    
    args = parser.parse_args()
    
    analyzer = StatisticalAnalyzer(
        prometheus_url=args.prometheus_url,
        significance_level=args.significance
    )
    
    logger.info(f"Running comparison: {args.version} vs {args.baseline}")
    report = analyzer.run_full_comparison(args.version, args.baseline)
    
    # Convert to dict for JSON serialization
    report_dict = {
        "version_id": report.version_id,
        "baseline_id": report.baseline_id,
        "timestamp": report.timestamp,
        "overall_passed": report.overall_passed,
        "summary": report.summary,
        "tests": [
            {
                **asdict(t),
                "confidence_interval": list(t.confidence_interval)
            }
            for t in report.tests
        ]
    }
    
    # Handle NaN values for JSON
    def handle_nan(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    
    with open(args.output, "w") as f:
        json.dump(report_dict, f, indent=2, default=handle_nan)
    
    logger.info(f"Results written to {args.output}")
    logger.info(f"Overall result: {'PASSED' if report.overall_passed else 'FAILED'}")
    
    # Exit with appropriate code
    sys.exit(0 if report.overall_passed else 1)


if __name__ == "__main__":
    main()
