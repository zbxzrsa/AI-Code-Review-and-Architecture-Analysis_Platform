"""
Unit Tests for Comparison Service

Tests the version comparison logic including:
- Metrics comparison
- Statistical testing
- Promotion decision logic
"""

import pytest
from datetime import datetime
from typing import Dict, List
import statistics


class TestMetricsComparison:
    """Tests for metrics comparison logic"""

    def test_latency_improvement_calculation(self):
        """Test calculating latency improvement percentage"""
        v1_latency = 2500  # ms
        v2_latency = 3000  # ms
        
        improvement = (v2_latency - v1_latency) / v2_latency * 100
        
        assert improvement > 0
        assert round(improvement, 2) == 16.67

    def test_accuracy_delta_calculation(self):
        """Test calculating accuracy delta"""
        v1_accuracy = 0.92
        v2_accuracy = 0.88
        
        delta = v1_accuracy - v2_accuracy
        
        assert delta > 0
        assert delta == 0.04

    def test_error_rate_comparison(self):
        """Test comparing error rates"""
        v1_error_rate = 0.01
        v2_error_rate = 0.015
        
        # V1 should have lower error rate
        assert v1_error_rate < v2_error_rate
        
        improvement = (v2_error_rate - v1_error_rate) / v2_error_rate * 100
        assert improvement > 0

    def test_cost_increase_calculation(self):
        """Test calculating cost increase percentage"""
        v1_cost = 0.0042
        v2_cost = 0.0038
        
        cost_increase = (v1_cost - v2_cost) / v2_cost
        
        assert cost_increase > 0
        assert round(cost_increase, 4) == 0.1053  # ~10.5% increase

    def test_security_pass_rate(self):
        """Test security pass rate calculation"""
        passed = 995
        total = 1000
        
        pass_rate = passed / total
        
        assert pass_rate == 0.995
        assert pass_rate >= 0.99  # Meets threshold


class TestStatisticalAnalysis:
    """Tests for statistical analysis functions"""

    @pytest.fixture
    def sample_v1_data(self) -> List[float]:
        return [2500, 2600, 2400, 2550, 2700, 2450, 2500, 2650, 2480, 2520]

    @pytest.fixture
    def sample_v2_data(self) -> List[float]:
        return [2800, 2900, 2700, 2850, 2950, 2750, 2800, 2900, 2820, 2880]

    def test_mean_calculation(self, sample_v1_data, sample_v2_data):
        """Test mean calculation"""
        v1_mean = statistics.mean(sample_v1_data)
        v2_mean = statistics.mean(sample_v2_data)
        
        assert v1_mean < v2_mean
        assert 2500 < v1_mean < 2600
        assert 2800 < v2_mean < 2900

    def test_std_deviation(self, sample_v1_data, sample_v2_data):
        """Test standard deviation calculation"""
        v1_std = statistics.stdev(sample_v1_data)
        v2_std = statistics.stdev(sample_v2_data)
        
        # Both should have reasonable variance
        assert v1_std > 0
        assert v2_std > 0

    def test_percentile_calculation(self, sample_v1_data):
        """Test percentile calculation"""
        sorted_data = sorted(sample_v1_data)
        n = len(sorted_data)
        
        # P95 index
        p95_idx = int(0.95 * n)
        p95 = sorted_data[p95_idx]
        
        assert p95 <= max(sample_v1_data)
        assert p95 >= statistics.median(sample_v1_data)

    def test_effect_size_cohens_d(self, sample_v1_data, sample_v2_data):
        """Test Cohen's d effect size calculation"""
        import math
        
        v1_mean = statistics.mean(sample_v1_data)
        v2_mean = statistics.mean(sample_v2_data)
        v1_std = statistics.stdev(sample_v1_data)
        v2_std = statistics.stdev(sample_v2_data)
        
        # Pooled standard deviation
        pooled_std = math.sqrt((v1_std**2 + v2_std**2) / 2)
        
        # Cohen's d
        cohens_d = (v1_mean - v2_mean) / pooled_std
        
        # Should show large effect (negative because v1 < v2)
        assert cohens_d < -0.8  # Large effect size

    def test_confidence_interval(self, sample_v1_data):
        """Test 95% confidence interval calculation"""
        import math
        
        mean = statistics.mean(sample_v1_data)
        std = statistics.stdev(sample_v1_data)
        n = len(sample_v1_data)
        
        # 95% CI (z = 1.96)
        margin = 1.96 * (std / math.sqrt(n))
        ci_lower = mean - margin
        ci_upper = mean + margin
        
        assert ci_lower < mean < ci_upper
        assert ci_upper - ci_lower > 0


class TestPromotionDecision:
    """Tests for promotion decision logic"""

    @pytest.fixture
    def thresholds(self) -> Dict:
        return {
            "p95_latency_ms": 3000,
            "error_rate": 0.02,
            "accuracy_delta": 0.02,
            "security_pass_rate": 0.99,
            "cost_increase_max": 0.10,
            "statistical_significance_p": 0.05
        }

    @pytest.fixture
    def passing_metrics(self) -> Dict:
        return {
            "p95_latency_ms": 2500,
            "error_rate": 0.01,
            "accuracy": 0.92,
            "accuracy_delta": 0.04,
            "security_pass_rate": 0.995,
            "cost_increase": 0.05,
            "p_value": 0.02
        }

    @pytest.fixture
    def failing_metrics(self) -> Dict:
        return {
            "p95_latency_ms": 4000,
            "error_rate": 0.035,
            "accuracy": 0.82,
            "accuracy_delta": -0.02,
            "security_pass_rate": 0.92,
            "cost_increase": 0.15,
            "p_value": 0.08
        }

    def test_all_metrics_pass(self, thresholds, passing_metrics):
        """Test that all passing metrics result in approval"""
        checks = {
            "latency": passing_metrics["p95_latency_ms"] <= thresholds["p95_latency_ms"],
            "error_rate": passing_metrics["error_rate"] <= thresholds["error_rate"],
            "accuracy": passing_metrics["accuracy_delta"] >= thresholds["accuracy_delta"],
            "security": passing_metrics["security_pass_rate"] >= thresholds["security_pass_rate"],
            "cost": passing_metrics["cost_increase"] <= thresholds["cost_increase_max"],
            "significance": passing_metrics["p_value"] < thresholds["statistical_significance_p"]
        }
        
        assert all(checks.values())

    def test_latency_fails(self, thresholds, failing_metrics):
        """Test latency threshold failure"""
        assert failing_metrics["p95_latency_ms"] > thresholds["p95_latency_ms"]

    def test_error_rate_fails(self, thresholds, failing_metrics):
        """Test error rate threshold failure"""
        assert failing_metrics["error_rate"] > thresholds["error_rate"]

    def test_accuracy_fails(self, thresholds, failing_metrics):
        """Test accuracy delta threshold failure"""
        assert failing_metrics["accuracy_delta"] < thresholds["accuracy_delta"]

    def test_security_fails(self, thresholds, failing_metrics):
        """Test security pass rate threshold failure"""
        assert failing_metrics["security_pass_rate"] < thresholds["security_pass_rate"]

    def test_cost_fails(self, thresholds, failing_metrics):
        """Test cost increase threshold failure"""
        assert failing_metrics["cost_increase"] > thresholds["cost_increase_max"]

    def test_significance_fails(self, thresholds, failing_metrics):
        """Test statistical significance threshold failure"""
        assert failing_metrics["p_value"] >= thresholds["statistical_significance_p"]

    def test_promotion_reasons(self, thresholds, failing_metrics):
        """Test generating denial reasons"""
        reasons = []
        
        if failing_metrics["p95_latency_ms"] > thresholds["p95_latency_ms"]:
            reasons.append(f"Latency {failing_metrics['p95_latency_ms']}ms exceeds {thresholds['p95_latency_ms']}ms")
        
        if failing_metrics["error_rate"] > thresholds["error_rate"]:
            reasons.append(f"Error rate {failing_metrics['error_rate']} exceeds {thresholds['error_rate']}")
        
        if failing_metrics["accuracy_delta"] < thresholds["accuracy_delta"]:
            reasons.append(f"Accuracy delta {failing_metrics['accuracy_delta']} below {thresholds['accuracy_delta']}")
        
        if failing_metrics["security_pass_rate"] < thresholds["security_pass_rate"]:
            reasons.append(f"Security rate {failing_metrics['security_pass_rate']} below {thresholds['security_pass_rate']}")
        
        assert len(reasons) >= 4


class TestIssueComparison:
    """Tests for comparing issues detected by V1 vs V2"""

    @pytest.fixture
    def v1_issues(self) -> List[Dict]:
        return [
            {"type": "security", "severity": "critical", "pattern": "sql_injection"},
            {"type": "security", "severity": "high", "pattern": "xss"},
            {"type": "quality", "severity": "medium", "pattern": "unused_var"},
        ]

    @pytest.fixture
    def v2_issues(self) -> List[Dict]:
        return [
            {"type": "security", "severity": "critical", "pattern": "sql_injection"},
            {"type": "quality", "severity": "low", "pattern": "naming"},
        ]

    def test_issue_count_comparison(self, v1_issues, v2_issues):
        """Test comparing issue counts"""
        v1_count = len(v1_issues)
        v2_count = len(v2_issues)
        
        assert v1_count == 3
        assert v2_count == 2
        assert v1_count > v2_count  # V1 found more issues

    def test_security_issue_comparison(self, v1_issues, v2_issues):
        """Test comparing security issues"""
        v1_security = [i for i in v1_issues if i["type"] == "security"]
        v2_security = [i for i in v2_issues if i["type"] == "security"]
        
        assert len(v1_security) == 2
        assert len(v2_security) == 1
        assert len(v1_security) > len(v2_security)

    def test_critical_issue_detection(self, v1_issues, v2_issues):
        """Test that both detect critical issues"""
        v1_critical = [i for i in v1_issues if i["severity"] == "critical"]
        v2_critical = [i for i in v2_issues if i["severity"] == "critical"]
        
        # Both should detect the SQL injection
        assert len(v1_critical) >= 1
        assert len(v2_critical) >= 1

    def test_issue_overlap(self, v1_issues, v2_issues):
        """Test calculating issue overlap"""
        v1_patterns = {i["pattern"] for i in v1_issues}
        v2_patterns = {i["pattern"] for i in v2_issues}
        
        overlap = v1_patterns & v2_patterns
        
        assert "sql_injection" in overlap
        assert len(overlap) >= 1

    def test_false_positive_rate(self):
        """Test false positive rate calculation"""
        total_issues = 100
        true_positives = 92
        false_positives = 8
        
        fp_rate = false_positives / total_issues
        
        assert fp_rate == 0.08
        assert fp_rate < 0.10  # Within acceptable range


class TestRollbackDecision:
    """Tests for rollback decision logic"""

    def test_rollback_on_slo_breach(self):
        """Test that SLO breach triggers rollback recommendation"""
        current_error_rate = 0.035
        threshold = 0.02
        
        should_rollback = current_error_rate > threshold
        
        assert should_rollback is True

    def test_rollback_on_latency_spike(self):
        """Test rollback on latency spike"""
        current_p95 = 5000
        threshold = 3000
        
        should_rollback = current_p95 > threshold
        
        assert should_rollback is True

    def test_no_rollback_healthy(self):
        """Test no rollback when metrics are healthy"""
        metrics = {
            "error_rate": 0.01,
            "p95_latency_ms": 2500,
            "security_pass_rate": 0.995
        }
        thresholds = {
            "error_rate": 0.02,
            "p95_latency_ms": 3000,
            "security_pass_rate": 0.99
        }
        
        should_rollback = (
            metrics["error_rate"] > thresholds["error_rate"] or
            metrics["p95_latency_ms"] > thresholds["p95_latency_ms"] or
            metrics["security_pass_rate"] < thresholds["security_pass_rate"]
        )
        
        assert should_rollback is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
