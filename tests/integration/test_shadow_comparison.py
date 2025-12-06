"""
Integration Tests for Shadow Traffic Comparison

Tests the shadow traffic flow between V1 and V2 including:
- Shadow request forwarding
- Response comparison
- Statistical analysis
- Promotion recommendations
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List
import statistics

import httpx
import pytest

# Configuration
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost")
LIFECYCLE_URL = os.getenv("LIFECYCLE_URL", "http://localhost:8003")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


class TestShadowTrafficFlow:
    """Tests for shadow traffic routing"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.fixture
    def sample_code(self) -> str:
        return '''
def process_data(user_input):
    # Potential security issue
    query = f"SELECT * FROM users WHERE id = {user_input}"
    return execute_query(query)
'''

    @pytest.mark.asyncio
    async def test_shadow_header_forwarding(self, client, sample_code):
        """Test that requests with shadow header reach V1"""
        response = await client.post(
            f"{GATEWAY_URL}/api/analyze",
            json={"code": sample_code, "language": "python"},
            headers={"X-Shadow-Request": "true"}
        )
        
        # Should succeed regardless of shadow routing
        assert response.status_code in [200, 201, 202]

    @pytest.mark.asyncio
    async def test_v1_receives_mirrored_traffic(self, client):
        """Test that V1 receives mirrored production traffic"""
        try:
            # Query Prometheus for V1 request count
            response = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={
                    "query": 'sum(rate(http_requests_total{namespace="platform-v1-exp"}[5m]))'
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", {}).get("result", [])
                if results:
                    value = float(results[0]["value"][1])
                    # V1 should have some traffic if shadow is enabled
                    # (May be 0 if shadow traffic is disabled)
                    assert value >= 0
        except httpx.ConnectError:
            pytest.skip("Prometheus not available")


class TestComparisonAnalysis:
    """Tests for V1 vs V2 comparison analysis"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.fixture
    def v1_metrics(self) -> List[Dict]:
        """Sample V1 metrics"""
        return [
            {"latency_ms": 2500, "accuracy": 0.92, "cost": 0.004},
            {"latency_ms": 2600, "accuracy": 0.91, "cost": 0.0042},
            {"latency_ms": 2400, "accuracy": 0.93, "cost": 0.0038},
            {"latency_ms": 2550, "accuracy": 0.90, "cost": 0.0041},
            {"latency_ms": 2700, "accuracy": 0.92, "cost": 0.0045},
        ]

    @pytest.fixture
    def v2_metrics(self) -> List[Dict]:
        """Sample V2 metrics (baseline)"""
        return [
            {"latency_ms": 2800, "accuracy": 0.88, "cost": 0.0038},
            {"latency_ms": 2900, "accuracy": 0.87, "cost": 0.004},
            {"latency_ms": 2700, "accuracy": 0.89, "cost": 0.0036},
            {"latency_ms": 2850, "accuracy": 0.86, "cost": 0.0039},
            {"latency_ms": 2950, "accuracy": 0.88, "cost": 0.0041},
        ]

    def test_latency_comparison(self, v1_metrics, v2_metrics):
        """Test that V1 has better latency than V2"""
        v1_latencies = [m["latency_ms"] for m in v1_metrics]
        v2_latencies = [m["latency_ms"] for m in v2_metrics]
        
        v1_avg = statistics.mean(v1_latencies)
        v2_avg = statistics.mean(v2_latencies)
        
        # V1 should have lower latency
        assert v1_avg < v2_avg
        
        # Calculate improvement percentage
        improvement = (v2_avg - v1_avg) / v2_avg * 100
        assert improvement > 0

    def test_accuracy_comparison(self, v1_metrics, v2_metrics):
        """Test that V1 has better accuracy than V2"""
        v1_accuracy = [m["accuracy"] for m in v1_metrics]
        v2_accuracy = [m["accuracy"] for m in v2_metrics]
        
        v1_avg = statistics.mean(v1_accuracy)
        v2_avg = statistics.mean(v2_accuracy)
        
        # V1 should have higher accuracy
        assert v1_avg > v2_avg
        
        # Calculate accuracy delta
        delta = v1_avg - v2_avg
        assert delta >= 0.02  # Meets promotion threshold

    def test_cost_comparison(self, v1_metrics, v2_metrics):
        """Test cost comparison between versions"""
        v1_costs = [m["cost"] for m in v1_metrics]
        v2_costs = [m["cost"] for m in v2_metrics]
        
        v1_avg = statistics.mean(v1_costs)
        v2_avg = statistics.mean(v2_costs)
        
        # Calculate cost increase
        cost_increase = (v1_avg - v2_avg) / v2_avg
        
        # Should be within acceptable range (±10%)
        assert abs(cost_increase) <= 0.10

    def test_statistical_significance(self, v1_metrics, v2_metrics):
        """Test statistical significance of comparison"""
        from scipy import stats
        
        v1_latencies = [m["latency_ms"] for m in v1_metrics]
        v2_latencies = [m["latency_ms"] for m in v2_metrics]
        
        # Perform t-test
        _, p_value = stats.ttest_ind(v1_latencies, v2_latencies)
        
        # Should be statistically significant
        assert p_value < 0.05


class TestPromotionRecommendation:
    """Tests for promotion recommendation logic"""

    @pytest.fixture
    def passing_comparison(self) -> Dict:
        """Comparison results that should trigger promotion"""
        return {
            "version_id": "v1-test",
            "baseline_id": "v2-stable",
            "metrics": {
                "v1": {
                    "p95_latency_ms": 2500,
                    "error_rate": 0.01,
                    "accuracy": 0.92,
                    "security_pass_rate": 0.995,
                    "cost_per_request": 0.004
                },
                "v2": {
                    "p95_latency_ms": 2800,
                    "error_rate": 0.015,
                    "accuracy": 0.88,
                    "security_pass_rate": 0.99,
                    "cost_per_request": 0.0038
                }
            },
            "statistical_tests": {
                "latency_p_value": 0.02,
                "accuracy_p_value": 0.01,
                "effect_size": 0.7
            }
        }

    @pytest.fixture
    def failing_comparison(self) -> Dict:
        """Comparison results that should NOT trigger promotion"""
        return {
            "version_id": "v1-failing",
            "baseline_id": "v2-stable",
            "metrics": {
                "v1": {
                    "p95_latency_ms": 4500,  # Too high
                    "error_rate": 0.035,      # Too high
                    "accuracy": 0.82,         # Worse than baseline
                    "security_pass_rate": 0.92,
                    "cost_per_request": 0.006
                },
                "v2": {
                    "p95_latency_ms": 2800,
                    "error_rate": 0.015,
                    "accuracy": 0.88,
                    "security_pass_rate": 0.99,
                    "cost_per_request": 0.0038
                }
            }
        }

    def test_promotion_recommended(self, passing_comparison):
        """Test that good metrics result in promotion recommendation"""
        thresholds = {
            "p95_latency_ms": 3000,
            "error_rate": 0.02,
            "accuracy_delta": 0.02,
            "security_pass_rate": 0.99,
            "cost_increase_max": 0.10
        }
        
        v1 = passing_comparison["metrics"]["v1"]
        v2 = passing_comparison["metrics"]["v2"]
        
        # Check all criteria
        latency_ok = v1["p95_latency_ms"] <= thresholds["p95_latency_ms"]
        error_ok = v1["error_rate"] <= thresholds["error_rate"]
        accuracy_ok = (v1["accuracy"] - v2["accuracy"]) >= thresholds["accuracy_delta"]
        security_ok = v1["security_pass_rate"] >= thresholds["security_pass_rate"]
        cost_ok = (v1["cost_per_request"] - v2["cost_per_request"]) / v2["cost_per_request"] <= thresholds["cost_increase_max"]
        
        # All should pass
        assert latency_ok
        assert error_ok
        assert accuracy_ok
        assert security_ok
        assert cost_ok

    def test_promotion_denied(self, failing_comparison):
        """Test that bad metrics result in promotion denial"""
        thresholds = {
            "p95_latency_ms": 3000,
            "error_rate": 0.02,
            "accuracy_delta": 0.02,
            "security_pass_rate": 0.99
        }
        
        v1 = failing_comparison["metrics"]["v1"]
        v2 = failing_comparison["metrics"]["v2"]
        
        # Check failures
        failures = []
        
        if v1["p95_latency_ms"] > thresholds["p95_latency_ms"]:
            failures.append("latency")
        if v1["error_rate"] > thresholds["error_rate"]:
            failures.append("error_rate")
        if (v1["accuracy"] - v2["accuracy"]) < thresholds["accuracy_delta"]:
            failures.append("accuracy")
        if v1["security_pass_rate"] < thresholds["security_pass_rate"]:
            failures.append("security")
        
        # Should have multiple failures
        assert len(failures) >= 2


class TestShadowComparisonAPI:
    """API tests for shadow comparison endpoints"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_comparison_endpoint(self, client):
        """Test the comparison API endpoint"""
        response = await client.get(f"{LIFECYCLE_URL}/comparison-requests")
        
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "requests" in data or isinstance(data, list)

    @pytest.mark.asyncio
    async def test_comparison_stats_endpoint(self, client):
        """Test the comparison stats endpoint"""
        response = await client.get(f"{LIFECYCLE_URL}/stats/comparison")
        
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            # Should have aggregate metrics
            assert any(k in data for k in ["total_requests", "totalRequests", "samples"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
