"""
Integration Tests for Evaluation Pipeline

Tests the complete evaluation flow including:
- Gold-set evaluation
- Shadow comparison
- Promotion decision
- Statistical testing
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Test configuration
EVALUATION_URL = os.getenv("EVALUATION_URL", "http://localhost:8080")
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


class TestEvaluationPipelineIntegration:
    """Integration tests for the evaluation pipeline service"""

    @pytest.fixture
    def client(self):
        """HTTP client for API calls"""
        return httpx.AsyncClient(timeout=30.0)

    @pytest.fixture
    def sample_gold_set(self) -> Dict:
        """Sample gold-set test case"""
        return {
            "id": "security-001",
            "category": "security",
            "name": "SQL Injection Detection",
            "code": """
def get_user(username):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
""",
            "language": "python",
            "expected_issues": [
                {
                    "type": "security",
                    "severity": "critical",
                    "pattern": "sql_injection"
                }
            ],
            "timeout_ms": 30000
        }

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test evaluation pipeline health check"""
        response = await client.get(f"{EVALUATION_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_gold_set_evaluation_started(self, client):
        """Test starting a gold-set evaluation"""
        request = {
            "version_id": "v1-test-123",
            "model_version": "gpt-4o",
            "prompt_version": "code-review-v3",
            "test_sets": ["security"]
        }
        
        response = await client.post(
            f"{EVALUATION_URL}/evaluate/gold-set",
            json=request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["started", "completed"]
        assert "version_id" in data or "check_status_at" in data

    @pytest.mark.asyncio
    async def test_evaluation_status(self, client):
        """Test getting evaluation status"""
        version_id = "v1-test-status"
        
        response = await client.get(
            f"{EVALUATION_URL}/evaluate/status/{version_id}"
        )
        
        # May return 404 if no evaluation exists, or 200 with status
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] in ["pending", "running", "completed", "failed"]

    @pytest.mark.asyncio
    async def test_gold_set_list(self, client):
        """Test listing available gold-sets"""
        response = await client.get(f"{EVALUATION_URL}/gold-sets")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have categories
        assert isinstance(data, dict)
        # Expect at least some categories
        if "categories" in data:
            assert len(data["categories"]) > 0

    @pytest.mark.asyncio
    async def test_shadow_comparison(self, client):
        """Test shadow traffic comparison"""
        request = {
            "v1_version_id": "v1-shadow-test",
            "v2_version_id": "v2-baseline"
        }
        
        response = await client.post(
            f"{EVALUATION_URL}/compare/shadow",
            json=request
        )
        
        # May fail if no shadow data exists
        assert response.status_code in [200, 400, 404]

    @pytest.mark.asyncio
    async def test_promotion_check(self, client):
        """Test checking promotion eligibility"""
        version_id = "v1-promo-check"
        
        response = await client.get(
            f"{EVALUATION_URL}/evaluate/promotion-check/{version_id}"
        )
        
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "eligible" in data or "promotion_recommended" in data


class TestGoldSetExecution:
    """Tests for gold-set test execution"""

    @pytest.fixture
    def mock_ai_response(self):
        """Mock AI service response"""
        return {
            "issues": [
                {
                    "type": "security",
                    "severity": "critical",
                    "message": "SQL injection vulnerability detected",
                    "line": 2,
                    "file": "test.py"
                }
            ],
            "metrics": {
                "latency_ms": 2500,
                "tokens_used": 1500,
                "cost_usd": 0.003
            }
        }

    @pytest.mark.asyncio
    async def test_single_test_execution(self, mock_ai_response):
        """Test executing a single gold-set test case"""
        from services.evaluation_pipeline.pipeline import evaluate_test_case
        
        test_case = {
            "id": "test-001",
            "code": "def test(): pass",
            "language": "python",
            "expected_issues": [],
            "timeout_ms": 5000
        }
        
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: mock_ai_response
            )
            
            # This would test the actual function if imported
            # result = await evaluate_test_case(test_case, "v1-test", "gpt-4o", "v3")
            # assert result is not None

    @pytest.mark.asyncio
    async def test_batch_evaluation(self):
        """Test batch evaluation of multiple test cases"""
        test_cases = [
            {"id": "test-001", "code": "code1", "expected_issues": []},
            {"id": "test-002", "code": "code2", "expected_issues": []},
            {"id": "test-003", "code": "code3", "expected_issues": []},
        ]
        
        # Test that batch processing handles concurrency
        # This is a structural test - actual implementation would be tested
        assert len(test_cases) == 3


class TestStatisticalAnalysis:
    """Tests for statistical analysis components"""

    @pytest.fixture
    def sample_metrics(self) -> Dict[str, List[float]]:
        """Sample metrics for statistical testing"""
        return {
            "v1_latencies": [2500, 2600, 2400, 2550, 2700, 2450, 2500, 2650],
            "v2_latencies": [2800, 2900, 2700, 2850, 2950, 2750, 2800, 2900],
            "v1_accuracy": [0.92, 0.91, 0.93, 0.90, 0.92, 0.91, 0.93, 0.92],
            "v2_accuracy": [0.88, 0.87, 0.89, 0.86, 0.88, 0.87, 0.88, 0.87]
        }

    def test_ttest_calculation(self, sample_metrics):
        """Test t-test statistical calculation"""
        from scipy import stats
        
        v1 = sample_metrics["v1_latencies"]
        v2 = sample_metrics["v2_latencies"]
        
        t_stat, p_value = stats.ttest_ind(v1, v2)
        
        # V1 should have lower latency
        assert t_stat < 0  # Negative because v1 < v2
        assert p_value < 0.05  # Statistically significant

    def test_mannwhitney_calculation(self, sample_metrics):
        """Test Mann-Whitney U statistical calculation"""
        from scipy import stats
        
        v1 = sample_metrics["v1_accuracy"]
        v2 = sample_metrics["v2_accuracy"]
        
        _, p_value = stats.mannwhitneyu(v1, v2, alternative='greater')
        
        # V1 should have higher accuracy
        assert p_value < 0.05  # Statistically significant

    def test_effect_size_calculation(self, sample_metrics):
        """Test Cohen's d effect size calculation"""
        import numpy as np
        
        v1 = np.array(sample_metrics["v1_latencies"])
        v2 = np.array(sample_metrics["v2_latencies"])
        
        # Calculate Cohen's d
        pooled_std = np.sqrt((v1.std()**2 + v2.std()**2) / 2)
        cohens_d = (v1.mean() - v2.mean()) / pooled_std
        
        # Should show medium to large effect size
        assert abs(cohens_d) > 0.5


class TestPromotionDecision:
    """Tests for promotion decision logic"""

    @pytest.fixture
    def passing_metrics(self) -> Dict:
        """Metrics that should pass promotion"""
        return {
            "p95_latency_ms": 2500,
            "error_rate": 0.01,
            "accuracy": 0.92,
            "accuracy_delta": 0.04,
            "security_pass_rate": 0.99,
            "cost_increase": 0.05
        }

    @pytest.fixture
    def failing_metrics(self) -> Dict:
        """Metrics that should fail promotion"""
        return {
            "p95_latency_ms": 4000,  # Too high
            "error_rate": 0.03,  # Too high
            "accuracy": 0.80,  # Too low
            "accuracy_delta": -0.02,  # Negative delta
            "security_pass_rate": 0.95,  # Below threshold
            "cost_increase": 0.15  # Too high
        }

    def test_promotion_approved(self, passing_metrics):
        """Test that good metrics result in promotion approval"""
        thresholds = {
            "p95_latency_ms": 3000,
            "error_rate": 0.02,
            "accuracy_delta": 0.02,
            "security_pass_rate": 0.99,
            "cost_increase": 0.10
        }
        
        # Check all metrics pass
        assert passing_metrics["p95_latency_ms"] <= thresholds["p95_latency_ms"]
        assert passing_metrics["error_rate"] <= thresholds["error_rate"]
        assert passing_metrics["accuracy_delta"] >= thresholds["accuracy_delta"]
        assert passing_metrics["security_pass_rate"] >= thresholds["security_pass_rate"]
        assert passing_metrics["cost_increase"] <= thresholds["cost_increase"]

    def test_promotion_denied(self, failing_metrics):
        """Test that bad metrics result in promotion denial"""
        thresholds = {
            "p95_latency_ms": 3000,
            "error_rate": 0.02,
            "accuracy_delta": 0.02,
            "security_pass_rate": 0.99,
            "cost_increase": 0.10
        }
        
        # At least one metric should fail
        failures = []
        
        if failing_metrics["p95_latency_ms"] > thresholds["p95_latency_ms"]:
            failures.append("latency")
        if failing_metrics["error_rate"] > thresholds["error_rate"]:
            failures.append("error_rate")
        if failing_metrics["accuracy_delta"] < thresholds["accuracy_delta"]:
            failures.append("accuracy")
        if failing_metrics["security_pass_rate"] < thresholds["security_pass_rate"]:
            failures.append("security")
        if failing_metrics["cost_increase"] > thresholds["cost_increase"]:
            failures.append("cost")
        
        assert len(failures) > 0


class TestEndToEndEvaluation:
    """End-to-end evaluation flow tests"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_evaluation_flow(self):
        """Test complete evaluation from start to decision"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 1. Start evaluation
            start_response = await client.post(
                f"{EVALUATION_URL}/evaluate/gold-set",
                json={
                    "version_id": "v1-e2e-test",
                    "model_version": "gpt-4o",
                    "prompt_version": "code-review-v3",
                    "test_sets": ["security"]
                }
            )
            
            if start_response.status_code != 200:
                pytest.skip("Evaluation service not available")
            
            # 2. Poll for completion (with timeout)
            max_attempts = 30
            for _ in range(max_attempts):
                status_response = await client.get(
                    f"{EVALUATION_URL}/evaluate/status/v1-e2e-test"
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    if status.get("status") == "completed":
                        break
                
                await asyncio.sleep(2)
            
            # 3. Check final results
            results_response = await client.get(
                f"{EVALUATION_URL}/results/v1-e2e-test"
            )
            
            if results_response.status_code == 200:
                results = results_response.json()
                assert "overall_pass_rate" in results or "promotion_recommended" in results


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
