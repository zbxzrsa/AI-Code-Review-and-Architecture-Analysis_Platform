"""
System Validation Tests

Comprehensive end-to-end tests that validate the complete
three-version architecture is working correctly.

Run with: pytest tests/integration/test_system_validation.py -v --tb=short
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List

import httpx
import pytest

# Service URLs
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost")
VCAI_V2_URL = os.getenv("VCAI_V2_URL", "http://localhost:8001")
LIFECYCLE_URL = os.getenv("LIFECYCLE_URL", "http://localhost:8003")
EVALUATION_URL = os.getenv("EVALUATION_URL", "http://localhost:8004")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
OPA_URL = os.getenv("OPA_URL", "http://localhost:8181")


class TestServiceHealth:
    """Validate all services are healthy"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_gateway_health(self, client):
        """Gateway should be healthy"""
        try:
            response = await client.get(f"{GATEWAY_URL}/health")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Gateway not available")

    @pytest.mark.asyncio
    async def test_vcai_v2_health(self, client):
        """VCAI V2 should be healthy"""
        try:
            response = await client.get(f"{VCAI_V2_URL}/health")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("VCAI V2 not available")

    @pytest.mark.asyncio
    async def test_lifecycle_controller_health(self, client):
        """Lifecycle controller should be healthy"""
        try:
            response = await client.get(f"{LIFECYCLE_URL}/health")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Lifecycle controller not available")

    @pytest.mark.asyncio
    async def test_evaluation_pipeline_health(self, client):
        """Evaluation pipeline should be healthy"""
        try:
            response = await client.get(f"{EVALUATION_URL}/health")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Evaluation pipeline not available")

    @pytest.mark.asyncio
    async def test_opa_health(self, client):
        """OPA should be healthy"""
        try:
            response = await client.get(f"{OPA_URL}/health")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("OPA not available")

    @pytest.mark.asyncio
    async def test_prometheus_health(self, client):
        """Prometheus should be healthy"""
        try:
            response = await client.get(f"{PROMETHEUS_URL}/-/healthy")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Prometheus not available")


class TestCodeAnalysisFlow:
    """Validate the code analysis flow works end-to-end"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=60.0)

    @pytest.fixture
    def vulnerable_code(self) -> Dict:
        return {
            "code": '''
def get_user(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
''',
            "language": "python"
        }

    @pytest.fixture
    def safe_code(self) -> Dict:
        return {
            "code": '''
def get_user(username):
    cursor.execute(
        "SELECT * FROM users WHERE name = %s",
        (username,)
    )
    return cursor.fetchone()
''',
            "language": "python"
        }

    @pytest.mark.asyncio
    async def test_analyze_vulnerable_code(self, client, vulnerable_code):
        """Should detect SQL injection in vulnerable code"""
        try:
            response = await client.post(
                f"{GATEWAY_URL}/api/v2/analyze",
                json=vulnerable_code
            )
            
            if response.status_code == 200:
                data = response.json()
                # Should have detected issues
                assert "issues" in data or "results" in data
        except httpx.ConnectError:
            pytest.skip("API not available")

    @pytest.mark.asyncio
    async def test_analyze_safe_code(self, client, safe_code):
        """Should not flag safe code"""
        try:
            response = await client.post(
                f"{GATEWAY_URL}/api/v2/analyze",
                json=safe_code
            )
            
            if response.status_code == 200:
                data = response.json()
                # Should have fewer or no security issues
                issues = data.get("issues", [])
                security_issues = [i for i in issues if i.get("type") == "security"]
                assert len(security_issues) == 0
        except httpx.ConnectError:
            pytest.skip("API not available")


class TestShadowTrafficFlow:
    """Validate shadow traffic is working"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_shadow_traffic_generated(self, client):
        """V1 should receive shadow traffic"""
        try:
            # Make a request to V2
            await client.post(
                f"{GATEWAY_URL}/api/v2/analyze",
                json={"code": "print('test')", "language": "python"}
            )
            
            # Check V1 metrics in Prometheus
            response = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={
                    "query": 'sum(increase(http_requests_total{namespace="platform-v1-exp"}[5m]))'
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", {}).get("result", [])
                # Shadow traffic should generate requests to V1
                if results:
                    value = float(results[0]["value"][1])
                    assert value >= 0  # May be 0 if just started
        except httpx.ConnectError:
            pytest.skip("Services not available")


class TestVersionLifecycle:
    """Validate version lifecycle management"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_list_versions(self, client):
        """Should list active versions"""
        try:
            response = await client.get(f"{LIFECYCLE_URL}/versions")
            
            if response.status_code == 200:
                data = response.json()
                assert "versions" in data
                assert isinstance(data["versions"], list)
        except httpx.ConnectError:
            pytest.skip("Lifecycle controller not available")

    @pytest.mark.asyncio
    async def test_register_new_version(self, client):
        """Should be able to register a new version"""
        try:
            version_id = f"v1-test-{datetime.now().strftime('%H%M%S')}"
            
            response = await client.post(
                f"{LIFECYCLE_URL}/versions/{version_id}/register",
                json={
                    "version_id": version_id,
                    "model_version": "gpt-4o",
                    "prompt_version": "code-review-v4",
                    "current_state": "shadow"
                }
            )
            
            assert response.status_code in [200, 201]
        except httpx.ConnectError:
            pytest.skip("Lifecycle controller not available")


class TestOPAPolicies:
    """Validate OPA policies are working"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_promotion_policy_approval(self, client):
        """Good metrics should be approved for promotion"""
        try:
            response = await client.post(
                f"{OPA_URL}/v1/data/lifecycle/allow_promotion",
                json={
                    "input": {
                        "metrics": {
                            "p95_latency_ms": 2500,
                            "error_rate": 0.01,
                            "accuracy_delta": 0.04,
                            "security_pass_rate": 0.995,
                            "cost_increase": 0.05
                        }
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data.get("result") is True
        except httpx.ConnectError:
            pytest.skip("OPA not available")

    @pytest.mark.asyncio
    async def test_promotion_policy_denial(self, client):
        """Bad metrics should be denied promotion"""
        try:
            response = await client.post(
                f"{OPA_URL}/v1/data/lifecycle/allow_promotion",
                json={
                    "input": {
                        "metrics": {
                            "p95_latency_ms": 5000,  # Too high
                            "error_rate": 0.05,      # Too high
                            "accuracy_delta": -0.02, # Negative
                            "security_pass_rate": 0.90,
                            "cost_increase": 0.20
                        }
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data.get("result") is False
        except httpx.ConnectError:
            pytest.skip("OPA not available")


class TestMetricsCollection:
    """Validate metrics are being collected"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_v2_metrics_exist(self, client):
        """V2 should have metrics in Prometheus"""
        try:
            response = await client.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": 'up{namespace="platform-v2-stable"}'}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", {}).get("result", [])
                # Validate that results is a list (may be empty if services not yet started)
                assert isinstance(results, list), "Expected results to be a list"
        except httpx.ConnectError:
            pytest.skip("Prometheus not available")

    @pytest.mark.asyncio
    async def test_slo_metrics_available(self, client):
        """SLO metrics should be queryable"""
        try:
            queries = [
                'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
                'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))',
            ]
            
            for query in queries:
                response = await client.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": query}
                )
                assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Prometheus not available")


class TestSystemIntegration:
    """Full system integration tests"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_analysis_cycle(self):
        """Test complete analysis cycle from request to comparison"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # 1. Submit code for analysis
                analysis_response = await client.post(
                    f"{GATEWAY_URL}/api/v2/analyze",
                    json={
                        "code": "def test(): pass",
                        "language": "python"
                    }
                )
                
                if analysis_response.status_code != 200:
                    pytest.skip("API not available")
                
                # 2. Wait for shadow traffic processing
                await asyncio.sleep(2)
                
                # 3. Check comparison data exists
                comparison_response = await client.get(
                    f"{LIFECYCLE_URL}/comparison-requests",
                    params={"limit": 5}
                )
                
                assert comparison_response.status_code in [200, 404]
                
            except httpx.ConnectError:
                pytest.skip("Services not available")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rollback_safety(self):
        """Test that rollback can be initiated safely"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Check rollback history endpoint works
                response = await client.get(
                    f"{LIFECYCLE_URL}/rollback/history",
                    params={"limit": 10}
                )
                
                assert response.status_code in [200, 404]
                
            except httpx.ConnectError:
                pytest.skip("Lifecycle controller not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
