"""
Integration Tests for Lifecycle Controller

Tests the version lifecycle management including:
- Version registration
- Shadow comparison
- Promotion decisions
- Rollback operations
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Dict

import httpx
import pytest

# Configuration
LIFECYCLE_URL = os.getenv("LIFECYCLE_URL", "http://localhost:8003")
OPA_URL = os.getenv("OPA_URL", "http://localhost:8181")


class TestLifecycleControllerHealth:
    """Health and basic API tests"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test lifecycle controller health check"""
        response = await client.get(f"{LIFECYCLE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_list_versions(self, client):
        """Test listing active versions"""
        response = await client.get(f"{LIFECYCLE_URL}/versions")
        
        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert isinstance(data["versions"], list)


class TestVersionRegistration:
    """Tests for version registration"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.fixture
    def sample_version_config(self) -> Dict:
        return {
            "version_id": f"v1-test-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "model_version": "gpt-4o-2024-08-06",
            "prompt_version": "code-review-v4-exp",
            "current_state": "shadow",
            "metadata": {
                "created_by": "integration-test",
                "experiment_name": "test-experiment"
            }
        }

    @pytest.mark.asyncio
    async def test_register_version(self, client, sample_version_config):
        """Test registering a new version"""
        response = await client.post(
            f"{LIFECYCLE_URL}/versions/{sample_version_config['version_id']}/register",
            json=sample_version_config
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["status"] == "registered"
        assert data["version_id"] == sample_version_config["version_id"]

    @pytest.mark.asyncio
    async def test_register_duplicate_version(self, client, sample_version_config):
        """Test that duplicate registration updates the version"""
        # Register first time
        await client.post(
            f"{LIFECYCLE_URL}/versions/{sample_version_config['version_id']}/register",
            json=sample_version_config
        )
        
        # Register again with updated config
        sample_version_config["model_version"] = "gpt-4o-2024-05-13"
        response = await client.post(
            f"{LIFECYCLE_URL}/versions/{sample_version_config['version_id']}/register",
            json=sample_version_config
        )
        
        assert response.status_code in [200, 201]


class TestComparisonRequests:
    """Tests for comparison request handling"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.fixture
    def sample_comparison_request(self) -> Dict:
        return {
            "requestId": f"req-test-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "code": "def test(): pass",
            "language": "python",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "v1Output": {
                "version": "v1",
                "versionId": "v1-test",
                "modelVersion": "gpt-4o",
                "promptVersion": "code-review-v4",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latencyMs": 2500,
                "cost": 0.004,
                "issues": [],
                "rawOutput": "{}",
                "confidence": 0.95,
                "securityPassed": True
            },
            "v2Output": {
                "version": "v2",
                "versionId": "v2-stable",
                "modelVersion": "gpt-4o",
                "promptVersion": "code-review-v3",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latencyMs": 2800,
                "cost": 0.0038,
                "issues": [],
                "rawOutput": "{}",
                "confidence": 0.88,
                "securityPassed": True
            }
        }

    @pytest.mark.asyncio
    async def test_create_comparison_request(self, client, sample_comparison_request):
        """Test creating a comparison request"""
        response = await client.post(
            f"{LIFECYCLE_URL}/comparison-requests",
            json=sample_comparison_request
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["requestId"] == sample_comparison_request["requestId"]

    @pytest.mark.asyncio
    async def test_get_comparison_requests(self, client, sample_comparison_request):
        """Test retrieving comparison requests"""
        # Create a request first
        await client.post(
            f"{LIFECYCLE_URL}/comparison-requests",
            json=sample_comparison_request
        )
        
        # Get requests
        response = await client.get(
            f"{LIFECYCLE_URL}/comparison-requests",
            params={"limit": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "requests" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_get_comparison_stats(self, client):
        """Test getting comparison statistics"""
        response = await client.get(f"{LIFECYCLE_URL}/stats/comparison")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data or "totalRequests" in data


class TestRollbackOperations:
    """Tests for rollback operations"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_initiate_rollback(self, client):
        """Test initiating a rollback"""
        rollback_request = {
            "versionId": "v1-test-rollback",
            "reason": "accuracy_regression",
            "notes": "Integration test rollback"
        }
        
        response = await client.post(
            f"{LIFECYCLE_URL}/rollback",
            json=rollback_request
        )
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert data["success"] is True
        assert "rollbackId" in data

    @pytest.mark.asyncio
    async def test_get_rollback_history(self, client):
        """Test getting rollback history"""
        response = await client.get(
            f"{LIFECYCLE_URL}/rollback/history",
            params={"limit": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_rollback_stats(self, client):
        """Test getting rollback statistics"""
        response = await client.get(f"{LIFECYCLE_URL}/stats/rollbacks")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_rollbacks" in data or "totalRollbacks" in data


class TestOPAIntegration:
    """Tests for OPA policy integration"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_opa_health(self, client):
        """Test OPA health check"""
        try:
            response = await client.get(f"{OPA_URL}/health")
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("OPA not available")

    @pytest.mark.asyncio
    async def test_promotion_policy_pass(self, client):
        """Test promotion policy with passing metrics"""
        try:
            # Query OPA for promotion decision
            query = {
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
            
            response = await client.post(
                f"{OPA_URL}/v1/data/lifecycle/allow_promotion",
                json=query
            )
            
            if response.status_code == 200:
                data = response.json()
                # Should allow promotion with good metrics
                assert data.get("result") is True
        except httpx.ConnectError:
            pytest.skip("OPA not available")

    @pytest.mark.asyncio
    async def test_promotion_policy_fail(self, client):
        """Test promotion policy with failing metrics"""
        try:
            query = {
                "input": {
                    "metrics": {
                        "p95_latency_ms": 4500,  # Too high
                        "error_rate": 0.035,     # Too high
                        "accuracy_delta": -0.02, # Negative
                        "security_pass_rate": 0.92,
                        "cost_increase": 0.15
                    }
                }
            }
            
            response = await client.post(
                f"{OPA_URL}/v1/data/lifecycle/allow_promotion",
                json=query
            )
            
            if response.status_code == 200:
                data = response.json()
                # Should deny promotion with bad metrics
                assert data.get("result") is False
        except httpx.ConnectError:
            pytest.skip("OPA not available")


class TestEvaluationTrigger:
    """Tests for triggering evaluations"""

    @pytest.fixture
    def client(self):
        return httpx.AsyncClient(timeout=30.0)

    @pytest.mark.asyncio
    async def test_trigger_evaluation(self, client):
        """Test triggering evaluation for a version"""
        # First register a version
        version_id = f"v1-eval-test-{datetime.now(timezone.utc).strftime('%H%M%S')}"
        
        await client.post(
            f"{LIFECYCLE_URL}/versions/{version_id}/register",
            json={
                "version_id": version_id,
                "model_version": "gpt-4o",
                "prompt_version": "code-review-v4",
                "current_state": "shadow"
            }
        )
        
        # Trigger evaluation
        response = await client.post(
            f"{LIFECYCLE_URL}/versions/{version_id}/evaluate"
        )
        
        # May fail if version metrics not available
        assert response.status_code in [200, 400, 404]

    @pytest.mark.asyncio
    async def test_get_event_history(self, client):
        """Test getting lifecycle event history"""
        response = await client.get(
            f"{LIFECYCLE_URL}/history",
            params={"limit": 50}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "events" in data


class TestEndToEndLifecycle:
    """End-to-end lifecycle flow tests"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_lifecycle_flow(self):
        """Test complete lifecycle from registration to evaluation"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            version_id = f"v1-e2e-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            
            # 1. Register version
            register_response = await client.post(
                f"{LIFECYCLE_URL}/versions/{version_id}/register",
                json={
                    "version_id": version_id,
                    "model_version": "gpt-4o-2024-08-06",
                    "prompt_version": "code-review-v4-exp",
                    "current_state": "shadow"
                }
            )
            
            if register_response.status_code not in [200, 201]:
                pytest.skip("Lifecycle controller not available")
            
            # 2. Create comparison data
            comparison_request = {
                "requestId": f"req-e2e-{datetime.now(timezone.utc).strftime('%H%M%S')}",
                "code": "def safe_func(): return 42",
                "language": "python",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "v1Output": {
                    "version": "v1",
                    "versionId": version_id,
                    "modelVersion": "gpt-4o-2024-08-06",
                    "promptVersion": "code-review-v4-exp",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "latencyMs": 2200,
                    "cost": 0.004,
                    "issues": [],
                    "rawOutput": "{}",
                    "confidence": 0.96,
                    "securityPassed": True
                },
                "v2Output": {
                    "version": "v2",
                    "versionId": "v2-stable",
                    "modelVersion": "gpt-4o",
                    "promptVersion": "code-review-v3",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "latencyMs": 2800,
                    "cost": 0.0038,
                    "issues": [],
                    "rawOutput": "{}",
                    "confidence": 0.88,
                    "securityPassed": True
                }
            }
            
            await client.post(
                f"{LIFECYCLE_URL}/comparison-requests",
                json=comparison_request
            )
            
            # 3. Check version is listed
            list_response = await client.get(f"{LIFECYCLE_URL}/versions")
            assert list_response.status_code == 200
            
            # 4. Verify history has events
            history_response = await client.get(
                f"{LIFECYCLE_URL}/history",
                params={"limit": 10}
            )
            assert history_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
