"""
Test version promotion workflow
Tests for experiment creation, evaluation, promotion, and access control
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json


class TestVersionPromotion:
    """Test version promotion workflow"""
    
    @pytest.fixture
    def admin_client(self, client: TestClient):
        """Login as admin and return authenticated client"""
        response = client.post("/api/auth/login", json={
            "email": "admin@example.com",
            "password": "test_password",
            "invitation_code": "ZBXzbx123"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        client.headers = {"Authorization": f"Bearer {token}"}
        return client
    
    @pytest.fixture
    def user_client(self, client: TestClient):
        """Login as regular user and return authenticated client"""
        response = client.post("/api/auth/login", json={
            "email": "user@example.com",
            "password": "test_password"
        })
        assert response.status_code == 200
        token = response.json()["access_token"]
        client.headers = {"Authorization": f"Bearer {token}"}
        return client
    
    def create_experiment(self, client: TestClient, metrics: dict) -> str:
        """Helper to create experiment with specific metrics"""
        # Create experiment
        response = client.post("/api/experiments", json={
            "name": f"test-experiment-{metrics.get('accuracy', 0)}",
            "config": {
                "model": "gpt-4-turbo",
                "temperature": 0.7,
                "prompt_template": "security_expert_v2"
            },
            "dataset_id": "test-dataset-100"
        })
        assert response.status_code == 201
        experiment_id = response.json()["id"]
        
        # Mock evaluation with specific metrics
        with patch('services.experiment_service.evaluate_experiment') as mock_eval:
            mock_eval.return_value = metrics
            response = client.post(f"/api/experiments/{experiment_id}/evaluate")
            assert response.status_code == 200
        
        return experiment_id
    
    def test_create_v1_experiment(self, admin_client):
        """Admin can create experiments in v1"""
        response = admin_client.post("/api/experiments", json={
            "name": "gpt-4-turbo-experiment",
            "config": {
                "model": "gpt-4-turbo",
                "temperature": 0.7,
                "prompt_template": "security_expert_v2"
            },
            "dataset_id": "test-dataset-100"
        })
        
        assert response.status_code == 201
        experiment = response.json()
        assert experiment["status"] == "pending"
        assert experiment["config"]["model"] == "gpt-4-turbo"
    
    def test_start_experiment(self, admin_client):
        """Admin can start an experiment"""
        # Create experiment
        create_response = admin_client.post("/api/experiments", json={
            "name": "start-test-experiment",
            "config": {
                "model": "gpt-4",
                "temperature": 0.5,
                "prompt_template": "code_quality_v1"
            },
            "dataset_id": "test-dataset-100"
        })
        experiment_id = create_response.json()["id"]
        
        # Start experiment
        response = admin_client.post(f"/api/experiments/{experiment_id}/start")
        
        assert response.status_code == 200
        assert response.json()["status"] == "running"
    
    def test_promotion_requires_passing_metrics(self, admin_client):
        """Version promotion requires passing OPA gates"""
        # Create experiment with poor metrics
        experiment_id = self.create_experiment(admin_client, {
            "accuracy": 0.70,  # Below threshold of 0.85
            "error_rate": 0.10  # Above threshold of 0.05
        })
        
        # Attempt promotion
        response = admin_client.post(
            f"/api/experiments/{experiment_id}/promote"
        )
        
        assert response.status_code == 403
        assert "does not meet promotion criteria" in response.json()["detail"]
    
    def test_promotion_fails_high_error_rate(self, admin_client):
        """Promotion fails when error rate exceeds threshold"""
        experiment_id = self.create_experiment(admin_client, {
            "accuracy": 0.90,  # Good accuracy
            "error_rate": 0.08  # Above threshold of 0.05
        })
        
        response = admin_client.post(
            f"/api/experiments/{experiment_id}/promote"
        )
        
        assert response.status_code == 403
        assert "error_rate" in response.json()["detail"].lower()
    
    def test_promotion_fails_high_cost_increase(self, admin_client):
        """Promotion fails when cost increase exceeds 20%"""
        experiment_id = self.create_experiment(admin_client, {
            "accuracy": 0.90,
            "error_rate": 0.02,
            "cost_increase": 0.25  # 25% increase, above 20% threshold
        })
        
        response = admin_client.post(
            f"/api/experiments/{experiment_id}/promote"
        )
        
        assert response.status_code == 403
        assert "cost" in response.json()["detail"].lower()
    
    def test_successful_promotion(self, admin_client):
        """Experiment with good metrics can be promoted"""
        experiment_id = self.create_experiment(admin_client, {
            "accuracy": 0.90,
            "error_rate": 0.02,
            "latency_p95": 2.1,
            "cost_per_analysis": 0.04
        })
        
        response = admin_client.post(
            f"/api/experiments/{experiment_id}/promote"
        )
        
        assert response.status_code == 200
        assert response.json()["new_version"] == "v2"
        
        # Verify audit log
        audit_response = admin_client.get(
            "/api/audit?entity=version&action=promoted"
        )
        assert audit_response.status_code == 200
        audit_logs = audit_response.json()
        assert len(audit_logs) > 0
        assert any(log["entity_id"] == experiment_id for log in audit_logs)
    
    def test_user_cannot_access_v1(self, user_client):
        """Regular users cannot access v1"""
        # Try to analyze with v1
        response = user_client.post("/api/analyze", json={
            "code": "def test(): pass",
            "version": "v1"
        })
        
        assert response.status_code == 403
        assert "not authorized" in response.json()["detail"].lower()
    
    def test_user_can_access_v2(self, user_client):
        """Regular users can access v2"""
        response = user_client.post("/api/analyze", json={
            "code": "def test(): pass",
            "version": "v2"
        })
        
        # Should succeed (200) or be accepted (202)
        assert response.status_code in [200, 202]
    
    def test_admin_can_access_all_versions(self, admin_client):
        """Admin can access all versions"""
        for version in ["v1", "v2", "v3"]:
            response = admin_client.post("/api/analyze", json={
                "code": "def test(): pass",
                "version": version
            })
            assert response.status_code in [200, 202], f"Failed for {version}"
    
    def test_quarantine_experiment(self, admin_client):
        """Admin can quarantine a failed experiment"""
        # Create and fail an experiment
        experiment_id = self.create_experiment(admin_client, {
            "accuracy": 0.50,
            "error_rate": 0.20
        })
        
        response = admin_client.post(
            f"/api/experiments/{experiment_id}/quarantine",
            json={"reason": "Poor performance in testing"}
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "quarantined"
        
        # Verify it's in v3
        get_response = admin_client.get(f"/api/experiments/{experiment_id}")
        assert get_response.json()["version"] == "v3"
    
    def test_rollback_promotion(self, admin_client):
        """Admin can rollback a promoted version"""
        # First promote an experiment
        experiment_id = self.create_experiment(admin_client, {
            "accuracy": 0.90,
            "error_rate": 0.02
        })
        
        promote_response = admin_client.post(
            f"/api/experiments/{experiment_id}/promote"
        )
        assert promote_response.status_code == 200
        
        # Now rollback
        rollback_response = admin_client.post(
            f"/api/versions/{experiment_id}/rollback"
        )
        
        assert rollback_response.status_code == 200
        assert "rolled back" in rollback_response.json()["message"].lower()


class TestAIOutputQuality:
    """Test AI model output quality"""
    
    @pytest.fixture
    def security_expert(self):
        """Mock security expert analyzer"""
        from services.ai_orchestrator import SecurityExpert
        return SecurityExpert()
    
    @pytest.fixture
    def code_ai_v2(self):
        """Mock v2 code analyzer"""
        from services.ai_orchestrator import CodeAnalyzerV2
        return CodeAnalyzerV2()
    
    def test_security_expert_detects_vulnerabilities(self, security_expert):
        """Security expert should detect known vulnerabilities"""
        test_cases = [
            ("password = 'hardcoded123'", ["hardcoded_credential"]),
            ("eval(user_input)", ["code_injection"]),
            ("os.system(user_command)", ["command_injection"]),
            ("cursor.execute(f'SELECT * FROM users WHERE id={user_id}')", ["sql_injection"]),
            ("pickle.loads(untrusted_data)", ["insecure_deserialization"]),
        ]
        
        for code, expected_issues in test_cases:
            result = security_expert.analyze(code)
            detected = [issue["type"] for issue in result.issues]
            
            for expected in expected_issues:
                assert expected in detected, \
                    f"Failed to detect {expected} in: {code}"
    
    def test_no_false_positives_on_clean_code(self, security_expert):
        """Security expert should not flag clean code"""
        clean_code = """
        import hashlib
        import secrets
        
        def hash_password(password: str) -> str:
            salt = secrets.token_hex(16)
            return hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt.encode(),
                100000
            ).hex()
        """
        
        result = security_expert.analyze(clean_code)
        
        # Should have no high-severity issues
        high_severity = [i for i in result.issues if i["severity"] in ["error", "critical"]]
        assert len(high_severity) == 0, f"False positives: {high_severity}"
    
    def test_v2_consistency(self, code_ai_v2):
        """v2 should produce consistent results"""
        sample_code = """
        def calculate_sum(numbers):
            total = 0
            for n in numbers:
                total += n
            return total
        """
        
        results = []
        for _ in range(10):
            result = code_ai_v2.analyze(sample_code)
            results.append(set(i["type"] for i in result.issues))
        
        # Calculate consistency
        if len(results) > 1:
            intersection = set.intersection(*results)
            union = set.union(*results)
            consistency = len(intersection) / len(union) if union else 1.0
            
            assert consistency > 0.9, f"v2 consistency too low: {consistency}"
    
    def test_ai_provides_actionable_fixes(self, code_ai_v2):
        """AI should provide actionable fix suggestions"""
        vulnerable_code = "password = 'admin123'"
        
        result = code_ai_v2.analyze(vulnerable_code)
        
        # Find the hardcoded credential issue
        credential_issues = [i for i in result.issues if "credential" in i["type"].lower()]
        assert len(credential_issues) > 0
        
        # Check that fix is provided
        for issue in credential_issues:
            assert issue.get("fix") is not None, "No fix provided"
            assert len(issue["fix"]) > 0, "Empty fix"
            # Fix should suggest environment variable or secrets manager
            assert any(keyword in issue["fix"].lower() 
                      for keyword in ["env", "environment", "secret", "vault", "config"])


class TestChaosEngineering:
    """Chaos engineering tests for resilience"""
    
    @pytest.fixture
    def client(self):
        """Test client"""
        from main import app
        return TestClient(app)
    
    @pytest.mark.chaos
    def test_v2_resilience_under_db_latency(self, client):
        """v2 should handle database latency gracefully"""
        # Inject 500ms latency to database
        with patch('database.get_connection') as mock_db:
            import time
            original_execute = mock_db.return_value.execute
            
            def slow_execute(*args, **kwargs):
                time.sleep(0.5)  # 500ms latency
                return original_execute(*args, **kwargs)
            
            mock_db.return_value.execute = slow_execute
            
            response = client.post("/api/analyze", json={
                "code": "def test(): pass",
                "version": "v2"
            })
            
            # Should still respond (possibly with cached result or graceful degradation)
            assert response.status_code in [200, 202, 503]
            
            if response.status_code == 503:
                # Should provide helpful error message
                assert "temporarily unavailable" in response.json()["detail"].lower()
    
    @pytest.mark.chaos
    def test_failover_to_backup_model(self, client):
        """System should failover when primary model fails"""
        # Mark primary model as unhealthy
        with patch('services.provider_health.is_provider_healthy') as mock_health:
            mock_health.side_effect = lambda provider: provider != "openai"
            
            response = client.post("/api/analyze", json={
                "code": "def test(): pass",
                "provider": "openai"  # User requested OpenAI
            })
            
            # Should succeed with fallback model
            assert response.status_code in [200, 202]
            result = response.json()
            
            # Should indicate fallback was used
            if "metadata" in result:
                assert result["metadata"]["provider"] != "openai"
                assert "fallback" in result["metadata"]
    
    @pytest.mark.chaos
    def test_circuit_breaker_activation(self, client):
        """Circuit breaker should activate after repeated failures"""
        failure_count = 0
        
        with patch('services.ai_client.call_model') as mock_call:
            def failing_call(*args, **kwargs):
                nonlocal failure_count
                failure_count += 1
                if failure_count <= 5:
                    raise Exception("Model unavailable")
                return {"result": "success"}
            
            mock_call.side_effect = failing_call
            
            # Make multiple requests to trigger circuit breaker
            for i in range(10):
                response = client.post("/api/analyze", json={
                    "code": "def test(): pass"
                })
            
            # After circuit breaker activates, should fail fast
            # (not actually calling the model)
            assert failure_count <= 6  # Circuit should open after ~5 failures
    
    @pytest.mark.chaos
    def test_graceful_degradation_under_load(self, client):
        """System should degrade gracefully under high load"""
        import concurrent.futures
        
        def make_request():
            return client.post("/api/analyze", json={
                "code": "def test(): pass"
            })
        
        # Simulate 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # At least 80% should succeed
        success_count = sum(1 for r in responses if r.status_code in [200, 202])
        success_rate = success_count / len(responses)
        
        assert success_rate >= 0.8, f"Success rate too low: {success_rate}"
        
        # Failed requests should have proper error messages
        for response in responses:
            if response.status_code not in [200, 202]:
                assert response.status_code in [429, 503]  # Rate limit or unavailable
                assert "detail" in response.json()
    
    @pytest.mark.chaos
    def test_recovery_after_failure(self, client):
        """System should recover after transient failures"""
        call_count = 0
        
        with patch('services.ai_client.call_model') as mock_call:
            def intermittent_failure(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 3:
                    raise Exception("Transient failure")
                return {"result": "success"}
            
            mock_call.side_effect = intermittent_failure
            
            # First few requests may fail
            for _ in range(3):
                client.post("/api/analyze", json={"code": "def test(): pass"})
            
            # System should recover
            response = client.post("/api/analyze", json={
                "code": "def test(): pass"
            })
            
            # Should eventually succeed
            assert response.status_code in [200, 202]
