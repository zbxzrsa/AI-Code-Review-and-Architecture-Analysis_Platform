"""Tests for AIOrchestration_V2 Features"""

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_balancer import LoadBalancer, LoadBalancingStrategy, HealthStatus
from src.circuit_breaker import CircuitBreaker, CircuitConfig, CircuitState, CircuitOpenError


class TestLoadBalancer:
    @pytest.fixture
    def balancer(self):
        lb = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
        lb.add_endpoint("provider-1", weight=1, max_connections=10)
        lb.add_endpoint("provider-2", weight=2, max_connections=10)
        return lb

    def test_select_endpoint(self, balancer):
        endpoint = balancer.select_endpoint()
        assert endpoint in ["provider-1", "provider-2"]

    def test_least_connections(self, balancer):
        # Acquire connections on provider-1
        balancer.acquire_connection("provider-1")
        balancer.acquire_connection("provider-1")

        # Should select provider-2 (fewer connections)
        selected = balancer.select_endpoint()
        assert selected == "provider-2"

    def test_health_tracking(self, balancer):
        # Record failures
        for _ in range(10):
            balancer.release_connection("provider-1", 100, success=False)

        stats = balancer.get_stats()
        assert stats["endpoints"]["provider-1"]["health"] == "unhealthy"

    def test_capacity_limit(self, balancer):
        # Fill provider-1
        for _ in range(10):
            balancer.acquire_connection("provider-1")

        # Should not select provider-1 (no capacity)
        for _ in range(5):
            selected = balancer.select_endpoint()
            assert selected == "provider-2"


class TestCircuitBreaker:
    @pytest.fixture
    def breaker(self):
        config = CircuitConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        return CircuitBreaker(config)

    def test_initial_state(self, breaker):
        assert breaker.state == CircuitState.CLOSED

    def test_open_on_failures(self, breaker):
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

    def test_reject_when_open(self, breaker):
        for _ in range(3):
            breaker.record_failure()

        assert not breaker.allow_request()

    def test_half_open_after_timeout(self, breaker):
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        import time
        time.sleep(0.15)

        # Should transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN

    def test_close_on_successes(self, breaker):
        for _ in range(3):
            breaker.record_failure()

        import time
        time.sleep(0.15)

        # Record successes in half-open
        breaker.allow_request()
        breaker.record_success()
        breaker.allow_request()
        breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    def test_stats(self, breaker):
        breaker.record_success()
        breaker.record_failure()

        stats = breaker.get_stats()
        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 1

    @pytest.mark.asyncio
    async def test_execute_success(self, breaker):
        async def success_func():
            return "success"

        result = await breaker.execute(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_circuit_open(self, breaker):
        for _ in range(3):
            breaker.record_failure()

        async def func():
            return "result"

        with pytest.raises(CircuitOpenError):
            await breaker.execute(func)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
