"""Tests for AIOrchestration_V1"""

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import Orchestrator, AITask, MockProvider, TaskStatus
from src.provider_router import ProviderRouter, RoutingStrategy
from src.fallback_chain import FallbackChain, FallbackConfig


class TestOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        orch = Orchestrator()
        orch.register_provider(MockProvider("mock", "gpt-4"))
        return orch

    @pytest.mark.asyncio
    async def test_execute_task(self, orchestrator):
        task = AITask(
            task_id="test-1",
            prompt="Hello world",
            model="gpt-4"
        )

        result = await orchestrator.execute(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_batch_execute(self, orchestrator):
        tasks = [
            AITask(task_id=f"batch-{i}", prompt=f"Prompt {i}", model="gpt-4")
            for i in range(3)
        ]

        results = await orchestrator.execute_batch(tasks)

        assert len(results) == 3
        assert all(r.status == TaskStatus.COMPLETED for r in results)

    def test_metrics(self, orchestrator):
        metrics = orchestrator.get_metrics()

        assert "total_tasks" in metrics
        assert "success_rate" in metrics


class TestProviderRouter:
    @pytest.fixture
    def router(self):
        r = ProviderRouter(strategy=RoutingStrategy.ROUND_ROBIN)
        r.register_provider("openai")
        r.register_provider("anthropic")
        return r

    def test_round_robin(self, router):
        first = router.select_provider()
        second = router.select_provider()

        assert first != second

    def test_exclude_provider(self, router):
        selected = router.select_provider(exclude=["openai"])

        assert selected == "anthropic"

    def test_mark_unhealthy(self, router):
        router.mark_unhealthy("openai")

        selected = router.select_provider()
        assert selected == "anthropic"

    def test_record_success(self, router):
        router.record_success("openai", latency_ms=100, cost=0.01)

        stats = router.get_stats("openai")
        assert stats.successful_requests == 1
        assert stats.avg_latency == 100


class TestFallbackChain:
    @pytest.fixture
    def chain(self):
        config = FallbackConfig(max_retries=2, retry_delay_seconds=0.1)
        c = FallbackChain(config)

        async def success_handler(prompt):
            return f"Success: {prompt}"

        async def fail_handler(prompt):
            raise Exception("Provider failed")

        c.add_provider("primary", success_handler, priority=0)
        c.add_provider("backup", success_handler, priority=1)

        return c

    @pytest.mark.asyncio
    async def test_execute_success(self, chain):
        result, provider = await chain.execute("test prompt")

        assert "Success" in result
        assert provider == "primary"

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        config = FallbackConfig(max_retries=1, retry_delay_seconds=0)
        chain = FallbackChain(config)

        async def fail_handler(prompt):
            raise Exception("Failed")

        async def success_handler(prompt):
            return "Backup success"

        chain.add_provider("primary", fail_handler, priority=0)
        chain.add_provider("backup", success_handler, priority=1)

        result, provider = await chain.execute("test")

        assert result == "Backup success"
        assert provider == "backup"

    def test_chain_status(self, chain):
        status = chain.get_chain_status()

        assert "primary" in status
        assert "backup" in status
        assert status["primary"]["available"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
