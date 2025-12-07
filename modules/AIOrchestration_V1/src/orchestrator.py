"""
AIOrchestration_V1 - Orchestrator

AI request orchestration with routing and execution.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AITask:
    """AI processing task"""
    task_id: str
    prompt: str
    model: str
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "model": self.model,
            "priority": self.priority.value,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class AIResult:
    """AI task result"""
    task_id: str
    status: TaskStatus
    output: Optional[str] = None
    error: Optional[str] = None
    provider: Optional[str] = None
    latency_ms: float = 0
    tokens_used: int = 0
    cost: float = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output[:200] + "..." if self.output and len(self.output) > 200 else self.output,
            "error": self.error,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "timestamp": self.timestamp.isoformat(),
        }


class AIProvider:
    """Base AI provider interface"""

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self.is_healthy = True

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from AI"""
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check provider health"""
        return self.is_healthy


class MockProvider(AIProvider):
    """Mock provider for testing"""

    async def generate(self, prompt: str, **kwargs) -> str:
        await asyncio.sleep(0.1)  # Simulate latency
        return f"Mock response for: {prompt[:50]}..."


class Orchestrator:
    """
    AI request orchestrator.

    Features:
    - Task queue management
    - Provider selection
    - Request execution
    - Result aggregation
    """

    def __init__(
        self,
        default_timeout: int = 30,
        max_concurrent: int = 10,
    ):
        self.default_timeout = default_timeout
        self.max_concurrent = max_concurrent

        self._providers: Dict[str, AIProvider] = {}
        self._pending_tasks: List[AITask] = []
        self._results: Dict[str, AIResult] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Metrics
        self._total_tasks = 0
        self._successful_tasks = 0
        self._failed_tasks = 0
        self._total_latency = 0.0

    def register_provider(self, provider: AIProvider):
        """Register AI provider"""
        self._providers[provider.name] = provider
        logger.info(f"Registered provider: {provider.name}")

    async def execute(self, task: AITask) -> AIResult:
        """Execute AI task"""
        import time
        import uuid

        self._total_tasks += 1

        if not task.task_id:
            task.task_id = str(uuid.uuid4())

        # Select provider
        provider = self._select_provider(task.model)

        if not provider:
            result = AIResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error="No available provider",
            )
            self._failed_tasks += 1
            return result

        start_time = time.time()

        try:
            async with self._semaphore:
                # Execute with timeout
                output = await asyncio.wait_for(
                    provider.generate(task.prompt),
                    timeout=task.timeout_seconds,
                )

            latency = (time.time() - start_time) * 1000

            result = AIResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                provider=provider.name,
                latency_ms=latency,
            )

            self._successful_tasks += 1
            self._total_latency += latency

        except asyncio.TimeoutError:
            result = AIResult(
                task_id=task.task_id,
                status=TaskStatus.TIMEOUT,
                error=f"Timeout after {task.timeout_seconds}s",
                provider=provider.name,
                latency_ms=(time.time() - start_time) * 1000,
            )
            self._failed_tasks += 1

        except Exception as e:
            result = AIResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                provider=provider.name,
                latency_ms=(time.time() - start_time) * 1000,
            )
            self._failed_tasks += 1

        self._results[task.task_id] = result
        return result

    def _select_provider(self, model: str) -> Optional[AIProvider]:
        """Select provider for model"""
        for provider in self._providers.values():
            if provider.is_healthy and provider.model == model:
                return provider

        # Fallback to any healthy provider
        for provider in self._providers.values():
            if provider.is_healthy:
                return provider

        return None

    async def execute_batch(
        self,
        tasks: List[AITask],
    ) -> List[AIResult]:
        """Execute multiple tasks"""
        results = await asyncio.gather(
            *[self.execute(task) for task in tasks],
            return_exceptions=True,
        )

        return [
            r if isinstance(r, AIResult) else AIResult(
                task_id="unknown",
                status=TaskStatus.FAILED,
                error=str(r),
            )
            for r in results
        ]

    def get_result(self, task_id: str) -> Optional[AIResult]:
        """Get result for task"""
        return self._results.get(task_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return {
            "total_tasks": self._total_tasks,
            "successful_tasks": self._successful_tasks,
            "failed_tasks": self._failed_tasks,
            "success_rate": self._successful_tasks / max(1, self._total_tasks),
            "avg_latency_ms": self._total_latency / max(1, self._successful_tasks),
            "providers": len(self._providers),
            "healthy_providers": sum(1 for p in self._providers.values() if p.is_healthy),
        }
