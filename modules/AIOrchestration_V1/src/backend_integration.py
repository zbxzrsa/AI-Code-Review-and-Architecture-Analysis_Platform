"""
AIOrchestration_V1 - Backend Integration Bridge

Integrates with backend/services/ai-orchestrator implementations.
"""

import sys
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

# Add backend path for imports
_backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

# Import backend implementations
try:
    from services.ai_orchestrator.src.orchestrator import (
        AIOrchestrator as BackendOrchestrator,
        OrchestratorConfig as BackendConfig,
    )
    ORCHESTRATOR_BACKEND_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_BACKEND_AVAILABLE = False
    BackendOrchestrator = None

try:
    from shared.services.ai_client import (
        UnifiedAIClient,
        AIProvider,
        AIRequest,
        AIResponse,
    )
    AI_CLIENT_AVAILABLE = True
except ImportError:
    AI_CLIENT_AVAILABLE = False
    UnifiedAIClient = None


BACKEND_AVAILABLE = ORCHESTRATOR_BACKEND_AVAILABLE or AI_CLIENT_AVAILABLE


class IntegratedOrchestrator:
    """
    V1 AI Orchestrator with backend integration.
    """

    def __init__(
        self,
        default_provider: str = "openai",
        timeout_seconds: int = 30,
        use_backend: bool = True,
    ):
        self.default_provider = default_provider
        self.timeout_seconds = timeout_seconds
        self.use_backend = use_backend and ORCHESTRATOR_BACKEND_AVAILABLE

        if self.use_backend:
            config = BackendConfig(
                default_provider=default_provider,
                timeout=timeout_seconds,
            )
            self._backend = BackendOrchestrator(config)
        else:
            from .orchestrator import Orchestrator
            self._local = Orchestrator(default_timeout=timeout_seconds)

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute AI request."""
        if self.use_backend:
            result = await self._backend.execute(
                prompt=prompt,
                model=model,
                provider=provider or self.default_provider,
                **kwargs,
            )
            return result.__dict__ if hasattr(result, '__dict__') else result

        from .orchestrator import AITask
        task = AITask(
            task_id=f"task-{id(prompt)}",
            prompt=prompt,
            model=model or "gpt-4",
        )
        result = await self._local.execute(task)
        return {
            "task_id": result.task_id,
            "status": result.status.value,
            "result": result.result,
            "error": result.error,
        }

    async def execute_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute batch AI requests."""
        if self.use_backend:
            results = await self._backend.execute_batch(prompts, model, **kwargs)
            return [r.__dict__ if hasattr(r, '__dict__') else r for r in results]

        from .orchestrator import AITask
        tasks = [
            AITask(task_id=f"batch-{i}", prompt=p, model=model or "gpt-4")
            for i, p in enumerate(prompts)
        ]
        results = await self._local.execute_batch(tasks)
        return [
            {"task_id": r.task_id, "status": r.status.value, "result": r.result}
            for r in results
        ]


class IntegratedAIClient:
    """
    V1 Unified AI Client with backend integration.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and AI_CLIENT_AVAILABLE

        if self.use_backend:
            self._backend = UnifiedAIClient()
        else:
            self._providers: Dict[str, Any] = {}

    def register_provider(self, name: str, provider: Any):
        """Register an AI provider."""
        if self.use_backend:
            self._backend.register_provider(name, provider)
        else:
            self._providers[name] = provider

    async def complete(
        self,
        prompt: str,
        provider: str = "openai",
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Get completion from AI provider."""
        if self.use_backend:
            request = AIRequest(prompt=prompt, model=model)
            response = await self._backend.complete(provider, request)
            return response.content

        if provider in self._providers:
            return await self._providers[provider].complete(prompt, model, **kwargs)

        # Mock response for testing
        return f"Mock response for: {prompt[:50]}..."

    async def chat(
        self,
        messages: List[Dict[str, str]],
        provider: str = "openai",
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Chat with AI provider."""
        if self.use_backend:
            response = await self._backend.chat(provider, messages, model)
            return response.content

        if provider in self._providers:
            return await self._providers[provider].chat(messages, model, **kwargs)

        # Mock response
        last_msg = messages[-1].get("content", "") if messages else ""
        return f"Mock chat response for: {last_msg[:50]}..."


class IntegratedProviderRouter:
    """
    V1 Provider Router with backend integration.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and AI_CLIENT_AVAILABLE

        if not self.use_backend:
            from .provider_router import ProviderRouter
            self._local = ProviderRouter()

    def add_provider(self, name: str, config: Dict[str, Any]):
        """Add provider to router."""
        if self.use_backend:
            # Backend handles this internally
            pass
        else:
            self._local.register_provider(name, config)

    def select_provider(self, strategy: str = "round_robin") -> str:
        """Select provider based on strategy."""
        if self.use_backend:
            return "openai"  # Backend manages selection
        return self._local.select_provider(strategy)

    def update_stats(self, provider: str, latency: float, success: bool):
        """Update provider statistics."""
        if not self.use_backend:
            self._local.update_provider_stats(provider, latency, success)


class IntegratedFallbackChain:
    """
    V1 Fallback Chain with backend integration.
    """

    def __init__(self, providers: List[str], use_backend: bool = True):
        self.providers = providers
        self.use_backend = use_backend and AI_CLIENT_AVAILABLE

        if not self.use_backend:
            from .fallback_chain import FallbackChain
            self._local = FallbackChain(providers)

    async def execute_with_fallback(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute with fallback chain."""
        if self.use_backend:
            # Backend handles fallback internally
            return await func(*args, **kwargs)
        return await self._local.execute(func, *args, **kwargs)


# Factory functions
def get_orchestrator(use_backend: bool = True) -> IntegratedOrchestrator:
    """Get integrated orchestrator."""
    return IntegratedOrchestrator(use_backend=use_backend)


def get_ai_client(use_backend: bool = True) -> IntegratedAIClient:
    """Get integrated AI client."""
    return IntegratedAIClient(use_backend)


def get_provider_router(use_backend: bool = True) -> IntegratedProviderRouter:
    """Get integrated provider router."""
    return IntegratedProviderRouter(use_backend)


def get_fallback_chain(providers: List[str], use_backend: bool = True) -> IntegratedFallbackChain:
    """Get integrated fallback chain."""
    return IntegratedFallbackChain(providers, use_backend)


__all__ = [
    "BACKEND_AVAILABLE",
    "ORCHESTRATOR_BACKEND_AVAILABLE",
    "AI_CLIENT_AVAILABLE",
    "IntegratedOrchestrator",
    "IntegratedAIClient",
    "IntegratedProviderRouter",
    "IntegratedFallbackChain",
    "get_orchestrator",
    "get_ai_client",
    "get_provider_router",
    "get_fallback_chain",
]
