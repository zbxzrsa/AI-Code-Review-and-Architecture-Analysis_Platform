"""
AIOrchestration_V2 - Fallback Chain

Production fallback chain with retry and exponential backoff.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, TypeVar


T = TypeVar('T')


@dataclass
class FallbackResult:
    """Result from fallback chain execution."""
    success: bool
    result: Any
    provider_used: str
    attempts: int
    total_time_ms: float
    errors: List[str]


class FallbackChain:
    """
    Production Fallback Chain.

    Features:
    - Ordered fallback providers
    - Exponential backoff
    - Retry logic
    - Error tracking
    """

    def __init__(
        self,
        providers: List[str],
        max_retries: int = 3,
        initial_backoff_ms: float = 100,
        max_backoff_ms: float = 5000,
        backoff_multiplier: float = 2.0,
    ):
        self.providers = providers
        self.max_retries = max_retries
        self.initial_backoff_ms = initial_backoff_ms
        self.max_backoff_ms = max_backoff_ms
        self.backoff_multiplier = backoff_multiplier

        self._provider_handlers: Dict[str, Callable] = {}
        self._failed_providers: Dict[str, datetime] = {}

    def register_handler(self, provider: str, handler: Callable):
        """Register handler for provider."""
        self._provider_handlers[provider] = handler

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> FallbackResult:
        """Execute with fallback chain."""
        import time

        start_time = time.time()
        errors: List[str] = []
        attempts = 0

        for provider in self.providers:
            # Skip recently failed providers
            if self._is_provider_cooling_down(provider):
                continue

            handler = self._provider_handlers.get(provider)
            if not handler:
                continue

            backoff = self.initial_backoff_ms

            for retry in range(self.max_retries):
                attempts += 1

                try:
                    result = await self._execute_with_handler(
                        handler, func, *args, **kwargs
                    )

                    total_time = (time.time() - start_time) * 1000

                    return FallbackResult(
                        success=True,
                        result=result,
                        provider_used=provider,
                        attempts=attempts,
                        total_time_ms=total_time,
                        errors=errors,
                    )

                except Exception as e:
                    error_msg = f"{provider} (attempt {retry + 1}): {str(e)}"
                    errors.append(error_msg)

                    if retry < self.max_retries - 1:
                        await asyncio.sleep(backoff / 1000)
                        backoff = min(backoff * self.backoff_multiplier, self.max_backoff_ms)

            # Mark provider as failed
            self._failed_providers[provider] = datetime.now(timezone.utc)

        total_time = (time.time() - start_time) * 1000

        return FallbackResult(
            success=False,
            result=None,
            provider_used="none",
            attempts=attempts,
            total_time_ms=total_time,
            errors=errors,
        )

    async def _execute_with_handler(
        self,
        handler: Callable,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with handler context."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(func, *args, **kwargs)
        return handler(func, *args, **kwargs)

    def _is_provider_cooling_down(self, provider: str) -> bool:
        """Check if provider is in cooldown."""
        if provider not in self._failed_providers:
            return False

        failed_time = self._failed_providers[provider]
        cooldown_seconds = 30  # 30 second cooldown

        elapsed = (datetime.now(timezone.utc) - failed_time).total_seconds()

        if elapsed >= cooldown_seconds:
            del self._failed_providers[provider]
            return False

        return True

    def reset_provider(self, provider: str):
        """Reset provider cooldown."""
        if provider in self._failed_providers:
            del self._failed_providers[provider]

    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return [
            p for p in self.providers
            if not self._is_provider_cooling_down(p)
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get chain statistics."""
        return {
            "total_providers": len(self.providers),
            "available_providers": len(self.get_available_providers()),
            "cooling_down": len(self._failed_providers),
            "max_retries": self.max_retries,
        }


__all__ = ["FallbackResult", "FallbackChain"]
