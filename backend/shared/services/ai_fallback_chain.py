"""
AI Model Fallback Chain

Implements:
- Primary/Secondary/Tertiary model fallback
- Circuit breaker integration
- Token limit validation
- Cost tracking
- Response caching
"""

import asyncio
import logging
import hashlib
import time
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """AI model provider."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    LOCAL = "local"


class ModelStatus(str, Enum):
    """Model availability status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    provider: ModelProvider
    api_key: str
    endpoint: Optional[str] = None
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    timeout: float = 30.0
    priority: int = 0  # Lower = higher priority


@dataclass
class ModelHealth:
    """Model health status."""
    model: str
    status: ModelStatus
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    avg_latency_ms: float = 0
    error_rate: float = 0


@dataclass
class TokenUsage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0


@dataclass
class ModelResponse:
    """Model API response."""
    content: str
    model: str
    provider: ModelProvider
    latency_ms: float
    tokens: TokenUsage
    from_cache: bool = False
    fallback_used: bool = False


class TokenLimitError(Exception):
    """Token limit exceeded."""
    pass


class AllModelsFailedError(Exception):
    """All models in chain failed."""
    pass


class AIFallbackChain:
    """
    AI model fallback chain with circuit breakers.
    
    Features:
    - Automatic failover between models
    - Circuit breaker pattern
    - Token validation before API calls
    - Cost tracking
    - Response caching
    """
    
    # Token limits by model
    TOKEN_LIMITS = {
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-2": 100000,
    }
    
    # Approximate tokens per character
    CHARS_PER_TOKEN = 4
    
    def __init__(
        self,
        models: List[ModelConfig],
        cache_ttl: int = 3600,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        max_concurrent: int = 10,
    ):
        self.models = sorted(models, key=lambda m: m.priority)
        self.cache_ttl = cache_ttl
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        # Circuit breaker state
        self._health: Dict[str, ModelHealth] = {
            m.name: ModelHealth(model=m.name, status=ModelStatus.HEALTHY)
            for m in models
        }
        
        # Response cache
        self._cache: Dict[str, tuple] = {}  # (response, expiry)
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Cost tracking
        self._total_cost: float = 0
        self._cost_by_model: Dict[str, float] = {}
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.CHARS_PER_TOKEN + 1
    
    def validate_token_limit(
        self,
        prompt: str,
        model: str,
        max_output_tokens: int = 1024,
    ) -> bool:
        """Validate prompt doesn't exceed model's token limit."""
        input_tokens = self.estimate_tokens(prompt)
        total_required = input_tokens + max_output_tokens
        
        limit = self.TOKEN_LIMITS.get(model, 4096)
        
        if total_required > limit:
            raise TokenLimitError(
                f"Estimated tokens ({total_required}) exceed model limit ({limit}). "
                f"Input: {input_tokens}, Output: {max_output_tokens}"
            )
        
        return True
    
    def _get_cache_key(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate cache key."""
        content = f"{system or ''}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _check_cache(self, key: str) -> Optional[ModelResponse]:
        """Check response cache."""
        if key in self._cache:
            response, expiry = self._cache[key]
            if datetime.now(timezone.utc) < expiry:
                response.from_cache = True
                return response
            else:
                del self._cache[key]
        return None
    
    def _update_cache(self, key: str, response: ModelResponse):
        """Update response cache."""
        expiry = datetime.now(timezone.utc) + timedelta(seconds=self.cache_ttl)
        self._cache[key] = (response, expiry)
    
    def _is_model_available(self, model: str) -> bool:
        """Check if model is available (circuit breaker)."""
        health = self._health.get(model)
        if not health:
            return False
        
        if health.status == ModelStatus.HEALTHY:
            return True
        
        if health.status == ModelStatus.UNHEALTHY:
            # Check if recovery timeout has passed
            if health.last_failure:
                recovery_time = health.last_failure + timedelta(seconds=self.recovery_timeout)
                if datetime.now(timezone.utc) >= recovery_time:
                    # Allow a test request
                    return True
            return False
        
        # Degraded - allow with caution
        return True
    
    def _record_success(self, model: str, latency_ms: float):
        """Record successful request."""
        health = self._health[model]
        health.consecutive_successes += 1
        health.consecutive_failures = 0
        health.last_success = datetime.now(timezone.utc)
        
        # Update average latency
        if health.avg_latency_ms == 0:
            health.avg_latency_ms = latency_ms
        else:
            health.avg_latency_ms = (health.avg_latency_ms * 0.9) + (latency_ms * 0.1)
        
        # Recover from degraded/unhealthy
        if health.consecutive_successes >= 3:
            health.status = ModelStatus.HEALTHY
    
    def _record_failure(self, model: str, error: str):  # noqa: ARG002 - error for future logging
        """Record failed request."""
        health = self._health[model]
        health.consecutive_failures += 1
        health.consecutive_successes = 0
        health.last_failure = datetime.now(timezone.utc)
        
        # Update status based on failures
        if health.consecutive_failures >= self.failure_threshold:
            health.status = ModelStatus.UNHEALTHY
            logger.warning(f"Model {model} marked unhealthy after {health.consecutive_failures} failures")
        elif health.consecutive_failures >= 2:
            health.status = ModelStatus.DEGRADED
    
    def _calculate_cost(
        self,
        model_config: ModelConfig,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost for API call."""
        input_cost = (input_tokens / 1000) * model_config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model_config.cost_per_1k_output
        return input_cost + output_cost
    
    async def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_output_tokens: int = 1024,
        use_cache: bool = True,
    ) -> ModelResponse:
        """
        Call AI model with automatic fallback.
        
        Tries models in priority order, using circuit breaker
        to skip unhealthy models.
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(prompt, system_prompt)
            cached = self._check_cache(cache_key)
            if cached:
                logger.debug("Cache hit for prompt")
                return cached
        
        errors = []
        
        async with self._semaphore:  # Rate limiting
            for model_config in self.models:
                # Check circuit breaker
                if not self._is_model_available(model_config.name):
                    logger.debug(f"Skipping unavailable model {model_config.name}")
                    continue
                
                # Validate token limit
                try:
                    self.validate_token_limit(
                        prompt,
                        model_config.name,
                        max_output_tokens,
                    )
                except TokenLimitError as e:
                    logger.warning(f"Token limit exceeded for {model_config.name}: {e}")
                    continue
                
                # Try calling model
                try:
                    response = await self._call_model(
                        model_config,
                        prompt,
                        system_prompt,
                        max_output_tokens,
                    )
                    
                    # Cache response
                    if use_cache:
                        self._update_cache(cache_key, response)
                    
                    return response
                    
                except Exception as e:
                    error_msg = f"{model_config.name}: {str(e)}"
                    errors.append(error_msg)
                    self._record_failure(model_config.name, str(e))
                    logger.warning(f"Model {model_config.name} failed: {e}")
                    continue
        
        # All models failed
        raise AllModelsFailedError(
            f"All {len(self.models)} models failed. Errors: {'; '.join(errors)}"
        )
    
    async def _call_model(
        self,
        config: ModelConfig,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
    ) -> ModelResponse:
        """Call specific model API."""
        start_time = time.time()
        
        # In production, call actual API
        # This is a mock implementation
        await asyncio.sleep(0.1)  # Simulate API call
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Mock response
        input_tokens = self.estimate_tokens(prompt)
        output_tokens = 100  # Mock
        
        cost = self._calculate_cost(config, input_tokens, output_tokens)
        self._total_cost += cost
        self._cost_by_model[config.name] = self._cost_by_model.get(config.name, 0) + cost
        
        self._record_success(config.name, latency_ms)
        
        return ModelResponse(
            content=f"Mock response from {config.name}",
            model=config.name,
            provider=config.provider,
            latency_ms=latency_ms,
            tokens=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                estimated_cost=cost,
            ),
            fallback_used=config.priority > 0,
        )
    
    def get_health_status(self) -> Dict[str, ModelHealth]:
        """Get health status of all models."""
        return self._health.copy()
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "total_cost": self._total_cost,
            "cost_by_model": self._cost_by_model.copy(),
        }
    
    def clear_cache(self):
        """Clear response cache."""
        self._cache.clear()
    
    def reset_circuit_breakers(self):
        """Reset all circuit breakers to healthy."""
        for model in self._health:
            self._health[model] = ModelHealth(
                model=model,
                status=ModelStatus.HEALTHY,
            )


def create_default_fallback_chain(
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
) -> AIFallbackChain:
    """Create default fallback chain with common models."""
    models = []
    
    if openai_key:
        models.extend([
            ModelConfig(
                name="gpt-4-turbo",
                provider=ModelProvider.OPENAI,
                api_key=openai_key,
                max_tokens=128000,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.03,
                priority=0,
            ),
            ModelConfig(
                name="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                api_key=openai_key,
                max_tokens=16385,
                cost_per_1k_input=0.0005,
                cost_per_1k_output=0.0015,
                priority=2,
            ),
        ])
    
    if anthropic_key:
        models.append(ModelConfig(
            name="claude-3-sonnet",
            provider=ModelProvider.ANTHROPIC,
            api_key=anthropic_key,
            max_tokens=200000,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            priority=1,
        ))
    
    if not models:
        raise ValueError("At least one API key required")
    
    return AIFallbackChain(models)
