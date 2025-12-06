"""
Provider-Specific Circuit Breakers

Independent circuit breaker instances for each AI service provider.

Features:
- Per-provider isolation
- Provider-specific configurations
- Automatic fallback chain
- Health-aware routing
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TypeVar
from enum import Enum

from .enhanced_circuit_breaker import (
    EnhancedCircuitBreaker,
    DynamicThresholdConfig,
    CircuitState,
    CircuitOpenError,
    FailureType,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ProviderType(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    CUSTOM = "custom"


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""
    provider_type: ProviderType
    name: str
    endpoint: str
    
    # Circuit breaker config
    failure_rate_threshold: float = 0.50
    window_seconds: int = 30
    recovery_timeout_seconds: float = 30.0
    minimum_requests: int = 10
    
    # Request config
    timeout_seconds: float = 30.0
    max_retries: int = 3
    
    # Priority (lower = higher priority)
    priority: int = 1
    
    # Weight for load balancing (0-100)
    weight: int = 100
    
    # Is this provider enabled?
    enabled: bool = True


@dataclass
class ProviderHealth:
    """Health status of a provider."""
    provider_name: str
    is_healthy: bool
    circuit_state: CircuitState
    failure_rate: float
    avg_latency_ms: float
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "is_healthy": self.is_healthy,
            "circuit_state": self.circuit_state.value,
            "failure_rate": round(self.failure_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "consecutive_failures": self.consecutive_failures,
        }


class ProviderCircuitBreakerManager:
    """
    Manages circuit breakers for all AI providers.
    
    Features:
    - Independent circuit breaker per provider
    - Automatic fallback routing
    - Health-based provider selection
    - Real-time monitoring dashboard
    
    Target metrics:
    - Fault isolation rate: 99.9%
    - Interception delay: < 100ms
    
    Usage:
        manager = ProviderCircuitBreakerManager()
        manager.register_provider(ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="openai_primary",
            endpoint="https://api.openai.com/v1"
        ))
        
        result = await manager.execute_with_fallback(
            call_ai_provider,
            prompt="Hello"
        )
    """
    
    def __init__(
        self,
        global_config: Optional[DynamicThresholdConfig] = None,
        on_state_change: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
    ):
        self._global_config = global_config or DynamicThresholdConfig()
        self._on_state_change = on_state_change
        self._on_failure = on_failure
        
        self._providers: Dict[str, ProviderConfig] = {}
        self._breakers: Dict[str, EnhancedCircuitBreaker] = {}
        self._fallback_chains: Dict[ProviderType, List[str]] = {}
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_requests = 0
        self._isolated_failures = 0
        self._fallback_invocations = 0
    
    def register_provider(
        self,
        config: ProviderConfig,
        fallback_providers: Optional[List[str]] = None
    ):
        """
        Register a provider with its circuit breaker.
        
        Args:
            config: Provider configuration
            fallback_providers: List of provider names to use as fallbacks
        """
        # Create circuit breaker config from provider config
        cb_config = DynamicThresholdConfig(
            failure_rate_threshold=config.failure_rate_threshold,
            window_seconds=config.window_seconds,
            recovery_timeout_seconds=config.recovery_timeout_seconds,
            minimum_requests=config.minimum_requests,
        )
        
        # Create circuit breaker
        breaker = EnhancedCircuitBreaker(
            name=config.name,
            config=cb_config,
            on_state_change=self._handle_state_change,
            on_failure=self._handle_failure,
        )
        
        self._providers[config.name] = config
        self._breakers[config.name] = breaker
        
        # Set up fallback chain
        if fallback_providers:
            self._fallback_chains[config.name] = fallback_providers
        
        logger.info(f"Registered provider: {config.name} ({config.provider_type.value})")
    
    def unregister_provider(self, provider_name: str):
        """Remove a provider and its circuit breaker."""
        if provider_name in self._providers:
            del self._providers[provider_name]
        if provider_name in self._breakers:
            del self._breakers[provider_name]
        if provider_name in self._fallback_chains:
            del self._fallback_chains[provider_name]
        
        logger.info(f"Unregistered provider: {provider_name}")
    
    async def _handle_state_change(
        self,
        provider_name: str,
        old_state: CircuitState,
        new_state: CircuitState
    ):
        """Handle circuit breaker state changes."""
        logger.info(
            f"Provider {provider_name}: Circuit state {old_state.value} -> {new_state.value}"
        )
        
        if new_state == CircuitState.OPEN:
            self._isolated_failures += 1
        
        if self._on_state_change:
            if asyncio.iscoroutinefunction(self._on_state_change):
                await self._on_state_change(provider_name, old_state, new_state)
            else:
                self._on_state_change(provider_name, old_state, new_state)
    
    async def _handle_failure(
        self,
        provider_name: str,
        failure_type: FailureType,
        error_message: Optional[str]
    ):
        """Handle provider failures."""
        logger.warning(
            f"Provider {provider_name}: Failure - {failure_type.value}: {error_message}"
        )
        
        if self._on_failure:
            if asyncio.iscoroutinefunction(self._on_failure):
                await self._on_failure(provider_name, failure_type, error_message)
            else:
                self._on_failure(provider_name, failure_type, error_message)
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers with closed or half-open circuits."""
        available = []
        for name, breaker in self._breakers.items():
            config = self._providers.get(name)
            if config and config.enabled and breaker.state != CircuitState.OPEN:
                available.append(name)
        return available
    
    def get_healthy_providers(self) -> List[str]:
        """Get list of providers with closed circuits and low failure rate."""
        healthy = []
        for name, breaker in self._breakers.items():
            config = self._providers.get(name)
            if (config and config.enabled and 
                breaker.state == CircuitState.CLOSED and
                breaker.metrics.current_failure_rate < 0.1):
                healthy.append(name)
        return healthy
    
    def select_provider(self, preferred: Optional[str] = None) -> Optional[str]:
        """
        Select the best provider based on health and priority.
        
        Args:
            preferred: Preferred provider name (used if healthy)
            
        Returns:
            Selected provider name or None if all unavailable
        """
        # Try preferred provider first
        if preferred and preferred in self._breakers:
            breaker = self._breakers[preferred]
            config = self._providers.get(preferred)
            if config and config.enabled and breaker.state != CircuitState.OPEN:
                return preferred
        
        # Get available providers sorted by priority
        available = []
        for name in self.get_available_providers():
            config = self._providers[name]
            breaker = self._breakers[name]
            
            # Score based on health and priority
            score = config.priority * 100
            if breaker.state == CircuitState.CLOSED:
                score -= 50  # Prefer closed circuits
            score += breaker.metrics.current_failure_rate * 100
            
            available.append((name, score))
        
        if not available:
            return None
        
        # Return provider with lowest score
        available.sort(key=lambda x: x[1])
        return available[0][0]
    
    async def execute(
        self,
        provider_name: str,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Execute function through a specific provider's circuit breaker.
        
        Args:
            provider_name: Name of the provider to use
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            ValueError: If provider not found
        """
        if provider_name not in self._breakers:
            raise ValueError(f"Provider not found: {provider_name}")
        
        async with self._lock:
            self._total_requests += 1
        
        config = self._providers[provider_name]
        breaker = self._breakers[provider_name]
        
        return await breaker.execute(
            func,
            *args,
            timeout_seconds=config.timeout_seconds,
            **kwargs
        )
    
    async def execute_with_fallback(
        self,
        func: Callable[..., T],
        *args,
        preferred_provider: Optional[str] = None,
        fallback_chain: Optional[List[str]] = None,
        **kwargs
    ) -> T:
        """
        Execute function with automatic fallback to other providers.
        
        Args:
            func: Function to execute (receives provider_name as first arg)
            *args, **kwargs: Additional function arguments
            preferred_provider: First provider to try
            fallback_chain: Custom fallback chain (overrides default)
            
        Returns:
            Function result from first successful provider
            
        Raises:
            CircuitOpenError: If all providers are unavailable
        """
        async with self._lock:
            self._total_requests += 1
        
        # Build provider chain
        chain = []
        
        if preferred_provider:
            chain.append(preferred_provider)
            
            # Add fallback chain
            if fallback_chain:
                chain.extend(fallback_chain)
            elif preferred_provider in self._fallback_chains:
                chain.extend(self._fallback_chains[preferred_provider])
        
        # Add remaining available providers
        for name in self.get_available_providers():
            if name not in chain:
                chain.append(name)
        
        if not chain:
            raise CircuitOpenError("All providers are unavailable")
        
        last_error = None
        
        for provider_name in chain:
            if provider_name not in self._breakers:
                continue
            
            breaker = self._breakers[provider_name]
            config = self._providers[provider_name]
            
            # Skip if circuit is open
            if breaker.state == CircuitState.OPEN:
                continue
            
            try:
                result = await breaker.execute(
                    func,
                    provider_name,
                    *args,
                    timeout_seconds=config.timeout_seconds,
                    **kwargs
                )
                return result
                
            except CircuitOpenError:
                # Try next provider
                async with self._lock:
                    self._fallback_invocations += 1
                continue
                
            except Exception as e:
                last_error = e
                async with self._lock:
                    self._fallback_invocations += 1
                logger.warning(f"Provider {provider_name} failed: {e}, trying fallback")
                continue
        
        # All providers failed
        raise CircuitOpenError(
            f"All providers failed. Last error: {last_error}"
        )
    
    def get_provider_health(self, provider_name: str) -> Optional[ProviderHealth]:
        """Get health status of a specific provider."""
        if provider_name not in self._breakers:
            return None
        
        breaker = self._breakers[provider_name]
        metrics = breaker.metrics
        
        is_healthy = (
            breaker.state == CircuitState.CLOSED and
            metrics.current_failure_rate < 0.1
        )
        
        return ProviderHealth(
            provider_name=provider_name,
            is_healthy=is_healthy,
            circuit_state=breaker.state,
            failure_rate=metrics.current_failure_rate,
            avg_latency_ms=metrics.avg_latency_ms,
            last_success=metrics.last_success_time,
            last_failure=metrics.last_failure_time,
            consecutive_failures=metrics.consecutive_failures,
        )
    
    def get_all_provider_health(self) -> List[ProviderHealth]:
        """Get health status of all providers."""
        return [
            self.get_provider_health(name)
            for name in self._providers
        ]
    
    async def reset_provider(self, provider_name: str):
        """Reset a provider's circuit breaker."""
        if provider_name in self._breakers:
            await self._breakers[provider_name].reset()
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get manager-level metrics."""
        total_providers = len(self._providers)
        healthy_providers = len(self.get_healthy_providers())
        available_providers = len(self.get_available_providers())
        
        # Calculate fault isolation rate
        if self._total_requests > 0:
            isolation_rate = (
                (self._total_requests - self._isolated_failures) / self._total_requests
            )
        else:
            isolation_rate = 1.0
        
        return {
            "total_providers": total_providers,
            "healthy_providers": healthy_providers,
            "available_providers": available_providers,
            "total_requests": self._total_requests,
            "isolated_failures": self._isolated_failures,
            "fallback_invocations": self._fallback_invocations,
            "fault_isolation_rate": round(isolation_rate, 4),
            "target_isolation_rate": 0.999,
            "providers": {
                name: self._breakers[name].get_status()
                for name in self._providers
            }
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            "overview": self.get_metrics(),
            "providers": [
                {
                    "name": name,
                    "type": self._providers[name].provider_type.value,
                    "enabled": self._providers[name].enabled,
                    "health": self.get_provider_health(name).to_dict(),
                    "circuit_breaker": self._breakers[name].get_status(),
                }
                for name in self._providers
            ],
            "fallback_chains": {
                name: chain
                for name, chain in self._fallback_chains.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# Pre-configured Provider Circuit Breakers
# =============================================================================

def create_default_provider_manager() -> ProviderCircuitBreakerManager:
    """Create manager with default provider configurations."""
    manager = ProviderCircuitBreakerManager()
    
    # OpenAI - Primary
    manager.register_provider(
        ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="openai_primary",
            endpoint="https://api.openai.com/v1",
            failure_rate_threshold=0.50,
            window_seconds=30,
            recovery_timeout_seconds=30.0,
            timeout_seconds=30.0,
            priority=1,
        ),
        fallback_providers=["anthropic_primary", "local_llm"]
    )
    
    # Anthropic - Secondary
    manager.register_provider(
        ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            name="anthropic_primary",
            endpoint="https://api.anthropic.com/v1",
            failure_rate_threshold=0.50,
            window_seconds=30,
            recovery_timeout_seconds=30.0,
            timeout_seconds=45.0,
            priority=2,
        ),
        fallback_providers=["openai_primary", "local_llm"]
    )
    
    # Local LLM - Fallback
    manager.register_provider(
        ProviderConfig(
            provider_type=ProviderType.LOCAL,
            name="local_llm",
            endpoint="http://localhost:11434",
            failure_rate_threshold=0.70,  # Higher tolerance for local
            window_seconds=60,
            recovery_timeout_seconds=10.0,
            timeout_seconds=60.0,
            priority=3,
        )
    )
    
    return manager


# Global manager instance
_provider_manager: Optional[ProviderCircuitBreakerManager] = None


def get_provider_manager() -> ProviderCircuitBreakerManager:
    """Get or create global provider manager."""
    global _provider_manager
    if _provider_manager is None:
        _provider_manager = create_default_provider_manager()
    return _provider_manager
