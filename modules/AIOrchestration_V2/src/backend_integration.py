"""
AIOrchestration_V2 - Production Backend Integration

Enhanced orchestration with circuit breakers, load balancing, and SLO enforcement.
"""

import sys
import asyncio
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Add backend path
_backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

# Import backend implementations
try:
    from services.ai_orchestrator.src.orchestrator import AIOrchestrator as BackendOrchestrator
    from shared.services.ai_client import UnifiedAIClient
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


class RequestPriority(str, Enum):
    """Request priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OrchestrationResult:
    """Result from AI orchestration."""
    request_id: str
    provider: str
    model: str
    response: str
    latency_ms: float
    tokens_used: int
    cost: float
    from_cache: bool
    circuit_state: str


class ProductionOrchestrator:
    """
    V2 Production AI Orchestrator with SLO enforcement.

    Features:
    - Circuit breaker per provider
    - Load balancing with health awareness
    - Request prioritization
    - SLO enforcement (p95 latency < 3s)
    """

    def __init__(
        self,
        slo_latency_ms: float = 3000,
        slo_error_rate: float = 0.02,
        use_backend: bool = True,
    ):
        self.slo_latency_ms = slo_latency_ms
        self.slo_error_rate = slo_error_rate
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            self._backend = BackendOrchestrator()
        else:
            from .orchestrator import Orchestrator
            self._local = Orchestrator()

        # Load balancer
        from .load_balancer import LoadBalancer, LoadBalancingStrategy
        self._load_balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME)

        # Circuit breakers per provider
        from .circuit_breaker import CircuitBreaker, CircuitConfig
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._default_circuit_config = CircuitConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout_seconds=30,
        )

        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._latencies: List[float] = []

    def register_provider(
        self,
        name: str,
        weight: int = 1,
        max_connections: int = 100,
    ):
        """Register an AI provider."""
        self._load_balancer.add_endpoint(name, weight, max_connections)

        from .circuit_breaker import CircuitBreaker
        self._circuit_breakers[name] = CircuitBreaker(self._default_circuit_config)

    async def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_override: Optional[float] = None,
    ) -> OrchestrationResult:
        """Execute AI request with production safeguards."""
        import time
        import uuid

        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Select provider via load balancer
        provider = self._load_balancer.select_endpoint()
        if not provider:
            raise RuntimeError("No healthy providers available")

        # Check circuit breaker
        breaker = self._circuit_breakers.get(provider)
        if breaker and not breaker.allow_request():
            # Try fallback provider
            provider = self._select_fallback_provider(provider)
            if not provider:
                raise RuntimeError("All providers circuit-broken")
            breaker = self._circuit_breakers.get(provider)

        # Apply timeout based on SLO
        timeout = timeout_override or (self.slo_latency_ms / 1000)

        try:
            self._load_balancer.acquire_connection(provider)

            # Execute request
            if self.use_backend:
                result = await asyncio.wait_for(
                    self._backend.execute(prompt, model, provider),
                    timeout=timeout,
                )
                response = result.response if hasattr(result, 'response') else str(result)
            else:
                from .orchestrator import AITask
                task = AITask(task_id=request_id, prompt=prompt, model=model or "gpt-4")
                result = await asyncio.wait_for(
                    self._local.execute(task),
                    timeout=timeout,
                )
                response = result.result or ""

            latency = (time.time() - start_time) * 1000

            # Record success
            if breaker:
                breaker.record_success()
            self._load_balancer.release_connection(provider, latency, success=True)
            self._record_request(latency, success=True)

            return OrchestrationResult(
                request_id=request_id,
                provider=provider,
                model=model or "gpt-4",
                response=response,
                latency_ms=latency,
                tokens_used=len(response.split()) * 2,  # Estimate
                cost=0.001,  # Estimate
                from_cache=False,
                circuit_state=breaker.state.value if breaker else "unknown",
            )

        except asyncio.TimeoutError:
            latency = (time.time() - start_time) * 1000

            if breaker:
                breaker.record_failure()
            self._load_balancer.release_connection(provider, latency, success=False)
            self._record_request(latency, success=False)

            raise TimeoutError(f"Request exceeded SLO timeout of {timeout}s")

        except Exception as e:
            latency = (time.time() - start_time) * 1000

            if breaker:
                breaker.record_failure()
            self._load_balancer.release_connection(provider, latency, success=False)
            self._record_request(latency, success=False)

            raise

    def _select_fallback_provider(self, exclude: str) -> Optional[str]:
        """Select fallback provider excluding the failed one."""
        for name, breaker in self._circuit_breakers.items():
            if name != exclude and breaker.allow_request():
                return name
        return None

    def _record_request(self, latency: float, success: bool):
        """Record request for SLO tracking."""
        self._request_count += 1
        self._latencies.append(latency)

        if not success:
            self._error_count += 1

        # Keep last 1000 latencies
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]

    def get_slo_status(self) -> Dict[str, Any]:
        """Get current SLO status."""
        if not self._latencies:
            return {"compliant": True, "samples": 0}

        sorted_latencies = sorted(self._latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[p95_index] if sorted_latencies else 0

        error_rate = self._error_count / self._request_count if self._request_count > 0 else 0

        return {
            "p95_latency_ms": p95_latency,
            "p95_slo_ms": self.slo_latency_ms,
            "latency_compliant": p95_latency <= self.slo_latency_ms,
            "error_rate": error_rate,
            "error_rate_slo": self.slo_error_rate,
            "error_rate_compliant": error_rate <= self.slo_error_rate,
            "total_requests": self._request_count,
            "compliant": p95_latency <= self.slo_latency_ms and error_rate <= self.slo_error_rate,
        }

    def get_provider_status(self) -> Dict[str, Dict]:
        """Get status of all providers."""
        status = {}
        for name, breaker in self._circuit_breakers.items():
            stats = breaker.get_stats()
            status[name] = {
                "circuit_state": breaker.state.value,
                "total_calls": stats["total_calls"],
                "failure_rate": stats.get("failure_rate", 0),
            }
        return status


class ProductionLoadBalancer:
    """
    V2 Production Load Balancer wrapper.
    """

    def __init__(self, strategy: str = "least_response_time"):
        from .load_balancer import LoadBalancer, LoadBalancingStrategy

        strategy_map = {
            "round_robin": LoadBalancingStrategy.ROUND_ROBIN,
            "weighted": LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
            "least_connections": LoadBalancingStrategy.LEAST_CONNECTIONS,
            "least_response_time": LoadBalancingStrategy.LEAST_RESPONSE_TIME,
            "random": LoadBalancingStrategy.RANDOM,
        }

        self._balancer = LoadBalancer(strategy=strategy_map.get(strategy, LoadBalancingStrategy.ROUND_ROBIN))

    def add_endpoint(self, name: str, weight: int = 1, max_connections: int = 100):
        self._balancer.add_endpoint(name, weight, max_connections)

    def select_endpoint(self) -> Optional[str]:
        return self._balancer.select_endpoint()

    def get_stats(self) -> Dict[str, Any]:
        return self._balancer.get_stats()


# Factory functions
def get_orchestrator(
    slo_latency_ms: float = 3000,
    slo_error_rate: float = 0.02,
    use_backend: bool = True,
) -> ProductionOrchestrator:
    """Get production orchestrator."""
    return ProductionOrchestrator(slo_latency_ms, slo_error_rate, use_backend)


def get_load_balancer(strategy: str = "least_response_time") -> ProductionLoadBalancer:
    """Get production load balancer."""
    return ProductionLoadBalancer(strategy)


__all__ = [
    "BACKEND_AVAILABLE",
    "RequestPriority",
    "OrchestrationResult",
    "ProductionOrchestrator",
    "ProductionLoadBalancer",
    "get_orchestrator",
    "get_load_balancer",
]
