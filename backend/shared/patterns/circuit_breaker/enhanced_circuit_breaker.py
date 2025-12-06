"""
Enhanced Circuit Breaker Implementation

Features:
- Dynamic failure threshold (50% failure rate over 30s window)
- Automatic recovery with health probes
- Exception interception delay < 100ms
- 99.9% fault isolation rate target
"""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from functools import wraps
import statistics

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, reject requests fast
    HALF_OPEN = "half_open" # Testing recovery


class FailureType(str, Enum):
    """Types of failures tracked."""
    TIMEOUT = "timeout"
    ERROR = "error"
    REJECTION = "rejection"
    RATE_LIMIT = "rate_limit"


@dataclass
class RequestMetric:
    """Single request metric."""
    timestamp: datetime
    success: bool
    duration_ms: float
    failure_type: Optional[FailureType] = None
    error_message: Optional[str] = None


@dataclass
class DynamicThresholdConfig:
    """
    Dynamic threshold configuration.
    
    Default: Opens when failure rate > 50% over 30 second window.
    """
    # Failure rate threshold (50%)
    failure_rate_threshold: float = 0.50
    
    # Time window for calculating failure rate (30 seconds)
    window_seconds: int = 30
    
    # Minimum requests before evaluating (prevent false positives)
    minimum_requests: int = 10
    
    # Time before attempting recovery (half-open)
    recovery_timeout_seconds: float = 30.0
    
    # Requests to test in half-open state
    half_open_requests: int = 5
    
    # Successes needed to close circuit
    recovery_success_threshold: int = 3
    
    # Consecutive failures to immediately open
    consecutive_failure_threshold: int = 5
    
    # Adaptive thresholds based on time of day
    adaptive_thresholds: bool = True
    
    # Increase threshold during peak hours
    peak_hour_threshold_increase: float = 0.1


@dataclass
class CircuitBreakerMetrics:
    """Comprehensive circuit breaker metrics."""
    state: CircuitState = CircuitState.CLOSED
    
    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    
    # Timing
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    circuit_opened_at: Optional[datetime] = None
    
    # Current window metrics
    window_requests: int = 0
    window_failures: int = 0
    current_failure_rate: float = 0.0
    
    # Consecutive tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # Latency tracking
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Recovery attempts
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "current_failure_rate": round(self.current_failure_rate, 4),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "recovery_attempts": self.recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_state_change": self.last_state_change.isoformat(),
        }


class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker with dynamic thresholds and monitoring.
    
    Key features:
    - Dynamic failure rate calculation over sliding window
    - Automatic recovery with health probing
    - Real-time metrics collection
    - Fault isolation with < 100ms interception delay
    
    Usage:
        breaker = EnhancedCircuitBreaker(
            name="openai",
            config=DynamicThresholdConfig(failure_rate_threshold=0.50)
        )
        
        async with breaker:
            result = await call_openai()
    """
    
    # Target: < 100ms interception delay
    INTERCEPTION_DELAY_TARGET_MS = 100
    
    def __init__(
        self,
        name: str,
        config: Optional[DynamicThresholdConfig] = None,
        fallback: Optional[Callable] = None,
        on_state_change: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
    ):
        self.name = name
        self.config = config or DynamicThresholdConfig()
        self.fallback = fallback
        self.on_state_change = on_state_change
        self.on_failure = on_failure
        
        self._state = CircuitState.CLOSED
        self._metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        
        # Sliding window for request metrics
        self._request_window: deque = deque()
        self._latencies: deque = deque(maxlen=1000)
        
        # Half-open state tracking
        self._half_open_requests = 0
        self._half_open_successes = 0
        
        # Recovery probing
        self._last_probe_time: Optional[datetime] = None
        self._probe_interval_seconds = 10.0
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def metrics(self) -> CircuitBreakerMetrics:
        return self._metrics
    
    async def _check_and_update_state(self) -> bool:
        """
        Check current state and determine if request should proceed.
        
        Returns:
            True if request can proceed, False to reject
        """
        now = datetime.now(timezone.utc)
        
        if self._state == CircuitState.CLOSED:
            # Check if we need to open
            self._calculate_window_metrics()
            
            if self._should_open():
                await self._transition_to(CircuitState.OPEN)
                return False
            return True
        
        elif self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._metrics.circuit_opened_at:
                elapsed = (now - self._metrics.circuit_opened_at).total_seconds()
                if elapsed >= self.config.recovery_timeout_seconds:
                    await self._transition_to(CircuitState.HALF_OPEN)
                    return True
            return False
        
        else:  # HALF_OPEN
            # Allow limited requests for testing
            if self._half_open_requests < self.config.half_open_requests:
                self._half_open_requests += 1
                return True
            return False
    
    def _calculate_window_metrics(self):
        """Calculate metrics over the sliding window."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.config.window_seconds)
        
        # Clean old metrics from window
        while self._request_window and self._request_window[0].timestamp < window_start:
            self._request_window.popleft()
        
        if not self._request_window:
            self._metrics.window_requests = 0
            self._metrics.window_failures = 0
            self._metrics.current_failure_rate = 0.0
            return
        
        # Calculate current failure rate
        total = len(self._request_window)
        failures = sum(1 for m in self._request_window if not m.success)
        
        self._metrics.window_requests = total
        self._metrics.window_failures = failures
        self._metrics.current_failure_rate = failures / total if total > 0 else 0.0
    
    def _should_open(self) -> bool:
        """Determine if circuit should open."""
        # Check consecutive failures first
        if self._metrics.consecutive_failures >= self.config.consecutive_failure_threshold:
            logger.warning(
                f"Circuit {self.name}: Opening due to {self._metrics.consecutive_failures} "
                f"consecutive failures"
            )
            return True
        
        # Check failure rate if minimum requests met
        if self._metrics.window_requests >= self.config.minimum_requests:
            threshold = self._get_current_threshold()
            
            if self._metrics.current_failure_rate > threshold:
                logger.warning(
                    f"Circuit {self.name}: Opening due to failure rate "
                    f"{self._metrics.current_failure_rate:.2%} > {threshold:.2%}"
                )
                return True
        
        return False
    
    def _get_current_threshold(self) -> float:
        """Get current failure threshold, potentially adjusted for time of day."""
        threshold = self.config.failure_rate_threshold
        
        if self.config.adaptive_thresholds:
            hour = datetime.now(timezone.utc).hour
            # Peak hours: 9-17 UTC
            if 9 <= hour <= 17:
                threshold += self.config.peak_hour_threshold_increase
        
        return min(threshold, 0.99)  # Cap at 99%
    
    async def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        
        if old_state == new_state:
            return
        
        self._state = new_state
        self._metrics.state = new_state
        self._metrics.last_state_change = datetime.now(timezone.utc)
        
        if new_state == CircuitState.OPEN:
            self._metrics.circuit_opened_at = datetime.now(timezone.utc)
            self._metrics.recovery_attempts += 1
        
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_requests = 0
            self._half_open_successes = 0
        
        elif new_state == CircuitState.CLOSED:
            self._metrics.circuit_opened_at = None
            self._metrics.consecutive_failures = 0
            
            if old_state == CircuitState.HALF_OPEN:
                self._metrics.successful_recoveries += 1
        
        logger.info(f"Circuit {self.name}: State changed {old_state.value} -> {new_state.value}")
        
        # Notify callback
        if self.on_state_change:
            try:
                if asyncio.iscoroutinefunction(self.on_state_change):
                    await self.on_state_change(self.name, old_state, new_state)
                else:
                    self.on_state_change(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    async def record_success(self, duration_ms: float):
        """Record a successful request."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            
            self._metrics.total_requests += 1
            self._metrics.successful_requests += 1
            self._metrics.consecutive_successes += 1
            self._metrics.consecutive_failures = 0
            self._metrics.last_success_time = now
            
            # Add to window
            self._request_window.append(RequestMetric(
                timestamp=now,
                success=True,
                duration_ms=duration_ms
            ))
            
            # Track latency
            self._latencies.append(duration_ms)
            self._update_latency_metrics()
            
            # Handle half-open success
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.recovery_success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
    
    async def record_failure(
        self,
        duration_ms: float,
        failure_type: FailureType = FailureType.ERROR,
        error_message: Optional[str] = None
    ):
        """Record a failed request."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            
            self._metrics.total_requests += 1
            self._metrics.failed_requests += 1
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
            self._metrics.last_failure_time = now
            
            # Add to window
            self._request_window.append(RequestMetric(
                timestamp=now,
                success=False,
                duration_ms=duration_ms,
                failure_type=failure_type,
                error_message=error_message
            ))
            
            # Track latency
            self._latencies.append(duration_ms)
            self._update_latency_metrics()
            
            # Notify callback
            if self.on_failure:
                try:
                    if asyncio.iscoroutinefunction(self.on_failure):
                        await self.on_failure(self.name, failure_type, error_message)
                    else:
                        self.on_failure(self.name, failure_type, error_message)
                except Exception as e:
                    logger.error(f"Failure callback error: {e}")
            
            # Handle state transitions
            if self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open reopens circuit
                await self._transition_to(CircuitState.OPEN)
            
            elif self._state == CircuitState.CLOSED:
                self._calculate_window_metrics()
                if self._should_open():
                    await self._transition_to(CircuitState.OPEN)
    
    async def record_rejection(self):
        """Record a rejected request (circuit open)."""
        async with self._lock:
            self._metrics.rejected_requests += 1
    
    def _update_latency_metrics(self):
        """Update latency percentile metrics."""
        if not self._latencies:
            return
        
        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)
        
        self._metrics.avg_latency_ms = statistics.mean(sorted_latencies)
        self._metrics.p95_latency_ms = sorted_latencies[int(n * 0.95)] if n > 20 else self._metrics.avg_latency_ms
        self._metrics.p99_latency_ms = sorted_latencies[int(n * 0.99)] if n > 100 else self._metrics.p95_latency_ms
    
    async def execute(
        self,
        func: Callable[..., T],
        *args,
        timeout_seconds: float = 30.0,
        **kwargs
    ) -> T:
        """
        Execute function with circuit breaker protection.
        
        Interception delay target: < 100ms
        """
        start_time = time.perf_counter()
        
        # Check if request can proceed
        async with self._lock:
            can_execute = await self._check_and_update_state()
        
        interception_time = (time.perf_counter() - start_time) * 1000
        
        if interception_time > self.INTERCEPTION_DELAY_TARGET_MS:
            logger.warning(
                f"Circuit {self.name}: Interception delay {interception_time:.2f}ms "
                f"exceeded target {self.INTERCEPTION_DELAY_TARGET_MS}ms"
            )
        
        if not can_execute:
            await self.record_rejection()
            
            # Try fallback
            if self.fallback:
                logger.info(f"Circuit {self.name}: Using fallback")
                if asyncio.iscoroutinefunction(self.fallback):
                    return await self.fallback(*args, **kwargs)
                return self.fallback(*args, **kwargs)
            
            raise CircuitOpenError(
                f"Circuit {self.name} is open. "
                f"Failure rate: {self._metrics.current_failure_rate:.2%}"
            )
        
        # Execute function
        execution_start = time.perf_counter()
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            else:
                result = func(*args, **kwargs)
            
            duration_ms = (time.perf_counter() - execution_start) * 1000
            await self.record_success(duration_ms)
            
            return result
            
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - execution_start) * 1000
            await self.record_failure(
                duration_ms,
                FailureType.TIMEOUT,
                f"Timeout after {timeout_seconds}s"
            )
            raise
            
        except Exception as e:
            duration_ms = (time.perf_counter() - execution_start) * 1000
            await self.record_failure(
                duration_ms,
                FailureType.ERROR,
                str(e)
            )
            raise
    
    async def probe_health(self, health_check: Callable) -> bool:
        """
        Probe service health for potential recovery.
        
        Used when circuit is open to check if service recovered.
        """
        try:
            if asyncio.iscoroutinefunction(health_check):
                await asyncio.wait_for(health_check(), timeout=5.0)
            else:
                health_check()
            
            logger.info(f"Circuit {self.name}: Health probe succeeded")
            return True
            
        except Exception as e:
            logger.debug(f"Circuit {self.name}: Health probe failed: {e}")
            return False
    
    async def reset(self):
        """Reset circuit breaker to initial state."""
        async with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._metrics = CircuitBreakerMetrics()
            self._request_window.clear()
            self._latencies.clear()
            self._half_open_requests = 0
            self._half_open_successes = 0
            
            logger.info(f"Circuit {self.name}: Reset from {old_state.value} to CLOSED")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "metrics": self._metrics.to_dict(),
            "config": {
                "failure_rate_threshold": self.config.failure_rate_threshold,
                "window_seconds": self.config.window_seconds,
                "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
                "minimum_requests": self.config.minimum_requests,
            },
            "has_fallback": self.fallback is not None,
        }
    
    # Context manager support
    async def __aenter__(self):
        async with self._lock:
            can_execute = await self._check_and_update_state()
        
        if not can_execute:
            await self.record_rejection()
            raise CircuitOpenError(f"Circuit {self.name} is open")
        
        self._execution_start = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._execution_start) * 1000
        
        if exc_type is None:
            await self.record_success(duration_ms)
        else:
            failure_type = FailureType.TIMEOUT if exc_type == asyncio.TimeoutError else FailureType.ERROR
            await self.record_failure(duration_ms, failure_type, str(exc_val))
        
        return False  # Don't suppress exceptions


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def circuit_breaker(
    name: str,
    config: Optional[DynamicThresholdConfig] = None,
    fallback: Optional[Callable] = None,
):
    """
    Decorator for applying circuit breaker to async functions.
    
    Usage:
        @circuit_breaker("openai", fallback=fallback_func)
        async def call_openai(prompt: str) -> str:
            ...
    """
    breaker = EnhancedCircuitBreaker(name=name, config=config, fallback=fallback)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await breaker.execute(func, *args, **kwargs)
        
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator
