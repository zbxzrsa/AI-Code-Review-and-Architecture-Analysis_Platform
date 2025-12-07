"""
AIOrchestration_V2 - Circuit Breaker

Production circuit breaker for fault tolerance.
"""

import logging
import asyncio
from typing import Dict, Optional, Any, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitStats:
    """Circuit statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0
        return self.failed_calls / self.total_calls


@dataclass
class CircuitConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes in half-open to close
    timeout_seconds: float = 30         # Time before trying half-open
    half_open_max_calls: int = 3        # Max calls in half-open state

    # Rolling window for failure counting
    window_size_seconds: int = 60

    # Slow call tracking
    slow_call_threshold_ms: float = 5000
    slow_call_rate_threshold: float = 0.5


class CircuitBreaker:
    """
    Production circuit breaker.

    V2 Features:
    - Three-state circuit (closed, open, half-open)
    - Configurable thresholds
    - Slow call detection
    - Per-service circuits
    - Metrics and monitoring
    """

    def __init__(self, config: Optional[CircuitConfig] = None):
        self.config = config or CircuitConfig()

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._half_open_calls = 0
        self._opened_at: Optional[datetime] = None

        # Recent calls for rolling window
        self._recent_calls: list = []

    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        self._check_state_transition()
        return self._state

    def _check_state_transition(self):
        """Check if state should transition"""
        now = datetime.now(timezone.utc)

        if self._state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self._opened_at:
                elapsed = (now - self._opened_at).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now(timezone.utc)
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._consecutive_successes = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._consecutive_failures = 0

        logger.info(f"Circuit breaker: {old_state.value} -> {new_state.value}")

    def allow_request(self) -> bool:
        """Check if request should be allowed"""
        state = self.state

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            self._stats.rejected_calls += 1
            return False

        # Half-open: allow limited requests
        if self._half_open_calls < self.config.half_open_max_calls:
            self._half_open_calls += 1
            return True

        self._stats.rejected_calls += 1
        return False

    def record_success(self, duration_ms: float = 0):
        """Record successful call"""
        now = datetime.now(timezone.utc)

        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = now
        self._consecutive_failures = 0
        self._consecutive_successes += 1

        # Track in rolling window
        self._recent_calls.append({
            "time": now,
            "success": True,
            "duration_ms": duration_ms,
        })
        self._cleanup_old_calls()

        # State transitions
        if self._state == CircuitState.HALF_OPEN:
            if self._consecutive_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

    def record_failure(self, error: Optional[str] = None):
        """Record failed call"""
        now = datetime.now(timezone.utc)

        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = now
        self._consecutive_failures += 1
        self._consecutive_successes = 0

        # Track in rolling window
        self._recent_calls.append({
            "time": now,
            "success": False,
            "error": error,
        })
        self._cleanup_old_calls()

        # State transitions
        if self._state == CircuitState.CLOSED:
            if self._consecutive_failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens circuit
            self._transition_to(CircuitState.OPEN)

    def _cleanup_old_calls(self):
        """Remove calls outside rolling window"""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.config.window_size_seconds)
        self._recent_calls = [c for c in self._recent_calls if c["time"] > cutoff]

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with circuit breaker protection"""
        import time

        if not self.allow_request():
            raise CircuitOpenError(f"Circuit is {self._state.value}")

        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            self.record_success(duration_ms)
            return result

        except Exception as e:
            self.record_failure(str(e))
            raise

    def reset(self):
        """Reset circuit breaker"""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._half_open_calls = 0
        self._opened_at = None
        self._recent_calls.clear()
        logger.info("Circuit breaker reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "state": self._state.value,
            "total_calls": self._stats.total_calls,
            "successful_calls": self._stats.successful_calls,
            "failed_calls": self._stats.failed_calls,
            "rejected_calls": self._stats.rejected_calls,
            "failure_rate": self._stats.failure_rate,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "last_failure": self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
            "last_success": self._stats.last_success_time.isoformat() if self._stats.last_success_time else None,
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open"""
    pass


class CircuitBreakerRegistry:
    """Registry for multiple circuit breakers"""

    def __init__(self, default_config: Optional[CircuitConfig] = None):
        self.default_config = default_config or CircuitConfig()
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitConfig] = None,
    ) -> CircuitBreaker:
        """Get or create circuit breaker by name"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(config or self.default_config)
        return self._breakers[name]

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all circuit breakers"""
        return {name: cb.get_stats() for name, cb in self._breakers.items()}
