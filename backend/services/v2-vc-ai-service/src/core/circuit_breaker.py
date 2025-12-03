"""
V2 VC-AI Circuit Breaker

Production-grade circuit breaker for model failover and resilience.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 60.0       # Time before half-open
    half_open_requests: int = 3         # Requests to try in half-open
    
    # Timeouts
    call_timeout_seconds: float = 5.0   # Individual call timeout
    
    # Metrics
    metrics_window_seconds: float = 300.0  # 5-minute window


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    timeouts: int = 0
    
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    
    # Per-window metrics
    window_start: datetime = field(default_factory=datetime.utcnow)
    window_requests: int = 0
    window_failures: int = 0


class CircuitBreaker:
    """
    Production-grade circuit breaker implementation.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is open, requests fail fast
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        self._half_open_count = 0
        self._opened_at: Optional[datetime] = None
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics"""
        return self._metrics
    
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)"""
        return self._state == CircuitState.CLOSED
    
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)"""
        return self._state == CircuitState.OPEN
    
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)"""
        return self._state == CircuitState.HALF_OPEN
    
    async def can_execute(self) -> bool:
        """Check if a request can be executed"""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._opened_at:
                    elapsed = (datetime.utcnow() - self._opened_at).total_seconds()
                    if elapsed >= self.config.timeout_seconds:
                        await self._transition_to(CircuitState.HALF_OPEN)
                        return True
                return False
            
            if self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open
                if self._half_open_count < self.config.half_open_requests:
                    self._half_open_count += 1
                    return True
                return False
            
            return False
    
    async def record_success(self) -> None:
        """Record a successful request"""
        async with self._lock:
            self._metrics.total_requests += 1
            self._metrics.successful_requests += 1
            self._metrics.consecutive_successes += 1
            self._metrics.consecutive_failures = 0
            self._metrics.last_success_time = datetime.utcnow()
            self._metrics.window_requests += 1
            
            if self._state == CircuitState.HALF_OPEN:
                if self._metrics.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
            
            logger.debug(
                f"Circuit {self.name}: success recorded, "
                f"consecutive_successes={self._metrics.consecutive_successes}"
            )
    
    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed request"""
        async with self._lock:
            self._metrics.total_requests += 1
            self._metrics.failed_requests += 1
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
            self._metrics.last_failure_time = datetime.utcnow()
            self._metrics.window_requests += 1
            self._metrics.window_failures += 1
            
            if self._state == CircuitState.CLOSED:
                if self._metrics.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
            
            elif self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open reopens circuit
                await self._transition_to(CircuitState.OPEN)
            
            logger.warning(
                f"Circuit {self.name}: failure recorded, "
                f"consecutive_failures={self._metrics.consecutive_failures}, "
                f"error={error}"
            )
    
    async def record_timeout(self) -> None:
        """Record a timeout"""
        self._metrics.timeouts += 1
        await self.record_failure(Exception("Request timeout"))
    
    async def record_rejected(self) -> None:
        """Record a rejected request (circuit open)"""
        async with self._lock:
            self._metrics.rejected_requests += 1
            logger.debug(f"Circuit {self.name}: request rejected (circuit open)")
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state"""
        old_state = self._state
        if old_state == new_state:
            return
        
        self._state = new_state
        self._metrics.last_state_change = datetime.utcnow()
        
        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.utcnow()
            self._half_open_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_count = 0
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._half_open_count = 0
            self._metrics.consecutive_failures = 0
        
        logger.info(f"Circuit {self.name}: state changed {old_state} -> {new_state}")
        
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    async def execute(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Exception: If function fails
        """
        if not await self.can_execute():
            await self.record_rejected()
            raise CircuitOpenError(f"Circuit {self.name} is open")
        
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.call_timeout_seconds,
            )
            await self.record_success()
            return result
        except asyncio.TimeoutError:
            await self.record_timeout()
            raise
        except Exception as e:
            await self.record_failure(e)
            raise
    
    async def reset(self) -> None:
        """Reset circuit breaker to initial state"""
        async with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._metrics = CircuitBreakerMetrics()
            self._half_open_count = 0
            self._opened_at = None
            
            logger.info(f"Circuit {self.name}: reset from {old_state} to CLOSED")
    
    def get_status(self) -> dict:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self._state.value,
            "metrics": {
                "total_requests": self._metrics.total_requests,
                "successful_requests": self._metrics.successful_requests,
                "failed_requests": self._metrics.failed_requests,
                "rejected_requests": self._metrics.rejected_requests,
                "timeouts": self._metrics.timeouts,
                "consecutive_failures": self._metrics.consecutive_failures,
                "consecutive_successes": self._metrics.consecutive_successes,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
            },
            "timestamps": {
                "last_failure": self._metrics.last_failure_time.isoformat() if self._metrics.last_failure_time else None,
                "last_success": self._metrics.last_success_time.isoformat() if self._metrics.last_success_time else None,
                "last_state_change": self._metrics.last_state_change.isoformat() if self._metrics.last_state_change else None,
            },
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass
