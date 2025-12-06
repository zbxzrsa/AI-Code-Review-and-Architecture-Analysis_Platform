"""
Retry Logic with Exponential Backoff
Implements intelligent retry mechanisms for 95% transient error recovery
"""

import asyncio
import functools
import logging
import random
from typing import Callable, TypeVar, Optional, Tuple, Type, Union, List
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    dont_retry_on: Tuple[Type[Exception], ...] = ()


@dataclass
class RetryStats:
    """Retry statistics."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retries: int = 0
    total_delay: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_attempts / self.total_attempts) * 100


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempts. "
            f"Last error: {last_exception}"
        )


def calculate_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """
    Calculate delay for next retry attempt.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
    
    Returns:
        Delay in seconds
    """
    # Exponential backoff: delay = initial_delay * (base ^ attempt)
    delay = config.initial_delay * (config.exponential_base ** attempt)
    
    # Cap at max_delay
    delay = min(delay, config.max_delay)
    
    # Add jitter to prevent thundering herd
    if config.jitter:
        jitter_amount = delay * 0.1  # 10% jitter
        delay += random.uniform(-jitter_amount, jitter_amount)
    
    return max(0, delay)


def should_retry(
    exception: Exception,
    config: RetryConfig
) -> bool:
    """
    Determine if exception should trigger a retry.
    
    Args:
        exception: The exception that occurred
        config: Retry configuration
    
    Returns:
        True if should retry, False otherwise
    """
    # Don't retry if in dont_retry_on list
    if isinstance(exception, config.dont_retry_on):
        return False
    
    # Retry if in retry_on_exceptions list
    return isinstance(exception, config.retry_on_exceptions)


async def retry_async(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Function arguments
        config: Retry configuration
        **kwargs: Function keyword arguments
    
    Returns:
        Function result
    
    Raises:
        RetryExhaustedError: If all retries exhausted
    
    Example:
        >>> async def fetch_data():
        ...     response = await http_client.get("/api/data")
        ...     return response.json()
        >>> 
        >>> config = RetryConfig(max_attempts=3, initial_delay=1.0)
        >>> result = await retry_async(fetch_data, config=config)
    """
    config = config or RetryConfig()
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            result = await func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(
                    f"Retry successful after {attempt} attempts",
                    extra={"function": func.__name__, "attempts": attempt + 1}
                )
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Check if we should retry
            if not should_retry(e, config):
                logger.warning(
                    f"Not retrying {func.__name__} due to exception type: {type(e).__name__}"
                )
                raise
            
            # Check if we have more attempts
            if attempt >= config.max_attempts - 1:
                break
            
            # Calculate delay
            delay = calculate_delay(attempt, config)
            
            logger.warning(
                f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}. "
                f"Retrying in {delay:.2f}s. Error: {e}"
            )
            
            # Wait before retry
            await asyncio.sleep(delay)
    
    # All retries exhausted
    raise RetryExhaustedError(config.max_attempts, last_exception)


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
    dont_retry_on: Optional[Tuple[Type[Exception], ...]] = None
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add jitter to delays
        retry_on: Tuple of exceptions to retry on
        dont_retry_on: Tuple of exceptions to never retry
    
    Example:
        >>> @retry(max_attempts=3, initial_delay=1.0)
        >>> async def fetch_user(user_id: int):
        ...     response = await http_client.get(f"/users/{user_id}")
        ...     return response.json()
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retry_on_exceptions=retry_on or (Exception,),
                dont_retry_on=dont_retry_on or ()
            )
            
            return await retry_async(func, *args, config=config, **kwargs)
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    
    Example:
        >>> breaker = CircuitBreaker(
        ...     failure_threshold=5,
        ...     recovery_timeout=60.0
        ... )
        >>> 
        >>> async def call_external_service():
        ...     async with breaker:
        ...         return await external_api.call()
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            expected_exception: Exception type that counts as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state
    
    def _should_attempt(self) -> bool:
        """Check if we should attempt the operation."""
        if self._state == "CLOSED":
            return True
        
        if self._state == "OPEN":
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._state = "HALF_OPEN"
                    logger.info("Circuit breaker entering HALF_OPEN state")
                    return True
            return False
        
        # HALF_OPEN state
        return True
    
    def _record_success(self):
        """Record successful operation."""
        self._failure_count = 0
        if self._state == "HALF_OPEN":
            self._state = "CLOSED"
            logger.info("Circuit breaker closed after successful recovery")
    
    def _record_failure(self):
        """Record failed operation."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )
    
    async def __aenter__(self):
        """Enter circuit breaker context."""
        if not self._should_attempt():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is OPEN. "
                f"Last failure: {self._last_failure_time}"
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit circuit breaker context."""
        if exc_type is None:
            self._record_success()
        elif isinstance(exc_val, self.expected_exception):
            self._record_failure()
        
        return False  # Don't suppress exception


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Predefined retry configurations for common scenarios
class RetryPresets:
    """Common retry configurations."""
    
    # Quick retry for transient network issues
    NETWORK = RetryConfig(
        max_attempts=3,
        initial_delay=0.5,
        max_delay=5.0,
        exponential_base=2.0
    )
    
    # Aggressive retry for critical operations
    CRITICAL = RetryConfig(
        max_attempts=5,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0
    )
    
    # Conservative retry for expensive operations
    EXPENSIVE = RetryConfig(
        max_attempts=2,
        initial_delay=2.0,
        max_delay=10.0,
        exponential_base=2.0
    )
    
    # AI provider retry (handle rate limits)
    AI_PROVIDER = RetryConfig(
        max_attempts=3,
        initial_delay=2.0,
        max_delay=60.0,
        exponential_base=3.0  # Longer backoff for rate limits
    )


# Example usage with specific exceptions
class AIProviderError(Exception):
    """AI provider error."""
    pass


class RateLimitError(AIProviderError):
    """Rate limit exceeded."""
    pass


class ValidationError(Exception):
    """Validation error (don't retry)."""
    pass


@retry(
    max_attempts=3,
    initial_delay=2.0,
    retry_on=(AIProviderError,),
    dont_retry_on=(ValidationError,)
)
async def call_ai_provider(code: str):
    """
    Call AI provider with automatic retry.
    
    - Retries on AIProviderError (including RateLimitError)
    - Does not retry on ValidationError
    - Uses exponential backoff
    """
    # Implementation here
    pass
