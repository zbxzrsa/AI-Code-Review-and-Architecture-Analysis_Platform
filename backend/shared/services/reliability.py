"""
Reliability Patterns Module

Implements:
- Circuit Breaker pattern for fault tolerance
- Retry logic with exponential backoff
- Request deduplication (hash-based)
- Dynamic batching for requests
"""

import asyncio
import hashlib
import logging
import random  # FIXED: Moved from inside function
import time
from typing import Optional, Dict, Any, List, Callable, Awaitable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

class CircuitState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 3       # Successes before closing
    recovery_timeout: int = 60       # Seconds before half-open
    expected_exceptions: tuple = (Exception,)  # Exceptions to count as failures


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.utcnow)
    total_requests: int = 0
    total_failures: int = 0
    total_rejected: int = 0


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        @breaker
        async def call_external_service():
            ...
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        recovery_timeout: int = 60,
        expected_exceptions: tuple = (Exception,),
        name: str = "default",
    ):
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
            expected_exceptions=expected_exceptions,
        )
        self.name = name
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self.metrics.state
    
    async def _should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.metrics.state == CircuitState.CLOSED:
            return True
        
        if self.metrics.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.metrics.last_failure_time:
                elapsed = (datetime.utcnow() - self.metrics.last_failure_time).total_seconds()
                if elapsed >= self.config.recovery_timeout:
                    await self._transition_to(CircuitState.HALF_OPEN)
                    return True
            return False
        
        if self.metrics.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    async def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self.metrics.state
        self.metrics.state = new_state
        self.metrics.last_state_change = datetime.utcnow()
        
        if new_state == CircuitState.CLOSED:
            self.metrics.failure_count = 0
            self.metrics.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.metrics.success_count = 0
        
        logger.info(f"Circuit breaker '{self.name}': {old_state.value} → {new_state.value}")
    
    async def _record_success(self):
        """Record successful request."""
        self.metrics.success_count += 1
        
        if self.metrics.state == CircuitState.HALF_OPEN:
            if self.metrics.success_count >= self.config.success_threshold:
                await self._transition_to(CircuitState.CLOSED)
    
    async def _record_failure(self, exception: Exception):
        """Record failed request."""
        self.metrics.failure_count += 1
        self.metrics.total_failures += 1
        self.metrics.last_failure_time = datetime.utcnow()
        
        if self.metrics.state == CircuitState.HALF_OPEN:
            await self._transition_to(CircuitState.OPEN)
        elif self.metrics.state == CircuitState.CLOSED:
            if self.metrics.failure_count >= self.config.failure_threshold:
                await self._transition_to(CircuitState.OPEN)
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            self.metrics.total_requests += 1
            
            if not await self._should_allow_request():
                self.metrics.total_rejected += 1
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Retry after {self.config.recovery_timeout}s"
                )
        
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                await self._record_success()
            return result
            
        except self.config.expected_exceptions as e:
            async with self._lock:
                await self._record_failure(e)
            raise
    
    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator for circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = self
        return wrapper
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self.metrics.state.value,
            "failure_count": self.metrics.failure_count,
            "success_count": self.metrics.success_count,
            "total_requests": self.metrics.total_requests,
            "total_failures": self.metrics.total_failures,
            "total_rejected": self.metrics.total_rejected,
            "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_state_change": self.metrics.last_state_change.isoformat(),
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# Retry with Exponential Backoff
# =============================================================================

@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 5
    initial_delay: float = 1.0      # seconds
    max_delay: float = 60.0         # seconds
    multiplier: float = 2.0         # exponential multiplier
    jitter: float = 0.1             # random jitter factor (0-1)
    retry_exceptions: tuple = (Exception,)


class RetryWithBackoff:
    """
    Retry logic with exponential backoff.
    
    Usage:
        @retry_with_backoff(max_attempts=5, initial_delay=1.0)
        async def unreliable_operation():
            ...
    """
    
    def __init__(
        self,
        max_attempts: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: float = 0.1,
        retry_exceptions: tuple = (Exception,),
        on_retry: Optional[Callable] = None,
    ):
        self.config = RetryConfig(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            multiplier=multiplier,
            jitter=jitter,
            retry_exceptions=retry_exceptions,
        )
        self.on_retry = on_retry
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry with jitter."""
        # FIXED: random import moved to module level
        delay = self.config.initial_delay * (self.config.multiplier ** attempt)
        delay = min(delay, self.config.max_delay)
        
        # Add jitter
        jitter_range = delay * self.config.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
                
            except self.config.retry_exceptions as e:
                last_exception = e
                
                if attempt + 1 >= self.config.max_attempts:
                    logger.error(f"All {self.config.max_attempts} retry attempts failed")
                    raise
                
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                if self.on_retry:
                    await self.on_retry(attempt, e, delay)
                
                await asyncio.sleep(delay)
        
        # FIXED: Guard against None exception
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry failed with no exception captured")
    
    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator for retry logic."""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.execute(func, *args, **kwargs)
        return wrapper


def retry_with_backoff(
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    retry_exceptions: tuple = (Exception,),
):
    """Decorator factory for retry with exponential backoff."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        retry = RetryWithBackoff(
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            multiplier=multiplier,
            retry_exceptions=retry_exceptions,
        )
        return retry(func)
    return decorator


# =============================================================================
# Request Deduplication
# =============================================================================

@dataclass
class DeduplicationEntry:
    """Deduplication cache entry."""
    request_hash: str
    result: Any
    created_at: datetime
    expires_at: datetime


class RequestDeduplicator:
    """
    Request deduplication using content hashing.
    
    Prevents duplicate API calls for identical requests.
    """
    
    def __init__(
        self,
        ttl: int = 300,  # 5 minutes
        max_size: int = 10000,
        redis_client = None,
    ):
        self.ttl = ttl
        self.max_size = max_size
        self.redis = redis_client
        
        self._cache: Dict[str, DeduplicationEntry] = {}
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
    
    def _hash_request(self, *args, **kwargs) -> str:
        """Generate hash for request."""
        import json
        content = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def deduplicate(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute with deduplication."""
        request_hash = self._hash_request(*args, **kwargs)
        
        async with self._lock:
            # Check cache
            if request_hash in self._cache:
                entry = self._cache[request_hash]
                if datetime.utcnow() < entry.expires_at:
                    logger.debug(f"Dedup cache hit: {request_hash[:16]}...")
                    return entry.result
                else:
                    del self._cache[request_hash]
            
            # Check if request is already pending
            if request_hash in self._pending:
                logger.debug(f"Dedup pending hit: {request_hash[:16]}...")
                return await self._pending[request_hash]
            
            # Create pending future
            # FIXED: Use get_running_loop() instead of deprecated get_event_loop()
            future = asyncio.get_running_loop().create_future()
            self._pending[request_hash] = future
        
        try:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            async with self._lock:
                self._cache[request_hash] = DeduplicationEntry(
                    request_hash=request_hash,
                    result=result,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(seconds=self.ttl),
                )
                
                # Evict if needed
                if len(self._cache) > self.max_size:
                    oldest = min(self._cache.values(), key=lambda e: e.created_at)
                    del self._cache[oldest.request_hash]
                
                # Resolve pending future
                future.set_result(result)
                del self._pending[request_hash]
            
            return result
            
        except Exception as e:
            async with self._lock:
                future.set_exception(e)
                del self._pending[request_hash]
            raise
    
    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorator for deduplication."""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.deduplicate(func, *args, **kwargs)
        return wrapper


def deduplicate(ttl: int = 300):
    """Decorator factory for request deduplication."""
    deduplicator = RequestDeduplicator(ttl=ttl)
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        return deduplicator(func)
    return decorator


# =============================================================================
# Dynamic Batching
# =============================================================================

@dataclass
class BatchItem:
    """Item in batch queue."""
    request: Any
    future: asyncio.Future
    added_at: datetime


class DynamicBatcher:
    """
    Dynamic request batching for efficient processing.
    
    Collects requests and processes them in batches for efficiency.
    """
    
    def __init__(
        self,
        batch_processor: Callable[[List[Any]], Awaitable[List[Any]]],
        max_batch_size: int = 32,
        max_wait_ms: int = 100,
    ):
        self.batch_processor = batch_processor
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self._queue: deque[BatchItem] = deque()
        self._lock = asyncio.Lock()
        self._processing = False
        self._process_task: Optional[asyncio.Task] = None
    
    async def add(self, request: Any) -> Any:
        """Add request to batch queue."""
        # FIXED: Use get_running_loop() instead of deprecated get_event_loop()
        future = asyncio.get_running_loop().create_future()
        
        async with self._lock:
            self._queue.append(BatchItem(
                request=request,
                future=future,
                added_at=datetime.utcnow(),
            ))
            
            # Start processing if not already running
            if not self._processing:
                self._process_task = asyncio.create_task(self._process_loop())
        
        return await future
    
    async def _process_loop(self):
        """Process batches continuously."""
        self._processing = True
        
        try:
            while True:
                await asyncio.sleep(self.max_wait_ms / 1000)
                
                async with self._lock:
                    if not self._queue:
                        self._processing = False
                        break
                    
                    # Collect batch
                    batch: List[BatchItem] = []
                    while self._queue and len(batch) < self.max_batch_size:
                        batch.append(self._queue.popleft())
                
                if batch:
                    await self._process_batch(batch)
                    
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self._processing = False
    
    async def _process_batch(self, batch: List[BatchItem]):
        """Process a single batch."""
        try:
            requests = [item.request for item in batch]
            results = await self.batch_processor(requests)
            
            # Distribute results
            for item, result in zip(batch, results):
                if isinstance(result, Exception):
                    item.future.set_exception(result)
                else:
                    item.future.set_result(result)
                    
        except Exception as e:
            # Set exception for all items
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)


# =============================================================================
# Combined Reliability Wrapper
# =============================================================================

class ReliableService:
    """
    Combined reliability wrapper with all patterns.
    
    Usage:
        service = ReliableService(
            name="ai_service",
            circuit_breaker_config={...},
            retry_config={...},
            dedupe_ttl=300,
        )
        
        result = await service.call(ai_function, prompt)
    """
    
    def __init__(
        self,
        name: str,
        circuit_breaker_config: Optional[Dict] = None,
        retry_config: Optional[Dict] = None,
        dedupe_ttl: int = 300,
        enable_deduplication: bool = True,
    ):
        self.name = name
        
        # Circuit breaker
        cb_config = circuit_breaker_config or {}
        self.circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=cb_config.get("failure_threshold", 5),
            success_threshold=cb_config.get("success_threshold", 3),
            recovery_timeout=cb_config.get("recovery_timeout", 60),
        )
        
        # Retry logic
        r_config = retry_config or {}
        self.retry = RetryWithBackoff(
            max_attempts=r_config.get("max_attempts", 3),
            initial_delay=r_config.get("initial_delay", 1.0),
            max_delay=r_config.get("max_delay", 30.0),
            multiplier=r_config.get("multiplier", 2.0),
        )
        
        # Deduplication
        self.enable_deduplication = enable_deduplication
        if enable_deduplication:
            self.deduplicator = RequestDeduplicator(ttl=dedupe_ttl)
    
    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute with all reliability patterns."""
        
        async def wrapped():
            return await self.circuit_breaker.call(func, *args, **kwargs)
        
        # Apply retry
        retried = lambda: self.retry.execute(wrapped)
        
        # Apply deduplication if enabled
        if self.enable_deduplication:
            return await self.deduplicator.deduplicate(retried)
        else:
            return await retried()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "name": self.name,
            "circuit_breaker": self.circuit_breaker.get_metrics(),
        }
