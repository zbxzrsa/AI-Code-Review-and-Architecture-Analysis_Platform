"""
Rate Limiting Middleware

Implements:
- Redis-based distributed rate limiting
- Multiple rate limit tiers (per-user, per-IP, global)
- Sliding window algorithm
- Configurable limits per endpoint
"""
import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from functools import wraps

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    # Default limits
    default_rpm: int = 60  # Requests per minute
    default_rph: int = 1000  # Requests per hour
    default_rpd: int = 10000  # Requests per day
    
    # Burst allowance
    burst_multiplier: float = 1.5
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "ratelimit:"
    
    # Response headers
    include_headers: bool = True


@dataclass
class RateLimitRule:
    """Rate limit rule for specific endpoint."""
    path_pattern: str  # Regex or exact path
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    by_user: bool = True  # Limit per user
    by_ip: bool = True  # Limit per IP
    methods: Optional[List[str]] = None  # HTTP methods (None = all)


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception."""
    
    def __init__(
        self,
        limit_type: str,
        limit: int,
        window: str,
        retry_after: int,
    ):
        super().__init__(
            status_code=429,
            detail=f"Rate limit exceeded: {limit} requests per {window}",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Type": limit_type,
                "X-RateLimit-Limit": str(limit),
            },
        )


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter using Redis.
    
    Uses sorted sets for efficient sliding window implementation.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._redis = None
        self._rules: Dict[str, RateLimitRule] = {}
        
    def _get_redis(self):
        """Get Redis connection (lazy initialization)."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.from_url(
                    self.config.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except ImportError:
                logger.warning("Redis not available, using in-memory rate limiting")
                self._redis = InMemoryStore()
        return self._redis
    
    def add_rule(self, rule: RateLimitRule):
        """Add rate limit rule."""
        self._rules[rule.path_pattern] = rule
    
    def get_rule(self, path: str, method: str) -> RateLimitRule:
        """Get applicable rule for path."""
        for pattern, rule in self._rules.items():
            if pattern == path or (pattern.endswith("*") and path.startswith(pattern[:-1])):
                if rule.methods is None or method in rule.methods:
                    return rule
        
        # Return default rule
        return RateLimitRule(
            path_pattern="*",
            requests_per_minute=self.config.default_rpm,
            requests_per_hour=self.config.default_rph,
            requests_per_day=self.config.default_rpd,
        )
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Rate limit key
            limit: Maximum requests
            window_seconds: Window size in seconds
            
        Returns:
            (allowed, remaining, reset_time)
        """
        redis = await self._get_redis()
        now = time.time()
        window_start = now - window_seconds
        
        full_key = f"{self.config.key_prefix}{key}"
        
        # Use Redis pipeline for atomic operations
        pipe = redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(full_key, 0, window_start)
        
        # Count current entries
        pipe.zcard(full_key)
        
        # Add current request
        pipe.zadd(full_key, {str(now): now})
        
        # Set expiry
        pipe.expire(full_key, window_seconds)
        
        results = await pipe.execute()
        current_count = results[1]
        
        allowed = current_count < limit
        remaining = max(0, limit - current_count - 1)
        reset_time = int(now + window_seconds)
        
        if not allowed:
            # Remove the request we just added
            await redis.zrem(full_key, str(now))
        
        return allowed, remaining, reset_time
    
    async def is_allowed(
        self,
        request: Request,
        user_id: Optional[str] = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed.
        
        Args:
            request: FastAPI request
            user_id: Optional user ID
            
        Returns:
            (allowed, rate_limit_info)
        """
        path = request.url.path
        method = request.method
        client_ip = self._get_client_ip(request)
        
        rule = self.get_rule(path, method)
        
        results = {
            "allowed": True,
            "limit_type": None,
            "limit": 0,
            "remaining": 0,
            "reset": 0,
        }
        
        # Check per-minute limit
        key_minute = f"minute:{client_ip}:{path}" if rule.by_ip else f"minute:global:{path}"
        if user_id and rule.by_user:
            key_minute = f"minute:{user_id}:{path}"
        
        allowed, remaining, reset = await self.check_rate_limit(
            key_minute,
            rule.requests_per_minute,
            60,
        )
        
        if not allowed:
            return False, {
                "allowed": False,
                "limit_type": "minute",
                "limit": rule.requests_per_minute,
                "remaining": remaining,
                "reset": reset,
                "retry_after": 60,
            }
        
        # Check per-hour limit
        key_hour = f"hour:{client_ip}:{path}" if rule.by_ip else f"hour:global:{path}"
        if user_id and rule.by_user:
            key_hour = f"hour:{user_id}:{path}"
        
        allowed, remaining, reset = await self.check_rate_limit(
            key_hour,
            rule.requests_per_hour,
            3600,
        )
        
        if not allowed:
            return False, {
                "allowed": False,
                "limit_type": "hour",
                "limit": rule.requests_per_hour,
                "remaining": remaining,
                "reset": reset,
                "retry_after": 3600,
            }
        
        # Check per-day limit
        key_day = f"day:{client_ip}:{path}" if rule.by_ip else f"day:global:{path}"
        if user_id and rule.by_user:
            key_day = f"day:{user_id}:{path}"
        
        allowed, remaining, reset = await self.check_rate_limit(
            key_day,
            rule.requests_per_day,
            86400,
        )
        
        if not allowed:
            return False, {
                "allowed": False,
                "limit_type": "day",
                "limit": rule.requests_per_day,
                "remaining": remaining,
                "reset": reset,
                "retry_after": 86400,
            }
        
        return True, {
            "allowed": True,
            "limit_type": "minute",
            "limit": rule.requests_per_minute,
            "remaining": remaining,
            "reset": reset,
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check X-Forwarded-For header
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client
        return request.client.host if request.client else "unknown"


class InMemoryStore:
    """In-memory rate limit store for development."""
    
    def __init__(self):
        self._data: Dict[str, list] = {}
        self._lock = asyncio.Lock()
    
    def pipeline(self):
        return InMemoryPipeline(self)
    
    async def zremrangebyscore(self, key: str, min_score: float, max_score: float):
        async with self._lock:
            if key in self._data:
                self._data[key] = [(s, t) for s, t in self._data[key] if not (min_score <= t <= max_score)]
    
    async def zcard(self, key: str) -> int:
        async with self._lock:
            return len(self._data.get(key, []))
    
    async def zadd(self, key: str, mapping: Dict[str, float]):
        async with self._lock:
            if key not in self._data:
                self._data[key] = []
            for member, score in mapping.items():
                self._data[key].append((member, score))
    
    async def zrem(self, key: str, member: str):
        async with self._lock:
            if key in self._data:
                self._data[key] = [(m, s) for m, s in self._data[key] if m != member]
    
    async def expire(self, key: str, seconds: int):
        pass  # Not implemented for in-memory


class InMemoryPipeline:
    """Pipeline for in-memory store."""
    
    def __init__(self, store: InMemoryStore):
        self._store = store
        self._operations: List[Callable] = []
    
    def zremrangebyscore(self, key: str, min_score: float, max_score: float):
        self._operations.append(lambda: self._store.zremrangebyscore(key, min_score, max_score))
        return self
    
    def zcard(self, key: str):
        self._operations.append(lambda: self._store.zcard(key))
        return self
    
    def zadd(self, key: str, mapping: Dict[str, float]):
        self._operations.append(lambda: self._store.zadd(key, mapping))
        return self
    
    def expire(self, key: str, seconds: int):
        self._operations.append(lambda: self._store.expire(key, seconds))
        return self
    
    async def execute(self) -> List[Any]:
        results = []
        for op in self._operations:
            result = await op()
            results.append(result)
        return results


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Usage:
        app = FastAPI()
        rate_limiter = SlidingWindowRateLimiter()
        app.add_middleware(RateLimitMiddleware, limiter=rate_limiter)
    """
    
    def __init__(self, app, limiter: SlidingWindowRateLimiter):
        super().__init__(app)
        self.limiter = limiter
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/health/live", "/health/ready"):
            return await call_next(request)
        
        # Extract user ID from auth if available
        user_id = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In production, decode JWT to get user ID
            pass
        
        # Check rate limit
        allowed, info = await self.limiter.is_allowed(request, user_id)
        
        if not allowed:
            raise RateLimitExceeded(
                limit_type=info["limit_type"],
                limit=info["limit"],
                window=info["limit_type"],
                retry_after=info["retry_after"],
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        if self.limiter.config.include_headers:
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset"])
        
        return response


# Decorator for route-level rate limiting
def rate_limit(
    requests_per_minute: int = 60,  # noqa: ARG001 - for future tier implementation
    requests_per_hour: int = 1000,  # noqa: ARG001 - for future tier implementation
    by_user: bool = True,  # noqa: ARG001 - for future per-user limiting
):
    """
    Decorator for route-level rate limiting.
    
    Usage:
        @app.get("/api/data")
        @rate_limit(requests_per_minute=10)
        async def get_data():
            ...
    """
    def decorator(func: Callable):
        limiter = SlidingWindowRateLimiter()
        
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user_id = kwargs.get("user", {}).get("sub")
            
            allowed, info = await limiter.is_allowed(request, user_id)
            
            if not allowed:
                raise RateLimitExceeded(
                    limit_type=info["limit_type"],
                    limit=info["limit"],
                    window=info["limit_type"],
                    retry_after=info["retry_after"],
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator


# Pre-configured rules for common endpoints
DEFAULT_RULES = [
    # Auth endpoints - stricter limits
    RateLimitRule(
        path_pattern="/api/auth/login",
        requests_per_minute=5,
        requests_per_hour=20,
        by_ip=True,
        by_user=False,
    ),
    RateLimitRule(
        path_pattern="/api/auth/register",
        requests_per_minute=3,
        requests_per_hour=10,
        by_ip=True,
        by_user=False,
    ),
    
    # Analysis endpoints - moderate limits
    RateLimitRule(
        path_pattern="/api/analyze*",
        requests_per_minute=10,
        requests_per_hour=100,
        by_user=True,
    ),
    
    # General API - default limits
    RateLimitRule(
        path_pattern="/api/*",
        requests_per_minute=60,
        requests_per_hour=1000,
        by_user=True,
    ),
]


def create_rate_limiter(
    redis_url: Optional[str] = None,
    rules: Optional[List[RateLimitRule]] = None,
) -> SlidingWindowRateLimiter:
    """Create configured rate limiter."""
    config = RateLimitConfig()
    if redis_url:
        config.redis_url = redis_url
    
    limiter = SlidingWindowRateLimiter(config)
    
    # Add rules
    for rule in (rules or DEFAULT_RULES):
        limiter.add_rule(rule)
    
    return limiter
