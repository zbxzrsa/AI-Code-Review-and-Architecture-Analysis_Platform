"""
Redis-based Rate Limiter Middleware

Provides distributed rate limiting with:
- Sliding window algorithm
- Different limits for different endpoints
- IP and user-based limiting
- Proper 429 responses with Retry-After header
"""

import time
import hashlib
from typing import Optional, Callable
from functools import wraps

from fastapi import Request, HTTPException, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis

from app.core.config import settings


# ============================================
# Rate Limit Configuration
# ============================================

class RateLimitConfig:
    """Rate limit configuration for an endpoint"""
    def __init__(
        self,
        requests: int,
        window_seconds: int,
        key_prefix: str = "",
        per_user: bool = True,
        per_ip: bool = True,
    ):
        self.requests = requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self.per_user = per_user
        self.per_ip = per_ip


# Pre-configured rate limits
RATE_LIMITS = {
    # Authentication endpoints
    "/api/auth/login": RateLimitConfig(
        requests=5,
        window_seconds=15 * 60,  # 5 per 15 minutes
        key_prefix="login",
        per_ip=True,
        per_user=False,
    ),
    "/api/auth/register": RateLimitConfig(
        requests=3,
        window_seconds=60,  # 3 per minute
        key_prefix="register",
        per_ip=True,
        per_user=False,
    ),
    "/api/auth/2fa/verify": RateLimitConfig(
        requests=5,
        window_seconds=60,  # 5 per minute
        key_prefix="2fa",
        per_ip=True,
        per_user=False,
    ),
    
    # Password endpoints
    "/api/user/password/change": RateLimitConfig(
        requests=3,
        window_seconds=24 * 60 * 60,  # 3 per 24 hours
        key_prefix="pwd_change",
        per_user=True,
        per_ip=False,
    ),
    "/api/user/password/reset": RateLimitConfig(
        requests=3,
        window_seconds=5 * 60,  # 3 per 5 minutes
        key_prefix="pwd_reset",
        per_ip=True,
        per_user=False,
    ),
    
    # API key generation
    "/api/user/api-keys": RateLimitConfig(
        requests=10,
        window_seconds=60 * 60,  # 10 per hour
        key_prefix="api_keys",
        per_user=True,
        per_ip=False,
    ),
}

# Default rate limit for general API
DEFAULT_RATE_LIMIT = RateLimitConfig(
    requests=100,
    window_seconds=60,  # 100 per minute
    key_prefix="general",
    per_user=True,
    per_ip=True,
)


# ============================================
# Redis Rate Limiter
# ============================================

class RedisRateLimiter:
    """
    Distributed rate limiter using Redis with sliding window algorithm
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis: Optional[redis.Redis] = None
    
    async def get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis
    
    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None
    
    def _generate_key(
        self,
        config: RateLimitConfig,
        request: Request,
        user_id: Optional[str] = None,
    ) -> str:
        """Generate unique rate limit key"""
        parts = [f"rate_limit:{config.key_prefix}"]
        
        if config.per_ip:
            # Get client IP (handle proxies)
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                ip = forwarded.split(",")[0].strip()
            else:
                ip = request.client.host if request.client else "unknown"
            parts.append(f"ip:{hashlib.md5(ip.encode()).hexdigest()[:12]}")
        
        if config.per_user and user_id:
            parts.append(f"user:{user_id}")
        
        # Add email for login attempts
        if config.key_prefix == "login":
            body = getattr(request.state, "body", {})
            if isinstance(body, dict) and "email" in body:
                email_hash = hashlib.md5(body["email"].encode()).hexdigest()[:12]
                parts.append(f"email:{email_hash}")
        
        return ":".join(parts)
    
    async def check_rate_limit(
        self,
        request: Request,
        config: RateLimitConfig,
        user_id: Optional[str] = None,
    ) -> tuple[bool, int, int]:
        """
        Check if request is rate limited
        
        Returns:
            tuple: (is_allowed, remaining_requests, reset_time_seconds)
        """
        r = await self.get_redis()
        key = self._generate_key(config, request, user_id)
        now = time.time()
        window_start = now - config.window_seconds
        
        # Use pipeline for atomic operations
        pipe = r.pipeline()
        
        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiry
        pipe.expire(key, config.window_seconds)
        
        # Execute pipeline
        results = await pipe.execute()
        current_count = results[1]  # zcard result
        
        remaining = max(0, config.requests - current_count - 1)
        reset_time = int(now + config.window_seconds)
        
        is_allowed = current_count < config.requests
        
        if not is_allowed:
            # Remove the request we just added
            await r.zrem(key, str(now))
        
        return is_allowed, remaining, reset_time
    
    async def get_limit_info(
        self,
        request: Request,
        config: RateLimitConfig,
        user_id: Optional[str] = None,
    ) -> dict:
        """Get current rate limit info for headers"""
        r = await self.get_redis()
        key = self._generate_key(config, request, user_id)
        now = time.time()
        window_start = now - config.window_seconds
        
        # Clean old entries and get count
        pipe = r.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        results = await pipe.execute()
        
        current_count = results[1]
        remaining = max(0, config.requests - current_count)
        reset_time = int(now + config.window_seconds)
        
        return {
            "limit": config.requests,
            "remaining": remaining,
            "reset": reset_time,
            "window": config.window_seconds,
        }


# Global rate limiter instance
rate_limiter = RedisRateLimiter()


# ============================================
# Middleware
# ============================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for OPTIONS requests
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Get rate limit config for this endpoint
        path = request.url.path
        config = RATE_LIMITS.get(path, DEFAULT_RATE_LIMIT)
        
        # Get user ID from auth if available
        user_id = getattr(request.state, "user_id", None)
        
        # Check rate limit
        is_allowed, remaining, reset_time = await rate_limiter.check_rate_limit(
            request, config, user_id
        )
        
        if not is_allowed:
            retry_after = reset_time - int(time.time())
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(config.requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(config.requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


# ============================================
# Decorator for specific endpoints
# ============================================

def rate_limit(
    requests: int = 100,
    window_seconds: int = 60,
    key_prefix: str = "",
    per_user: bool = True,
    per_ip: bool = True,
):
    """
    Rate limit decorator for specific endpoints
    
    Usage:
        @app.post("/api/login")
        @rate_limit(requests=5, window_seconds=900)
        async def login(request: Request):
            ...
    """
    config = RateLimitConfig(
        requests=requests,
        window_seconds=window_seconds,
        key_prefix=key_prefix,
        per_user=per_user,
        per_ip=per_ip,
    )
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user_id = getattr(request.state, "user_id", None)
            
            is_allowed, remaining, reset_time = await rate_limiter.check_rate_limit(
                request, config, user_id
            )
            
            if not is_allowed:
                retry_after = reset_time - int(time.time())
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Too Many Requests",
                        "message": f"Rate limit exceeded. Try again in {retry_after} seconds.",
                        "retry_after": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(config.requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset_time),
                    },
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator


# ============================================
# Cleanup
# ============================================

async def shutdown_rate_limiter():
    """Call this on application shutdown"""
    await rate_limiter.close()
