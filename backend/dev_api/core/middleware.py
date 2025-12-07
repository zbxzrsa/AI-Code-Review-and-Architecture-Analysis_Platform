"""
Middleware Implementations

Custom middleware for:
- Request logging
- Rate limiting
- Error handling
- Performance monitoring

Module Size: ~200 lines (target < 2000)
"""

import time
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, Any
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import status

from .config import get_settings

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all requests.
    
    Logs:
    - Request method and path
    - Response status code
    - Request duration
    - Client IP
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = f"req-{int(start_time * 1000)}"
        
        # Add request ID to state
        request.state.request_id = request_id
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request
            logger.info(
                f"{request.method} {request.url.path} "
                f"status={response.status_code} "
                f"duration={duration_ms:.1f}ms "
                f"client={client_ip} "
                f"request_id={request_id}"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"{request.method} {request.url.path} "
                f"error={str(e)} "
                f"duration={duration_ms:.1f}ms "
                f"request_id={request_id}"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.
    
    Uses a simple in-memory sliding window algorithm.
    For production, use Redis-backed rate limiting.
    """
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._request_counts: Dict[str, list] = defaultdict(list)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/", "/docs", "/openapi.json"):
            return await call_next(request)
        
        # Get client identifier (IP or API key)
        client_id = self._get_client_id(request)
        
        # Check rate limit
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old entries
        self._request_counts[client_id] = [
            ts for ts in self._request_counts[client_id]
            if ts > window_start
        ]
        
        # Check limit
        if len(self._request_counts[client_id]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after_seconds": 60,
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )
        
        # Record request
        self._request_counts[client_id].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.requests_per_minute - len(self._request_counts[client_id])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:8]}"
        
        # Fall back to IP
        return f"ip:{request.client.host if request.client else 'unknown'}"


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling uncaught exceptions.
    
    Provides consistent error responses and logging.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
            
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            
            logger.exception(
                f"Unhandled exception: {str(e)} "
                f"path={request.url.path} "
                f"request_id={request_id}"
            )
            
            # Don't expose internal errors in production
            settings = get_settings()
            detail = str(e) if settings.debug else "Internal server error"
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": detail,
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
