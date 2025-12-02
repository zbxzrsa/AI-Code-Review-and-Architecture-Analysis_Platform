"""
Rate limiting middleware for auth service.
"""
import logging
import time
from typing import Callable, Dict, List
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter for auth endpoints.
    
    In production, use Redis-based rate limiting.
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._requests: Dict[str, List[float]] = {}
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        now = time.time()
        minute_ago = now - 60
        
        # Initialize or clean up old requests
        if client_ip not in self._requests:
            self._requests[client_ip] = []
        
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > minute_ago
        ]
        
        # Check limit
        if len(self._requests[client_ip]) >= self.rpm:
            return True
        
        # Record request
        self._requests[client_ip].append(now)
        return False
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)
