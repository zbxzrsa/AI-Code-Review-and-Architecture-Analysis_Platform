"""
Middleware Package

Security middleware for FastAPI:
- Rate limiting (Redis-based)
- Security headers
- CSRF protection
"""

from .rate_limiter import (
    RateLimitMiddleware,
    RateLimitConfig,
    rate_limit,
    rate_limiter,
    shutdown_rate_limiter,
    RATE_LIMITS,
    DEFAULT_RATE_LIMIT,
)

from .security_headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig,
    create_security_headers_middleware,
    get_csp_policy,
)

__all__ = [
    # Rate limiting
    "RateLimitMiddleware",
    "RateLimitConfig",
    "rate_limit",
    "rate_limiter",
    "shutdown_rate_limiter",
    "RATE_LIMITS",
    "DEFAULT_RATE_LIMIT",
    # Security headers
    "SecurityHeadersMiddleware",
    "SecurityHeadersConfig",
    "create_security_headers_middleware",
    "get_csp_policy",
]
