"""
FastAPI 应用中间件模块 (Middleware modules for FastAPI applications)

模块功能描述:
    提供 FastAPI 应用的各类中间件。

主要功能:
    - 速率限制中间件
    - 滑动窗口限流器
    - 请求速率控制
    - RBAC 访问控制中间件

主要组件:
    - SlidingWindowRateLimiter: 滑动窗口限流器
    - RateLimitMiddleware: 速率限制中间件
    - RateLimitConfig: 速率限制配置
    - RBACMiddleware: 角色访问控制中间件
    - DirectURLProtectionMiddleware: URL直接访问保护

最后修改日期: 2024-12-07
"""
from .rate_limiter import (
    SlidingWindowRateLimiter,
    RateLimitConfig,
    RateLimitRule,
    RateLimitMiddleware,
    RateLimitExceeded,
    rate_limit,
    create_rate_limiter,
    DEFAULT_RULES,
)

from .rbac_middleware import (
    RBACMiddleware,
    DirectURLProtectionMiddleware,
)

__all__ = [
    # Rate Limiting
    "SlidingWindowRateLimiter",
    "RateLimitConfig",
    "RateLimitRule",
    "RateLimitMiddleware",
    "RateLimitExceeded",
    "rate_limit",
    "create_rate_limiter",
    "DEFAULT_RULES",
    # RBAC
    "RBACMiddleware",
    "DirectURLProtectionMiddleware",
]
