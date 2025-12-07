"""
Core Infrastructure Components

Contains:
- config: System configuration management
- dependencies: Dependency injection management
- middleware: Custom middleware implementations

Usage:
    from dev_api.core import get_db, get_current_user, Settings
"""

from .config import Settings, get_settings
from .dependencies import (
    get_db,
    get_current_user,
    get_optional_user,
    require_admin,
    require_auth,
)
from .middleware import (
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    ErrorHandlerMiddleware,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Dependencies
    "get_db",
    "get_current_user",
    "get_optional_user",
    "require_admin",
    "require_auth",
    # Middleware
    "RequestLoggingMiddleware",
    "RateLimitMiddleware",
    "ErrorHandlerMiddleware",
]
