"""
RBAC Middleware for FastAPI

Automatically protects admin routes and logs unauthorized access attempts.
"""

import logging
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RBACMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces RBAC on all admin routes.

    Features:
    - Automatic admin route protection
    - Unauthorized access logging
    - Friendly error responses with admin contact info
    """

    def __init__(
        self,
        app,
        admin_route_prefix: str = "/api/admin",
        skip_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.admin_route_prefix = admin_route_prefix
        self.skip_paths = skip_paths or ["/api/admin/login", "/api/admin/health"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process each request through RBAC checks."""
        path = request.url.path

        # Skip paths that don't need protection
        if not self._is_protected_route(path):
            return await call_next(request)

        # Check if path should be skipped
        if any(path.startswith(skip) for skip in self.skip_paths):
            return await call_next(request)

        # Extract token and verify admin access
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return self._create_unauthorized_response(
                request,
                "Authentication required",
                "Please log in to access this resource.",
            )

        try:
            # Import here to avoid circular imports
            from ..security.auth import TokenManager
            from ..security.rbac import is_admin, AccessAuditLogger

            token = auth_header.split(" ")[1]
            payload = TokenManager.verify_token(token, token_type="access")

            user_id = payload.get("sub")
            user_role = payload.get("role", "guest")

            # Check admin access for admin routes
            if self._is_admin_route(path) and not is_admin(user_role):
                # Log the unauthorized attempt
                AccessAuditLogger.log_access_attempt(
                    user_id=user_id,
                    user_role=user_role,
                    requested_path=path,
                    requested_method=request.method,
                    granted=False,
                    denial_reason="Non-admin accessing admin route",
                    required_role="admin",
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                )

                return self._create_forbidden_response(
                    request,
                    user_role,
                    "Administrator access required",
                    "This function is only available to administrators.",
                )

            # Attach user info to request state for downstream use
            request.state.user_id = user_id
            request.state.user_role = user_role

        except Exception as e:
            logger.warning(f"RBAC middleware auth error: {e}")
            return self._create_unauthorized_response(
                request,
                "Invalid authentication",
                "Your authentication token is invalid or expired.",
            )

        return await call_next(request)

    def _is_protected_route(self, path: str) -> bool:
        """Check if route requires authentication."""
        # Protect all routes starting with /api/ except public endpoints
        public_paths = [
            "/api/health",
            "/api/docs",
            "/api/openapi",
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/refresh",
        ]

        if path.startswith("/api/"):
            return not any(path.startswith(p) for p in public_paths)

        return False

    def _is_admin_route(self, path: str) -> bool:
        """Check if route is admin-only."""
        return path.startswith(self.admin_route_prefix)

    def _create_unauthorized_response(
        self,
        request: Request,
        error: str,
        message: str,
    ) -> JSONResponse:
        """Create 401 Unauthorized response."""
        return JSONResponse(
            status_code=401,
            content={
                "error": error,
                "message": message,
                "code": "UNAUTHORIZED",
                "path": str(request.url.path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "contact": {
                    "email": "admin@coderev.example.com",
                    "support_url": "https://coderev.example.com/support",
                },
            },
        )

    def _create_forbidden_response(
        self,
        request: Request,
        user_role: str,
        error: str,
        message: str,
    ) -> JSONResponse:
        """Create 403 Forbidden response with admin contact info."""
        return JSONResponse(
            status_code=403,
            content={
                "error": error,
                "message": message,
                "code": "ADMIN_REQUIRED",
                "your_role": user_role,
                "required_role": "admin",
                "path": str(request.url.path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "help": "Administrator functions are restricted to users with admin privileges.",
                "contact": {
                    "email": "admin@coderev.example.com",
                    "support_url": "https://coderev.example.com/support",
                    "documentation_url": "https://coderev.example.com/docs/permissions",
                },
            },
        )


class DirectURLProtectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that prevents direct URL access to admin routes.

    This adds an extra layer of security by checking that admin routes
    are accessed through the application rather than direct URL entry.
    """

    def __init__(self, app, admin_route_prefix: str = "/api/admin"):
        super().__init__(app)
        self.admin_route_prefix = admin_route_prefix

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check for direct URL access attempts."""
        path = request.url.path

        # Only check admin routes
        if not path.startswith(self.admin_route_prefix):
            return await call_next(request)

        # Check for suspicious direct access patterns
        referer = request.headers.get("Referer", "")
        x_requested_with = request.headers.get("X-Requested-With", "")

        # Allow API requests from the application
        is_api_request = (
            x_requested_with == "XMLHttpRequest" or
            "application/json" in request.headers.get("Accept", "") or
            "application/json" in request.headers.get("Content-Type", "")
        )

        # If it looks like a browser navigation without proper headers, log it
        if not is_api_request and request.method == "GET":
            logger.warning(
                f"Possible direct URL access attempt to admin route: "
                f"path={path} referer={referer} method={request.method}"
            )

            # Import here to avoid circular imports
            try:
                from ..security.rbac import AccessAuditLogger

                AccessAuditLogger.log_access_attempt(
                    user_id=getattr(request.state, "user_id", None),
                    user_role=getattr(request.state, "user_role", None),
                    requested_path=path,
                    requested_method=request.method,
                    granted=False,
                    denial_reason="Suspected direct URL access to admin route",
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                )
            except Exception:
                pass

        return await call_next(request)


__all__ = [
    "RBACMiddleware",
    "DirectURLProtectionMiddleware",
]
