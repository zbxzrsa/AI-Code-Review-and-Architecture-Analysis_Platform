"""
Role-Based Access Control (RBAC) System

Comprehensive RBAC implementation with:
- Role hierarchy and permissions matrix
- Fine-grained permission checks
- Unauthorized access logging
- Admin function isolation
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from functools import wraps

from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)


# =============================================================================
# Role Definitions
# =============================================================================

class Role(str, Enum):
    """User roles with hierarchical levels."""
    SUPER_ADMIN = "super_admin"  # Full system access
    ADMIN = "admin"             # Administrative access
    USER = "user"               # Standard user access
    VIEWER = "viewer"           # Read-only access
    GUEST = "guest"             # Minimal access


# Role hierarchy - higher number = more privileges
ROLE_HIERARCHY: Dict[str, int] = {
    Role.SUPER_ADMIN: 100,
    Role.ADMIN: 80,
    Role.USER: 50,
    Role.VIEWER: 20,
    Role.GUEST: 0,
}


# =============================================================================
# Permission Definitions
# =============================================================================

class Permission(str, Enum):
    """Fine-grained permissions."""
    # Admin-only permissions
    ADMIN_ALL = "admin:all"
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_PROVIDERS = "admin:providers"
    ADMIN_EXPERIMENTS = "admin:experiments"
    ADMIN_AUDIT = "admin:audit"
    ADMIN_AI_MODELS = "admin:ai_models"
    ADMIN_SECURITY = "admin:security"

    # Project permissions
    PROJECT_CREATE = "project:create"
    PROJECT_READ = "project:read"
    PROJECT_UPDATE = "project:update"
    PROJECT_DELETE = "project:delete"
    PROJECT_ADMIN = "project:admin"

    # Analysis permissions
    ANALYSIS_CREATE = "analysis:create"
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_UPDATE = "analysis:update"
    ANALYSIS_DELETE = "analysis:delete"

    # Code review permissions
    REVIEW_CREATE = "review:create"
    REVIEW_READ = "review:read"
    REVIEW_APPROVE = "review:approve"
    REVIEW_REJECT = "review:reject"

    # Report permissions
    REPORT_CREATE = "report:create"
    REPORT_READ = "report:read"
    REPORT_EXPORT = "report:export"

    # Settings permissions
    SETTINGS_READ = "settings:read"
    SETTINGS_UPDATE = "settings:update"

    # API key permissions
    API_KEY_CREATE = "api_key:create"
    API_KEY_READ = "api_key:read"
    API_KEY_DELETE = "api_key:delete"


# Permission matrix by role
ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    Role.SUPER_ADMIN: {
        Permission.ADMIN_ALL,
        Permission.ADMIN_USERS,
        Permission.ADMIN_SYSTEM,
        Permission.ADMIN_PROVIDERS,
        Permission.ADMIN_EXPERIMENTS,
        Permission.ADMIN_AUDIT,
        Permission.ADMIN_AI_MODELS,
        Permission.ADMIN_SECURITY,
        Permission.PROJECT_CREATE,
        Permission.PROJECT_READ,
        Permission.PROJECT_UPDATE,
        Permission.PROJECT_DELETE,
        Permission.PROJECT_ADMIN,
        Permission.ANALYSIS_CREATE,
        Permission.ANALYSIS_READ,
        Permission.ANALYSIS_UPDATE,
        Permission.ANALYSIS_DELETE,
        Permission.REVIEW_CREATE,
        Permission.REVIEW_READ,
        Permission.REVIEW_APPROVE,
        Permission.REVIEW_REJECT,
        Permission.REPORT_CREATE,
        Permission.REPORT_READ,
        Permission.REPORT_EXPORT,
        Permission.SETTINGS_READ,
        Permission.SETTINGS_UPDATE,
        Permission.API_KEY_CREATE,
        Permission.API_KEY_READ,
        Permission.API_KEY_DELETE,
    },
    Role.ADMIN: {
        Permission.ADMIN_USERS,
        Permission.ADMIN_SYSTEM,
        Permission.ADMIN_PROVIDERS,
        Permission.ADMIN_EXPERIMENTS,
        Permission.ADMIN_AUDIT,
        Permission.ADMIN_AI_MODELS,
        Permission.PROJECT_CREATE,
        Permission.PROJECT_READ,
        Permission.PROJECT_UPDATE,
        Permission.PROJECT_DELETE,
        Permission.PROJECT_ADMIN,
        Permission.ANALYSIS_CREATE,
        Permission.ANALYSIS_READ,
        Permission.ANALYSIS_UPDATE,
        Permission.ANALYSIS_DELETE,
        Permission.REVIEW_CREATE,
        Permission.REVIEW_READ,
        Permission.REVIEW_APPROVE,
        Permission.REVIEW_REJECT,
        Permission.REPORT_CREATE,
        Permission.REPORT_READ,
        Permission.REPORT_EXPORT,
        Permission.SETTINGS_READ,
        Permission.SETTINGS_UPDATE,
        Permission.API_KEY_CREATE,
        Permission.API_KEY_READ,
        Permission.API_KEY_DELETE,
    },
    Role.USER: {
        Permission.PROJECT_CREATE,
        Permission.PROJECT_READ,
        Permission.PROJECT_UPDATE,
        Permission.ANALYSIS_CREATE,
        Permission.ANALYSIS_READ,
        Permission.ANALYSIS_UPDATE,
        Permission.REVIEW_CREATE,
        Permission.REVIEW_READ,
        Permission.REPORT_READ,
        Permission.SETTINGS_READ,
        Permission.SETTINGS_UPDATE,
        Permission.API_KEY_CREATE,
        Permission.API_KEY_READ,
        Permission.API_KEY_DELETE,
    },
    Role.VIEWER: {
        Permission.PROJECT_READ,
        Permission.ANALYSIS_READ,
        Permission.REVIEW_READ,
        Permission.REPORT_READ,
        Permission.SETTINGS_READ,
    },
    Role.GUEST: {
        Permission.PROJECT_READ,
    },
}


# =============================================================================
# Admin Functions Registry
# =============================================================================

# Routes that are admin-only (cannot be accessed by regular users)
ADMIN_ONLY_ROUTES: Set[str] = {
    "/api/admin/users",
    "/api/admin/users/stats",
    "/api/admin/users/suspend",
    "/api/admin/users/reactivate",
    "/api/admin/users/reset-password",
    "/api/admin/users/bulk",
    "/api/admin/providers",
    "/api/admin/providers/models",
    "/api/admin/providers/metrics",
    "/api/admin/providers/test",
    "/api/admin/experiments",
    "/api/admin/audit",
    "/api/admin/audit/analytics",
    "/api/admin/audit/security-alerts",
    "/api/admin/ai-models",
    "/api/admin/system",
    "/api/admin/system-health",
    "/api/admin/ai-testing",
}


# =============================================================================
# Unauthorized Access Logging
# =============================================================================

@dataclass
class AccessAttempt:
    """Record of an access attempt."""
    timestamp: str
    user_id: Optional[str]
    user_role: Optional[str]
    requested_path: str
    requested_method: str
    required_role: Optional[str]
    required_permission: Optional[str]
    granted: bool
    denial_reason: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]


class AccessAuditLogger:
    """Logger for unauthorized access attempts."""

    _attempts: List[AccessAttempt] = []
    _max_attempts: int = 10000

    @classmethod
    def log_access_attempt(
        cls,
        user_id: Optional[str],
        user_role: Optional[str],
        requested_path: str,
        requested_method: str,
        granted: bool,
        denial_reason: Optional[str] = None,
        required_role: Optional[str] = None,
        required_permission: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Log an access attempt."""
        attempt = AccessAttempt(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            user_role=user_role,
            requested_path=requested_path,
            requested_method=requested_method,
            required_role=required_role,
            required_permission=required_permission,
            granted=granted,
            denial_reason=denial_reason,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        cls._attempts.append(attempt)

        # Trim old entries
        if len(cls._attempts) > cls._max_attempts:
            cls._attempts = cls._attempts[-cls._max_attempts:]

        # Log to system logger
        if granted:
            logger.debug(
                f"Access granted: user={user_id} role={user_role} "
                f"path={requested_path} method={requested_method}"
            )
        else:
            logger.warning(
                f"Access DENIED: user={user_id} role={user_role} "
                f"path={requested_path} method={requested_method} "
                f"reason={denial_reason}"
            )

    @classmethod
    def get_recent_denials(cls, limit: int = 100) -> List[AccessAttempt]:
        """Get recent access denials."""
        denials = [a for a in cls._attempts if not a.granted]
        return denials[-limit:]

    @classmethod
    def get_user_denials(cls, user_id: str, limit: int = 50) -> List[AccessAttempt]:
        """Get denials for a specific user."""
        denials = [a for a in cls._attempts if not a.granted and a.user_id == user_id]
        return denials[-limit:]

    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Get access statistics."""
        total = len(cls._attempts)
        denials = sum(1 for a in cls._attempts if not a.granted)

        return {
            "total_attempts": total,
            "denied_attempts": denials,
            "denial_rate": denials / total if total > 0 else 0,
            "recent_denials": len(cls.get_recent_denials(100)),
        }


# =============================================================================
# RBAC Core Functions
# =============================================================================

def get_role_level(role: str) -> int:
    """Get the hierarchy level for a role."""
    return ROLE_HIERARCHY.get(role, 0)


def get_role_permissions(role: str) -> Set[str]:
    """Get all permissions for a role."""
    return ROLE_PERMISSIONS.get(role, set())


def has_role(user_role: str, required_role: str) -> bool:
    """Check if user role meets or exceeds required role."""
    user_level = get_role_level(user_role)
    required_level = get_role_level(required_role)
    return user_level >= required_level


def has_permission(user_role: str, permission: str) -> bool:
    """Check if user role has the required permission."""
    user_permissions = get_role_permissions(user_role)

    # Check for admin:all which grants all permissions
    if Permission.ADMIN_ALL in user_permissions:
        return True

    return permission in user_permissions


def has_any_permission(user_role: str, permissions: List[str]) -> bool:
    """Check if user role has any of the required permissions."""
    return any(has_permission(user_role, p) for p in permissions)


def has_all_permissions(user_role: str, permissions: List[str]) -> bool:
    """Check if user role has all required permissions."""
    return all(has_permission(user_role, p) for p in permissions)


def is_admin_route(path: str) -> bool:
    """Check if a route is admin-only."""
    # Exact match
    if path in ADMIN_ONLY_ROUTES:
        return True

    # Prefix match for admin routes
    if path.startswith("/api/admin/"):
        return True

    return False


def is_admin(user_role: str) -> bool:
    """Check if user has admin privileges."""
    return has_role(user_role, Role.ADMIN)


# =============================================================================
# FastAPI Dependencies
# =============================================================================

security = HTTPBearer()


class RBACDependency:
    """FastAPI dependency for RBAC checks."""

    def __init__(
        self,
        required_role: Optional[str] = None,
        required_permission: Optional[str] = None,
        required_permissions: Optional[List[str]] = None,
        require_all: bool = True,
    ):
        self.required_role = required_role
        self.required_permission = required_permission
        self.required_permissions = required_permissions or []
        self.require_all = require_all

    async def __call__(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> Dict[str, Any]:
        """Verify access and return user info."""
        from .auth import TokenManager

        # Extract user info from token
        try:
            token = credentials.credentials
            payload = TokenManager.verify_token(token, token_type="access")
            user_id = payload.get("sub")
            user_role = payload.get("role", "guest")
        except Exception:
            AccessAuditLogger.log_access_attempt(
                user_id=None,
                user_role=None,
                requested_path=str(request.url.path),
                requested_method=request.method,
                granted=False,
                denial_reason="Invalid or missing token",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication token",
            )

        # Check admin route protection
        if is_admin_route(str(request.url.path)) and not is_admin(user_role):
            AccessAuditLogger.log_access_attempt(
                user_id=user_id,
                user_role=user_role,
                requested_path=str(request.url.path),
                requested_method=request.method,
                granted=False,
                denial_reason="Admin route accessed by non-admin",
                required_role="admin",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Administrator access required",
                    "message": "This function is only available to administrators.",
                    "contact": "Please contact your system administrator for access.",
                    "code": "ADMIN_REQUIRED",
                },
            )

        # Check role requirement
        if self.required_role and not has_role(user_role, self.required_role):
            AccessAuditLogger.log_access_attempt(
                user_id=user_id,
                user_role=user_role,
                requested_path=str(request.url.path),
                requested_method=request.method,
                granted=False,
                denial_reason=f"Insufficient role: requires {self.required_role}",
                required_role=self.required_role,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "message": f"This action requires {self.required_role} role or higher.",
                    "your_role": user_role,
                    "required_role": self.required_role,
                    "contact": "Contact your administrator if you need elevated access.",
                    "code": "ROLE_REQUIRED",
                },
            )

        # Check single permission
        if self.required_permission and not has_permission(user_role, self.required_permission):
            AccessAuditLogger.log_access_attempt(
                user_id=user_id,
                user_role=user_role,
                requested_path=str(request.url.path),
                requested_method=request.method,
                granted=False,
                denial_reason=f"Missing permission: {self.required_permission}",
                required_permission=self.required_permission,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Permission denied",
                    "message": f"You don't have the required permission: {self.required_permission}",
                    "contact": "Contact your administrator to request this permission.",
                    "code": "PERMISSION_DENIED",
                },
            )

        # Check multiple permissions
        if self.required_permissions:
            if self.require_all:
                has_perms = has_all_permissions(user_role, self.required_permissions)
            else:
                has_perms = has_any_permission(user_role, self.required_permissions)

            if not has_perms:
                AccessAuditLogger.log_access_attempt(
                    user_id=user_id,
                    user_role=user_role,
                    requested_path=str(request.url.path),
                    requested_method=request.method,
                    granted=False,
                    denial_reason=f"Missing permissions: {self.required_permissions}",
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "Permission denied",
                        "message": "You don't have the required permissions for this action.",
                        "required_permissions": self.required_permissions,
                        "contact": "Contact your administrator to request these permissions.",
                        "code": "PERMISSIONS_DENIED",
                    },
                )

        # Access granted
        AccessAuditLogger.log_access_attempt(
            user_id=user_id,
            user_role=user_role,
            requested_path=str(request.url.path),
            requested_method=request.method,
            granted=True,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return {
            "id": user_id,
            "role": user_role,
            "permissions": list(get_role_permissions(user_role)),
        }


# Convenience dependencies
def require_admin():
    """Require admin role."""
    return RBACDependency(required_role=Role.ADMIN)


def require_user():
    """Require user role or higher."""
    return RBACDependency(required_role=Role.USER)


def require_viewer():
    """Require viewer role or higher."""
    return RBACDependency(required_role=Role.VIEWER)


def require_permission(permission: str):
    """Require specific permission."""
    return RBACDependency(required_permission=permission)


def require_any_permission(*permissions: str):
    """Require any of the specified permissions."""
    return RBACDependency(required_permissions=list(permissions), require_all=False)


def require_all_permissions(*permissions: str):
    """Require all specified permissions."""
    return RBACDependency(required_permissions=list(permissions), require_all=True)


# =============================================================================
# API Response for Permission Info
# =============================================================================

def get_user_permissions_response(user_role: str) -> Dict[str, Any]:
    """Get user permissions as API response."""
    permissions = get_role_permissions(user_role)

    return {
        "role": user_role,
        "role_level": get_role_level(user_role),
        "is_admin": is_admin(user_role),
        "permissions": list(permissions),
        "admin_permissions": [p for p in permissions if str(p).startswith("admin:")],
        "can_access_admin_panel": is_admin(user_role),
    }


# =============================================================================
# Admin Contact Information
# =============================================================================

ADMIN_CONTACT_INFO = {
    "email": "admin@coderev.example.com",
    "support_url": "https://coderev.example.com/support",
    "documentation_url": "https://coderev.example.com/docs/permissions",
}


def get_access_denied_response(
    error_code: str,
    message: str,
    required_role: Optional[str] = None,
    required_permission: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate standardized access denied response."""
    return {
        "error": "Access Denied",
        "code": error_code,
        "message": message,
        "required_role": required_role,
        "required_permission": required_permission,
        "contact": ADMIN_CONTACT_INFO,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    # Enums
    "Role",
    "Permission",
    # Constants
    "ROLE_HIERARCHY",
    "ROLE_PERMISSIONS",
    "ADMIN_ONLY_ROUTES",
    "ADMIN_CONTACT_INFO",
    # Functions
    "get_role_level",
    "get_role_permissions",
    "has_role",
    "has_permission",
    "has_any_permission",
    "has_all_permissions",
    "is_admin_route",
    "is_admin",
    "get_user_permissions_response",
    "get_access_denied_response",
    # Classes
    "AccessAttempt",
    "AccessAuditLogger",
    "RBACDependency",
    # Dependencies
    "require_admin",
    "require_user",
    "require_viewer",
    "require_permission",
    "require_any_permission",
    "require_all_permissions",
]
