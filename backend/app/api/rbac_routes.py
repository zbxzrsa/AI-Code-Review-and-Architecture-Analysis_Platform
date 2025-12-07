"""
RBAC API Routes

API endpoints for role-based access control:
- Permission checks
- Access denial logging
- Admin contact info
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel

# Import RBAC modules
import sys
from pathlib import Path
_backend_path = Path(__file__).parent.parent.parent / "shared"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

from security.rbac import (
    Role,
    Permission,
    get_role_permissions,
    get_user_permissions_response,
    AccessAuditLogger,
    ADMIN_CONTACT_INFO,
    ADMIN_ONLY_ROUTES,
    require_admin,
    require_user,
    RBACDependency,
)
from security.auth import CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rbac", tags=["RBAC"])


# =============================================================================
# Request/Response Models
# =============================================================================

class AccessDenialLog(BaseModel):
    """Frontend access denial log."""
    userId: Optional[str] = None
    userRole: Optional[str] = None
    attemptedPath: str
    requiredRole: Optional[str] = None
    requiredPermission: Optional[str] = None
    timestamp: str
    featureId: Optional[str] = None


class PermissionCheckRequest(BaseModel):
    """Permission check request."""
    permission: str


class PermissionCheckResponse(BaseModel):
    """Permission check response."""
    has_permission: bool
    permission: str
    user_role: str


class UserPermissionsResponse(BaseModel):
    """User permissions response."""
    role: str
    role_level: int
    is_admin: bool
    permissions: List[str]
    admin_permissions: List[str]
    can_access_admin_panel: bool


class AdminContactResponse(BaseModel):
    """Admin contact information response."""
    email: str
    support_url: str
    documentation_url: str


class AccessStatisticsResponse(BaseModel):
    """Access statistics response."""
    total_attempts: int
    denied_attempts: int
    denial_rate: float
    recent_denials: int


# =============================================================================
# Public Endpoints
# =============================================================================

@router.get("/admin-contact", response_model=AdminContactResponse)
async def get_admin_contact():
    """
    Get administrator contact information.

    Available to all users for requesting access.
    """
    return AdminContactResponse(
        email=ADMIN_CONTACT_INFO["email"],
        support_url=ADMIN_CONTACT_INFO["support_url"],
        documentation_url=ADMIN_CONTACT_INFO["documentation_url"],
    )


@router.get("/admin-routes")
async def get_admin_routes():
    """
    Get list of admin-only routes.

    Useful for frontend to know which routes require admin access.
    """
    return {
        "admin_routes": list(ADMIN_ONLY_ROUTES),
        "admin_prefix": "/api/admin/",
    }


# =============================================================================
# Authenticated Endpoints
# =============================================================================

@router.get("/my-permissions", response_model=UserPermissionsResponse)
async def get_my_permissions(
    user: dict = Depends(CurrentUser.get_current_user)
):
    """
    Get current user's permissions.

    Returns role, permissions, and admin access status.
    """
    return get_user_permissions_response(user.get("role", "guest"))


@router.post("/check-permission", response_model=PermissionCheckResponse)
async def check_permission(
    request: PermissionCheckRequest,
    user: dict = Depends(CurrentUser.get_current_user)
):
    """
    Check if current user has a specific permission.
    """
    from security.rbac import has_permission

    user_role = user.get("role", "guest")

    return PermissionCheckResponse(
        has_permission=has_permission(user_role, request.permission),
        permission=request.permission,
        user_role=user_role,
    )


@router.post("/check-admin-access")
async def check_admin_access(
    user: dict = Depends(CurrentUser.get_current_user)
):
    """
    Check if current user has admin access.

    Returns 200 if admin, 403 if not.
    """
    from security.rbac import is_admin

    user_role = user.get("role", "guest")

    if not is_admin(user_role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Administrator access required",
                "message": "This function is only available to administrators.",
                "your_role": user_role,
                "required_role": "admin",
                "contact": ADMIN_CONTACT_INFO,
            }
        )

    return {
        "has_admin_access": True,
        "role": user_role,
    }


# =============================================================================
# Audit Logging Endpoints
# =============================================================================

@router.post("/log-access-denial")
async def log_access_denial(
    log: AccessDenialLog,
    request: Request,
):
    """
    Log an access denial from the frontend.

    This endpoint accepts unauthenticated requests to log
    access attempts that were denied on the frontend.
    """
    # Log the denial
    AccessAuditLogger.log_access_attempt(
        user_id=log.userId,
        user_role=log.userRole,
        requested_path=log.attemptedPath,
        requested_method="GET",  # Frontend typically logs GET denials
        granted=False,
        denial_reason=f"Frontend denial: requires {log.requiredRole or log.requiredPermission}",
        required_role=log.requiredRole,
        required_permission=log.requiredPermission,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    logger.warning(
        f"Frontend access denial logged: user={log.userId} "
        f"role={log.userRole} path={log.attemptedPath}"
    )

    return {"status": "logged"}


# Alias for frontend compatibility
@router.post("/audit/access-denial")
async def audit_access_denial(log: AccessDenialLog, request: Request):
    """Alias endpoint for frontend logging."""
    return await log_access_denial(log, request)


# =============================================================================
# Admin-Only Endpoints
# =============================================================================

@router.get(
    "/access-statistics",
    response_model=AccessStatisticsResponse,
    dependencies=[Depends(require_admin())]
)
async def get_access_statistics():
    """
    Get access statistics (admin only).

    Shows total attempts, denials, and denial rate.
    """
    stats = AccessAuditLogger.get_statistics()
    return AccessStatisticsResponse(**stats)


@router.get(
    "/recent-denials",
    dependencies=[Depends(require_admin())]
)
async def get_recent_denials(limit: int = 100):
    """
    Get recent access denials (admin only).

    Shows the most recent access denial attempts.
    """
    denials = AccessAuditLogger.get_recent_denials(limit)
    return {
        "denials": [
            {
                "timestamp": d.timestamp,
                "user_id": d.user_id,
                "user_role": d.user_role,
                "path": d.requested_path,
                "method": d.requested_method,
                "required_role": d.required_role,
                "required_permission": d.required_permission,
                "denial_reason": d.denial_reason,
                "ip_address": d.ip_address,
            }
            for d in denials
        ],
        "total": len(denials),
    }


@router.get(
    "/user-denials/{user_id}",
    dependencies=[Depends(require_admin())]
)
async def get_user_denials(user_id: str, limit: int = 50):
    """
    Get access denials for a specific user (admin only).
    """
    denials = AccessAuditLogger.get_user_denials(user_id, limit)
    return {
        "user_id": user_id,
        "denials": [
            {
                "timestamp": d.timestamp,
                "path": d.requested_path,
                "method": d.requested_method,
                "required_role": d.required_role,
                "denial_reason": d.denial_reason,
            }
            for d in denials
        ],
        "total": len(denials),
    }


@router.get(
    "/roles",
    dependencies=[Depends(require_admin())]
)
async def get_all_roles():
    """
    Get all available roles and their permissions (admin only).
    """
    from security.rbac import ROLE_PERMISSIONS, ROLE_HIERARCHY

    return {
        "roles": [
            {
                "name": role,
                "level": ROLE_HIERARCHY.get(role, 0),
                "permissions": list(permissions),
            }
            for role, permissions in ROLE_PERMISSIONS.items()
        ]
    }


@router.get(
    "/permissions",
    dependencies=[Depends(require_admin())]
)
async def get_all_permissions():
    """
    Get all available permissions (admin only).
    """
    return {
        "permissions": [p.value for p in Permission],
        "admin_permissions": [p.value for p in Permission if p.value.startswith("admin:")],
        "user_permissions": [p.value for p in Permission if not p.value.startswith("admin:")],
    }


__all__ = ["router"]
