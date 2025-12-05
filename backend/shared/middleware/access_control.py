"""
Access Control Middleware

Enforces the three-version access control rules:
- Users: Can ONLY access Code Review AI (CR-AI) on V2 (stable)
- Admins: Can access both CR-AI and VC-AI on all versions
- System: Full access for self-evolution cycle

Version Control AI (VC-AI) is NEVER accessible to regular users.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime, timezone import datetime, timezone
from enum import Enum

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles for access control."""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"


class AIModelType(str, Enum):
    """AI model types."""
    CODE_REVIEW = "cr-ai"      # User-facing
    VERSION_CONTROL = "vc-ai"  # Admin-only


# =============================================================================
# Access Control Rules
# =============================================================================

ACCESS_RULES = {
    # V1 Experimentation - Admin/System only
    "/api/v1/cr-ai": [UserRole.ADMIN, UserRole.SYSTEM],
    "/api/v1/vc-ai": [UserRole.ADMIN, UserRole.SYSTEM],
    
    # V2 Production - Users can access CR-AI, Admins can access both
    "/api/v2/cr-ai": [UserRole.USER, UserRole.ADMIN, UserRole.SYSTEM],
    "/api/v2/vc-ai": [UserRole.ADMIN, UserRole.SYSTEM],  # NEVER user access
    
    # V3 Quarantine - Admin/System only
    "/api/v3/cr-ai": [UserRole.ADMIN, UserRole.SYSTEM],
    "/api/v3/vc-ai": [UserRole.ADMIN, UserRole.SYSTEM],
}

# Public endpoints (no auth required)
PUBLIC_ENDPOINTS = [
    "/health",
    "/ready",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
]


class AccessControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce AI model access control.
    
    Critical Rules:
    1. Regular users can ONLY access /api/v2/cr-ai/*
    2. Version Control AI is NEVER accessible to regular users
    3. V1 and V3 are admin-only zones
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process request with access control."""
        path = request.url.path
        
        # Skip public endpoints
        if self._is_public_endpoint(path):
            return await call_next(request)
        
        # Get user role
        user_role = self._get_user_role(request)
        
        # Check access
        allowed = self._check_access(path, user_role)
        
        if not allowed:
            # Log access denial
            logger.warning(
                f"Access denied: role={user_role.value}, path={path}, "
                f"ip={request.client.host}"
            )
            
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Access Denied",
                    "message": self._get_denial_message(path, user_role),
                    "path": path,
                    "your_role": user_role.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        
        # Log successful access
        logger.debug(f"Access granted: role={user_role.value}, path={path}")
        
        # Add role header for downstream services
        response = await call_next(request)
        response.headers["X-User-Role"] = user_role.value
        
        return response
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public."""
        return any(path.endswith(ep) for ep in PUBLIC_ENDPOINTS)
    
    def _get_user_role(self, request: Request) -> UserRole:
        """Extract user role from request."""
        # Check Authorization header for JWT
        auth_header = request.headers.get("Authorization", "")
        
        if auth_header.startswith("Bearer "):
            # In production, decode JWT and extract role
            # For now, check X-User-Role header (set by auth service)
            pass
        
        # Get role from header (set by auth service or gateway)
        # FIXED: Default to guest, not user (secure by default)
        role_header = request.headers.get("X-User-Role", "guest")
        
        # Also check for system API key
        api_key = request.headers.get("X-API-Key", "")
        if api_key and self._is_system_api_key(api_key):
            return UserRole.SYSTEM
        
        try:
            return UserRole(role_header.lower())
        except ValueError:
            return UserRole.USER
    
    def _is_system_api_key(self, api_key: str) -> bool:
        """Check if API key is a system key."""
        import os
        # FIXED: Validate against actual stored keys, not just prefix
        valid_system_keys = os.getenv("SYSTEM_API_KEYS", "").split(",")
        valid_system_keys = [k.strip() for k in valid_system_keys if k.strip()]
        return api_key in valid_system_keys
    
    def _check_access(self, path: str, user_role: UserRole) -> bool:
        """Check if role can access path."""
        # Find matching rule
        for endpoint, allowed_roles in ACCESS_RULES.items():
            if path.startswith(endpoint):
                return user_role in allowed_roles
        
        # FIXED: Default deny for unknown paths (secure by default)
        logger.warning(f"Access denied for unknown path: {path}")
        return False
    
    def _get_denial_message(self, path: str, user_role: UserRole) -> str:
        """Get appropriate denial message."""
        if "/vc-ai" in path:
            return (
                "Version Control AI is restricted to administrators. "
                "Regular users can only access Code Review AI on V2."
            )
        
        if "/v1/" in path:
            return (
                "V1 (Experimentation) is restricted to administrators. "
                "Please use V2 (Production) for code reviews."
            )
        
        if "/v3/" in path:
            return (
                "V3 (Quarantine) is restricted to administrators. "
                "This version contains archived/deprecated features."
            )
        
        return "You do not have permission to access this resource."


# =============================================================================
# Route Guards
# =============================================================================

def require_admin(request: Request):
    """Dependency to require admin role."""
    role_header = request.headers.get("X-User-Role", "user")
    
    if role_header not in ["admin", "system"]:
        raise HTTPException(
            status_code=403,
            detail="Admin access required",
        )


def require_system(request: Request):
    """Dependency to require system role."""
    role_header = request.headers.get("X-User-Role", "user")
    
    if role_header != "system":
        raise HTTPException(
            status_code=403,
            detail="System access required",
        )


def require_user_or_above(request: Request):
    """Dependency to require at least user role."""
    role_header = request.headers.get("X-User-Role", "guest")
    
    if role_header == "guest":
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )


# =============================================================================
# Access Control Helper Functions
# =============================================================================

def get_user_accessible_endpoints(user_role: UserRole) -> Dict[str, List[str]]:
    """Get all endpoints accessible by a user role."""
    accessible = {}
    
    for endpoint, allowed_roles in ACCESS_RULES.items():
        if user_role in allowed_roles:
            version = endpoint.split("/")[2]  # e.g., "v1", "v2", "v3"
            
            if version not in accessible:
                accessible[version] = []
            
            accessible[version].append(endpoint)
    
    return accessible


def get_user_version() -> str:
    """Get the version available to regular users."""
    return "v2"


def can_user_access_vc_ai(user_role: UserRole) -> bool:
    """Check if user can access Version Control AI."""
    return user_role in [UserRole.ADMIN, UserRole.SYSTEM]


def can_user_access_cr_ai(user_role: UserRole, version: str) -> bool:
    """Check if user can access Code Review AI on a specific version."""
    if version == "v2":
        return user_role in [UserRole.USER, UserRole.ADMIN, UserRole.SYSTEM]
    else:
        return user_role in [UserRole.ADMIN, UserRole.SYSTEM]


# =============================================================================
# Audit Logging
# =============================================================================

async def log_access_attempt(
    request: Request,
    user_role: UserRole,
    path: str,
    allowed: bool,
):
    """Log access attempt for audit trail."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_role": user_role.value,
        "path": path,
        "method": request.method,
        "client_ip": request.client.host,
        "allowed": allowed,
        "user_agent": request.headers.get("User-Agent", "unknown"),
    }
    
    # In production, send to audit log service
    if not allowed:
        logger.warning(f"Access denied: {log_entry}")
    else:
        logger.info(f"Access granted: {log_entry}")
