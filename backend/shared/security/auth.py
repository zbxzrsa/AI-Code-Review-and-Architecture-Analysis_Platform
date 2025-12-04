"""
Authentication and authorization utilities.

JWT token management, role-based access control, and security dependencies.
"""
import hashlib
import logging
import os
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from functools import wraps

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Configuration - SECURITY: No default fallback for secrets
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY environment variable must be set")

ALGORITHM = "HS256"
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "code-review-platform")
JWT_ISSUER = os.getenv("JWT_ISSUER", "auth-service")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

security = HTTPBearer()


class TokenManager:
    """Manage JWT tokens."""

    @staticmethod
    def create_access_token(
        user_id: str,
        role: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create access token."""
        try:
            if expires_delta is None:
                expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

            expire = datetime.now(timezone.utc) + expires_delta
            to_encode = {
                "sub": user_id,
                "role": role,
                "type": "access",
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "jti": str(uuid.uuid4()),  # JWT ID for revocation tracking
                "aud": JWT_AUDIENCE,
                "iss": JWT_ISSUER,
            }

            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            logger.debug(f"Access token created for user {user_id}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise

    @staticmethod
    def create_refresh_token(
        user_id: str,
        role: str = "user",
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create refresh token."""
        try:
            if expires_delta is None:
                expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

            expire = datetime.now(timezone.utc) + expires_delta
            to_encode = {
                "sub": user_id,
                "role": role,  # Include role for token refresh
                "type": "refresh",
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "jti": str(uuid.uuid4()),
                "aud": JWT_AUDIENCE,
                "iss": JWT_ISSUER,
            }

            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            logger.debug(f"Refresh token created for user {user_id}")
            return encoded_jwt
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise

    @staticmethod
    def create_tokens(user_id: str, role: str) -> tuple:
        """Create both access and refresh tokens."""
        try:
            access_token = TokenManager.create_access_token(user_id, role)
            refresh_token = TokenManager.create_refresh_token(user_id, role)
            return access_token, refresh_token
        except Exception as e:
            logger.error(f"Failed to create tokens: {e}")
            raise

    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode token with full validation."""
        try:
            payload = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=[ALGORITHM],
                options={
                    "verify_aud": True,
                    "verify_iss": True,
                    "require": ["exp", "sub", "type", "iat", "jti"]
                },
                audience=JWT_AUDIENCE,
                issuer=JWT_ISSUER,
            )

            # Verify token type
            if payload.get("type") != token_type:
                raise ValueError(f"Invalid token type: {payload.get('type')}")

            user_id = payload.get("sub")
            if user_id is None:
                raise ValueError("Invalid token: missing user_id")

            return payload
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    @staticmethod
    def refresh_access_token(refresh_token: str) -> str:
        """Create new access token from refresh token."""
        try:
            payload = TokenManager.verify_token(refresh_token, token_type="refresh")
            user_id = payload.get("sub")
            role = payload.get("role", "user")

            return TokenManager.create_access_token(user_id, role)
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            raise


class CurrentUser:
    """Current user dependency."""

    @staticmethod
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> Dict[str, Any]:
        """Get current user from token."""
        try:
            token = credentials.credentials
            payload = TokenManager.verify_token(token, token_type="access")

            user_id = payload.get("sub")
            role = payload.get("role")

            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )

            return {
                "id": user_id,
                "role": role,
                "exp": payload.get("exp")
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get current user: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

    @staticmethod
    async def get_current_admin(
        user: Dict[str, Any] = Depends(CurrentUser.get_current_user)
    ) -> Dict[str, Any]:
        """Get current user with admin role."""
        if user.get("role") != "admin":
            logger.warning(f"Non-admin user {user.get('id')} attempted admin access")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return user

    @staticmethod
    async def get_current_user_optional(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
    ) -> Optional[Dict[str, Any]]:
        """Get current user (optional - returns None if no token)."""
        if credentials is None:
            return None

        try:
            token = credentials.credentials
            payload = TokenManager.verify_token(token, token_type="access")
            return {
                "id": payload.get("sub"),
                "role": payload.get("role"),
                "exp": payload.get("exp")
            }
        except Exception:
            return None


class RoleBasedAccess:
    """Role-based access control."""

    ROLE_HIERARCHY = {
        "admin": 3,
        "user": 2,
        "viewer": 1,
        "guest": 0
    }

    @staticmethod
    def require_role(required_role: str):
        """Decorator to require specific role."""
        async def role_checker(
            user: Dict[str, Any] = Depends(CurrentUser.get_current_user)
        ) -> Dict[str, Any]:
            user_role = user.get("role", "guest")
            user_level = RoleBasedAccess.ROLE_HIERARCHY.get(user_role, 0)
            required_level = RoleBasedAccess.ROLE_HIERARCHY.get(required_role, 0)

            if user_level < required_level:
                logger.warning(
                    f"User {user.get('id')} with role {user_role} "
                    f"attempted to access {required_role} resource"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"{required_role} access required"
                )
            return user

        return role_checker

    @classmethod
    def require_admin(cls):
        """Require admin role dependency."""
        return cls.require_role("admin")

    @classmethod
    def require_user(cls):
        """Require user role or higher dependency."""
        return cls.require_role("user")


class PermissionManager:
    """Manage fine-grained permissions."""

    @staticmethod
    def check_permission(
        user: Dict[str, Any],
        resource: str,
        action: str,
        permissions: Dict[str, list]
    ) -> bool:
        """Check if user has permission for action on resource."""
        try:
            user_role = user.get("role", "guest")
            allowed_actions = permissions.get(resource, {}).get(user_role, [])

            has_permission = action in allowed_actions
            if not has_permission:
                logger.warning(
                    f"User {user.get('id')} denied {action} on {resource}"
                )

            return has_permission
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False

    @staticmethod
    def require_permission(resource: str, action: str, permissions: Dict[str, list]):
        """Decorator to require specific permission."""
        async def permission_checker(
            user: Dict[str, Any] = Depends(CurrentUser.get_current_user)
        ) -> Dict[str, Any]:
            if not PermissionManager.check_permission(user, resource, action, permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied for {action} on {resource}"
                )
            return user

        return permission_checker


class SecurityAudit:
    """Audit security events."""

    @staticmethod
    def log_auth_event(
        user_id: str,
        event_type: str,
        status: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Log authentication event."""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "event_type": event_type,
                "status": status,
                "details": details or {}
            }

            if status == "success":
                logger.info(f"Auth event: {event_type} for user {user_id}")
            else:
                logger.warning(f"Auth event: {event_type} failed for user {user_id}")

            # TODO: Store in audit log database
        except Exception as e:
            logger.error(f"Failed to log auth event: {e}")

    @staticmethod
    def log_access_event(
        user_id: str,
        resource: str,
        action: str,
        status: str,
        details: Dict[str, Any] = None
    ) -> None:
        """Log access event."""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "status": status,
                "details": details or {}
            }

            if status == "success":
                logger.debug(f"Access: {action} on {resource} by {user_id}")
            else:
                logger.warning(f"Access denied: {action} on {resource} by {user_id}")

            # TODO: Store in audit log database
        except Exception as e:
            logger.error(f"Failed to log access event: {e}")


class SessionManager:
    """Manage user sessions."""

    def __init__(self, redis_client):
        """Initialize session manager."""
        self.redis = redis_client

    def create_session(
        self,
        user_id: str,
        role: str,
        device_info: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """Create user session."""
        try:
            access_token, refresh_token = TokenManager.create_tokens(user_id, role)

            # Store session in Redis with secure hash
            token_hash = hashlib.sha256(access_token.encode()).hexdigest()[:32]
            session_key = f"session:{user_id}:{token_hash}"
            session_data = {
                "user_id": user_id,
                "role": role,
                "device_info": device_info,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat()
            }

            self.redis.set_global_cache(
                session_key,
                session_data,
                ttl=ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )

            logger.info(f"Session created for user {user_id}")

            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    def invalidate_session(self, user_id: str, token: str) -> bool:
        """Invalidate user session."""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
            session_key = f"session:{user_id}:{token_hash}"
            self.redis.delete_global_cache(session_key)
            logger.info(f"Session invalidated for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to invalidate session: {e}")
            return False

    def invalidate_all_sessions(self, user_id: str) -> bool:
        """Invalidate all sessions for user."""
        try:
            pattern = f"session:{user_id}:*"
            self.redis.delete_by_pattern(pattern)
            logger.info(f"All sessions invalidated for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to invalidate all sessions: {e}")
            return False

    def get_session(self, user_id: str, token: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
            session_key = f"session:{user_id}:{token_hash}"
            return self.redis.get_global_cache(session_key)
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    def update_session_activity(self, user_id: str, token: str) -> bool:
        """Update session last activity."""
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
            session_key = f"session:{user_id}:{token_hash}"
            session = self.redis.get_global_cache(session_key)

            if session:
                session["last_activity"] = datetime.now(timezone.utc).isoformat()
                self.redis.set_global_cache(
                    session_key,
                    session,
                    ttl=ACCESS_TOKEN_EXPIRE_MINUTES * 60
                )
                return True

            return False
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
            return False
