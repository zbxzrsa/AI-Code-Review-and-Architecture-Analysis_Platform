"""
Authentication and authorization utilities - FIXED VERSION.

Addresses all issues identified in granular code audit:
- Line 19: Removed hardcoded default secret
- Line 41: Fixed deprecated datetime.utcnow()
- Line 47: Added JTI claim for token revocation
- Line 97: Added proper JWT validation options
- Line 363: Fixed weak token truncation for session key
"""
import logging
import os
import hashlib
import uuid
from typing import Optional, Dict, Any, Set
from datetime import datetime, timedelta, timezone
from functools import wraps

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration - FIXED: No default secret
# =============================================================================

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise ValueError(
        "JWT_SECRET_KEY environment variable must be set. "
        "Generate with: openssl rand -hex 32"
    )

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "code-review-platform")
JWT_ISSUER = os.getenv("JWT_ISSUER", "auth-service")

security = HTTPBearer()

# Token blacklist (in production, use Redis)
_revoked_tokens: Set[str] = set()


class TokenManager:
    """Manage JWT tokens with security fixes."""

    @staticmethod
    def create_access_token(
        user_id: str,
        role: str,
        expires_delta: Optional[timedelta] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create access token.
        
        FIXES:
        - Uses timezone-aware datetime
        - Includes JTI for revocation
        - Includes audience and issuer
        """
        try:
            if expires_delta is None:
                expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

            # FIXED: Use timezone-aware datetime
            now = datetime.now(timezone.utc)
            expire = now + expires_delta
            
            # FIXED: Add JTI for token revocation
            jti = str(uuid.uuid4())
            
            to_encode = {
                "sub": user_id,
                "role": role,
                "type": "access",
                "exp": expire,
                "iat": now,
                "jti": jti,  # JWT ID for revocation
                "aud": JWT_AUDIENCE,  # Audience
                "iss": JWT_ISSUER,    # Issuer
            }
            
            # Add any additional claims
            if additional_claims:
                to_encode.update(additional_claims)

            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            logger.debug(f"Access token created for user {user_id}, jti={jti}")
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
        """
        Create refresh token.
        
        FIXES:
        - Uses timezone-aware datetime
        - Includes role for token refresh
        - Includes JTI
        """
        try:
            if expires_delta is None:
                expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

            now = datetime.now(timezone.utc)
            expire = now + expires_delta
            jti = str(uuid.uuid4())
            
            to_encode = {
                "sub": user_id,
                "role": role,  # Include role for refresh
                "type": "refresh",
                "exp": expire,
                "iat": now,
                "jti": jti,
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
        """
        Verify and decode token.
        
        FIXES:
        - Validates audience and issuer
        - Requires essential claims
        - Checks token revocation
        """
        try:
            # FIXED: Add proper validation options
            payload = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=[ALGORITHM],
                options={
                    "verify_aud": True,
                    "verify_iss": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "require": ["exp", "sub", "type", "iat", "jti"]
                },
                audience=JWT_AUDIENCE,
                issuer=JWT_ISSUER,
            )

            # Verify token type
            if payload.get("type") != token_type:
                raise ValueError(f"Invalid token type: expected {token_type}, got {payload.get('type')}")

            user_id = payload.get("sub")
            if user_id is None:
                raise ValueError("Invalid token: missing user_id")

            # Check if token is revoked
            jti = payload.get("jti")
            if jti and TokenManager.is_token_revoked(jti):
                raise ValueError("Token has been revoked")

            return payload
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        except ValueError as e:
            logger.warning(f"Token validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed"
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

    @staticmethod
    def revoke_token(jti: str) -> None:
        """Revoke a token by its JTI."""
        _revoked_tokens.add(jti)
        logger.info(f"Token revoked: {jti}")

    @staticmethod
    def is_token_revoked(jti: str) -> bool:
        """Check if token is revoked."""
        return jti in _revoked_tokens


class CurrentUser:
    """Current user dependency with fixes."""

    @staticmethod
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> Dict[str, Any]:
        """Get current user from token."""
        try:
            token = credentials.credentials
            payload = TokenManager.verify_token(token, token_type="access")

            return {
                "id": payload.get("sub"),
                "role": payload.get("role"),
                "exp": payload.get("exp"),
                "jti": payload.get("jti"),
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
        user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Get current user with admin role."""
        if user.get("role") not in ["admin", "system"]:
            logger.warning(f"Non-admin user {user.get('id')} attempted admin access")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return user

    @staticmethod
    async def get_current_user_optional(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[Dict[str, Any]]:
        """Get current user (optional)."""
        if credentials is None:
            return None

        try:
            return await CurrentUser.get_current_user(credentials)
        except HTTPException:
            return None


class SessionManager:
    """
    Manage user sessions with fixes.
    
    FIXES:
    - Uses secure hash for session key instead of truncation
    - Better error handling
    """

    def __init__(self, redis_client):
        """Initialize session manager."""
        self.redis = redis_client

    def _get_session_key(self, user_id: str, token: str) -> str:
        """
        Generate secure session key.
        
        FIXED: Use hash instead of truncation
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
        return f"session:{user_id}:{token_hash}"

    def create_session(
        self,
        user_id: str,
        role: str,
        device_info: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """Create user session."""
        try:
            access_token, refresh_token = TokenManager.create_tokens(user_id, role)

            # Store session in Redis
            session_key = self._get_session_key(user_id, access_token)
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
            session_key = self._get_session_key(user_id, token)
            self.redis.delete_global_cache(session_key)
            
            # Also revoke the token
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
                if jti := payload.get("jti"):
                    TokenManager.revoke_token(jti)
            except:
                pass
            
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
            session_key = self._get_session_key(user_id, token)
            return self.redis.get_global_cache(session_key)
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    def update_session_activity(self, user_id: str, token: str) -> bool:
        """Update session last activity."""
        try:
            session_key = self._get_session_key(user_id, token)
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


# =============================================================================
# Role-based access control
# =============================================================================

class RoleBasedAccess:
    """Role-based access control."""

    ROLE_HIERARCHY = {
        "system": 4,
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

    @staticmethod
    def require_admin():
        """Require admin role."""
        return RoleBasedAccess.require_role("admin")

    @staticmethod
    def require_user():
        """Require user role or higher."""
        return RoleBasedAccess.require_role("user")

    @staticmethod
    def require_system():
        """Require system role."""
        return RoleBasedAccess.require_role("system")
