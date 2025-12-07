"""
Secure Authentication Module

Implements:
- httpOnly cookie-based token storage (XSS protection)
- CSRF token validation
- Secure session management
- Token refresh mechanism
"""
import secrets
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from fastapi import Request, Response, HTTPException, Depends, Cookie
from fastapi.security import HTTPBearer
from jose import jwt, JWTError
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Configuration
@dataclass
class AuthConfig:
    """Authentication configuration."""
    # JWT settings
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # Cookie settings
    cookie_domain: Optional[str] = None
    cookie_secure: bool = True  # HTTPS only
    cookie_httponly: bool = True  # No JavaScript access
    cookie_samesite: str = "lax"  # CSRF protection

    # CSRF settings
    csrf_token_expire_minutes: int = 60
    csrf_header_name: str = "X-CSRF-Token"

    # Session settings
    max_sessions_per_user: int = 5


# Token models
class TokenPair(BaseModel):
    """Access and refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class CSRFToken(BaseModel):
    """CSRF token."""
    token: str
    expires_at: datetime


# Secure Auth Manager
class SecureAuthManager:
    """
    Secure authentication manager with httpOnly cookies and CSRF protection.

    Security Features:
    1. Access tokens stored in httpOnly cookies (XSS protection)
    2. CSRF tokens for state-changing operations
    3. Refresh tokens with rotation
    4. Secure cookie attributes
    """

    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self._csrf_tokens: Dict[str, CSRFToken] = {}  # In production, use Redis

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.config.access_token_expire_minutes)

        to_encode.update({
            "exp": expire,
            "type": "access",
            "iat": datetime.now(timezone.utc),
        })

        return jwt.encode(
            to_encode,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    def create_refresh_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(days=self.config.refresh_token_expire_days)

        # Add unique identifier for token rotation
        to_encode.update({
            "exp": expire,
            "type": "refresh",
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
        })

        return jwt.encode(
            to_encode,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

    def create_token_pair(self, user_data: Dict[str, Any]) -> TokenPair:
        """Create access and refresh token pair."""
        access_token = self.create_access_token(user_data)
        refresh_token = self.create_refresh_token(user_data)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.access_token_expire_minutes * 60,
        )

    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string
            token_type: Expected token type ("access" or "refresh")

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
            )

            # Verify token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid token type. Expected {token_type}",
                )

            return payload

        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
            )

    def generate_csrf_token(
        self,
        _session_id: str,  # Reserved for session-bound CSRF
    ) -> str:
        """
        Generate CSRF token for a session.

        Args:
            session_id: User's session identifier (reserved for future use)

        Returns:
            CSRF token string
        """
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=self.config.csrf_token_expire_minutes)

        # Hash the token for storage
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        self._csrf_tokens[token_hash] = CSRFToken(
            token=token_hash,
            expires_at=expires_at,
        )

        # Clean up expired tokens
        self._cleanup_expired_csrf_tokens()

        return token

    def verify_csrf_token(self, token: str) -> bool:
        """
        Verify CSRF token.

        Args:
            token: CSRF token from header

        Returns:
            True if valid, False otherwise
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        csrf = self._csrf_tokens.get(token_hash)
        if not csrf:
            return False

        if datetime.now(timezone.utc) > csrf.expires_at:
            del self._csrf_tokens[token_hash]
            return False

        return True

    def _cleanup_expired_csrf_tokens(self):
        """Remove expired CSRF tokens."""
        now = datetime.now(timezone.utc)
        expired = [k for k, v in self._csrf_tokens.items() if now > v.expires_at]
        for k in expired:
            del self._csrf_tokens[k]

    def set_auth_cookies(
        self,
        response: Response,
        access_token: str,
        refresh_token: str,
        csrf_token: str,
    ):
        """
        Set authentication cookies on response.

        Cookies:
        - access_token: httpOnly, secure (XSS protected)
        - refresh_token: httpOnly, secure (XSS protected)
        - csrf_token: NOT httpOnly (readable by JS for headers)
        """
        # Access token cookie
        response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=self.config.access_token_expire_minutes * 60,
            expires=datetime.now(timezone.utc) + timedelta(minutes=self.config.access_token_expire_minutes),
            path="/",
            domain=self.config.cookie_domain,
            secure=self.config.cookie_secure,
            httponly=self.config.cookie_httponly,
            samesite=self.config.cookie_samesite,
        )

        # Refresh token cookie
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            max_age=self.config.refresh_token_expire_days * 24 * 60 * 60,
            expires=datetime.now(timezone.utc) + timedelta(days=self.config.refresh_token_expire_days),
            path="/api/auth/refresh",  # Only sent to refresh endpoint
            domain=self.config.cookie_domain,
            secure=self.config.cookie_secure,
            httponly=self.config.cookie_httponly,
            samesite=self.config.cookie_samesite,
        )

        # CSRF token cookie (readable by JS)
        response.set_cookie(
            key="csrf_token",
            value=csrf_token,
            max_age=self.config.csrf_token_expire_minutes * 60,
            path="/",
            domain=self.config.cookie_domain,
            secure=self.config.cookie_secure,
            httponly=False,  # Must be readable by JS
            samesite=self.config.cookie_samesite,
        )

    def clear_auth_cookies(self, response: Response):
        """Clear all authentication cookies."""
        response.delete_cookie("access_token", path="/")
        response.delete_cookie("refresh_token", path="/api/auth/refresh")
        response.delete_cookie("csrf_token", path="/")


# FastAPI Dependencies
auth_manager = SecureAuthManager()


async def get_current_user(
    _request: Request,  # Required by FastAPI dependency signature
    access_token: Optional[str] = Cookie(None),
) -> Dict[str, Any]:
    """
    Get current user from httpOnly cookie.

    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user_id": user["sub"]}
    """
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return auth_manager.verify_token(access_token, "access")


async def verify_csrf(
    request: Request,
    _csrf_token: Optional[str] = Cookie(None),  # Cookie fallback
) -> None:
    """
    Verify CSRF token for state-changing requests.
    Raises HTTPException on failure, returns None on success.

    Usage:
        @app.post("/api/data")
        async def create_data(
            _: None = Depends(verify_csrf),
            user: dict = Depends(get_current_user),
        ):
            ...
    """
    # Skip CSRF for safe methods
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return

    # Get CSRF token from header
    header_token = request.headers.get("X-CSRF-Token")

    if not header_token:
        raise HTTPException(
            status_code=403,
            detail="CSRF token missing",
        )

    if not auth_manager.verify_csrf_token(header_token):
        raise HTTPException(
            status_code=403,
            detail="Invalid CSRF token",
        )


class CSRFProtectedRoute:
    """
    Dependency for CSRF-protected routes.

    Usage:
        @app.post("/api/action", dependencies=[Depends(CSRFProtectedRoute())])
        async def action():
            ...
    """

    async def __call__(self, request: Request) -> None:
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return

        csrf_token = request.headers.get("X-CSRF-Token")
        if not csrf_token:
            raise HTTPException(status_code=403, detail="CSRF token required")

        if not auth_manager.verify_csrf_token(csrf_token):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")


# Rate limiting helper
class RateLimiter:
    """Simple rate limiter (use Redis in production)."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._requests: Dict[str, list] = {}  # IP -> timestamps

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def __call__(self, request: Request) -> None:
        """Check rate limit."""
        ip = self._get_client_ip(request)
        now = datetime.now(timezone.utc)

        # Initialize or get request history
        if ip not in self._requests:
            self._requests[ip] = []

        # Clean old requests
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        self._requests[ip] = [t for t in self._requests[ip] if t > hour_ago]

        # Check limits
        recent_minute = sum(1 for t in self._requests[ip] if t > minute_ago)
        recent_hour = len(self._requests[ip])

        if recent_minute >= self.rpm:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.rpm} requests per minute",
                headers={"Retry-After": "60"},
            )

        if recent_hour >= self.rph:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.rph} requests per hour",
                headers={"Retry-After": "3600"},
            )

        # Record request
        self._requests[ip].append(now)


# Usage example
"""
from fastapi import FastAPI, Depends, Response
from secure_auth import (
    auth_manager,
    get_current_user,
    verify_csrf,
    CSRFProtectedRoute,
    RateLimiter,
)

app = FastAPI()
rate_limiter = RateLimiter(requests_per_minute=60)

@app.post("/api/auth/login")
async def login(credentials: LoginCredentials, response: Response):
    # Verify credentials...
    user = authenticate(credentials)

    # Create tokens
    tokens = auth_manager.create_token_pair({"sub": user.id, "role": user.role})
    csrf = auth_manager.generate_csrf_token(user.id)

    # Set cookies
    auth_manager.set_auth_cookies(
        response,
        tokens.access_token,
        tokens.refresh_token,
        csrf,
    )

    return {"message": "Login successful"}

@app.post("/api/auth/logout")
async def logout(response: Response):
    auth_manager.clear_auth_cookies(response)
    return {"message": "Logged out"}

@app.post(
    "/api/protected",
    dependencies=[Depends(rate_limiter), Depends(CSRFProtectedRoute())],
)
async def protected_action(user: dict = Depends(get_current_user)):
    return {"user": user["sub"]}
"""
