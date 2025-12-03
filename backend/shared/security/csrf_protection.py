"""
CSRF Protection Module

Implements:
- Double-submit cookie pattern
- Origin/Referer validation
- SameSite cookie attributes
- Token-based CSRF protection
"""

import secrets
import hashlib
import hmac
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import wraps

from fastapi import Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# Configuration
CSRF_SECRET_KEY = secrets.token_urlsafe(32)
CSRF_TOKEN_LENGTH = 32
CSRF_TOKEN_EXPIRE_MINUTES = 60
CSRF_HEADER_NAME = "X-CSRF-Token"
CSRF_COOKIE_NAME = "csrf_token"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


class CSRFToken:
    """CSRF token management."""
    
    @staticmethod
    def generate_token(session_id: str) -> str:
        """Generate CSRF token bound to session."""
        timestamp = datetime.utcnow().isoformat()
        random_part = secrets.token_urlsafe(CSRF_TOKEN_LENGTH)
        
        # Create signed token
        message = f"{session_id}:{timestamp}:{random_part}"
        signature = hmac.new(
            CSRF_SECRET_KEY.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{message}:{signature}"
    
    @staticmethod
    def validate_token(token: str, session_id: str) -> bool:
        """Validate CSRF token."""
        try:
            parts = token.rsplit(":", 1)
            if len(parts) != 2:
                return False
            
            message, signature = parts
            
            # Verify signature
            expected_signature = hmac.new(
                CSRF_SECRET_KEY.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("CSRF token signature mismatch")
                return False
            
            # Parse message
            msg_parts = message.split(":")
            if len(msg_parts) != 3:
                return False
            
            token_session_id, timestamp_str, _ = msg_parts
            
            # Verify session binding
            if token_session_id != session_id:
                logger.warning("CSRF token session mismatch")
                return False
            
            # Verify expiration
            timestamp = datetime.fromisoformat(timestamp_str)
            if datetime.utcnow() - timestamp > timedelta(minutes=CSRF_TOKEN_EXPIRE_MINUTES):
                logger.warning("CSRF token expired")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"CSRF token validation error: {e}")
            return False


class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CSRF protection middleware.
    
    Implements double-submit cookie pattern with origin validation.
    """
    
    def __init__(
        self,
        app,
        allowed_origins: list = None,
        exempt_paths: list = None,
        cookie_secure: bool = True,
        cookie_httponly: bool = False,  # Need JS access for AJAX
        cookie_samesite: str = "strict",
    ):
        super().__init__(app)
        self.allowed_origins = allowed_origins or []
        self.exempt_paths = exempt_paths or [
            "/health",
            "/metrics",
            "/api/auth/login",
            "/api/auth/register",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.cookie_samesite = cookie_samesite
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip for safe methods
        if request.method in SAFE_METHODS:
            response = await call_next(request)
            return self._set_csrf_cookie(request, response)
        
        # Skip for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)
        
        # Validate origin/referer
        if not self._validate_origin(request):
            logger.warning(f"CSRF origin validation failed for {request.url.path}")
            raise HTTPException(
                status_code=403,
                detail="CSRF validation failed: invalid origin"
            )
        
        # Validate CSRF token
        if not await self._validate_csrf_token(request):
            logger.warning(f"CSRF token validation failed for {request.url.path}")
            raise HTTPException(
                status_code=403,
                detail="CSRF validation failed: invalid token"
            )
        
        response = await call_next(request)
        return self._set_csrf_cookie(request, response)
    
    def _validate_origin(self, request: Request) -> bool:
        """Validate request origin."""
        # Check Origin header
        origin = request.headers.get("Origin")
        if origin:
            if not self.allowed_origins:
                return True
            return origin in self.allowed_origins
        
        # Check Referer header as fallback
        referer = request.headers.get("Referer")
        if referer:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            referer_origin = f"{parsed.scheme}://{parsed.netloc}"
            if not self.allowed_origins:
                return True
            return referer_origin in self.allowed_origins
        
        # No origin information - block for safety
        return False
    
    async def _validate_csrf_token(self, request: Request) -> bool:
        """Validate CSRF token from header and cookie."""
        # Get token from header
        header_token = request.headers.get(CSRF_HEADER_NAME)
        if not header_token:
            return False
        
        # Get token from cookie
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        if not cookie_token:
            return False
        
        # Get session ID
        session_id = self._get_session_id(request)
        if not session_id:
            session_id = request.client.host if request.client else "unknown"
        
        # Validate both tokens
        return (
            CSRFToken.validate_token(header_token, session_id) and
            CSRFToken.validate_token(cookie_token, session_id) and
            hmac.compare_digest(header_token, cookie_token)
        )
    
    def _get_session_id(self, request: Request) -> Optional[str]:
        """Extract session ID from request."""
        # Try to get from JWT
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In production, decode JWT to get session ID
            token = auth_header[7:]
            return hashlib.sha256(token.encode()).hexdigest()[:16]
        
        # Fall back to client IP
        return request.client.host if request.client else None
    
    def _set_csrf_cookie(self, request: Request, response: Response) -> Response:
        """Set CSRF token cookie."""
        session_id = self._get_session_id(request)
        if not session_id:
            session_id = request.client.host if request.client else "unknown"
        
        token = CSRFToken.generate_token(session_id)
        
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=token,
            max_age=CSRF_TOKEN_EXPIRE_MINUTES * 60,
            secure=self.cookie_secure,
            httponly=self.cookie_httponly,
            samesite=self.cookie_samesite,
        )
        
        # Also set in response header for JavaScript access
        response.headers[CSRF_HEADER_NAME] = token
        
        return response


def csrf_protect(func):
    """Decorator for CSRF protection on specific routes."""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        if request.method not in SAFE_METHODS:
            # Validate CSRF token
            header_token = request.headers.get(CSRF_HEADER_NAME)
            cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
            
            if not header_token or not cookie_token:
                raise HTTPException(
                    status_code=403,
                    detail="CSRF token required"
                )
            
            if not hmac.compare_digest(header_token, cookie_token):
                raise HTTPException(
                    status_code=403,
                    detail="CSRF token mismatch"
                )
        
        return await func(request, *args, **kwargs)
    
    return wrapper


class CSRFTokenDependency:
    """FastAPI dependency for CSRF token validation."""
    
    async def __call__(self, request: Request) -> str:
        """Validate and return CSRF token."""
        header_token = request.headers.get(CSRF_HEADER_NAME)
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        
        if not header_token:
            raise HTTPException(
                status_code=403,
                detail=f"Missing {CSRF_HEADER_NAME} header"
            )
        
        if not cookie_token:
            raise HTTPException(
                status_code=403,
                detail="Missing CSRF cookie"
            )
        
        if not hmac.compare_digest(header_token, cookie_token):
            raise HTTPException(
                status_code=403,
                detail="CSRF token mismatch"
            )
        
        return header_token


# Export dependency
require_csrf = CSRFTokenDependency()
