"""
Authentication_V1 - Backend Integration Bridge

Integrates with backend/shared/auth and backend/shared/security implementations.
"""

import sys
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add backend path for imports
_backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

# Import backend implementations
try:
    from shared.auth.oauth_providers import (
        OAuthProvider as BackendOAuthProvider,
        OAuthConfig as BackendOAuthConfig,
        OAuthToken as BackendOAuthToken,
        OAuthUserInfo as BackendOAuthUserInfo,
    )
    AUTH_BACKEND_AVAILABLE = True
except ImportError:
    AUTH_BACKEND_AVAILABLE = False
    BackendOAuthProvider = None

try:
    from shared.security import (
        JWTManager,
        PasswordHasher,
        RateLimiter,
    )
    SECURITY_BACKEND_AVAILABLE = True
except ImportError:
    SECURITY_BACKEND_AVAILABLE = False


BACKEND_AVAILABLE = AUTH_BACKEND_AVAILABLE or SECURITY_BACKEND_AVAILABLE


class IntegratedAuthManager:
    """
    V1 Auth Manager with backend security integration.
    """

    def __init__(
        self,
        secret_key: str = "change-me",
        access_token_ttl: int = 3600,
        use_backend: bool = True,
    ):
        self.secret_key = secret_key
        self.access_token_ttl = access_token_ttl
        self.use_backend = use_backend and SECURITY_BACKEND_AVAILABLE

        if self.use_backend:
            self._jwt_manager = JWTManager(secret_key)
            self._password_hasher = PasswordHasher()
        else:
            from .auth_manager import AuthManager
            self._local = AuthManager()

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        if self.use_backend:
            return self._password_hasher.hash(password)
        return self._local.hash_password(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        if self.use_backend:
            return self._password_hasher.verify(password, hashed)
        return self._local.verify_password(password, hashed)

    def create_access_token(self, user_id: str, claims: Optional[Dict] = None) -> str:
        """Create JWT access token."""
        if self.use_backend:
            return self._jwt_manager.create_token(
                user_id=user_id,
                claims=claims or {},
                ttl=self.access_token_ttl,
            )
        return self._local.create_token(user_id, claims)

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token."""
        if self.use_backend:
            return self._jwt_manager.verify_token(token)
        return self._local.verify_token(token)


class IntegratedOAuthProvider:
    """
    V1 OAuth Provider with backend integration.
    """

    def __init__(
        self,
        provider_type: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        use_backend: bool = True,
    ):
        self.provider_type = provider_type
        self.use_backend = use_backend and AUTH_BACKEND_AVAILABLE

        if self.use_backend:
            config = BackendOAuthConfig(
                provider=provider_type,
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
            )
            self._backend = BackendOAuthProvider(config)
        else:
            from modules.Authentication_V2.src.oauth_provider import OAuthProvider
            self._local = OAuthProvider(redirect_uri)

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Get OAuth authorization URL."""
        if self.use_backend:
            return self._backend.get_authorization_url(state)
        url, _ = self._local.get_authorization_url(self.provider_type)
        return url

    async def exchange_code(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        if self.use_backend:
            token = await self._backend.exchange_code(code)
            return token.__dict__ if hasattr(token, '__dict__') else token
        return await self._local.handle_callback(code, "")

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user info from OAuth provider."""
        if self.use_backend:
            user_info = await self._backend.get_user_info(access_token)
            return user_info.__dict__ if hasattr(user_info, '__dict__') else user_info
        return {"token": access_token}


class IntegratedRateLimiter:
    """
    V1 Rate Limiter with backend integration.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        use_backend: bool = True,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.use_backend = use_backend and SECURITY_BACKEND_AVAILABLE

        if self.use_backend:
            self._backend = RateLimiter(max_requests, window_seconds)
        else:
            self._counters: Dict[str, List] = {}

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        if self.use_backend:
            return self._backend.is_allowed(key)

        # Local implementation
        import time
        now = time.time()

        if key not in self._counters:
            self._counters[key] = []

        # Clean old entries
        self._counters[key] = [
            t for t in self._counters[key]
            if now - t < self.window_seconds
        ]

        if len(self._counters[key]) >= self.max_requests:
            return False

        self._counters[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in window."""
        if self.use_backend:
            return self._backend.get_remaining(key)

        import time
        now = time.time()

        if key not in self._counters:
            return self.max_requests

        valid = [t for t in self._counters[key] if now - t < self.window_seconds]
        return max(0, self.max_requests - len(valid))


# Factory functions
def get_auth_manager(use_backend: bool = True) -> IntegratedAuthManager:
    """Get integrated auth manager."""
    return IntegratedAuthManager(use_backend=use_backend)


def get_oauth_provider(
    provider_type: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    use_backend: bool = True,
) -> IntegratedOAuthProvider:
    """Get integrated OAuth provider."""
    return IntegratedOAuthProvider(
        provider_type, client_id, client_secret, redirect_uri, use_backend
    )


def get_rate_limiter(
    max_requests: int = 100,
    window_seconds: int = 60,
    use_backend: bool = True,
) -> IntegratedRateLimiter:
    """Get integrated rate limiter."""
    return IntegratedRateLimiter(max_requests, window_seconds, use_backend)


__all__ = [
    "BACKEND_AVAILABLE",
    "AUTH_BACKEND_AVAILABLE",
    "SECURITY_BACKEND_AVAILABLE",
    "IntegratedAuthManager",
    "IntegratedOAuthProvider",
    "IntegratedRateLimiter",
    "get_auth_manager",
    "get_oauth_provider",
    "get_rate_limiter",
]
