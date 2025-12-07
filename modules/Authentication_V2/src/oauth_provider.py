"""
Authentication_V2 - OAuth Provider

OAuth 2.0 integration for third-party authentication.
"""

import secrets
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class OAuthProviderType(str, Enum):
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"


@dataclass
class OAuthConfig:
    """OAuth provider configuration"""
    provider: OAuthProviderType
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: list

    @classmethod
    def google(cls, client_id: str, client_secret: str) -> "OAuthConfig":
        return cls(
            provider=OAuthProviderType.GOOGLE,
            client_id=client_id,
            client_secret=client_secret,
            authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
            scopes=["openid", "email", "profile"],
        )

    @classmethod
    def github(cls, client_id: str, client_secret: str) -> "OAuthConfig":
        return cls(
            provider=OAuthProviderType.GITHUB,
            client_id=client_id,
            client_secret=client_secret,
            authorize_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            userinfo_url="https://api.github.com/user",
            scopes=["read:user", "user:email"],
        )


@dataclass
class OAuthUser:
    """User info from OAuth provider"""
    provider: OAuthProviderType
    provider_id: str
    email: str
    name: Optional[str]
    avatar_url: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "provider_id": self.provider_id,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
        }


class OAuthProvider:
    """
    OAuth 2.0 provider integration.

    Features:
    - Multiple provider support (Google, GitHub)
    - State management (CSRF protection)
    - Token exchange
    - User info retrieval
    """

    def __init__(self, redirect_uri: str):
        self.redirect_uri = redirect_uri
        self._configs: Dict[OAuthProviderType, OAuthConfig] = {}
        self._states: Dict[str, Dict] = {}

    def register_provider(self, config: OAuthConfig):
        """Register OAuth provider"""
        self._configs[config.provider] = config
        logger.info(f"Registered OAuth provider: {config.provider.value}")

    def get_authorization_url(
        self,
        provider: OAuthProviderType,
        extra_params: Optional[Dict] = None,
    ) -> tuple[str, str]:
        """
        Get authorization URL for provider.

        Returns:
            Tuple of (authorization_url, state)
        """
        if provider not in self._configs:
            raise ValueError(f"Provider not configured: {provider}")

        config = self._configs[provider]

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        self._states[state] = {
            "provider": provider,
            "created_at": datetime.now(timezone.utc),
        }

        # Build authorization URL
        params = {
            "client_id": config.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(config.scopes),
            "state": state,
        }

        if extra_params:
            params.update(extra_params)

        url = f"{config.authorize_url}?{urlencode(params)}"

        return url, state

    async def handle_callback(
        self,
        code: str,
        state: str,
    ) -> Optional[OAuthUser]:
        """
        Handle OAuth callback.

        Args:
            code: Authorization code
            state: State parameter

        Returns:
            OAuthUser if successful
        """
        # Verify state
        if state not in self._states:
            logger.warning("Invalid OAuth state")
            return None

        state_data = self._states.pop(state)
        provider = state_data["provider"]

        # Check state expiration (10 minutes)
        if datetime.now(timezone.utc) - state_data["created_at"] > timedelta(minutes=10):
            logger.warning("OAuth state expired")
            return None

        config = self._configs[provider]

        # Exchange code for token
        access_token = await self._exchange_code(config, code)

        if not access_token:
            return None

        # Get user info
        user = await self._get_user_info(config, access_token)

        return user

    async def _exchange_code(
        self,
        config: OAuthConfig,
        code: str,
    ) -> Optional[str]:
        """Exchange authorization code for access token"""
        # In production, use httpx or aiohttp
        # This is a mock implementation

        logger.info(f"Exchanging code for {config.provider.value}")

        # Mock token response
        return f"mock_access_token_{secrets.token_hex(8)}"

    async def _get_user_info(
        self,
        config: OAuthConfig,
        access_token: str,
    ) -> Optional[OAuthUser]:
        """Get user info from provider"""
        # In production, make actual HTTP request
        # This is a mock implementation

        logger.info(f"Getting user info from {config.provider.value}")

        # Mock user response
        return OAuthUser(
            provider=config.provider,
            provider_id=f"mock_{secrets.token_hex(8)}",
            email="oauth.user@example.com",
            name="OAuth User",
            avatar_url=None,
        )

    def cleanup_expired_states(self):
        """Clean up expired OAuth states"""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)

        expired = [
            state for state, data in self._states.items()
            if data["created_at"] < cutoff
        ]

        for state in expired:
            del self._states[state]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired OAuth states")
