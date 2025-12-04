"""
Shared Authentication Module

Provides OAuth providers and authentication utilities.
"""

from .oauth_providers import (
    OAuthProviderBase,
    OAuthProviderFactory,
    GitHubOAuth,
    GitLabOAuth,
    OAuthToken,
    OAuthUser,
    OAuthRepository,
)

__all__ = [
    "OAuthProviderBase",
    "OAuthProviderFactory",
    "GitHubOAuth",
    "GitLabOAuth",
    "OAuthToken",
    "OAuthUser",
    "OAuthRepository",
]
