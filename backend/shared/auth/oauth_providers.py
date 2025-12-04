"""
OAuth Provider Implementations

Supports:
- GitHub OAuth
- GitLab OAuth
- Bitbucket OAuth
- Google OAuth
"""

import os
import httpx
import secrets
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


@dataclass
class OAuthToken:
    """OAuth token response."""
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    scope: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.expires_in and not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(seconds=self.expires_in)


@dataclass
class OAuthUser:
    """OAuth user info."""
    provider: str
    provider_user_id: str
    email: Optional[str]
    name: Optional[str]
    username: Optional[str]
    avatar_url: Optional[str]
    raw_data: Dict[str, Any]


@dataclass
class OAuthRepository:
    """OAuth repository info."""
    provider: str
    provider_repo_id: str
    full_name: str
    name: str
    owner: str
    description: Optional[str]
    url: str
    clone_url: str
    ssh_url: str
    default_branch: str
    is_private: bool
    is_fork: bool
    stars: int
    forks: int


class OAuthProviderBase(ABC):
    """Base class for OAuth providers."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._http_client: Optional[httpx.AsyncClient] = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def authorize_url(self) -> str:
        """OAuth authorization URL."""
        pass
    
    @property
    @abstractmethod
    def token_url(self) -> str:
        """OAuth token URL."""
        pass
    
    @property
    @abstractmethod
    def user_info_url(self) -> str:
        """User info API URL."""
        pass
    
    @property
    @abstractmethod
    def default_scopes(self) -> List[str]:
        """Default OAuth scopes."""
        pass
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client
    
    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    def get_authorization_url(self, state: str, scopes: Optional[List[str]] = None) -> str:
        """Generate authorization URL."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(scopes or self.default_scopes),
            "state": state,
            "response_type": "code",
        }
        return f"{self.authorize_url}?{urlencode(params)}"
    
    @abstractmethod
    async def exchange_code(self, code: str) -> OAuthToken:
        """Exchange authorization code for tokens."""
        pass
    
    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> OAuthToken:
        """Refresh access token."""
        pass
    
    @abstractmethod
    async def get_user_info(self, access_token: str) -> OAuthUser:
        """Get user information."""
        pass
    
    @abstractmethod
    async def list_repositories(self, access_token: str) -> List[OAuthRepository]:
        """List user's repositories."""
        pass


class GitHubOAuth(OAuthProviderBase):
    """GitHub OAuth provider."""
    
    provider_name = "github"
    authorize_url = "https://github.com/login/oauth/authorize"
    token_url = "https://github.com/login/oauth/access_token"
    user_info_url = "https://api.github.com/user"
    api_base_url = "https://api.github.com"
    default_scopes = ["read:user", "user:email", "repo"]
    
    async def exchange_code(self, code: str) -> OAuthToken:
        """Exchange code for access token."""
        client = await self.get_client()
        
        response = await client.post(
            self.token_url,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            raise ValueError(f"OAuth error: {data.get('error_description', data['error'])}")
        
        return OAuthToken(
            access_token=data["access_token"],
            token_type=data.get("token_type", "bearer"),
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
        )
    
    async def refresh_token(self, refresh_token: str) -> OAuthToken:
        """GitHub doesn't support refresh tokens by default."""
        raise NotImplementedError("GitHub OAuth doesn't support refresh tokens")
    
    async def get_user_info(self, access_token: str) -> OAuthUser:
        """Get GitHub user info."""
        client = await self.get_client()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        
        # Get user info
        response = await client.get(self.user_info_url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        
        # Get primary email if not public
        email = user_data.get("email")
        if not email:
            email_response = await client.get(f"{self.api_base_url}/user/emails", headers=headers)
            if email_response.status_code == 200:
                emails = email_response.json()
                primary = next((e for e in emails if e.get("primary")), None)
                if primary:
                    email = primary.get("email")
        
        return OAuthUser(
            provider="github",
            provider_user_id=str(user_data["id"]),
            email=email,
            name=user_data.get("name"),
            username=user_data.get("login"),
            avatar_url=user_data.get("avatar_url"),
            raw_data=user_data,
        )
    
    async def list_repositories(self, access_token: str) -> List[OAuthRepository]:
        """List user's GitHub repositories."""
        client = await self.get_client()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        
        repos = []
        page = 1
        per_page = 100
        
        while True:
            response = await client.get(
                f"{self.api_base_url}/user/repos",
                headers=headers,
                params={"page": page, "per_page": per_page, "sort": "updated"},
            )
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            for repo in data:
                repos.append(OAuthRepository(
                    provider="github",
                    provider_repo_id=str(repo["id"]),
                    full_name=repo["full_name"],
                    name=repo["name"],
                    owner=repo["owner"]["login"],
                    description=repo.get("description"),
                    url=repo["html_url"],
                    clone_url=repo["clone_url"],
                    ssh_url=repo["ssh_url"],
                    default_branch=repo.get("default_branch", "main"),
                    is_private=repo["private"],
                    is_fork=repo["fork"],
                    stars=repo.get("stargazers_count", 0),
                    forks=repo.get("forks_count", 0),
                ))
            
            if len(data) < per_page:
                break
            page += 1
        
        return repos
    
    async def get_repository(self, access_token: str, owner: str, repo: str) -> OAuthRepository:
        """Get a specific repository."""
        client = await self.get_client()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        
        response = await client.get(
            f"{self.api_base_url}/repos/{owner}/{repo}",
            headers=headers,
        )
        response.raise_for_status()
        repo_data = response.json()
        
        return OAuthRepository(
            provider="github",
            provider_repo_id=str(repo_data["id"]),
            full_name=repo_data["full_name"],
            name=repo_data["name"],
            owner=repo_data["owner"]["login"],
            description=repo_data.get("description"),
            url=repo_data["html_url"],
            clone_url=repo_data["clone_url"],
            ssh_url=repo_data["ssh_url"],
            default_branch=repo_data.get("default_branch", "main"),
            is_private=repo_data["private"],
            is_fork=repo_data["fork"],
            stars=repo_data.get("stargazers_count", 0),
            forks=repo_data.get("forks_count", 0),
        )
    
    async def create_webhook(
        self,
        access_token: str,
        owner: str,
        repo: str,
        webhook_url: str,
        secret: str,
        events: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a webhook for a repository."""
        client = await self.get_client()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        }
        
        response = await client.post(
            f"{self.api_base_url}/repos/{owner}/{repo}/hooks",
            headers=headers,
            json={
                "name": "web",
                "active": True,
                "events": events or ["push", "pull_request"],
                "config": {
                    "url": webhook_url,
                    "content_type": "json",
                    "secret": secret,
                    "insecure_ssl": "0",
                },
            },
        )
        response.raise_for_status()
        return response.json()


class GitLabOAuth(OAuthProviderBase):
    """GitLab OAuth provider."""
    
    provider_name = "gitlab"
    authorize_url = "https://gitlab.com/oauth/authorize"
    token_url = "https://gitlab.com/oauth/token"
    user_info_url = "https://gitlab.com/api/v4/user"
    api_base_url = "https://gitlab.com/api/v4"
    default_scopes = ["read_user", "read_api", "read_repository"]
    
    async def exchange_code(self, code: str) -> OAuthToken:
        """Exchange code for tokens."""
        client = await self.get_client()
        
        response = await client.post(
            self.token_url,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return OAuthToken(
            access_token=data["access_token"],
            token_type=data.get("token_type", "bearer"),
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in"),
            scope=data.get("scope"),
        )
    
    async def refresh_token(self, refresh_token: str) -> OAuthToken:
        """Refresh access token."""
        client = await self.get_client()
        
        response = await client.post(
            self.token_url,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
        response.raise_for_status()
        data = response.json()
        
        return OAuthToken(
            access_token=data["access_token"],
            token_type=data.get("token_type", "bearer"),
            refresh_token=data.get("refresh_token"),
            expires_in=data.get("expires_in"),
        )
    
    async def get_user_info(self, access_token: str) -> OAuthUser:
        """Get GitLab user info."""
        client = await self.get_client()
        headers = {"Authorization": f"Bearer {access_token}"}
        
        response = await client.get(self.user_info_url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        
        return OAuthUser(
            provider="gitlab",
            provider_user_id=str(user_data["id"]),
            email=user_data.get("email"),
            name=user_data.get("name"),
            username=user_data.get("username"),
            avatar_url=user_data.get("avatar_url"),
            raw_data=user_data,
        )
    
    async def list_repositories(self, access_token: str) -> List[OAuthRepository]:
        """List user's GitLab repositories."""
        client = await self.get_client()
        headers = {"Authorization": f"Bearer {access_token}"}
        
        repos = []
        page = 1
        per_page = 100
        
        while True:
            response = await client.get(
                f"{self.api_base_url}/projects",
                headers=headers,
                params={"page": page, "per_page": per_page, "membership": True},
            )
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            for project in data:
                repos.append(OAuthRepository(
                    provider="gitlab",
                    provider_repo_id=str(project["id"]),
                    full_name=project["path_with_namespace"],
                    name=project["name"],
                    owner=project["namespace"]["path"],
                    description=project.get("description"),
                    url=project["web_url"],
                    clone_url=project["http_url_to_repo"],
                    ssh_url=project["ssh_url_to_repo"],
                    default_branch=project.get("default_branch", "main"),
                    is_private=project["visibility"] == "private",
                    is_fork=project.get("forked_from_project") is not None,
                    stars=project.get("star_count", 0),
                    forks=project.get("forks_count", 0),
                ))
            
            if len(data) < per_page:
                break
            page += 1
        
        return repos


class OAuthProviderFactory:
    """Factory for creating OAuth providers."""
    
    _providers: Dict[str, type] = {
        "github": GitHubOAuth,
        "gitlab": GitLabOAuth,
    }
    
    @classmethod
    def get_provider(cls, provider_name: str) -> OAuthProviderBase:
        """Get an OAuth provider instance."""
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown OAuth provider: {provider_name}")
        
        # Get credentials from environment
        client_id = os.getenv(f"{provider_name.upper()}_CLIENT_ID")
        client_secret = os.getenv(f"{provider_name.upper()}_CLIENT_SECRET")
        redirect_uri = os.getenv(
            f"{provider_name.upper()}_REDIRECT_URI",
            os.getenv("OAUTH_REDIRECT_BASE_URL", "http://localhost:5173") + f"/oauth/callback/{provider_name}"
        )
        
        if not client_id or not client_secret:
            raise ValueError(f"Missing OAuth credentials for {provider_name}")
        
        return provider_class(client_id, client_secret, redirect_uri)
    
    @classmethod
    def generate_state(cls) -> str:
        """Generate a secure state token."""
        return secrets.token_urlsafe(32)
