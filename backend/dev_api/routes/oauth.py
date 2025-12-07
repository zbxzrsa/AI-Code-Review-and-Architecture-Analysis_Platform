"""
OAuth Routes

OAuth authentication and repository integration endpoints.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException
from ..models import LoginRequest, RegisterRequest, TokenResponse
from ..mock_data import oauth_states, mock_oauth_connections, mock_repositories
from ..config import Literals, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, MOCK_MODE

router = APIRouter(tags=["OAuth"])


# ============================================
# Authentication
# ============================================

@router.post("/auth/login")
async def login(request: LoginRequest):
    """User login."""
    # In mock mode, accept any credentials
    if MOCK_MODE or request.email == Literals.DEMO_EMAIL:
        return TokenResponse(
            access_token=f"mock_access_{secrets.token_hex(16)}",
            refresh_token=f"mock_refresh_{secrets.token_hex(16)}",
            token_type="bearer",
            expires_in=3600
        )
    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/auth/register")
async def register(request: RegisterRequest):
    """User registration."""
    return TokenResponse(
        access_token=f"mock_access_{secrets.token_hex(16)}",
        refresh_token=f"mock_refresh_{secrets.token_hex(16)}",
        token_type="bearer",
        expires_in=3600
    )


@router.post("/auth/refresh")
async def refresh_token():
    """Refresh access token."""
    return TokenResponse(
        access_token=f"mock_access_{secrets.token_hex(16)}",
        refresh_token=f"mock_refresh_{secrets.token_hex(16)}",
        token_type="bearer",
        expires_in=3600
    )


@router.post("/auth/logout")
async def logout():
    """User logout."""
    return {"message": "Logged out successfully"}


@router.get("/auth/me")
async def get_current_user():
    """Get current user."""
    return {
        "id": "user_1",
        "email": Literals.DEMO_EMAIL,
        "name": Literals.DEMO_USER,
        "role": "admin",
        "permissions": ["read", "write", "admin"],
    }


# ============================================
# OAuth Providers
# ============================================

@router.get("/api/oauth/{provider}/authorize")
async def oauth_authorize(provider: str, redirect_uri: Optional[str] = None):
    """Get OAuth authorization URL."""
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {
        "provider": provider,
        "redirect_uri": redirect_uri,
        "created_at": datetime.now().isoformat()
    }
    
    if provider == "github":
        auth_url = f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&state={state}&scope=repo,user"
    elif provider == "gitlab":
        auth_url = f"https://gitlab.com/oauth/authorize?client_id=YOUR_CLIENT_ID&state={state}&scope=api"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    return {"authorization_url": auth_url, "state": state}


@router.post("/api/oauth/{provider}/callback")
async def oauth_callback(provider: str, code: str, state: str):
    """Handle OAuth callback."""
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    # In mock mode, return mock connection
    mock_oauth_connections.append({
        "provider": provider,
        "user_id": "user_1",
        "connected_at": datetime.now().isoformat()
    })
    
    return {
        "message": f"Connected to {provider}",
        "provider": provider,
        "username": f"mock_{provider}_user",
    }


@router.get("/api/user/oauth/connections")
async def get_oauth_connections():
    """Get user OAuth connections."""
    return {
        "items": [
            {
                "provider": "github",
                "username": "demo_user",
                "connected_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "scopes": ["repo", "user"],
            }
        ]
    }


@router.delete("/api/user/oauth/{provider}")
async def disconnect_oauth(provider: str):
    """Disconnect OAuth provider."""
    return {"message": f"Disconnected from {provider}"}


# ============================================
# Repositories
# ============================================

@router.get("/api/repositories")
async def list_repositories(
    provider: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    search: Optional[str] = None
):
    """List repositories from connected providers."""
    repos = mock_repositories
    
    if provider:
        repos = [r for r in repos if r["provider"] == provider]
    
    if search:
        repos = [r for r in repos if search.lower() in r["name"].lower()]
    
    return {
        "items": repos,
        "total": len(repos),
        "page": page,
        "limit": limit
    }


@router.get("/api/repositories/{repo_id}")
async def get_repository(repo_id: str):
    """Get repository details."""
    for repo in mock_repositories:
        if repo["id"] == repo_id:
            return repo
    raise HTTPException(status_code=404, detail="Repository not found")


@router.post("/api/repositories/{repo_id}/connect")
async def connect_repository(repo_id: str):
    """Connect repository to project."""
    return {
        "message": "Repository connected",
        "project_id": f"proj_{secrets.token_hex(4)}"
    }


@router.delete("/api/repositories/{repo_id}/disconnect")
async def disconnect_repository(repo_id: str):
    """Disconnect repository."""
    return {"message": "Repository disconnected"}


@router.get("/api/repositories/{repo_id}/branches")
async def get_repository_branches(repo_id: str):
    """Get repository branches."""
    return {
        "items": [
            {"name": "main", "protected": True, "default": True},
            {"name": "develop", "protected": False, "default": False},
            {"name": "feature/auth", "protected": False, "default": False},
        ]
    }


@router.get("/api/repositories/{repo_id}/commits")
async def get_repository_commits(repo_id: str, branch: str = "main"):
    """Get repository commits."""
    return {
        "items": [
            {
                "sha": "abc123",
                "message": "feat: add authentication",
                "author": Literals.JOHN_DOE,
                "date": (datetime.now() - timedelta(hours=2)).isoformat(),
            },
            {
                "sha": "def456",
                "message": "fix: resolve security issue",
                "author": Literals.JANE_SMITH,
                "date": (datetime.now() - timedelta(hours=5)).isoformat(),
            },
        ]
    }
