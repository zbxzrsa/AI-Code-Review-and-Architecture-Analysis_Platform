"""
OAuth Routes / OAuth 认证路由
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import (
    GITHUB_CLIENT_ID,
    GITLAB_CLIENT_ID,
    BITBUCKET_API_TOKEN,
)

router = APIRouter(prefix="/api/oauth", tags=["OAuth"])


class OAuthConnection(BaseModel):
    provider: str
    connected: bool
    username: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    connected_at: Optional[str] = None


# In-memory storage for demo
_oauth_connections: list[dict] = []


@router.get("/providers")
async def list_oauth_providers():
    """List available OAuth providers / 列出可用的 OAuth 提供商"""
    return {
        "providers": [
            {
                "id": "github",
                "name": "GitHub",
                "enabled": bool(GITHUB_CLIENT_ID),
                "oauth_url": f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&scope=repo,user"
                if GITHUB_CLIENT_ID else None
            },
            {
                "id": "gitlab",
                "name": "GitLab",
                "enabled": bool(GITLAB_CLIENT_ID),
                "oauth_url": f"https://gitlab.com/oauth/authorize?client_id={GITLAB_CLIENT_ID}&scope=read_user+read_repository"
                if GITLAB_CLIENT_ID else None
            },
            {
                "id": "bitbucket",
                "name": "Bitbucket",
                "enabled": bool(BITBUCKET_API_TOKEN),
                "oauth_url": None  # Uses API token
            }
        ]
    }


@router.get("/connections")
async def get_oauth_connections():
    """Get user's OAuth connections / 获取用户的 OAuth 连接"""
    return {
        "connections": [
            {
                "provider": "github",
                "connected": any(c["provider"] == "github" for c in _oauth_connections),
                "username": next((c["username"] for c in _oauth_connections if c["provider"] == "github"), None),
            },
            {
                "provider": "gitlab",
                "connected": any(c["provider"] == "gitlab" for c in _oauth_connections),
                "username": next((c["username"] for c in _oauth_connections if c["provider"] == "gitlab"), None),
            },
            {
                "provider": "bitbucket",
                "connected": bool(BITBUCKET_API_TOKEN),
                "username": "api_token_user" if BITBUCKET_API_TOKEN else None,
            }
        ]
    }


@router.post("/connect/{provider}")
async def connect_oauth(provider: str):
    """Initiate OAuth connection / 发起 OAuth 连接"""
    if provider == "github":
        if not GITHUB_CLIENT_ID:
            raise HTTPException(status_code=400, detail="GitHub OAuth not configured")
        return {
            "redirect_url": f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&scope=repo,user"
        }
    elif provider == "gitlab":
        if not GITLAB_CLIENT_ID:
            raise HTTPException(status_code=400, detail="GitLab OAuth not configured")
        return {
            "redirect_url": f"https://gitlab.com/oauth/authorize?client_id={GITLAB_CLIENT_ID}&scope=read_user+read_repository"
        }
    elif provider == "bitbucket":
        if not BITBUCKET_API_TOKEN:
            raise HTTPException(status_code=400, detail="Bitbucket API token not configured")
        # Bitbucket uses API token, no OAuth redirect
        _oauth_connections.append({
            "provider": "bitbucket",
            "username": "api_token_user",
            "connected_at": datetime.now().isoformat()
        })
        return {"message": "Connected via API token", "connected": True}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


@router.post("/callback/{provider}")
async def oauth_callback(provider: str, code: Optional[str] = None):
    """Handle OAuth callback / 处理 OAuth 回调"""
    # In production, exchange code for access token
    # For demo, just mark as connected
    _oauth_connections.append({
        "provider": provider,
        "username": f"{provider}_user",
        "connected_at": datetime.now().isoformat()
    })
    return {"message": f"Connected to {provider}", "connected": True}


@router.delete("/disconnect/{provider}")
async def disconnect_oauth(provider: str):
    """Disconnect OAuth provider / 断开 OAuth 连接"""
    global _oauth_connections
    _oauth_connections = [c for c in _oauth_connections if c["provider"] != provider]
    return {"message": f"Disconnected from {provider}"}
