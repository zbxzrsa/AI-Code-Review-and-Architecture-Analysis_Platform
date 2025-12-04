"""
OAuth Router - GitHub/GitLab OAuth Integration

Handles OAuth authorization flow:
1. Initiate OAuth flow
2. Handle callback with authorization code
3. Exchange code for tokens
4. Create/link user account
"""

import os
import secrets
import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, status, Response, Query, Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# Import shared modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from shared.database import get_async_session
from shared.database.models import User, OAuthConnection, UserSession, OAuthProvider, UserStatus, UserRole
from shared.auth.oauth_providers import OAuthProviderFactory, GitHubOAuth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/oauth", tags=["OAuth"])

# In-memory state storage (use Redis in production)
_oauth_states: dict = {}


class OAuthInitResponse(BaseModel):
    """OAuth initiation response."""
    authorization_url: str
    state: str


class OAuthCallbackResponse(BaseModel):
    """OAuth callback response."""
    success: bool
    message: str
    user_id: Optional[str] = None
    is_new_user: bool = False
    requires_linking: bool = False


class ConnectedAccount(BaseModel):
    """Connected OAuth account."""
    provider: str
    username: str
    email: Optional[str]
    connected_at: datetime


# Helper functions
def generate_state() -> str:
    """Generate a secure state token."""
    return secrets.token_urlsafe(32)


def store_state(state: str, data: dict, ttl_seconds: int = 600) -> None:
    """Store OAuth state (use Redis in production)."""
    _oauth_states[state] = {
        "data": data,
        "expires_at": datetime.utcnow() + timedelta(seconds=ttl_seconds),
    }


def get_state(state: str) -> Optional[dict]:
    """Get and validate OAuth state."""
    stored = _oauth_states.get(state)
    if not stored:
        return None
    
    if datetime.utcnow() > stored["expires_at"]:
        del _oauth_states[state]
        return None
    
    return stored["data"]


def clear_state(state: str) -> None:
    """Clear OAuth state after use."""
    _oauth_states.pop(state, None)


@router.get("/providers")
async def list_providers():
    """
    List available OAuth providers.
    
    Returns configured OAuth providers with their authorization URLs.
    """
    providers = []
    
    # Check GitHub
    if os.getenv("GITHUB_CLIENT_ID"):
        providers.append({
            "name": "github",
            "display_name": "GitHub",
            "icon": "github",
            "scopes": ["read:user", "user:email", "repo"],
        })
    
    # Check GitLab
    if os.getenv("GITLAB_CLIENT_ID"):
        providers.append({
            "name": "gitlab",
            "display_name": "GitLab",
            "icon": "gitlab",
            "scopes": ["read_user", "read_api", "read_repository"],
        })
    
    return {"providers": providers}


@router.get("/connect/{provider}", response_model=OAuthInitResponse)
async def initiate_oauth(
    provider: str,
    return_url: Optional[str] = Query(None, description="URL to redirect after OAuth"),
    link_to_user: Optional[str] = Query(None, description="User ID to link account to"),
):
    """
    Initiate OAuth authorization flow.
    
    Redirects user to the OAuth provider's authorization page.
    
    Args:
        provider: OAuth provider name (github, gitlab)
        return_url: URL to redirect after successful OAuth
        link_to_user: If provided, links the OAuth account to this user
    
    Returns:
        Authorization URL to redirect the user to
    """
    try:
        oauth = OAuthProviderFactory.get_provider(provider)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    
    # Generate and store state
    state = generate_state()
    store_state(state, {
        "provider": provider,
        "return_url": return_url or "/",
        "link_to_user": link_to_user,
    })
    
    # Generate authorization URL
    auth_url = oauth.get_authorization_url(state)
    
    return OAuthInitResponse(
        authorization_url=auth_url,
        state=state,
    )


@router.get("/callback/{provider}")
async def oauth_callback(
    provider: str,
    code: str = Query(..., description="Authorization code from OAuth provider"),
    state: str = Query(..., description="State token for CSRF protection"),
    response: Response = None,
    db: AsyncSession = Depends(get_async_session),
):
    """
    Handle OAuth callback from provider.
    
    Exchanges authorization code for tokens and creates/links user account.
    
    Args:
        provider: OAuth provider name
        code: Authorization code from provider
        state: State token for CSRF protection
        response: HTTP response for setting cookies
        db: Database session
    
    Returns:
        Redirects to frontend with success/error status
    """
    # Validate state
    state_data = get_state(state)
    if not state_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state",
        )
    
    if state_data["provider"] != provider:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provider mismatch in OAuth state",
        )
    
    clear_state(state)
    
    try:
        # Get OAuth provider
        oauth = OAuthProviderFactory.get_provider(provider)
        
        # Exchange code for tokens
        tokens = await oauth.exchange_code(code)
        
        # Get user info from provider
        oauth_user = await oauth.get_user_info(tokens.access_token)
        
        # Close OAuth client
        await oauth.close()
        
    except Exception as e:
        logger.error(f"OAuth error for {provider}: {e}")
        return_url = state_data.get("return_url", "/")
        return RedirectResponse(
            url=f"{return_url}?oauth_error={str(e)}",
            status_code=status.HTTP_302_FOUND,
        )
    
    # Check if OAuth connection exists
    oauth_connection = await db.execute(
        select(OAuthConnection).where(
            OAuthConnection.provider == OAuthProvider(provider),
            OAuthConnection.provider_user_id == oauth_user.provider_user_id,
        )
    )
    oauth_connection = oauth_connection.scalar_one_or_none()
    
    user = None
    is_new_user = False
    
    if oauth_connection:
        # Existing connection - get user
        user = await db.get(User, oauth_connection.user_id)
        
        # Update tokens
        oauth_connection.access_token_encrypted = tokens.access_token  # TODO: encrypt
        oauth_connection.token_expires_at = tokens.expires_at
        oauth_connection.updated_at = datetime.utcnow()
        
    else:
        # New connection
        link_to_user_id = state_data.get("link_to_user")
        
        if link_to_user_id:
            # Link to existing user
            user = await db.get(User, link_to_user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User to link not found",
                )
        else:
            # Check if email already exists
            if oauth_user.email:
                existing_user = await db.execute(
                    select(User).where(User.email == oauth_user.email)
                )
                user = existing_user.scalar_one_or_none()
            
            if not user:
                # Create new user
                user = User(
                    email=oauth_user.email or f"{oauth_user.username}@{provider}.oauth",
                    password_hash="oauth_user",  # OAuth users don't have passwords
                    name=oauth_user.name or oauth_user.username or "OAuth User",
                    avatar_url=oauth_user.avatar_url,
                    role=UserRole.USER,
                    status=UserStatus.ACTIVE,
                    email_verified=bool(oauth_user.email),
                )
                db.add(user)
                await db.flush()  # Get user ID
                is_new_user = True
        
        # Create OAuth connection
        oauth_connection = OAuthConnection(
            user_id=user.id,
            provider=OAuthProvider(provider),
            provider_user_id=oauth_user.provider_user_id,
            provider_username=oauth_user.username,
            provider_email=oauth_user.email,
            access_token_encrypted=tokens.access_token,  # TODO: encrypt
            refresh_token_encrypted=tokens.refresh_token,
            token_expires_at=tokens.expires_at,
            scopes=tokens.scope.split() if tokens.scope else [],
        )
        db.add(oauth_connection)
    
    # Update user login info
    user.last_login_at = datetime.utcnow()
    user.login_count += 1
    
    await db.commit()
    
    # Create session and set cookies
    # TODO: Implement proper JWT token generation
    session_token = secrets.token_urlsafe(32)
    
    response.set_cookie(
        key="access_token",
        value=session_token,
        httponly=True,
        secure=os.getenv("ENVIRONMENT", "development") != "development",
        samesite="lax",
        max_age=900,  # 15 minutes
    )
    
    # Redirect to frontend
    return_url = state_data.get("return_url", "/")
    params = f"oauth_success=true&provider={provider}"
    if is_new_user:
        params += "&new_user=true"
    
    return RedirectResponse(
        url=f"{return_url}?{params}",
        status_code=status.HTTP_302_FOUND,
    )


@router.get("/connections")
async def list_connections(
    db: AsyncSession = Depends(get_async_session),
    # TODO: Add current user dependency
):
    """
    List user's connected OAuth accounts.
    
    Returns all OAuth providers connected to the current user.
    """
    # TODO: Get current user from auth middleware
    current_user_id = "user_123"  # Placeholder
    
    result = await db.execute(
        select(OAuthConnection).where(OAuthConnection.user_id == current_user_id)
    )
    connections = result.scalars().all()
    
    return {
        "connections": [
            ConnectedAccount(
                provider=conn.provider.value,
                username=conn.provider_username,
                email=conn.provider_email,
                connected_at=conn.created_at,
            )
            for conn in connections
        ]
    }


@router.delete("/connections/{provider}")
async def disconnect_provider(
    provider: str,
    db: AsyncSession = Depends(get_async_session),
    # TODO: Add current user dependency
):
    """
    Disconnect an OAuth provider.
    
    Removes the OAuth connection for the specified provider.
    User must have another authentication method (password or other OAuth).
    """
    # TODO: Get current user from auth middleware
    current_user_id = "user_123"  # Placeholder
    
    # Get connection
    result = await db.execute(
        select(OAuthConnection).where(
            OAuthConnection.user_id == current_user_id,
            OAuthConnection.provider == OAuthProvider(provider),
        )
    )
    connection = result.scalar_one_or_none()
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {provider} connection found",
        )
    
    # Check if user has other auth methods
    user = await db.get(User, current_user_id)
    other_connections = await db.execute(
        select(OAuthConnection).where(
            OAuthConnection.user_id == current_user_id,
            OAuthConnection.provider != OAuthProvider(provider),
        )
    )
    
    if user.password_hash == "oauth_user" and not other_connections.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot disconnect last authentication method. Set a password first.",
        )
    
    # Delete connection
    await db.delete(connection)
    await db.commit()
    
    return {"message": f"Disconnected from {provider}"}


@router.get("/{provider}/repositories")
async def list_oauth_repositories(
    provider: str,
    db: AsyncSession = Depends(get_async_session),
    # TODO: Add current user dependency
):
    """
    List repositories from connected OAuth provider.
    
    Uses the stored OAuth tokens to fetch the user's repositories.
    """
    # TODO: Get current user from auth middleware
    current_user_id = "user_123"  # Placeholder
    
    # Get OAuth connection
    result = await db.execute(
        select(OAuthConnection).where(
            OAuthConnection.user_id == current_user_id,
            OAuthConnection.provider == OAuthProvider(provider),
        )
    )
    connection = result.scalar_one_or_none()
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {provider} connection found. Please connect first.",
        )
    
    # Check token expiration
    if connection.token_expires_at and connection.token_expires_at < datetime.utcnow():
        # TODO: Implement token refresh
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OAuth token expired. Please reconnect.",
        )
    
    try:
        # Get OAuth provider
        oauth = OAuthProviderFactory.get_provider(provider)
        
        # TODO: Decrypt access token
        access_token = connection.access_token_encrypted
        
        # Fetch repositories
        repos = await oauth.list_repositories(access_token)
        
        await oauth.close()
        
        return {
            "repositories": [
                {
                    "id": repo.provider_repo_id,
                    "full_name": repo.full_name,
                    "name": repo.name,
                    "owner": repo.owner,
                    "description": repo.description,
                    "url": repo.url,
                    "clone_url": repo.clone_url,
                    "default_branch": repo.default_branch,
                    "is_private": repo.is_private,
                    "stars": repo.stars,
                    "forks": repo.forks,
                }
                for repo in repos
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch repositories from {provider}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch repositories: {str(e)}",
        )
