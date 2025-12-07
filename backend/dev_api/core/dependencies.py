"""
Dependency Injection Management

FastAPI dependencies for:
- Database connections
- Authentication
- Authorization
- Service instances

Module Size: ~200 lines (target < 2000)
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Generator
from fastapi import Depends, HTTPException, Header, status
from functools import lru_cache

from .config import get_settings, Settings


# =============================================================================
# Database Dependencies
# =============================================================================

class MockDatabase:
    """Mock database for development."""
    
    def __init__(self):
        self.connected = True
        self._data: Dict[str, Any] = {}
    
    def execute(self, query: str, params: tuple = None):
        """Mock query execution."""
        return []
    
    def close(self):
        """Close connection."""
        self.connected = False


_db_instance: Optional[MockDatabase] = None


def get_db() -> Generator[MockDatabase, None, None]:
    """
    Get database connection.
    
    Yields:
        Database connection instance
        
    Usage:
        @router.get("/items")
        async def get_items(db = Depends(get_db)):
            return db.execute("SELECT * FROM items")
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = MockDatabase()
    
    try:
        yield _db_instance
    finally:
        pass  # Connection pooling handles cleanup


# =============================================================================
# Authentication Dependencies
# =============================================================================

# Mock user store (in production, would query database)
MOCK_USERS = {
    "user-001": {
        "id": "user-001",
        "email": "admin@example.com",
        "name": "Admin User",
        "role": "admin",
        "is_active": True,
    },
    "user-002": {
        "id": "user-002",
        "email": "user@example.com",
        "name": "Regular User",
        "role": "user",
        "is_active": True,
    },
}

# Mock session store
MOCK_SESSIONS: Dict[str, Dict] = {}


async def get_current_user(
    authorization: str = Header(...),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """
    Get currently authenticated user.
    
    Args:
        authorization: Bearer token from Authorization header
        
    Returns:
        User information dict
        
    Raises:
        HTTPException: If token is invalid or user not found
        
    Usage:
        @router.get("/me")
        async def get_me(user = Depends(get_current_user)):
            return user
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.split(" ")[1]
    
    # Check session
    session = MOCK_SESSIONS.get(token)
    if not session:
        # For development, accept any token and return mock user
        if settings.mock_mode:
            return MOCK_USERS.get("user-001", {})
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check expiration
    expires_at = datetime.fromisoformat(session.get("expires_at", ""))
    if datetime.now(timezone.utc) > expires_at:
        del MOCK_SESSIONS[token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user
    user_id = session.get("user_id")
    user = MOCK_USERS.get(user_id)
    
    if not user or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    return user


async def get_optional_user(
    authorization: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings),
) -> Optional[Dict[str, Any]]:
    """
    Get current user if authenticated, None otherwise.
    
    Useful for endpoints that work with or without authentication.
    """
    if not authorization:
        return None
    
    try:
        return await get_current_user(authorization, settings)
    except HTTPException:
        return None


async def require_auth(
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Require authentication (alias for get_current_user).
    """
    return user


async def require_admin(
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Require admin role.
    
    Raises:
        HTTPException: If user is not an admin
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


# =============================================================================
# Service Dependencies
# =============================================================================

@lru_cache()
def get_code_review_service():
    """Get code review service instance."""
    from ..services import CodeReviewService
    return CodeReviewService()


@lru_cache()
def get_vulnerability_service():
    """Get vulnerability service instance."""
    from ..services import VulnerabilityService
    return VulnerabilityService()


@lru_cache()
def get_analytics_service():
    """Get analytics service instance."""
    from ..services import AnalyticsService
    return AnalyticsService()
