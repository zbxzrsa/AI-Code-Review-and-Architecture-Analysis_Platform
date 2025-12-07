"""
Users Routes

User profile and settings endpoints.
"""

from datetime import datetime, timedelta
from fastapi import APIRouter
from ..models import UpdateProfileRequest, ChangePasswordRequest
from ..config import Literals

router = APIRouter(prefix="/api/user", tags=["Users"])


@router.get("/profile")
async def get_user_profile():
    """Get current user profile."""
    return {
        "id": "user_1",
        "email": Literals.DEMO_EMAIL,
        "name": Literals.DEMO_USER,
        "avatar": None,
        "role": "admin",
        "created_at": (datetime.now() - timedelta(days=90)).isoformat(),
        "settings": {
            "theme": "system",
            "language": "en",
            "notifications": True,
        }
    }


@router.put("/profile")
async def update_user_profile(request: UpdateProfileRequest):
    """Update user profile."""
    return {
        "id": "user_1",
        "email": Literals.DEMO_EMAIL,
        "name": request.name or Literals.DEMO_USER,
        "avatar": request.avatar,
        "bio": request.bio,
        "company": request.company,
        "location": request.location,
        "website": request.website,
        "updated_at": datetime.now().isoformat()
    }


@router.post("/password")
async def change_password(request: ChangePasswordRequest):
    """Change user password."""
    return {"message": "Password changed successfully"}


@router.get("/api-keys")
async def list_api_keys():
    """List user API keys."""
    return {
        "items": [
            {
                "id": "key_1",
                "name": "Development Key",
                "prefix": "sk_dev_****",
                "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "last_used": (datetime.now() - timedelta(hours=2)).isoformat(),
            },
            {
                "id": "key_2",
                "name": "Production Key",
                "prefix": "sk_prod_****",
                "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
                "last_used": datetime.now().isoformat(),
            }
        ]
    }


@router.post("/api-keys")
async def create_api_key():
    """Create new API key."""
    return {
        "id": "key_new",
        "name": "New Key",
        "key": "sk_live_1234567890abcdef",  # Only shown once
        "created_at": datetime.now().isoformat()
    }


@router.delete("/api-keys/{key_id}")
async def delete_api_key(key_id: str):
    """Delete API key."""
    return {"message": "API key deleted"}


# ============================================
# User Settings
# ============================================

@router.get("/settings/privacy")
async def get_privacy_settings():
    """Get privacy settings."""
    return {
        "profile_visibility": "public",
        "show_email": False,
        "show_activity": True,
        "allow_analytics": True,
    }


@router.put("/settings/privacy")
async def update_privacy_settings():
    """Update privacy settings."""
    return {"message": "Privacy settings updated"}


@router.get("/settings/notifications")
async def get_notification_settings():
    """Get notification settings."""
    return {
        "email_notifications": True,
        "push_notifications": True,
        "analysis_complete": True,
        "security_alerts": True,
        "weekly_digest": False,
    }


@router.put("/settings/notifications")
async def update_notification_settings():
    """Update notification settings."""
    return {"message": "Notification settings updated"}


@router.delete("/account")
async def delete_account():
    """Request account deletion."""
    return {"message": "Account deletion requested"}
