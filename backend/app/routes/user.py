"""
User Routes / 用户路由
"""

from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import secrets

router = APIRouter(prefix="/api/user", tags=["User"])


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None


class CreateApiKeyRequest(BaseModel):
    name: str
    scopes: List[str] = ["read"]
    expires_in_days: Optional[int] = 365


# In-memory storage for demo
_api_keys: list[dict] = []


@router.get("/profile")
async def get_user_profile():
    """Get current user profile / 获取当前用户资料"""
    return {
        "id": "user_1",
        "email": "demo@example.com",
        "name": "Demo User",
        "avatar_url": "https://api.dicebear.com/7.x/avataaars/svg?seed=demo",
        "role": "admin",
        "bio": "AI Code Review Platform enthusiast",
        "created_at": (datetime.now() - timedelta(days=90)).isoformat(),
        "last_login": datetime.now().isoformat(),
        "preferences": {
            "theme": "light",
            "language": "en",
            "notifications": True
        }
    }


@router.put("/profile")
async def update_user_profile(request: UpdateProfileRequest):
    """Update user profile / 更新用户资料"""
    return {
        "message": "Profile updated successfully",
        "updated_fields": [k for k, v in request.dict().items() if v is not None]
    }


@router.get("/settings/privacy")
async def get_privacy_settings():
    """Get privacy settings / 获取隐私设置"""
    return {
        "profile_visibility": "public",
        "show_email": False,
        "show_activity": True,
        "allow_analytics": True,
        "two_factor_enabled": False
    }


@router.put("/settings/privacy")
async def update_privacy_settings(settings: dict):
    """Update privacy settings / 更新隐私设置"""
    return {"message": "Privacy settings updated", "settings": settings}


@router.get("/settings/notifications")
async def get_notification_settings():
    """Get notification settings / 获取通知设置"""
    return {
        "email_notifications": True,
        "push_notifications": True,
        "analysis_complete": True,
        "security_alerts": True,
        "weekly_digest": True,
        "marketing": False
    }


@router.put("/settings/notifications")
async def update_notification_settings(settings: dict):
    """Update notification settings / 更新通知设置"""
    return {"message": "Notification settings updated", "settings": settings}


@router.get("/api-keys")
async def list_api_keys():
    """List user's API keys / 列出用户的 API 密钥"""
    return {
        "keys": _api_keys,
        "total": len(_api_keys)
    }


@router.post("/api-keys")
async def create_api_key(request: CreateApiKeyRequest):
    """Create new API key / 创建新的 API 密钥"""
    key_value = f"sk_{secrets.token_urlsafe(32)}"
    new_key = {
        "id": f"key_{len(_api_keys) + 1}",
        "name": request.name,
        "key_preview": f"{key_value[:8]}...{key_value[-4:]}",
        "full_key": key_value,  # Only shown once
        "scopes": request.scopes,
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(days=request.expires_in_days)).isoformat()
        if request.expires_in_days else None,
        "last_used": None
    }
    _api_keys.append(new_key)
    
    return {
        "message": "API key created successfully",
        "key": new_key,
        "warning": "Save this key now. You won't be able to see it again."
    }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: str):
    """Revoke API key / 撤销 API 密钥"""
    global _api_keys
    original_count = len(_api_keys)
    _api_keys = [k for k in _api_keys if k["id"] != key_id]
    
    if len(_api_keys) == original_count:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return {"message": f"API key {key_id} revoked"}


@router.get("/activity")
async def get_user_activity(limit: int = 20):
    """Get user activity / 获取用户活动"""
    # Cap limit to prevent resource exhaustion from user-controlled input
    MAX_LIMIT = 100
    safe_limit = min(max(1, limit), MAX_LIMIT)
    
    activities = [
        {
            "id": f"activity_{i}",
            "type": ["analysis", "login", "project_create", "settings_update"][i % 4],
            "message": f"Activity {i} description",
            "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
            "metadata": {}
        }
        for i in range(1, safe_limit + 1)
    ]
    
    return {
        "activities": activities,
        "total": len(activities)
    }


@router.post("/2fa/enable")
async def enable_2fa():
    """Enable two-factor authentication / 启用双因素认证"""
    return {
        "message": "2FA setup initiated",
        "secret": secrets.token_urlsafe(20),
        "qr_code_url": "https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=otpauth://totp/AICodeReview:demo@example.com?secret=MOCK_SECRET"
    }


@router.post("/2fa/verify")
async def verify_2fa(code: str):
    """Verify 2FA code / 验证双因素认证码"""
    # In production, verify against stored secret
    if len(code) == 6 and code.isdigit():
        return {"message": "2FA enabled successfully", "verified": True}
    raise HTTPException(status_code=400, detail="Invalid 2FA code")


@router.post("/2fa/disable")
async def disable_2fa():
    """Disable two-factor authentication / 禁用双因素认证"""
    return {"message": "2FA disabled"}
