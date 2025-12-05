"""
User Service / 用户服务

Business logic for user management operations.
用户管理操作的业务逻辑。
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import secrets
import hashlib


class UserService:
    """Service for user management / 用户管理服务"""
    
    def __init__(self):
        self._users: Dict[str, Dict[str, Any]] = {}
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        
        # Create default demo user
        self._create_demo_user()
    
    def _create_demo_user(self):
        """Create demo user for development / 创建演示用户"""
        self._users["user_1"] = {
            "id": "user_1",
            "email": "demo@example.com",
            "name": "Demo User",
            "role": "admin",
            "status": "active",
            "avatar_url": "https://api.dicebear.com/7.x/avataaars/svg?seed=demo",
            "created_at": datetime.now() - timedelta(days=90),
            "last_login": datetime.now(),
            "preferences": {
                "theme": "light",
                "language": "en",
                "notifications": True
            }
        }
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID / 根据 ID 获取用户"""
        return self._users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email / 根据邮箱获取用户"""
        for user in self._users.values():
            if user["email"] == email:
                return user
        return None
    
    def update_user(
        self,
        user_id: str,
        **updates
    ) -> Optional[Dict[str, Any]]:
        """Update user profile / 更新用户资料"""
        user = self._users.get(user_id)
        if not user:
            return None
        
        allowed_fields = {"name", "avatar_url", "bio"}
        for field, value in updates.items():
            if field in allowed_fields and value is not None:
                user[field] = value
        
        return user
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: List[str],
        expires_in_days: Optional[int] = 365
    ) -> Dict[str, Any]:
        """Create API key for user / 为用户创建 API 密钥"""
        key_value = f"sk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()
        key_id = f"key_{len(self._api_keys) + 1}"
        
        key_data = {
            "id": key_id,
            "user_id": user_id,
            "name": name,
            "key_hash": key_hash,
            "key_preview": f"{key_value[:8]}...{key_value[-4:]}",
            "scopes": scopes,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=expires_in_days)
            if expires_in_days else None,
            "last_used": None
        }
        
        self._api_keys[key_id] = key_data
        
        # Return with full key (only shown once)
        return {
            **key_data,
            "full_key": key_value
        }
    
    def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """List user's API keys / 列出用户的 API 密钥"""
        return [
            {k: v for k, v in key.items() if k != "key_hash"}
            for key in self._api_keys.values()
            if key["user_id"] == user_id
        ]
    
    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke API key / 撤销 API 密钥"""
        key = self._api_keys.get(key_id)
        if key and key["user_id"] == user_id:
            del self._api_keys[key_id]
            return True
        return False
    
    def validate_api_key(self, key_value: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info / 验证 API 密钥并返回用户信息"""
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()
        
        for key in self._api_keys.values():
            if key["key_hash"] == key_hash:
                # Check expiration
                if key["expires_at"] and key["expires_at"] < datetime.now():
                    return None
                
                # Update last used
                key["last_used"] = datetime.now()
                
                # Return user info
                return self.get_user(key["user_id"])
        
        return None
    
    def create_session(
        self,
        user_id: str,
        device_info: Optional[Dict[str, str]] = None
    ) -> str:
        """Create user session / 创建用户会话"""
        session_id = secrets.token_urlsafe(32)
        
        self._sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "device_info": device_info or {},
            "expires_at": datetime.now() + timedelta(days=7)
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and return user / 验证会话并返回用户"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if session["expires_at"] < datetime.now():
            del self._sessions[session_id]
            return None
        
        # Update last activity
        session["last_activity"] = datetime.now()
        
        return self.get_user(session["user_id"])
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session / 使会话失效"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics / 获取用户统计"""
        users = list(self._users.values())
        return {
            "total": len(users),
            "active": sum(1 for u in users if u["status"] == "active"),
            "admins": sum(1 for u in users if u["role"] == "admin"),
            "api_keys_issued": len(self._api_keys),
            "active_sessions": len(self._sessions)
        }
