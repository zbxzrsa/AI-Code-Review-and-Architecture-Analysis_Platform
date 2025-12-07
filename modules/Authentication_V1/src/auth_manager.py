"""
Authentication_V1 - Auth Manager

Core authentication logic with JWT support.
"""

import logging
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User model"""
    user_id: str
    email: str
    role: str
    created_at: datetime
    last_login: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


@dataclass
class AuthResult:
    """Authentication result"""
    success: bool
    user: Optional[User]
    access_token: Optional[str]
    refresh_token: Optional[str]
    error: Optional[str] = None
    expires_at: Optional[datetime] = None


class AuthManager:
    """
    Authentication manager with JWT support.

    Features:
    - User authentication
    - Token generation
    - Role verification
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        secret_key: str = "dev-secret-key",
        access_token_ttl: int = 900,  # 15 minutes
        refresh_token_ttl: int = 604800,  # 7 days
    ):
        self.secret_key = secret_key
        self.access_token_ttl = access_token_ttl
        self.refresh_token_ttl = refresh_token_ttl

        # In-memory user store (for experimental version)
        self._users: Dict[str, Dict] = {}
        self._tokens: Dict[str, str] = {}

    async def register(
        self,
        email: str,
        password: str,
        role: str = "user",
    ) -> AuthResult:
        """Register new user"""
        if email in self._users:
            return AuthResult(success=False, user=None, access_token=None,
                            refresh_token=None, error="User already exists")

        user_id = secrets.token_hex(16)
        password_hash = self._hash_password(password)

        user = User(
            user_id=user_id,
            email=email,
            role=role,
            created_at=datetime.now(timezone.utc),
        )

        self._users[email] = {
            "user": user,
            "password_hash": password_hash,
        }

        tokens = self._generate_tokens(user)

        logger.info(f"User registered: {email}")

        return AuthResult(
            success=True,
            user=user,
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_at=tokens["expires_at"],
        )

    async def login(
        self,
        email: str,
        password: str,
    ) -> AuthResult:
        """Authenticate user"""
        if email not in self._users:
            return AuthResult(success=False, user=None, access_token=None,
                            refresh_token=None, error="Invalid credentials")

        user_data = self._users[email]

        if not self._verify_password(password, user_data["password_hash"]):
            return AuthResult(success=False, user=None, access_token=None,
                            refresh_token=None, error="Invalid credentials")

        user = user_data["user"]
        user.last_login = datetime.now(timezone.utc)

        tokens = self._generate_tokens(user)

        logger.info(f"User logged in: {email}")

        return AuthResult(
            success=True,
            user=user,
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_at=tokens["expires_at"],
        )

    async def verify_token(self, token: str) -> Optional[User]:
        """Verify access token and return user"""
        if token not in self._tokens:
            return None

        user_id = self._tokens[token]

        for user_data in self._users.values():
            if user_data["user"].user_id == user_id:
                return user_data["user"]

        return None

    async def logout(self, token: str) -> bool:
        """Invalidate token"""
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        hash_input = f"{salt}{password}{self.secret_key}"
        return f"{salt}${hashlib.sha256(hash_input.encode()).hexdigest()}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        salt, hash_value = stored_hash.split("$")
        hash_input = f"{salt}{password}{self.secret_key}"
        return hashlib.sha256(hash_input.encode()).hexdigest() == hash_value

    def _generate_tokens(self, user: User) -> Dict[str, Any]:
        """Generate access and refresh tokens"""
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.access_token_ttl)

        self._tokens[access_token] = user.user_id
        self._tokens[refresh_token] = user.user_id

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
        }

    def require_role(self, required_role: str):
        """Decorator for role-based access control"""
        def decorator(func):
            async def wrapper(user: User, *args, **kwargs):
                roles = {"admin": 3, "user": 2, "viewer": 1, "guest": 0}
                if roles.get(user.role, 0) < roles.get(required_role, 0):
                    raise PermissionError(f"Role '{required_role}' required")
                return await func(user, *args, **kwargs)
            return wrapper
        return decorator
