"""
Security utilities for authentication.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash
import pyotp
import qrcode
import io
import base64
import secrets
import hashlib

from src.config import settings

logger = logging.getLogger(__name__)

# Password hashing with Argon2id
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__memory_cost=65540,  # 64MB
    argon2__time_cost=3,
    argon2__parallelism=4,
)

argon2_hasher = PasswordHasher()


class SecurityManager:
    """Security utilities manager."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using Argon2id."""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    @staticmethod
    def hash_token(token: str) -> str:
        """Hash a token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def create_access_token(
        user_id: str,
        role: str,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create JWT access token."""
        if expires_delta is None:
            expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        expire = datetime.utcnow() + expires_delta
        to_encode = {
            "sub": user_id,
            "role": role,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }

        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        return encoded_jwt

    @staticmethod
    def create_refresh_token(user_id: str) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }

        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        return encoded_jwt

    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[dict]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM],
            )

            if payload.get("type") != token_type:
                logger.warning(f"Invalid token type: {payload.get('type')}")
                return None

            return payload
        except JWTError as e:
            logger.error(f"Token verification error: {e}")
            return None

    @staticmethod
    def setup_totp(user_id: str) -> tuple[str, str]:
        """
        Setup TOTP for 2FA.

        Returns:
            (secret, qr_code_url)
        """
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(
            totp.provisioning_uri(
                name=f"user_{user_id}",
                issuer_name="AI Code Review Platform",
            )
        )
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()

        return secret, f"data:image/png;base64,{qr_code_base64}"

    @staticmethod
    def verify_totp(secret: str, token: str) -> bool:
        """Verify TOTP token."""
        try:
            totp = pyotp.TOTP(secret)
            # Allow for time drift (±30 seconds)
            return totp.verify(token, valid_window=1)
        except Exception as e:
            logger.error(f"TOTP verification error: {e}")
            return False

    @staticmethod
    def generate_password_reset_token() -> str:
        """Generate password reset token."""
        return SecurityManager.generate_token(32)

    @staticmethod
    def generate_invitation_code() -> str:
        """Generate invitation code."""
        return SecurityManager.generate_token(16)


class RateLimiter:
    """Rate limiting utilities."""

    # In-memory store for rate limiting (use Redis in production)
    _login_attempts = {}

    @staticmethod
    def check_login_rate_limit(ip_address: str) -> bool:
        """
        Check if login attempts exceed rate limit.

        Returns:
            True if within limit, False if exceeded
        """
        key = f"login:{ip_address}"
        now = datetime.utcnow()

        if key not in RateLimiter._login_attempts:
            RateLimiter._login_attempts[key] = []

        # Remove old attempts (older than 15 minutes)
        attempts = [
            attempt for attempt in RateLimiter._login_attempts[key]
            if (now - attempt).total_seconds() < 900
        ]

        if len(attempts) >= 5:
            logger.warning(f"Rate limit exceeded for IP: {ip_address}")
            return False

        attempts.append(now)
        RateLimiter._login_attempts[key] = attempts
        return True

    @staticmethod
    def record_failed_login(ip_address: str) -> None:
        """Record failed login attempt."""
        key = f"login:{ip_address}"
        if key not in RateLimiter._login_attempts:
            RateLimiter._login_attempts[key] = []
        RateLimiter._login_attempts[key].append(datetime.utcnow())

    @staticmethod
    def reset_login_attempts(ip_address: str) -> None:
        """Reset login attempts for IP."""
        key = f"login:{ip_address}"
        if key in RateLimiter._login_attempts:
            del RateLimiter._login_attempts[key]


class AccountLockout:
    """Account lockout utilities."""

    # In-memory store (use Redis in production)
    _lockouts = {}

    @staticmethod
    def is_account_locked(user_id: str) -> bool:
        """Check if account is locked."""
        if user_id not in AccountLockout._lockouts:
            return False

        lockout_until = AccountLockout._lockouts[user_id]
        if datetime.utcnow() > lockout_until:
            del AccountLockout._lockouts[user_id]
            return False

        return True

    @staticmethod
    def lock_account(user_id: str, duration_minutes: int = 30) -> None:
        """Lock account for specified duration."""
        lockout_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        AccountLockout._lockouts[user_id] = lockout_until
        logger.warning(f"Account locked: {user_id} until {lockout_until}")

    @staticmethod
    def unlock_account(user_id: str) -> None:
        """Unlock account."""
        if user_id in AccountLockout._lockouts:
            del AccountLockout._lockouts[user_id]
            logger.info(f"Account unlocked: {user_id}")
