"""
Two-Factor Authentication Service

Implements TOTP (Time-based One-Time Password) using:
- pyotp for TOTP generation and verification
- qrcode for QR code generation
- Backup codes for account recovery
"""

import pyotp
import qrcode
import io
import base64
import secrets
import hashlib
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

from app.core.config import settings


class TwoFactorService:
    """
    Two-Factor Authentication Service
    
    Handles:
    - TOTP secret generation
    - QR code generation for authenticator apps
    - Code verification
    - Backup code management
    """
    
    # Application name shown in authenticator apps
    APP_NAME = "AI Code Review"
    
    # TOTP settings
    TOTP_DIGITS = 6
    TOTP_INTERVAL = 30  # seconds
    TOTP_VALID_WINDOW = 1  # Allow 1 interval before/after for clock drift
    
    # Backup codes settings
    BACKUP_CODE_COUNT = 10
    BACKUP_CODE_LENGTH = 8
    
    def generate_secret(self) -> str:
        """
        Generate a new TOTP secret
        
        Returns:
            Base32 encoded secret string
        """
        return pyotp.random_base32()
    
    def get_totp(self, secret: str) -> pyotp.TOTP:
        """
        Create TOTP instance from secret
        
        Args:
            secret: Base32 encoded secret
            
        Returns:
            TOTP instance
        """
        return pyotp.TOTP(
            secret,
            digits=self.TOTP_DIGITS,
            interval=self.TOTP_INTERVAL,
        )
    
    def generate_provisioning_uri(self, secret: str, email: str) -> str:
        """
        Generate provisioning URI for authenticator apps
        
        Args:
            secret: TOTP secret
            email: User's email address
            
        Returns:
            otpauth:// URI string
        """
        totp = self.get_totp(secret)
        return totp.provisioning_uri(
            name=email,
            issuer_name=self.APP_NAME,
        )
    
    def generate_qr_code(self, secret: str, email: str) -> str:
        """
        Generate QR code as base64 data URL
        
        Args:
            secret: TOTP secret
            email: User's email address
            
        Returns:
            Base64 encoded PNG image as data URL
        """
        uri = self.generate_provisioning_uri(secret, email)
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(uri)
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        base64_img = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{base64_img}"
    
    def verify_code(self, secret: str, code: str) -> bool:
        """
        Verify TOTP code
        
        Args:
            secret: TOTP secret
            code: 6-digit code from authenticator
            
        Returns:
            True if code is valid
        """
        if not code or len(code) != self.TOTP_DIGITS:
            return False
        
        # Remove any spaces or dashes
        code = code.replace(" ", "").replace("-", "")
        
        if not code.isdigit():
            return False
        
        totp = self.get_totp(secret)
        return totp.verify(code, valid_window=self.TOTP_VALID_WINDOW)
    
    def get_current_code(self, secret: str) -> str:
        """
        Get current TOTP code (for testing)
        
        Args:
            secret: TOTP secret
            
        Returns:
            Current 6-digit code
        """
        totp = self.get_totp(secret)
        return totp.now()
    
    def generate_backup_codes(self) -> List[str]:
        """
        Generate backup codes for account recovery
        
        Returns:
            List of backup codes
        """
        codes = []
        for _ in range(self.BACKUP_CODE_COUNT):
            # Generate random alphanumeric code
            code = secrets.token_hex(self.BACKUP_CODE_LENGTH // 2).upper()
            codes.append(code)
        return codes
    
    def hash_backup_code(self, code: str) -> str:
        """
        Hash a backup code for storage
        
        Args:
            code: Plain text backup code
            
        Returns:
            Hashed backup code
        """
        # Normalize: uppercase, remove spaces/dashes
        normalized = code.upper().replace(" ", "").replace("-", "")
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def verify_backup_code(self, code: str, hashed_codes: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Verify a backup code against stored hashes
        
        Args:
            code: Plain text backup code
            hashed_codes: List of hashed backup codes
            
        Returns:
            Tuple of (is_valid, matched_hash)
        """
        code_hash = self.hash_backup_code(code)
        
        for stored_hash in hashed_codes:
            if secrets.compare_digest(code_hash, stored_hash):
                return True, stored_hash
        
        return False, None
    
    def format_backup_code(self, code: str) -> str:
        """
        Format backup code for display (XXXX-XXXX)
        
        Args:
            code: Raw backup code
            
        Returns:
            Formatted backup code
        """
        code = code.upper()
        if len(code) >= 8:
            return f"{code[:4]}-{code[4:]}"
        return code
    
    def format_secret_for_manual_entry(self, secret: str) -> str:
        """
        Format secret for manual entry (groups of 4)
        
        Args:
            secret: Base32 secret
            
        Returns:
            Formatted secret with spaces
        """
        return " ".join([secret[i:i+4] for i in range(0, len(secret), 4)])


# Singleton instance
two_factor_service = TwoFactorService()


# ============================================
# Database Models (SQLAlchemy example)
# ============================================
"""
Example SQLAlchemy model additions for User:

class User(Base):
    # ... existing fields ...
    
    # 2FA fields
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String(32), nullable=True)  # Encrypted
    two_factor_backup_codes = Column(JSON, nullable=True)  # List of hashed codes
    two_factor_enabled_at = Column(DateTime, nullable=True)
    two_factor_last_used = Column(DateTime, nullable=True)
"""


# ============================================
# Encryption helpers for secret storage
# ============================================

from cryptography.fernet import Fernet
from app.core.config import settings


class SecretEncryption:
    """Encrypt/decrypt 2FA secrets for database storage"""
    
    def __init__(self, key: Optional[str] = None):
        # Use environment variable or generate key
        encryption_key = key or getattr(settings, "TWO_FACTOR_ENCRYPTION_KEY", None)
        if not encryption_key:
            raise ValueError("TWO_FACTOR_ENCRYPTION_KEY not configured")
        
        # Ensure key is proper Fernet format
        if len(encryption_key) == 32:
            encryption_key = base64.urlsafe_b64encode(encryption_key.encode()).decode()
        
        self.fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
    
    def encrypt(self, secret: str) -> str:
        """Encrypt a 2FA secret"""
        return self.fernet.encrypt(secret.encode()).decode()
    
    def decrypt(self, encrypted_secret: str) -> str:
        """Decrypt a 2FA secret"""
        return self.fernet.decrypt(encrypted_secret.encode()).decode()


# ============================================
# Rate limiting for 2FA attempts
# ============================================

class TwoFactorRateLimiter:
    """Rate limit 2FA verification attempts"""
    
    # Max attempts before lockout
    MAX_ATTEMPTS = 5
    
    # Lockout duration
    LOCKOUT_MINUTES = 15
    
    def __init__(self):
        # In production, use Redis
        self._attempts: dict[str, list[datetime]] = {}
        self._lockouts: dict[str, datetime] = {}
    
    def record_attempt(self, user_id: str, success: bool) -> None:
        """Record a 2FA attempt"""
        if success:
            # Clear attempts on success
            self._attempts.pop(user_id, None)
            self._lockouts.pop(user_id, None)
            return
        
        # Record failed attempt
        now = datetime.utcnow()
        if user_id not in self._attempts:
            self._attempts[user_id] = []
        
        # Clean old attempts
        cutoff = now - timedelta(minutes=self.LOCKOUT_MINUTES)
        self._attempts[user_id] = [
            t for t in self._attempts[user_id] if t > cutoff
        ]
        
        self._attempts[user_id].append(now)
        
        # Check for lockout
        if len(self._attempts[user_id]) >= self.MAX_ATTEMPTS:
            self._lockouts[user_id] = now + timedelta(minutes=self.LOCKOUT_MINUTES)
    
    def is_locked_out(self, user_id: str) -> Tuple[bool, Optional[int]]:
        """
        Check if user is locked out
        
        Returns:
            Tuple of (is_locked, seconds_remaining)
        """
        if user_id not in self._lockouts:
            return False, None
        
        lockout_until = self._lockouts[user_id]
        now = datetime.utcnow()
        
        if now >= lockout_until:
            # Lockout expired
            self._lockouts.pop(user_id, None)
            self._attempts.pop(user_id, None)
            return False, None
        
        seconds_remaining = int((lockout_until - now).total_seconds())
        return True, seconds_remaining
    
    def get_remaining_attempts(self, user_id: str) -> int:
        """Get remaining attempts before lockout"""
        if user_id not in self._attempts:
            return self.MAX_ATTEMPTS
        
        return max(0, self.MAX_ATTEMPTS - len(self._attempts[user_id]))


# Singleton rate limiter
two_factor_rate_limiter = TwoFactorRateLimiter()
