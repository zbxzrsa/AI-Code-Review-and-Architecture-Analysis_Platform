"""
Authentication_V2 - MFA Service

Multi-factor authentication with TOTP support.
"""

import secrets
import hashlib
import base64
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MFASetup:
    """MFA setup result"""
    secret: str
    qr_uri: str
    backup_codes: list

    def to_dict(self) -> Dict[str, Any]:
        return {
            "secret": self.secret,
            "qr_uri": self.qr_uri,
            "backup_codes": self.backup_codes,
        }


class MFAService:
    """
    Multi-factor authentication service.

    Features:
    - TOTP generation and verification
    - Backup codes
    - QR code URI generation
    """

    ALGORITHM = "SHA1"
    DIGITS = 6
    PERIOD = 30

    def __init__(self, issuer: str = "CodeReview Platform"):
        self.issuer = issuer
        self._backup_codes: Dict[str, list] = {}

    def generate_secret(self) -> str:
        """Generate TOTP secret"""
        return base64.b32encode(secrets.token_bytes(20)).decode('utf-8')

    def setup_mfa(self, user_id: str, email: str) -> MFASetup:
        """Set up MFA for user"""
        secret = self.generate_secret()

        # Generate QR code URI
        qr_uri = self._generate_qr_uri(secret, email)

        # Generate backup codes
        backup_codes = self._generate_backup_codes(user_id)

        logger.info(f"MFA setup for user: {user_id}")

        return MFASetup(
            secret=secret,
            qr_uri=qr_uri,
            backup_codes=backup_codes,
        )

    def verify_totp(self, secret: str, code: str, window: int = 1) -> bool:
        """
        Verify TOTP code.

        Args:
            secret: Base32-encoded secret
            code: 6-digit code to verify
            window: Time window tolerance

        Returns:
            True if code is valid
        """
        if len(code) != self.DIGITS:
            return False

        try:
            code_int = int(code)
        except ValueError:
            return False

        # Check current and adjacent time windows
        current_time = int(datetime.now(timezone.utc).timestamp())

        for offset in range(-window, window + 1):
            time_step = (current_time // self.PERIOD) + offset
            expected = self._generate_totp(secret, time_step)

            if expected == code_int:
                return True

        return False

    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume backup code"""
        if user_id not in self._backup_codes:
            return False

        codes = self._backup_codes[user_id]

        if code in codes:
            codes.remove(code)
            logger.info(f"Backup code used for user: {user_id}")
            return True

        return False

    def _generate_totp(self, secret: str, time_step: int) -> int:
        """Generate TOTP for time step"""
        try:
            key = base64.b32decode(secret)
        except Exception:
            return -1

        # HMAC-based OTP
        msg = time_step.to_bytes(8, byteorder='big')

        import hmac
        h = hmac.new(key, msg, hashlib.sha1).digest()

        # Dynamic truncation
        offset = h[-1] & 0x0F
        code = ((h[offset] & 0x7F) << 24 |
                (h[offset + 1] & 0xFF) << 16 |
                (h[offset + 2] & 0xFF) << 8 |
                (h[offset + 3] & 0xFF))

        return code % (10 ** self.DIGITS)

    def _generate_qr_uri(self, secret: str, email: str) -> str:
        """Generate otpauth URI for QR code"""
        import urllib.parse

        label = urllib.parse.quote(f"{self.issuer}:{email}")
        params = {
            "secret": secret,
            "issuer": self.issuer,
            "algorithm": self.ALGORITHM,
            "digits": str(self.DIGITS),
            "period": str(self.PERIOD),
        }

        param_str = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())

        return f"otpauth://totp/{label}?{param_str}"

    def _generate_backup_codes(self, user_id: str, count: int = 10) -> list:
        """Generate backup codes"""
        codes = [secrets.token_hex(4).upper() for _ in range(count)]
        self._backup_codes[user_id] = codes.copy()
        return codes

    def regenerate_backup_codes(self, user_id: str) -> list:
        """Regenerate backup codes"""
        return self._generate_backup_codes(user_id)

    def get_remaining_backup_codes(self, user_id: str) -> int:
        """Get count of remaining backup codes"""
        return len(self._backup_codes.get(user_id, []))
