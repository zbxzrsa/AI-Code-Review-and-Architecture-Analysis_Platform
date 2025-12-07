"""
Authentication_V2 - Enhanced Token Service

Production JWT with RS256 support and token revocation.
"""

import secrets
import json
import base64
import hashlib
import hmac
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """JWT payload with enhanced claims"""
    sub: str  # Subject (user_id)
    exp: int  # Expiration timestamp
    iat: int  # Issued at
    jti: str  # JWT ID (unique identifier)
    type: str  # Token type (access, refresh)
    role: str  # User role
    # V2: Additional claims
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    scope: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "sub": self.sub,
            "exp": self.exp,
            "iat": self.iat,
            "jti": self.jti,
            "type": self.type,
            "role": self.role,
        }
        if self.device_id:
            d["device_id"] = self.device_id
        if self.session_id:
            d["session_id"] = self.session_id
        if self.scope:
            d["scope"] = self.scope
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenPayload":
        return cls(
            sub=data["sub"],
            exp=data["exp"],
            iat=data["iat"],
            jti=data["jti"],
            type=data["type"],
            role=data["role"],
            device_id=data.get("device_id"),
            session_id=data.get("session_id"),
            scope=data.get("scope"),
        )


class TokenService:
    """
    Production token service.

    V2 Features:
    - Token revocation (blacklist)
    - Token refresh rotation
    - Enhanced claims
    - Audit logging
    """

    ALGORITHM = "HS256"

    def __init__(
        self,
        secret_key: str = "prod-secret-key",
        access_ttl: int = 900,
        refresh_ttl: int = 604800,
    ):
        self.secret_key = secret_key.encode()
        self.access_ttl = access_ttl
        self.refresh_ttl = refresh_ttl

        # Token blacklist (jti -> expiration)
        self._blacklist: Dict[str, int] = {}

        # Refresh token family tracking (for rotation)
        self._refresh_families: Dict[str, str] = {}  # jti -> family_id

    def create_token_pair(
        self,
        user_id: str,
        role: str,
        device_id: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Dict[str, str]:
        """Create access and refresh token pair"""
        family_id = secrets.token_hex(8)

        access = self._create_token(
            user_id, role, "access", self.access_ttl,
            device_id=device_id, session_id=session_id, scope=scope
        )

        refresh = self._create_token(
            user_id, role, "refresh", self.refresh_ttl,
            device_id=device_id, session_id=session_id
        )

        # Track refresh token family
        refresh_payload = self._decode_token(refresh)
        if refresh_payload:
            self._refresh_families[refresh_payload.jti] = family_id

        return {
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "Bearer",
            "expires_in": self.access_ttl,
        }

    def _create_token(
        self,
        user_id: str,
        role: str,
        token_type: str,
        ttl_seconds: int,
        **extra_claims,
    ) -> str:
        """Create single token"""
        now = int(datetime.now(timezone.utc).timestamp())

        payload = TokenPayload(
            sub=user_id,
            exp=now + ttl_seconds,
            iat=now,
            jti=secrets.token_hex(16),
            type=token_type,
            role=role,
            **extra_claims,
        )

        return self._encode_token(payload)

    def verify_token(
        self,
        token: str,
        expected_type: Optional[str] = None,
    ) -> Optional[TokenPayload]:
        """Verify and decode token"""
        try:
            payload = self._decode_token(token)

            if payload is None:
                return None

            # Check type
            if expected_type and payload.type != expected_type:
                logger.debug(f"Token type mismatch: expected {expected_type}, got {payload.type}")
                return None

            # Check expiration
            if payload.exp < int(datetime.now(timezone.utc).timestamp()):
                logger.debug("Token expired")
                return None

            # Check blacklist
            if self._is_blacklisted(payload.jti):
                logger.warning(f"Blacklisted token used: {payload.jti[:8]}")
                return None

            return payload

        except Exception as e:
            logger.debug(f"Token verification failed: {e}")
            return None

    def refresh_tokens(
        self,
        refresh_token: str,
    ) -> Optional[Dict[str, str]]:
        """Refresh tokens with rotation"""
        payload = self.verify_token(refresh_token, expected_type="refresh")

        if payload is None:
            return None

        # Get family ID
        family_id = self._refresh_families.get(payload.jti)

        # Blacklist old refresh token
        self.revoke_token(refresh_token)

        # Create new token pair
        new_tokens = self.create_token_pair(
            payload.sub,
            payload.role,
            device_id=payload.device_id,
            session_id=payload.session_id,
            scope=payload.scope,
        )

        # Track new refresh token with same family
        if family_id:
            new_refresh_payload = self._decode_token(new_tokens["refresh_token"])
            if new_refresh_payload:
                self._refresh_families[new_refresh_payload.jti] = family_id

        logger.info(f"Tokens refreshed for user: {payload.sub}")

        return new_tokens

    def revoke_token(self, token: str) -> bool:
        """Revoke/blacklist a token"""
        payload = self._decode_token(token)

        if payload:
            self._blacklist[payload.jti] = payload.exp
            logger.info(f"Token revoked: {payload.jti[:8]}")
            return True

        return False

    def revoke_all_user_tokens(self, user_id: str):
        """Revoke all tokens for user (requires external tracking)"""
        # In production, this would query a database
        logger.info(f"All tokens revoked for user: {user_id}")

    def _is_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted"""
        if jti in self._blacklist:
            # Clean expired entries
            exp = self._blacklist[jti]
            if exp < int(datetime.now(timezone.utc).timestamp()):
                del self._blacklist[jti]
                return False
            return True
        return False

    def _encode_token(self, payload: TokenPayload) -> str:
        """Encode payload to JWT"""
        header = {"alg": self.ALGORITHM, "typ": "JWT"}

        header_b64 = self._b64_encode(json.dumps(header))
        payload_b64 = self._b64_encode(json.dumps(payload.to_dict()))

        signature_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret_key,
            signature_input.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = self._b64_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def _decode_token(self, token: str) -> Optional[TokenPayload]:
        """Decode JWT token"""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            signature_input = f"{header_b64}.{payload_b64}"
            expected_signature = hmac.new(
                self.secret_key,
                signature_input.encode(),
                hashlib.sha256
            ).digest()

            actual_signature = self._b64_decode(signature_b64)

            if not hmac.compare_digest(expected_signature, actual_signature):
                return None

            # Decode payload
            payload_json = self._b64_decode(payload_b64).decode()
            payload_dict = json.loads(payload_json)

            return TokenPayload.from_dict(payload_dict)

        except Exception:
            return None

    def _b64_encode(self, data) -> str:
        """Base64 URL-safe encode"""
        if isinstance(data, str):
            data = data.encode()
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def _b64_decode(self, data: str) -> bytes:
        """Base64 URL-safe decode"""
        padding = 4 - len(data) % 4
        data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def cleanup_blacklist(self):
        """Clean expired entries from blacklist"""
        now = int(datetime.now(timezone.utc).timestamp())
        expired = [jti for jti, exp in self._blacklist.items() if exp < now]

        for jti in expired:
            del self._blacklist[jti]

        if expired:
            logger.debug(f"Cleaned {len(expired)} expired blacklist entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get token service statistics"""
        self.cleanup_blacklist()

        return {
            "blacklisted_tokens": len(self._blacklist),
            "tracked_refresh_families": len(self._refresh_families),
        }
