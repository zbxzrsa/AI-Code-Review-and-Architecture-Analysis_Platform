"""
Authentication_V1 - Token Service

JWT token generation and validation.
"""

import secrets
import json
import base64
import hashlib
import hmac
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """JWT payload"""
    sub: str  # Subject (user_id)
    exp: int  # Expiration timestamp
    iat: int  # Issued at
    type: str  # Token type (access, refresh)
    role: str  # User role

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub": self.sub,
            "exp": self.exp,
            "iat": self.iat,
            "type": self.type,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenPayload":
        return cls(
            sub=data["sub"],
            exp=data["exp"],
            iat=data["iat"],
            type=data["type"],
            role=data["role"],
        )


class TokenService:
    """
    JWT token service.

    Features:
    - Token generation (HS256)
    - Token validation
    - Token refresh
    """

    ALGORITHM = "HS256"

    def __init__(self, secret_key: str = "dev-secret-key"):
        self.secret_key = secret_key.encode()

    def create_access_token(
        self,
        user_id: str,
        role: str,
        ttl_seconds: int = 900,
    ) -> str:
        """Create access token"""
        return self._create_token(user_id, role, "access", ttl_seconds)

    def create_refresh_token(
        self,
        user_id: str,
        role: str,
        ttl_seconds: int = 604800,
    ) -> str:
        """Create refresh token"""
        return self._create_token(user_id, role, "refresh", ttl_seconds)

    def _create_token(
        self,
        user_id: str,
        role: str,
        token_type: str,
        ttl_seconds: int,
    ) -> str:
        """Create JWT token"""
        now = int(datetime.now(timezone.utc).timestamp())

        payload = TokenPayload(
            sub=user_id,
            exp=now + ttl_seconds,
            iat=now,
            type=token_type,
            role=role,
        )

        return self._encode_token(payload)

    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """Verify and decode token"""
        try:
            payload = self._decode_token(token)

            if payload is None:
                return None

            # Check expiration
            if payload.exp < int(datetime.now(timezone.utc).timestamp()):
                logger.debug("Token expired")
                return None

            return payload

        except Exception as e:
            logger.debug(f"Token verification failed: {e}")
            return None

    def refresh_tokens(
        self,
        refresh_token: str,
    ) -> Optional[Dict[str, str]]:
        """Refresh tokens using refresh token"""
        payload = self.verify_token(refresh_token)

        if payload is None or payload.type != "refresh":
            return None

        return {
            "access_token": self.create_access_token(payload.sub, payload.role),
            "refresh_token": self.create_refresh_token(payload.sub, payload.role),
        }

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
