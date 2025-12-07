"""
JWT Key Rotation

Implements automatic JWT key rotation with:
- Rotation cycle ≤ 7 days
- Support for multiple key versions
- Zero downtime during rotation
- Backward compatibility for existing tokens
"""

import os
import json
import asyncio
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    import jwt
except ImportError:
    jwt = None
    rsa = None

logger = logging.getLogger(__name__)


class KeyStatus(Enum):
    """Key status in rotation lifecycle."""
    ACTIVE = "active"          # Currently used for signing
    RETIRING = "retiring"      # Still valid for verification, not signing
    RETIRED = "retired"        # No longer valid


@dataclass
class JWTKey:
    """JWT signing key with metadata."""
    kid: str                              # Key ID
    algorithm: str = "RS256"              # Signing algorithm
    private_key: Optional[bytes] = None   # Private key (PEM)
    public_key: Optional[bytes] = None    # Public key (PEM)
    status: KeyStatus = KeyStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize key metadata (not private key)."""
        return {
            "kid": self.kid,
            "algorithm": self.algorithm,
            "public_key": self.public_key.decode() if self.public_key else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], private_key: Optional[bytes] = None) -> "JWTKey":
        """Deserialize key from dict."""
        return cls(
            kid=data["kid"],
            algorithm=data.get("algorithm", "RS256"),
            private_key=private_key,
            public_key=data["public_key"].encode() if data.get("public_key") else None,
            status=KeyStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            retired_at=datetime.fromisoformat(data["retired_at"]) if data.get("retired_at") else None,
        )


@dataclass
class RotationConfig:
    """JWT key rotation configuration."""
    # Rotation timing
    rotation_interval_days: int = 7       # Rotate every 7 days
    grace_period_days: int = 1            # Keep old key valid for 1 day after rotation

    # Key settings
    key_size: int = 4096                  # RSA key size
    algorithm: str = "RS256"              # Signing algorithm

    # Storage
    keys_directory: str = field(default_factory=lambda: os.getenv("JWT_KEYS_DIR", "/var/secrets/jwt"))

    # Limits
    max_active_keys: int = 3              # Maximum keys to keep

    # Auto-rotation
    auto_rotate: bool = True
    check_interval_hours: int = 1


class JWTKeyRotationManager:
    """
    Manages JWT key rotation with zero downtime.

    Features:
    - Automatic key rotation on schedule
    - Multiple key version support
    - Graceful key retirement
    - JWKS endpoint support

    Usage:
        manager = JWTKeyRotationManager()
        await manager.initialize()

        # Create token
        token = await manager.create_token({"user_id": "123"})

        # Verify token (works with any valid key)
        payload = await manager.verify_token(token)

        # Manual rotation
        await manager.rotate_keys()
    """

    def __init__(self, config: Optional[RotationConfig] = None, redis_client=None):
        self.config = config or RotationConfig()
        self.redis = redis_client
        self._keys: Dict[str, JWTKey] = {}
        self._active_kid: Optional[str] = None
        self._initialized = False
        self._rotation_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize key manager and load existing keys."""
        if self._initialized:
            return

        # Load existing keys
        await self._load_keys()

        # Create initial key if none exists
        if not self._keys:
            await self._generate_new_key()

        # Start auto-rotation task
        if self.config.auto_rotate:
            self._rotation_task = asyncio.create_task(self._auto_rotation_loop())

        self._initialized = True
        logger.info(f"JWT Key Manager initialized with {len(self._keys)} keys")

    async def shutdown(self) -> None:
        """Shutdown manager and cancel tasks."""
        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                # Intentionally not re-raised: we initiated the cancellation
                # during shutdown, so propagation is not needed
                logger.debug("Rotation task cancelled during shutdown")

    # ========================================
    # Token Operations
    # ========================================

    async def create_token(
        self,
        payload: Dict[str, Any],
        expires_in: int = 3600,
        token_type: str = "access"
    ) -> str:
        """
        Create a signed JWT token.

        Args:
            payload: Token payload
            expires_in: Expiration time in seconds
            token_type: Token type (access/refresh)

        Returns:
            Signed JWT token
        """
        if not self._initialized:
            await self.initialize()

        if jwt is None:
            raise RuntimeError("PyJWT not installed")

        active_key = self._get_active_key()
        if not active_key:
            raise RuntimeError("No active signing key available")

        now = datetime.now(timezone.utc)
        token_payload = {
            **payload,
            "iat": now,
            "exp": now + timedelta(seconds=expires_in),
            "type": token_type,
        }

        # Load private key
        private_key = serialization.load_pem_private_key(
            active_key.private_key,
            password=None,
            backend=default_backend()
        )

        token = jwt.encode(
            token_payload,
            private_key,
            algorithm=active_key.algorithm,
            headers={"kid": active_key.kid}
        )

        return token

    async def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.

        Supports tokens signed with any non-retired key.

        Args:
            token: JWT token to verify

        Returns:
            Decoded token payload

        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        if not self._initialized:
            await self.initialize()

        if jwt is None:
            raise RuntimeError("PyJWT not installed")

        # Get key ID from header
        try:
            unverified = jwt.get_unverified_header(token)
            kid = unverified.get("kid")
        except jwt.exceptions.DecodeError:
            raise jwt.InvalidTokenError("Invalid token format")

        # Find the key
        key = self._keys.get(kid)
        if not key:
            raise jwt.InvalidTokenError(f"Unknown key ID: {kid}")

        if key.status == KeyStatus.RETIRED:
            raise jwt.InvalidTokenError(f"Key {kid} has been retired")

        # Load public key
        public_key = serialization.load_pem_public_key(
            key.public_key,
            backend=default_backend()
        )

        # Verify and decode
        payload = jwt.decode(
            token,
            public_key,
            algorithms=[key.algorithm],
            options={"require": ["exp", "iat"]}
        )

        return payload

    async def refresh_token(
        self,
        refresh_token: str,
        access_expires_in: int = 3600,
        refresh_expires_in: int = 604800
    ) -> Tuple[str, str]:
        """
        Refresh tokens using a refresh token.

        Args:
            refresh_token: Current refresh token
            access_expires_in: New access token expiration
            refresh_expires_in: New refresh token expiration

        Returns:
            Tuple of (new_access_token, new_refresh_token)
        """
        # Verify refresh token
        payload = await self.verify_token(refresh_token)

        if payload.get("type") != "refresh":
            raise jwt.InvalidTokenError("Invalid token type")

        # Extract user data
        user_data = {k: v for k, v in payload.items() if k not in ["iat", "exp", "type"]}

        # Create new tokens
        new_access = await self.create_token(user_data, access_expires_in, "access")
        new_refresh = await self.create_token(user_data, refresh_expires_in, "refresh")

        return new_access, new_refresh

    # ========================================
    # Key Rotation
    # ========================================

    async def rotate_keys(self) -> JWTKey:
        """
        Perform key rotation.

        1. Generate new key
        2. Mark current key as retiring
        3. Retire old keys past grace period

        Returns:
            New active key
        """
        logger.info("Starting JWT key rotation")

        # Mark current active key as retiring
        if self._active_kid and self._active_kid in self._keys:
            old_key = self._keys[self._active_kid]
            old_key.status = KeyStatus.RETIRING
            old_key.retired_at = datetime.now(timezone.utc) + timedelta(days=self.config.grace_period_days)
            logger.info(f"Key {old_key.kid} marked as retiring")

        # Generate new key
        new_key = await self._generate_new_key()

        # Retire old keys past grace period
        await self._retire_old_keys()

        # Persist keys
        await self._save_keys()

        # Publish rotation event
        if self.redis:
            await self._publish_rotation_event(new_key.kid)

        logger.info(f"Key rotation complete. New active key: {new_key.kid}")
        return new_key

    async def _generate_new_key(self) -> JWTKey:
        """Generate a new RSA key pair."""
        if rsa is None:
            raise RuntimeError("cryptography not installed")

        # Generate RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.key_size,
            backend=default_backend()
        )

        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        # Create key object
        key = JWTKey(
            kid=f"key_{secrets.token_hex(8)}",
            algorithm=self.config.algorithm,
            private_key=private_pem,
            public_key=public_pem,
            status=KeyStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=self.config.rotation_interval_days),
        )

        # Store and set as active
        self._keys[key.kid] = key
        self._active_kid = key.kid

        logger.info(f"Generated new JWT key: {key.kid}")
        return key

    async def _retire_old_keys(self) -> None:
        """Retire keys past their grace period."""
        now = datetime.now(timezone.utc)

        for kid, key in list(self._keys.items()):
            if key.status == KeyStatus.RETIRING and key.retired_at and key.retired_at <= now:
                key.status = KeyStatus.RETIRED
                logger.info(f"Key {kid} retired")

        # Remove excess retired keys
        retired_keys = [k for k, v in self._keys.items() if v.status == KeyStatus.RETIRED]
        while len(self._keys) > self.config.max_active_keys and retired_keys:
            oldest = retired_keys.pop(0)
            del self._keys[oldest]
            logger.info(f"Removed old key: {oldest}")

    async def _auto_rotation_loop(self) -> None:
        """Background task for automatic key rotation."""
        while True:
            try:
                await asyncio.sleep(self.config.check_interval_hours * 3600)

                # Check if rotation needed
                active_key = self._get_active_key()
                if active_key and active_key.expires_at:
                    if datetime.now(timezone.utc) >= active_key.expires_at:
                        await self.rotate_keys()

            except asyncio.CancelledError:
                # Re-raise to properly propagate cancellation
                raise
            except Exception as e:
                logger.error(f"Auto-rotation error: {e}")

    # ========================================
    # Key Management
    # ========================================

    def _get_active_key(self) -> Optional[JWTKey]:
        """Get the current active signing key."""
        if self._active_kid:
            return self._keys.get(self._active_kid)

        # Fallback: find first active key
        for key in self._keys.values():
            if key.status == KeyStatus.ACTIVE:
                self._active_kid = key.kid
                return key

        return None

    def get_jwks(self) -> Dict[str, Any]:
        """
        Get JWKS (JSON Web Key Set) for public keys.

        Returns keys that can be used for token verification.
        """
        keys = []

        for key in self._keys.values():
            if key.status != KeyStatus.RETIRED and key.public_key:
                # Load public key to get components
                pub_key = serialization.load_pem_public_key(
                    key.public_key,
                    backend=default_backend()
                )

                # Get RSA components
                numbers = pub_key.public_numbers()

                # Convert to base64url
                import base64
                def to_base64url(num: int, length: int) -> str:
                    data = num.to_bytes(length, byteorder='big')
                    return base64.urlsafe_b64encode(data).rstrip(b'=').decode()

                keys.append({
                    "kty": "RSA",
                    "kid": key.kid,
                    "use": "sig",
                    "alg": key.algorithm,
                    "n": to_base64url(numbers.n, 512),  # 4096 bits = 512 bytes
                    "e": to_base64url(numbers.e, 3),
                })

        return {"keys": keys}

    def get_key_status(self) -> List[Dict[str, Any]]:
        """Get status of all keys."""
        return [
            {
                "kid": key.kid,
                "status": key.status.value,
                "created_at": key.created_at.isoformat(),
                "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                "is_active": key.kid == self._active_kid,
            }
            for key in self._keys.values()
        ]

    # ========================================
    # Persistence
    # ========================================

    async def _load_keys(self) -> None:
        """Load keys from storage."""
        # Try Redis first
        if self.redis:
            try:
                keys_data = await self.redis.get("jwt:keys:metadata")
                if keys_data:
                    metadata = json.loads(keys_data)
                    for kid, key_meta in metadata.items():
                        private_key = await self.redis.get(f"jwt:keys:private:{kid}")
                        if private_key:
                            key = JWTKey.from_dict(key_meta, private_key.encode())
                            self._keys[kid] = key
                            if key.status == KeyStatus.ACTIVE:
                                self._active_kid = kid
                    logger.info(f"Loaded {len(self._keys)} keys from Redis")
                    return
            except Exception as e:
                logger.warning(f"Failed to load keys from Redis: {e}")

        # Fallback to file storage
        keys_dir = Path(self.config.keys_directory)
        if keys_dir.exists():
            metadata_file = keys_dir / "keys.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    for kid, key_meta in metadata.items():
                        private_file = keys_dir / f"{kid}.pem"
                        if private_file.exists():
                            with open(private_file, "rb") as f:
                                private_key = f.read()
                            key = JWTKey.from_dict(key_meta, private_key)
                            self._keys[kid] = key
                            if key.status == KeyStatus.ACTIVE:
                                self._active_kid = kid

                    logger.info(f"Loaded {len(self._keys)} keys from file")
                except Exception as e:
                    logger.warning(f"Failed to load keys from file: {e}")

    async def _save_keys(self) -> None:
        """Save keys to storage."""
        metadata = {kid: key.to_dict() for kid, key in self._keys.items()}

        # Save to Redis
        if self.redis:
            try:
                await self.redis.set("jwt:keys:metadata", json.dumps(metadata))
                for kid, key in self._keys.items():
                    if key.private_key:
                        await self.redis.set(f"jwt:keys:private:{kid}", key.private_key.decode())
                logger.debug("Keys saved to Redis")
            except Exception as e:
                logger.warning(f"Failed to save keys to Redis: {e}")

        # Save to file
        keys_dir = Path(self.config.keys_directory)
        try:
            keys_dir.mkdir(parents=True, exist_ok=True)

            with open(keys_dir / "keys.json", "w") as f:
                json.dump(metadata, f, indent=2)

            for kid, key in self._keys.items():
                if key.private_key:
                    with open(keys_dir / f"{kid}.pem", "wb") as f:
                        f.write(key.private_key)
                    os.chmod(keys_dir / f"{kid}.pem", 0o600)

            logger.debug("Keys saved to file")
        except Exception as e:
            logger.warning(f"Failed to save keys to file: {e}")

    async def _publish_rotation_event(self, new_kid: str) -> None:
        """Publish key rotation event for other services."""
        if self.redis:
            try:
                await self.redis.publish("jwt:key:rotated", json.dumps({
                    "new_kid": new_kid,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }))
            except Exception as e:
                logger.warning(f"Failed to publish rotation event: {e}")


# Global instance
_jwt_manager: Optional[JWTKeyRotationManager] = None


async def get_jwt_manager(redis_client=None) -> JWTKeyRotationManager:
    """Get or create the global JWT manager."""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTKeyRotationManager(redis_client=redis_client)
        await _jwt_manager.initialize()
    return _jwt_manager
