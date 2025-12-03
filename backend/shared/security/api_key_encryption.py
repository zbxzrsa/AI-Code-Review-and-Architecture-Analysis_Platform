"""
API Key Encryption Module

Implements:
- AES-256-GCM encryption for API keys
- Key derivation using PBKDF2
- Secure key rotation
- Hardware Security Module (HSM) integration ready
"""

import os
import base64
import secrets
import hashlib
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


# Configuration
MASTER_KEY_ENV = "API_KEY_MASTER_SECRET"
DEFAULT_SALT_LENGTH = 16
DEFAULT_NONCE_LENGTH = 12
DEFAULT_KEY_LENGTH = 32  # 256 bits
PBKDF2_ITERATIONS = 100000


@dataclass
class EncryptedKey:
    """Encrypted API key structure."""
    ciphertext: bytes
    nonce: bytes
    salt: bytes
    version: int
    created_at: datetime
    
    def to_string(self) -> str:
        """Serialize to string for storage."""
        data = (
            f"{self.version}:"
            f"{base64.b64encode(self.salt).decode()}:"
            f"{base64.b64encode(self.nonce).decode()}:"
            f"{base64.b64encode(self.ciphertext).decode()}"
        )
        return data
    
    @classmethod
    def from_string(cls, data: str) -> "EncryptedKey":
        """Deserialize from string."""
        parts = data.split(":")
        if len(parts) != 4:
            raise ValueError("Invalid encrypted key format")
        
        version = int(parts[0])
        salt = base64.b64decode(parts[1])
        nonce = base64.b64decode(parts[2])
        ciphertext = base64.b64decode(parts[3])
        
        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
            version=version,
            created_at=datetime.utcnow(),
        )


class APIKeyEncryption:
    """
    Secure API key encryption using AES-256-GCM.
    
    Features:
    - Per-key salt for additional security
    - Key derivation using PBKDF2
    - Authenticated encryption (GCM mode)
    - Version support for key rotation
    """
    
    def __init__(
        self,
        master_key: Optional[str] = None,
        version: int = 1,
    ):
        """
        Initialize encryption.
        
        Args:
            master_key: Master encryption key (from env or HSM)
            version: Encryption version for rotation support
        """
        self.master_key = master_key or os.getenv(MASTER_KEY_ENV)
        if not self.master_key:
            # Generate and log warning for development
            self.master_key = secrets.token_urlsafe(32)
            logger.warning(
                "No master key provided, using generated key. "
                "Set API_KEY_MASTER_SECRET in production!"
            )
        
        self.version = version
        self._key_cache: dict = {}
    
    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key from master key using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=DEFAULT_KEY_LENGTH,
            salt=salt,
            iterations=PBKDF2_ITERATIONS,
            backend=default_backend(),
        )
        return kdf.derive(self.master_key.encode())
    
    def encrypt(self, plaintext: str) -> EncryptedKey:
        """
        Encrypt API key.
        
        Args:
            plaintext: API key to encrypt
            
        Returns:
            EncryptedKey object
        """
        try:
            # Generate random salt and nonce
            salt = secrets.token_bytes(DEFAULT_SALT_LENGTH)
            nonce = secrets.token_bytes(DEFAULT_NONCE_LENGTH)
            
            # Derive key
            key = self._derive_key(salt)
            
            # Encrypt using AES-GCM
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(
                nonce,
                plaintext.encode(),
                None  # Additional authenticated data (optional)
            )
            
            return EncryptedKey(
                ciphertext=ciphertext,
                nonce=nonce,
                salt=salt,
                version=self.version,
                created_at=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.error(f"API key encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted: EncryptedKey) -> str:
        """
        Decrypt API key.
        
        Args:
            encrypted: EncryptedKey object
            
        Returns:
            Decrypted API key
        """
        try:
            # Derive key using stored salt
            key = self._derive_key(encrypted.salt)
            
            # Decrypt using AES-GCM
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(
                encrypted.nonce,
                encrypted.ciphertext,
                None,
            )
            
            return plaintext.decode()
            
        except Exception as e:
            logger.error(f"API key decryption failed: {e}")
            raise
    
    def encrypt_to_string(self, plaintext: str) -> str:
        """Encrypt and return as string for database storage."""
        encrypted = self.encrypt(plaintext)
        return encrypted.to_string()
    
    def decrypt_from_string(self, data: str) -> str:
        """Decrypt from stored string."""
        encrypted = EncryptedKey.from_string(data)
        return self.decrypt(encrypted)
    
    def rotate_key(
        self,
        encrypted_data: str,
        new_master_key: str,
    ) -> str:
        """
        Rotate encryption key.
        
        Decrypts with current key and re-encrypts with new key.
        """
        # Decrypt with current key
        plaintext = self.decrypt_from_string(encrypted_data)
        
        # Create new encryptor with new key
        new_encryptor = APIKeyEncryption(
            master_key=new_master_key,
            version=self.version + 1,
        )
        
        # Re-encrypt
        return new_encryptor.encrypt_to_string(plaintext)
    
    def hash_for_lookup(self, api_key: str) -> str:
        """
        Create secure hash for API key lookup.
        
        Allows finding encrypted keys without decrypting all.
        """
        return hashlib.sha256(
            f"{self.master_key}:{api_key}".encode()
        ).hexdigest()


class SecureAPIKeyStore:
    """
    Secure storage for API keys.
    
    Integrates encryption with database storage.
    """
    
    def __init__(
        self,
        encryption: APIKeyEncryption,
        db_connection = None,
    ):
        self.encryption = encryption
        self.db = db_connection
        self._cache: dict = {}
    
    async def store_key(
        self,
        user_id: str,
        provider: str,
        api_key: str,
    ) -> str:
        """Store encrypted API key."""
        encrypted = self.encryption.encrypt_to_string(api_key)
        lookup_hash = self.encryption.hash_for_lookup(api_key)
        
        # Store in database (example)
        key_id = secrets.token_urlsafe(16)
        
        # In production, store to database:
        # INSERT INTO api_keys (id, user_id, provider, encrypted_key, lookup_hash)
        # VALUES (key_id, user_id, provider, encrypted, lookup_hash)
        
        logger.info(f"Stored encrypted API key for user {user_id}, provider {provider}")
        
        return key_id
    
    async def get_key(
        self,
        user_id: str,
        provider: str,
    ) -> Optional[str]:
        """Retrieve and decrypt API key."""
        # In production, fetch from database:
        # SELECT encrypted_key FROM api_keys
        # WHERE user_id = user_id AND provider = provider
        
        # encrypted = result.encrypted_key
        # return self.encryption.decrypt_from_string(encrypted)
        
        return None
    
    async def delete_key(
        self,
        user_id: str,
        provider: str,
    ) -> bool:
        """Delete API key."""
        # DELETE FROM api_keys WHERE user_id = user_id AND provider = provider
        logger.info(f"Deleted API key for user {user_id}, provider {provider}")
        return True
    
    async def rotate_all_keys(
        self,
        new_master_key: str,
    ) -> int:
        """Rotate all stored API keys to new master key."""
        rotated_count = 0
        
        # In production:
        # SELECT id, encrypted_key FROM api_keys
        # For each key:
        #   new_encrypted = self.encryption.rotate_key(encrypted_key, new_master_key)
        #   UPDATE api_keys SET encrypted_key = new_encrypted, version = version + 1
        #   rotated_count += 1
        
        logger.info(f"Rotated {rotated_count} API keys")
        return rotated_count


# Singleton instance
_encryption_instance: Optional[APIKeyEncryption] = None


def get_api_key_encryption() -> APIKeyEncryption:
    """Get singleton encryption instance."""
    global _encryption_instance
    if _encryption_instance is None:
        _encryption_instance = APIKeyEncryption()
    return _encryption_instance
