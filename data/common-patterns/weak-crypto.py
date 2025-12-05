# Common Weak Cryptography Pattern - For Cache Warming
# This pattern should be detected as a security vulnerability
#
# NOTE: This file intentionally contains insecure code patterns for training
# the code review AI to detect vulnerabilities. DO NOT use these patterns
# in production code. See the "SAFE ALTERNATIVES" section for secure examples.
# noqa: S324,S303 - Intentionally weak crypto for pattern detection

import hashlib
import random

# VULNERABLE: Using MD5 for password hashing
def hash_password_md5(password):
    """DANGEROUS: MD5 is cryptographically broken"""
    return hashlib.md5(password.encode()).hexdigest()


# VULNERABLE: Using SHA1 for security purposes
def hash_password_sha1(password):
    """DANGEROUS: SHA1 has known collisions"""
    return hashlib.sha1(password.encode()).hexdigest()


# VULNERABLE: Using random instead of secrets
def generate_token_insecure():
    """DANGEROUS: random module is not cryptographically secure"""
    return ''.join(random.choice('0123456789abcdef') for _ in range(32))


# VULNERABLE: Hardcoded salt
def hash_with_static_salt(password):
    """DANGEROUS: Same salt for all passwords"""
    salt = "static_salt_123"
    return hashlib.sha256((salt + password).encode()).hexdigest()


# VULNERABLE: No salt at all
def hash_without_salt(password):
    """DANGEROUS: Vulnerable to rainbow table attacks"""
    return hashlib.sha256(password.encode()).hexdigest()


# VULNERABLE: Using ECB mode for encryption
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

def encrypt_ecb_insecure(key, plaintext):
    """DANGEROUS: ECB mode reveals patterns in ciphertext"""
    cipher = Cipher(
        algorithms.AES(key),
        modes.ECB(),  # Insecure mode
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    return encryptor.update(plaintext) + encryptor.finalize()


# ============================================================
# SAFE ALTERNATIVES
# ============================================================

import secrets
import bcrypt
from argon2 import PasswordHasher


# SAFE: Using bcrypt for password hashing
def hash_password_bcrypt(password):
    """SAFE: bcrypt with automatic salt generation"""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode(), salt)


def verify_password_bcrypt(password, hashed):
    """SAFE: Constant-time comparison"""
    return bcrypt.checkpw(password.encode(), hashed)


# SAFE: Using Argon2 (recommended)
def hash_password_argon2(password):
    """SAFE: Argon2id is the current best practice"""
    ph = PasswordHasher(
        time_cost=3,
        memory_cost=65536,
        parallelism=4
    )
    return ph.hash(password)


def verify_password_argon2(password, hashed):
    """SAFE: Argon2 verification"""
    ph = PasswordHasher()
    try:
        return ph.verify(hashed, password)
    except Exception:  # Catch verification failures
        return False


# SAFE: Using secrets module for tokens
def generate_token_secure():
    """SAFE: Cryptographically secure random token"""
    return secrets.token_hex(32)


def generate_url_safe_token():
    """SAFE: URL-safe token"""
    return secrets.token_urlsafe(32)


# SAFE: Using AES-GCM mode with random IV
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def encrypt_aes_gcm(key, plaintext):
    """SAFE: AES-GCM provides authenticated encryption"""
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # Random 96-bit nonce
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext  # Prepend nonce for decryption


def decrypt_aes_gcm(key, ciphertext):
    """SAFE: AES-GCM decryption with authentication"""
    aesgcm = AESGCM(key)
    nonce = ciphertext[:12]
    encrypted = ciphertext[12:]
    return aesgcm.decrypt(nonce, encrypted, None)


# SAFE: Using PBKDF2 for key derivation
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

def derive_key_from_password(password, salt):
    """SAFE: PBKDF2 with sufficient iterations"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600000,  # OWASP recommended minimum
    )
    return kdf.derive(password.encode())
