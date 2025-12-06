"""
Security modules including authentication, authorization, rate limiting,
key management, and vulnerability management.
"""
from .secure_auth import (
    SecureAuthManager,
    AuthConfig,
    TokenPair,
    get_current_user,
    verify_csrf,
    CSRFProtectedRoute,
    RateLimiter,
)

# KMS integration (R-005 mitigation)
from .kms_manager import (
    KMSManager,
    KMSProvider,
    SecretType,
    RotationPolicy,
    get_kms_manager,
)

# Vulnerability management (R-006 mitigation)
from .vulnerability_manager import (
    VulnerabilityManager,
    VulnerabilitySeverity,
    VulnerabilityStatus,
    DependencyScanner,
    RemediationSLA,
)

__all__ = [
    # Auth
    "SecureAuthManager",
    "AuthConfig", 
    "TokenPair",
    "get_current_user",
    "verify_csrf",
    "CSRFProtectedRoute",
    "RateLimiter",
    # KMS
    "KMSManager",
    "KMSProvider",
    "SecretType",
    "RotationPolicy",
    "get_kms_manager",
    # Vulnerability
    "VulnerabilityManager",
    "VulnerabilitySeverity",
    "VulnerabilityStatus",
    "DependencyScanner",
    "RemediationSLA",
]
