"""
安全模块 (Security Module)

模块功能描述:
    提供认证、授权、速率限制、密钥管理和漏洞管理功能。

主要功能:
    - 安全认证管理
    - CSRF 保护
    - 速率限制
    - KMS 密钥管理
    - 漏洞管理

主要组件:
    - SecureAuthManager: 安全认证管理器
    - KMSManager: 密钥管理服务
    - VulnerabilityManager: 漏洞管理器
    - DependencyScanner: 依赖扫描器

最后修改日期: 2024-12-07
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

# RBAC - Role-Based Access Control
from .rbac import (
    Role,
    Permission,
    ROLE_HIERARCHY,
    ROLE_PERMISSIONS,
    ADMIN_ONLY_ROUTES,
    ADMIN_CONTACT_INFO,
    get_role_level,
    get_role_permissions,
    has_role,
    has_permission,
    has_any_permission,
    has_all_permissions,
    is_admin_route,
    is_admin,
    get_user_permissions_response,
    get_access_denied_response,
    AccessAttempt,
    AccessAuditLogger,
    RBACDependency,
    require_admin,
    require_user,
    require_viewer,
    require_permission,
    require_any_permission,
    require_all_permissions,
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
    # RBAC
    "Role",
    "Permission",
    "ROLE_HIERARCHY",
    "ROLE_PERMISSIONS",
    "ADMIN_ONLY_ROUTES",
    "ADMIN_CONTACT_INFO",
    "get_role_level",
    "get_role_permissions",
    "has_role",
    "has_permission",
    "has_any_permission",
    "has_all_permissions",
    "is_admin_route",
    "is_admin",
    "get_user_permissions_response",
    "get_access_denied_response",
    "AccessAttempt",
    "AccessAuditLogger",
    "RBACDependency",
    "require_admin",
    "require_user",
    "require_viewer",
    "require_permission",
    "require_any_permission",
    "require_all_permissions",
]
