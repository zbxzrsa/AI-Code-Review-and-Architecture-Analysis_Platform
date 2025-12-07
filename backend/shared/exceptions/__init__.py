"""
分层异常分类系统 (Hierarchical Exception Classification System)

模块功能描述:
    提供按领域组织的分层异常类系统。

主要功能:
    - 按领域组织的分层异常类
    - 带模块前缀的错误代码
    - 用于告警的严重级别
    - 自动上下文丰富
    - 与结构化日志集成

异常分类:
    - AuthError: 认证/授权异常
    - AnalysisError: 代码分析异常
    - ProviderError: AI 提供者异常
    - DataError: 数据验证异常
    - SystemError: 系统配置异常

使用示例:
    from backend.shared.exceptions import (
        AuthenticationError,
        AuthorizationError,
        AnalysisError,
        ProviderError,
    )
    
    # 带上下文抛出异常
    raise AuthenticationError(
        "无效凭据",
        user_id="user-123",
        method="password",
    )
    
    # 检查错误属性
    try:
        ...
    except CodeRevException as e:
        logger.error(e.message, code=e.code, severity=e.severity)

最后修改日期: 2024-12-07
"""

from .base import (
    CodeRevException,
    ErrorSeverity,
    ErrorCategory,
)
from .auth import (
    AuthError,
    AuthenticationError,
    AuthorizationError,
    TokenExpiredError,
    InvalidTokenError,
    SessionError,
    PermissionDeniedError,
)
from .analysis import (
    AnalysisError,
    CodeParsingError,
    PatternMatchError,
    TimeoutError as AnalysisTimeoutError,
    ResourceExhaustedError,
)
from .provider import (
    ProviderError,
    ProviderUnavailableError,
    ProviderRateLimitError,
    ProviderAuthError,
    ProviderTimeoutError,
    QuotaExceededError,
)
from .data import (
    DataError,
    ValidationError,
    NotFoundError,
    ConflictError,
    IntegrityError,
)
from .system import (
    SystemError as CodeRevSystemError,
    ConfigurationError,
    DependencyError,
    InitializationError,
)

__all__ = [
    # Base
    "CodeRevException",
    "ErrorSeverity",
    "ErrorCategory",
    # Auth
    "AuthError",
    "AuthenticationError",
    "AuthorizationError",
    "TokenExpiredError",
    "InvalidTokenError",
    "SessionError",
    "PermissionDeniedError",
    # Analysis
    "AnalysisError",
    "CodeParsingError",
    "PatternMatchError",
    "AnalysisTimeoutError",
    "ResourceExhaustedError",
    # Provider
    "ProviderError",
    "ProviderUnavailableError",
    "ProviderRateLimitError",
    "ProviderAuthError",
    "ProviderTimeoutError",
    "QuotaExceededError",
    # Data
    "DataError",
    "ValidationError",
    "NotFoundError",
    "ConflictError",
    "IntegrityError",
    # System
    "CodeRevSystemError",
    "ConfigurationError",
    "DependencyError",
    "InitializationError",
]
