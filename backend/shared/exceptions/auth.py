"""
Authentication and Authorization Exceptions

Error codes: AUTH001-AUTH099
"""

from .base import CodeRevException, ErrorSeverity, ErrorCategory


class AuthError(CodeRevException):
    """Base class for authentication/authorization errors."""
    default_code = "AUTH000"
    default_category = ErrorCategory.AUTHENTICATION
    default_http_status = 401


class AuthenticationError(AuthError):
    """
    Authentication failed.
    
    Raised when credentials are invalid or missing.
    """
    default_code = "AUTH001"
    default_severity = ErrorSeverity.MEDIUM
    default_message = "Authentication failed"
    default_http_status = 401


class AuthorizationError(AuthError):
    """
    Authorization failed.
    
    Raised when user lacks required permissions.
    """
    default_code = "AUTH002"
    default_severity = ErrorSeverity.MEDIUM
    default_category = ErrorCategory.AUTHORIZATION
    default_message = "Access denied"
    default_http_status = 403


class TokenExpiredError(AuthError):
    """
    Token has expired.
    
    Raised when JWT or session token is expired.
    """
    default_code = "AUTH003"
    default_severity = ErrorSeverity.LOW
    default_message = "Token has expired"
    default_http_status = 401
    
    @property
    def is_retryable(self) -> bool:
        """Token expiry is retryable with refresh."""
        return True


class InvalidTokenError(AuthError):
    """
    Token is invalid.
    
    Raised when token signature is invalid or malformed.
    """
    default_code = "AUTH004"
    default_severity = ErrorSeverity.MEDIUM
    default_message = "Invalid token"
    default_http_status = 401


class SessionError(AuthError):
    """
    Session error.
    
    Raised for session-related issues.
    """
    default_code = "AUTH005"
    default_severity = ErrorSeverity.MEDIUM
    default_message = "Session error"
    default_http_status = 401


class PermissionDeniedError(AuthError):
    """
    Specific permission denied.
    
    More specific than AuthorizationError, includes permission name.
    """
    default_code = "AUTH006"
    default_severity = ErrorSeverity.MEDIUM
    default_category = ErrorCategory.AUTHORIZATION
    default_message = "Permission denied"
    default_http_status = 403
    
    def __init__(self, permission: str, **kwargs):
        super().__init__(
            f"Permission denied: {permission}",
            permission=permission,
            **kwargs,
        )
