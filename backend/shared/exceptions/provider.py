"""
AI Provider Exceptions

Error codes: PRV001-PRV099
"""

from .base import CodeRevException, ErrorSeverity, ErrorCategory


class ProviderError(CodeRevException):
    """Base class for AI provider errors."""
    default_code = "PRV000"
    default_category = ErrorCategory.PROVIDER
    default_http_status = 502


class ProviderUnavailableError(ProviderError):
    """
    Provider is unavailable.
    
    Raised when AI provider cannot be reached.
    """
    default_code = "PRV001"
    default_severity = ErrorSeverity.HIGH
    default_message = "AI provider is unavailable"
    default_http_status = 503
    
    @property
    def is_retryable(self) -> bool:
        """Provider availability may change."""
        return True
    
    def __init__(self, provider: str = None, **kwargs):
        message = kwargs.pop("message", self.default_message)
        if provider:
            message = f"{message}: {provider}"
        
        super().__init__(
            message,
            provider=provider,
            **kwargs,
        )


class ProviderRateLimitError(ProviderError):
    """
    Provider rate limit exceeded.
    
    Raised when API rate limit is hit.
    """
    default_code = "PRV002"
    default_severity = ErrorSeverity.MEDIUM
    default_category = ErrorCategory.RATE_LIMIT
    default_message = "Provider rate limit exceeded"
    default_http_status = 429
    
    @property
    def is_retryable(self) -> bool:
        """Rate limits reset over time."""
        return True
    
    def __init__(self, provider: str = None, retry_after: int = None, **kwargs):
        message = kwargs.pop("message", self.default_message)
        if provider:
            message = f"{message} for {provider}"
        if retry_after:
            message = f"{message}, retry after {retry_after}s"
        
        super().__init__(
            message,
            provider=provider,
            retry_after=retry_after,
            **kwargs,
        )


class ProviderAuthError(ProviderError):
    """
    Provider authentication failed.
    
    Raised when API key is invalid or expired.
    """
    default_code = "PRV003"
    default_severity = ErrorSeverity.HIGH
    default_message = "Provider authentication failed"
    default_http_status = 401
    
    def __init__(self, provider: str = None, **kwargs):
        message = kwargs.pop("message", self.default_message)
        if provider:
            message = f"{message} for {provider}"
        
        super().__init__(
            message,
            provider=provider,
            **kwargs,
        )


class ProviderTimeoutError(ProviderError):
    """
    Provider request timed out.
    
    Raised when provider doesn't respond in time.
    """
    default_code = "PRV004"
    default_severity = ErrorSeverity.MEDIUM
    default_category = ErrorCategory.TIMEOUT
    default_message = "Provider request timed out"
    default_http_status = 504
    
    @property
    def is_retryable(self) -> bool:
        """Timeouts may succeed on retry."""
        return True
    
    def __init__(self, provider: str = None, timeout_seconds: float = None, **kwargs):
        message = kwargs.pop("message", self.default_message)
        if provider:
            message = f"{message} for {provider}"
        if timeout_seconds:
            message = f"{message} after {timeout_seconds}s"
        
        super().__init__(
            message,
            provider=provider,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )


class QuotaExceededError(ProviderError):
    """
    Quota exceeded.
    
    Raised when daily/monthly usage quota is exceeded.
    """
    default_code = "PRV005"
    default_severity = ErrorSeverity.HIGH
    default_category = ErrorCategory.QUOTA
    default_message = "Usage quota exceeded"
    default_http_status = 429
    
    def __init__(
        self,
        quota_type: str = None,  # "daily", "monthly", "tokens"
        limit: int = None,
        current: int = None,
        **kwargs,
    ):
        message = kwargs.pop("message", self.default_message)
        if quota_type:
            message = f"{quota_type.capitalize()} {message.lower()}"
        if limit and current:
            message = f"{message} ({current}/{limit})"
        
        super().__init__(
            message,
            quota_type=quota_type,
            limit=limit,
            current=current,
            **kwargs,
        )
