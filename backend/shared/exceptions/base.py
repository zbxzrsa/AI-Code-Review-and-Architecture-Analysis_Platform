"""
Base Exception Classes

Provides the foundation for all CodeRev exceptions with:
- Error codes
- Severity levels
- Context enrichment
- Serialization support
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import traceback


class ErrorSeverity(str, Enum):
    """
    Severity levels for exceptions.
    
    Used for alerting and prioritization:
    - CRITICAL: Requires immediate attention, service impacted
    - HIGH: Significant issue, should be addressed soon
    - MEDIUM: Notable issue, can be scheduled
    - LOW: Minor issue, informational
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ErrorCategory(str, Enum):
    """
    Categories for exception classification.
    
    Used for routing and aggregation.
    """
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    PROVIDER = "provider"
    TIMEOUT = "timeout"
    INTERNAL = "internal"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"


class CodeRevException(Exception):
    """
    Base exception class for all CodeRev exceptions.
    
    Features:
    - Unique error code with module prefix
    - Severity level for alerting
    - Category for classification
    - Context dictionary for additional info
    - HTTP status code mapping
    - Serialization to dict/JSON
    
    Attributes:
        message: Human-readable error message
        code: Unique error code (e.g., "AUTH001")
        severity: Error severity level
        category: Error category
        http_status: HTTP status code
        context: Additional context information
        timestamp: When the error occurred
        
    Example:
        raise CodeRevException(
            "Something went wrong",
            code="GEN001",
            severity=ErrorSeverity.MEDIUM,
            user_id="123",
        )
    """
    
    # Default values (override in subclasses)
    default_code: str = "GEN000"
    default_severity: ErrorSeverity = ErrorSeverity.MEDIUM
    default_category: ErrorCategory = ErrorCategory.INTERNAL
    default_http_status: int = 500
    default_message: str = "An unexpected error occurred"
    
    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[str] = None,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None,
        http_status: Optional[int] = None,
        cause: Optional[Exception] = None,
        **context: Any,
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message (uses default if not provided)
            code: Error code (uses class default if not provided)
            severity: Severity level (uses class default if not provided)
            category: Error category (uses class default if not provided)
            http_status: HTTP status code (uses class default if not provided)
            cause: Original exception that caused this error
            **context: Additional context to include
        """
        self.message = message or self.default_message
        self.code = code or self.default_code
        self.severity = severity or self.default_severity
        self.category = category or self.default_category
        self.http_status = http_status or self.default_http_status
        self.cause = cause
        self.context = context
        self.timestamp = datetime.now(timezone.utc)
        
        # Capture traceback
        self._traceback = traceback.format_exc() if cause else None
        
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return formatted error message."""
        return f"[{self.code}] {self.message}"
    
    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"code={self.code!r}, "
            f"message={self.message!r}, "
            f"severity={self.severity.value!r})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize exception to dictionary.
        
        Returns:
            Dictionary representation of the error
        """
        result = {
            "error": {
                "code": self.code,
                "message": self.message,
                "severity": self.severity.value,
                "category": self.category.value,
                "timestamp": self.timestamp.isoformat(),
            }
        }
        
        if self.context:
            result["error"]["context"] = self.context
        
        if self.cause:
            result["error"]["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }
        
        return result
    
    def to_http_response(self) -> Dict[str, Any]:
        """
        Create HTTP error response.
        
        Returns:
            Dictionary suitable for HTTP response body
        """
        return {
            "error": self.code,
            "message": self.message,
            "status": self.http_status,
        }
    
    @property
    def is_critical(self) -> bool:
        """Check if error is critical severity."""
        return self.severity == ErrorSeverity.CRITICAL
    
    @property
    def is_retryable(self) -> bool:
        """
        Check if operation can be retried.
        
        Override in subclasses for specific retry logic.
        """
        return False
    
    def with_context(self, **additional_context: Any) -> "CodeRevException":
        """
        Add additional context to the exception.
        
        Returns:
            Self for chaining
        """
        self.context.update(additional_context)
        return self
    
    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        message: Optional[str] = None,
        **context: Any,
    ) -> "CodeRevException":
        """
        Create CodeRevException from another exception.
        
        Args:
            exc: Original exception
            message: Override message (uses exc message if not provided)
            **context: Additional context
            
        Returns:
            New CodeRevException instance
        """
        return cls(
            message=message or str(exc),
            cause=exc,
            **context,
        )
