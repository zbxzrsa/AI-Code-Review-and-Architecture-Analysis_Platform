"""
Code Analysis Exceptions

Error codes: ANA001-ANA099
"""

from .base import CodeRevException, ErrorSeverity, ErrorCategory


class AnalysisError(CodeRevException):
    """Base class for code analysis errors."""
    default_code = "ANA000"
    default_category = ErrorCategory.INTERNAL
    default_http_status = 500


class CodeParsingError(AnalysisError):
    """
    Failed to parse code.
    
    Raised when source code cannot be parsed.
    """
    default_code = "ANA001"
    default_severity = ErrorSeverity.MEDIUM
    default_message = "Failed to parse code"
    default_http_status = 400
    
    def __init__(self, language: str = None, line: int = None, **kwargs):
        message = kwargs.pop("message", self.default_message)
        if language:
            message = f"{message} ({language})"
        if line:
            message = f"{message} at line {line}"
        
        super().__init__(
            message,
            language=language,
            line=line,
            **kwargs,
        )


class PatternMatchError(AnalysisError):
    """
    Pattern matching failed.
    
    Raised when pattern detection encounters an error.
    """
    default_code = "ANA002"
    default_severity = ErrorSeverity.MEDIUM
    default_message = "Pattern matching failed"
    default_http_status = 500


class TimeoutError(AnalysisError):
    """
    Analysis timed out.
    
    Raised when analysis exceeds time limit.
    """
    default_code = "ANA003"
    default_severity = ErrorSeverity.HIGH
    default_category = ErrorCategory.TIMEOUT
    default_message = "Analysis timed out"
    default_http_status = 504
    
    @property
    def is_retryable(self) -> bool:
        """Timeouts may succeed on retry."""
        return True
    
    def __init__(self, timeout_seconds: float = None, **kwargs):
        message = kwargs.pop("message", self.default_message)
        if timeout_seconds:
            message = f"{message} after {timeout_seconds}s"
        
        super().__init__(
            message,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )


class ResourceExhaustedError(AnalysisError):
    """
    Resource exhausted during analysis.
    
    Raised when memory, CPU, or other resources are exhausted.
    """
    default_code = "ANA004"
    default_severity = ErrorSeverity.HIGH
    default_message = "Resource exhausted during analysis"
    default_http_status = 503
    
    def __init__(self, resource: str = None, **kwargs):
        message = kwargs.pop("message", self.default_message)
        if resource:
            message = f"{message}: {resource}"
        
        super().__init__(
            message,
            resource=resource,
            **kwargs,
        )
