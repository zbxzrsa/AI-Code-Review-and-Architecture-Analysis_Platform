"""
System and Infrastructure Exceptions

Error codes: SYS001-SYS099
"""

from typing import List, Optional

from .base import CodeRevException, ErrorSeverity, ErrorCategory


class SystemError(CodeRevException):
    """Base class for system-level errors."""
    default_code = "SYS000"
    default_severity = ErrorSeverity.CRITICAL
    default_category = ErrorCategory.INTERNAL
    default_http_status = 500


class ConfigurationError(SystemError):
    """
    Configuration error.
    
    Raised when system configuration is invalid.
    """
    default_code = "SYS001"
    default_severity = ErrorSeverity.CRITICAL
    default_category = ErrorCategory.CONFIGURATION
    default_message = "Configuration error"
    default_http_status = 500
    
    def __init__(
        self,
        config_key: str = None,
        expected: str = None,
        actual: str = None,
        **kwargs,
    ):
        message = kwargs.pop("message", self.default_message)
        if config_key:
            message = f"{message}: {config_key}"
        if expected and actual:
            message = f"{message} (expected: {expected}, got: {actual})"
        
        super().__init__(
            message,
            config_key=config_key,
            expected=expected,
            actual=actual,
            **kwargs,
        )


class DependencyError(SystemError):
    """
    Dependency error.
    
    Raised when a required dependency is unavailable.
    """
    default_code = "SYS002"
    default_severity = ErrorSeverity.CRITICAL
    default_category = ErrorCategory.DEPENDENCY
    default_message = "Dependency unavailable"
    default_http_status = 503
    
    @property
    def is_retryable(self) -> bool:
        """Dependencies may become available."""
        return True
    
    def __init__(
        self,
        dependency: str = None,
        reason: str = None,
        **kwargs,
    ):
        message = kwargs.pop("message", self.default_message)
        if dependency:
            message = f"{message}: {dependency}"
        if reason:
            message = f"{message} ({reason})"
        
        super().__init__(
            message,
            dependency=dependency,
            reason=reason,
            **kwargs,
        )


class InitializationError(SystemError):
    """
    Initialization error.
    
    Raised when system component fails to initialize.
    """
    default_code = "SYS003"
    default_severity = ErrorSeverity.CRITICAL
    default_message = "Initialization failed"
    default_http_status = 500
    
    def __init__(
        self,
        component: str = None,
        phase: str = None,
        **kwargs,
    ):
        message = kwargs.pop("message", self.default_message)
        if component:
            message = f"{message}: {component}"
        if phase:
            message = f"{message} during {phase}"
        
        super().__init__(
            message,
            component=component,
            phase=phase,
            **kwargs,
        )
