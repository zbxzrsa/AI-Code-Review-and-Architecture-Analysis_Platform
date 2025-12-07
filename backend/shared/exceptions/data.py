"""
Data and Validation Exceptions

Error codes: DAT001-DAT099
"""

from typing import Any, Dict, List, Optional

from .base import CodeRevException, ErrorSeverity, ErrorCategory


class DataError(CodeRevException):
    """Base class for data-related errors."""
    default_code = "DAT000"
    default_category = ErrorCategory.VALIDATION
    default_http_status = 400


class ValidationError(DataError):
    """
    Data validation failed.
    
    Raised when input data fails validation.
    """
    default_code = "DAT001"
    default_severity = ErrorSeverity.LOW
    default_message = "Validation failed"
    default_http_status = 400
    
    def __init__(
        self,
        message: str = None,
        field: str = None,
        errors: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        msg = message or self.default_message
        if field:
            msg = f"{msg}: {field}"
        
        super().__init__(
            msg,
            field=field,
            errors=errors,
            **kwargs,
        )
    
    @classmethod
    def from_pydantic(cls, exc) -> "ValidationError":
        """Create from Pydantic ValidationError."""
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            })
        
        return cls(
            message=f"Validation failed: {len(errors)} error(s)",
            errors=errors,
        )


class NotFoundError(DataError):
    """
    Resource not found.
    
    Raised when requested resource doesn't exist.
    """
    default_code = "DAT002"
    default_severity = ErrorSeverity.LOW
    default_category = ErrorCategory.NOT_FOUND
    default_message = "Resource not found"
    default_http_status = 404
    
    def __init__(
        self,
        resource_type: str = None,
        resource_id: str = None,
        **kwargs,
    ):
        message = kwargs.pop("message", None)
        if not message:
            if resource_type and resource_id:
                message = f"{resource_type} not found: {resource_id}"
            elif resource_type:
                message = f"{resource_type} not found"
            else:
                message = self.default_message
        
        super().__init__(
            message,
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs,
        )


class ConflictError(DataError):
    """
    Resource conflict.
    
    Raised when operation conflicts with existing data.
    """
    default_code = "DAT003"
    default_severity = ErrorSeverity.MEDIUM
    default_category = ErrorCategory.CONFLICT
    default_message = "Resource conflict"
    default_http_status = 409
    
    def __init__(
        self,
        resource_type: str = None,
        conflict_field: str = None,
        **kwargs,
    ):
        message = kwargs.pop("message", self.default_message)
        if resource_type:
            message = f"{resource_type} conflict"
        if conflict_field:
            message = f"{message}: {conflict_field} already exists"
        
        super().__init__(
            message,
            resource_type=resource_type,
            conflict_field=conflict_field,
            **kwargs,
        )


class IntegrityError(DataError):
    """
    Data integrity error.
    
    Raised when data integrity constraints are violated.
    """
    default_code = "DAT004"
    default_severity = ErrorSeverity.HIGH
    default_category = ErrorCategory.INTERNAL
    default_message = "Data integrity error"
    default_http_status = 500
    
    def __init__(
        self,
        constraint: str = None,
        **kwargs,
    ):
        message = kwargs.pop("message", self.default_message)
        if constraint:
            message = f"{message}: {constraint}"
        
        super().__init__(
            message,
            constraint=constraint,
            **kwargs,
        )
