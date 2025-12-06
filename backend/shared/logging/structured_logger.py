"""
Standardized Structured Logging (TD-004)

Provides Winston-style logging for Python with:
- Standardized log levels
- Structured JSON logging
- Request tracking ID support
- Context enrichment
- Hierarchical query support

Log Levels:
- error: Critical errors requiring immediate attention
- warning: Warning conditions that should be investigated
- info: Informational messages about normal operation
- http: HTTP request/response logging
- debug: Debug information for development

Usage:
    from backend.shared.logging import logger, get_logger
    
    # Default logger
    logger.info("User logged in", user_id="123")
    
    # Module-specific logger
    log = get_logger(__name__)
    log.info("Processing request", endpoint="/api/users")
    
    # With request context
    with request_context(request_id="req-123"):
        logger.info("Handling request")
"""
import json
import logging
import sys
import traceback
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
import uuid


# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


# Custom log level for HTTP
HTTP_LEVEL = 25
logging.addLevelName(HTTP_LEVEL, 'HTTP')


@dataclass
class LogConfig:
    """Logger configuration."""
    level: str = "INFO"
    format: str = "json"  # json or text
    output: str = "stdout"  # stdout, file, or both
    file_path: Optional[str] = None
    include_timestamp: bool = True
    include_request_id: bool = True
    include_source: bool = True
    max_message_length: int = 10000


class StructuredFormatter(logging.Formatter):
    """
    Formats log records as structured JSON.
    
    Output format:
    {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "level": "info",
        "message": "User logged in",
        "source": "auth.service",
        "request_id": "req-123",
        "user_id": "user-456",
        "extra": {"key": "value"}
    }
    """
    
    def __init__(self, include_timestamp: bool = True, include_request_id: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_request_id = include_request_id
    
    def format(self, record: logging.LogRecord) -> str:
        # Build log entry
        entry: Dict[str, Any] = {}
        
        if self.include_timestamp:
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        entry["level"] = record.levelname.lower()
        entry["message"] = record.getMessage()
        entry["source"] = record.name
        
        # Add request context
        if self.include_request_id:
            request_id = request_id_var.get()
            if request_id:
                entry["request_id"] = request_id
            
            user_id = user_id_var.get()
            if user_id:
                entry["user_id"] = user_id
            
            session_id = session_id_var.get()
            if session_id:
                entry["session_id"] = session_id
        
        # Add extra fields
        extra = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'taskName'
            }:
                extra[key] = value
        
        if extra:
            entry["extra"] = extra
        
        # Add exception info
        if record.exc_info:
            entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[2] else None,
            }
        
        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    """
    Formats log records as human-readable text.
    
    Output format:
    [2024-01-15 10:30:00] [INFO] [auth.service] [req-123] User logged in
    """
    
    def __init__(self, include_timestamp: bool = True, include_request_id: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_request_id = include_request_id
    
    def format(self, record: logging.LogRecord) -> str:
        parts = []
        
        if self.include_timestamp:
            parts.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        
        parts.append(f"[{record.levelname}]")
        parts.append(f"[{record.name}]")
        
        if self.include_request_id:
            request_id = request_id_var.get()
            if request_id:
                parts.append(f"[{request_id[:8]}]")
        
        parts.append(record.getMessage())
        
        # Add extra fields
        extras = []
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'taskName'
            }:
                extras.append(f"{key}={value}")
        
        if extras:
            parts.append(f"| {' '.join(extras)}")
        
        message = " ".join(parts)
        
        # Add exception
        if record.exc_info:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))
        
        return message


class StructuredLogger(logging.Logger):
    """
    Enhanced logger with structured logging support.
    """
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
    
    def http(self, msg: str, *args, **kwargs):
        """Log HTTP request/response."""
        if self.isEnabledFor(HTTP_LEVEL):
            self._log(HTTP_LEVEL, msg, args, **kwargs)
    
    def with_context(self, **context) -> 'ContextualLogger':
        """Create a logger with additional context."""
        return ContextualLogger(self, context)
    
    def _log_with_extras(
        self,
        level: int,
        msg: str,
        args: tuple = (),
        exc_info: Any = None,
        **kwargs
    ):
        """Log with extra fields."""
        extra = kwargs.pop('extra', {})
        extra.update(kwargs)
        super()._log(level, msg, args, exc_info=exc_info, extra=extra)
    
    # Override methods to support keyword arguments as extra fields
    def debug(self, msg: str, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG):
            self._log_with_extras(logging.DEBUG, msg, args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log_with_extras(logging.INFO, msg, args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        if self.isEnabledFor(logging.WARNING):
            self._log_with_extras(logging.WARNING, msg, args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        exc_info = kwargs.pop('exc_info', None)
        if self.isEnabledFor(logging.ERROR):
            self._log_with_extras(logging.ERROR, msg, args, exc_info=exc_info, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        exc_info = kwargs.pop('exc_info', None)
        if self.isEnabledFor(logging.CRITICAL):
            self._log_with_extras(logging.CRITICAL, msg, args, exc_info=exc_info, **kwargs)


class ContextualLogger:
    """Logger wrapper with persistent context."""
    
    def __init__(self, logger: StructuredLogger, context: Dict[str, Any]):
        self._logger = logger
        self._context = context
    
    def _merge_context(self, kwargs: Dict) -> Dict:
        merged = {**self._context, **kwargs}
        return merged
    
    def debug(self, msg: str, **kwargs):
        self._logger.debug(msg, **self._merge_context(kwargs))
    
    def info(self, msg: str, **kwargs):
        self._logger.info(msg, **self._merge_context(kwargs))
    
    def warning(self, msg: str, **kwargs):
        self._logger.warning(msg, **self._merge_context(kwargs))
    
    def error(self, msg: str, **kwargs):
        self._logger.error(msg, **self._merge_context(kwargs))
    
    def http(self, msg: str, **kwargs):
        self._logger.http(msg, **self._merge_context(kwargs))


# Context managers for request tracking

class RequestContext:
    """Context manager for request tracking."""
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.session_id = session_id
        self._tokens = []
    
    def __enter__(self):
        self._tokens.append(request_id_var.set(self.request_id))
        if self.user_id:
            self._tokens.append(user_id_var.set(self.user_id))
        if self.session_id:
            self._tokens.append(session_id_var.set(self.session_id))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in reversed(self._tokens):
            try:
                if token.var == request_id_var:
                    request_id_var.reset(token)
                elif token.var == user_id_var:
                    user_id_var.reset(token)
                elif token.var == session_id_var:
                    session_id_var.reset(token)
            except ValueError:
                pass  # Token already reset


def request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> RequestContext:
    """Create a request context for logging."""
    return RequestContext(request_id, user_id, session_id)


def set_request_id(request_id: str):
    """Set request ID for current context."""
    request_id_var.set(request_id)


def set_user_id(user_id: str):
    """Set user ID for current context."""
    user_id_var.set(user_id)


def get_request_id() -> Optional[str]:
    """Get current request ID."""
    return request_id_var.get()


# Logger setup

def setup_logging(config: Optional[LogConfig] = None) -> StructuredLogger:
    """
    Setup structured logging.
    
    Args:
        config: Logger configuration
        
    Returns:
        Configured root logger
    """
    config = config or LogConfig()
    
    # Set logger class
    logging.setLoggerClass(StructuredLogger)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    if config.format == "json":
        formatter = StructuredFormatter(
            include_timestamp=config.include_timestamp,
            include_request_id=config.include_request_id,
        )
    else:
        formatter = TextFormatter(
            include_timestamp=config.include_timestamp,
            include_request_id=config.include_request_id,
        )
    
    # Add stdout handler
    if config.output in ("stdout", "both"):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)
    
    # Add file handler
    if config.output in ("file", "both") and config.file_path:
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> StructuredLogger:
    """
    Get a logger for a module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Structured logger instance
    """
    return logging.getLogger(name)


# Create default logger
logger = get_logger("app")


# Initialize logging on import
_initialized = False

def init_logging():
    """Initialize logging (called automatically)."""
    global _initialized
    if not _initialized:
        import os
        config = LogConfig(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format=os.environ.get("LOG_FORMAT", "json"),
        )
        setup_logging(config)
        _initialized = True


# Auto-initialize
init_logging()
