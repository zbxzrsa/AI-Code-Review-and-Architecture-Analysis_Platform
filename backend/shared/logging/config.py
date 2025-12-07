"""
Structured Logging Configuration

Provides standardized JSON logging across the entire platform.

Features:
- JSON-formatted logs for machine parsing
- Consistent timestamp format (ISO 8601)
- Request ID correlation
- Log level filtering
- Exception formatting
- Performance context (duration, memory)

Usage:
    from backend.shared.logging import get_logger, configure_logging

    # Configure at application startup
    configure_logging(level="INFO", json_format=True)

    # Get a logger
    logger = get_logger(__name__)

    # Log with context
    logger.info("user_login", user_id="123", ip="192.168.1.1")
    logger.error("api_error", error_code=500, endpoint="/api/review")
"""

import logging
import sys
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from contextvars import ContextVar
from functools import wraps
import traceback

# Try to import structlog, provide fallback if not available
try:
    import structlog
    from structlog.processors import JSONRenderer, TimeStamper, StackInfoRenderer
    from structlog.stdlib import (
        add_log_level,
        add_logger_name,
        filter_by_level,
        PositionalArgumentsFormatter,
        ProcessorFormatter,
    )
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None


# =============================================================================
# Context Variables for Request Correlation
# =============================================================================

# Request ID for log correlation
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

# User ID for user context
user_id_var: ContextVar[str] = ContextVar('user_id', default='')

# Session ID for session tracking
session_id_var: ContextVar[str] = ContextVar('session_id', default='')


# =============================================================================
# Custom Processors
# =============================================================================

def add_request_context(logger, method_name, event_dict):
    """Add request context from context variables."""
    request_id = request_id_var.get()
    if request_id:
        event_dict['request_id'] = request_id

    user_id = user_id_var.get()
    if user_id:
        event_dict['user_id'] = user_id

    session_id = session_id_var.get()
    if session_id:
        event_dict['session_id'] = session_id

    return event_dict


def add_service_context(service_name: str, version: str, environment: str):
    """Create processor that adds service context."""
    def processor(logger, method_name, event_dict):
        event_dict['service'] = service_name
        event_dict['version'] = version
        event_dict['environment'] = environment
        return event_dict
    return processor


def add_caller_info(logger, method_name, event_dict):
    """Add caller information (file, line, function)."""
    # Get the caller's frame (skip structlog internals)
    frame = sys._getframe()
    while frame:
        module = frame.f_globals.get('__name__', '')
        if not module.startswith('structlog') and not module.startswith('logging'):
            event_dict['caller'] = {
                'file': frame.f_code.co_filename.split(os.sep)[-1],
                'line': frame.f_lineno,
                'function': frame.f_code.co_name,
            }
            break
        frame = frame.f_back
    return event_dict


def format_exception_info(logger, method_name, event_dict):
    """Format exception information consistently."""
    exc_info = event_dict.pop('exc_info', None)
    if exc_info:
        if isinstance(exc_info, tuple):
            event_dict['exception'] = {
                'type': exc_info[0].__name__ if exc_info[0] else None,
                'message': str(exc_info[1]) if exc_info[1] else None,
                'traceback': ''.join(traceback.format_exception(*exc_info)),
            }
        elif exc_info is True:
            exc_info = sys.exc_info()
            if exc_info[0]:
                event_dict['exception'] = {
                    'type': exc_info[0].__name__,
                    'message': str(exc_info[1]),
                    'traceback': ''.join(traceback.format_exception(*exc_info)),
                }
    return event_dict


def censor_sensitive_data(logger, method_name, event_dict):
    """Censor sensitive data from logs."""
    sensitive_keys = {
        'password', 'token', 'api_key', 'apikey', 'secret',
        'authorization', 'auth', 'credential', 'private_key',
        'access_token', 'refresh_token', 'bearer',
    }

    def censor_dict(d: dict) -> dict:
        result = {}
        for key, value in d.items():
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                result[key] = '[REDACTED]'
            elif isinstance(value, dict):
                result[key] = censor_dict(value)
            elif isinstance(value, str) and len(value) > 100:
                # Truncate very long strings
                result[key] = value[:100] + '...[truncated]'
            else:
                result[key] = value
        return result

    return censor_dict(event_dict)


# =============================================================================
# Log Level Mapping
# =============================================================================

LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'WARN': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


# =============================================================================
# Configuration Functions
# =============================================================================

def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    service_name: str = "coderev-platform",
    version: str = "2.1.0",
    environment: str = "development",
    log_file: Optional[str] = None,
    censor_sensitive: bool = True,
    include_caller_info: bool = False,
):
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Output logs in JSON format
        service_name: Name of the service
        version: Service version
        environment: Deployment environment
        log_file: Optional file path for log output
        censor_sensitive: Censor sensitive data in logs
        include_caller_info: Include caller file/line info
    """
    log_level = LOG_LEVELS.get(level.upper(), logging.INFO)

    if STRUCTLOG_AVAILABLE:
        _configure_structlog(
            log_level=log_level,
            json_format=json_format,
            service_name=service_name,
            version=version,
            environment=environment,
            log_file=log_file,
            censor_sensitive=censor_sensitive,
            include_caller_info=include_caller_info,
        )
    else:
        _configure_stdlib_logging(
            log_level=log_level,
            json_format=json_format,
            service_name=service_name,
            log_file=log_file,
        )


def _configure_structlog(
    log_level: int,
    json_format: bool,
    service_name: str,
    version: str,
    environment: str,
    log_file: Optional[str],
    censor_sensitive: bool,
    include_caller_info: bool,
):
    """Configure structlog for structured logging."""

    # Build processor chain
    shared_processors: List = [
        filter_by_level,
        add_log_level,
        add_logger_name,
        PositionalArgumentsFormatter(),
        TimeStamper(fmt="iso"),
        add_request_context,
        add_service_context(service_name, version, environment),
    ]

    if include_caller_info:
        shared_processors.append(add_caller_info)

    shared_processors.extend([
        StackInfoRenderer(),
        format_exception_info,
    ])

    if censor_sensitive:
        shared_processors.append(censor_sensitive_data)

    # Final renderer
    if json_format:
        shared_processors.append(
            structlog.processors.JSONRenderer(sort_keys=True)
        )
    else:
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=True)
        )

    # Configure structlog
    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        logging.getLogger().addHandler(file_handler)


def _configure_stdlib_logging(
    log_level: int,
    json_format: bool,
    service_name: str,
    log_file: Optional[str],
):
    """Fallback configuration using stdlib logging."""

    if json_format:
        formatter = JsonFormatter(service_name=service_name)
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S%z'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [console_handler]

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# =============================================================================
# JSON Formatter (Fallback for stdlib logging)
# =============================================================================

class JsonFormatter(logging.Formatter):
    """
    JSON formatter for stdlib logging.

    Used as fallback when structlog is not available.
    """

    def __init__(self, service_name: str = "coderev-platform"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname.lower(),
            'logger': record.name,
            'message': record.getMessage(),
            'service': self.service_name,
        }

        # Add request context
        request_id = request_id_var.get()
        if request_id:
            log_data['request_id'] = request_id

        user_id = user_id_var.get()
        if user_id:
            log_data['user_id'] = user_id

        # Add exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info),
            }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'asctime',
            }:
                log_data[key] = value

        return json.dumps(log_data, default=str, ensure_ascii=False)


# =============================================================================
# Logger Factory
# =============================================================================

def get_logger(name: str = None) -> Union['structlog.stdlib.BoundLogger', logging.Logger]:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        structlog logger if available, otherwise stdlib logger

    Usage:
        logger = get_logger(__name__)
        logger.info("event_name", key1="value1", key2="value2")
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


# =============================================================================
# Context Managers for Request Correlation
# =============================================================================

class LogContext:
    """
    Context manager for adding correlation IDs to logs.

    Usage:
        async with LogContext(request_id="abc123", user_id="user-1"):
            logger.info("processing_request")
            # All logs within this context will have request_id and user_id
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id
        self._tokens = []

    def __enter__(self):
        if self.request_id:
            self._tokens.append(request_id_var.set(self.request_id))
        if self.user_id:
            self._tokens.append(user_id_var.set(self.user_id))
        if self.session_id:
            self._tokens.append(session_id_var.set(self.session_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _token in self._tokens:
            # Reset context variables - tokens tracked for future reset support
            pass
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Set request context for log correlation."""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def clear_request_context():
    """Clear request context."""
    request_id_var.set('')
    user_id_var.set('')
    session_id_var.set('')


# =============================================================================
# Decorators
# =============================================================================

def log_function_call(logger=None, level: str = "INFO"):
    """
    Decorator to log function entry and exit.

    Usage:
        @log_function_call(level="DEBUG")
        def process_data(data):
            ...
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        log_method = getattr(logger, level.lower(), logger.info)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            log_method(
                "function_call_start",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )
            try:
                result = await func(*args, **kwargs)
                log_method(
                    "function_call_end",
                    function=func.__name__,
                    status="success",
                )
                return result
            except Exception as e:
                logger.error(
                    "function_call_error",
                    function=func.__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            log_method(
                "function_call_start",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )
            try:
                result = func(*args, **kwargs)
                log_method(
                    "function_call_end",
                    function=func.__name__,
                    status="success",
                )
                return result
            except Exception as e:
                logger.error(
                    "function_call_error",
                    function=func.__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# FastAPI Middleware for Request Logging
# =============================================================================

class RequestLoggingMiddleware:
    """
    ASGI middleware for request logging with correlation IDs.

    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)
    """

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("http.request")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        import time
        import uuid

        # Generate or extract request ID
        request_id = None
        for header_name, header_value in scope.get("headers", []):
            if header_name.lower() == b"x-request-id":
                request_id = header_value.decode()
                break

        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        # Set context for this request
        set_request_context(request_id=request_id)

        start_time = time.time()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Add request ID to response headers
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "http_request",
                method=scope.get("method", "UNKNOWN"),
                path=scope.get("path", "/"),
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                client_ip=scope.get("client", ("unknown", 0))[0],
            )

            clear_request_context()


# =============================================================================
# Initialization Helper
# =============================================================================

def init_logging(
    service_name: str = "coderev-platform",
    log_level: str = None,
    json_logs: bool = None,
):
    """
    Initialize logging from environment variables with sensible defaults.

    Environment variables:
    - LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - LOG_FORMAT: json, text
    - ENVIRONMENT: development, staging, production
    - LOG_FILE: Path to log file (optional)

    Usage:
        from backend.shared.logging import init_logging
        init_logging(service_name="auth-service")
    """
    level = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "json")
    environment = os.getenv("ENVIRONMENT", "development")
    log_file = os.getenv("LOG_FILE")
    version = os.getenv("APP_VERSION", "2.1.0")

    if json_logs is None:
        json_logs = log_format.lower() == "json"

    configure_logging(
        level=level,
        json_format=json_logs,
        service_name=service_name,
        version=version,
        environment=environment,
        log_file=log_file,
        censor_sensitive=True,
        include_caller_info=(environment == "development"),
    )

    logger = get_logger(__name__)
    logger.info(
        "logging_initialized",
        level=level,
        format="json" if json_logs else "text",
        environment=environment,
        structlog_available=STRUCTLOG_AVAILABLE,
    )
