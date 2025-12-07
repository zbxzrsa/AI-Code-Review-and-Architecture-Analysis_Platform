"""
Structured Logging Module

Provides standardized structured logging with:
- JSON format output
- Request tracking ID
- Multi-level support
- Context propagation
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from contextvars import ContextVar
import uuid

# Context variable for request tracking
request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


@dataclass
class LogConfig:
    """Logging configuration."""
    # Output settings
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = "json"  # json or text
    output: str = "stdout"  # stdout, stderr, or file path
    
    # Content settings
    include_timestamp: bool = True
    include_level: bool = True
    include_logger: bool = True
    include_location: bool = True
    include_request_id: bool = True
    include_user_id: bool = True
    
    # Filtering
    exclude_paths: list = field(default_factory=lambda: ["/health", "/metrics"])
    
    # Performance
    async_logging: bool = False
    buffer_size: int = 1000


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Output format:
    {
        "timestamp": "2024-01-01T00:00:00.000Z",
        "level": "INFO",
        "logger": "app.service",
        "message": "Request processed",
        "request_id": "uuid",
        "user_id": "user_123",
        "data": {...},
        "location": {
            "file": "service.py",
            "line": 42,
            "function": "process"
        }
    }
    """
    
    def __init__(self, config: Optional[LogConfig] = None):
        super().__init__()
        self.config = config or LogConfig()
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {}
        
        # Timestamp
        if self.config.include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Level
        if self.config.include_level:
            log_entry["level"] = record.levelname
        
        # Logger name
        if self.config.include_logger:
            log_entry["logger"] = record.name
        
        # Message
        log_entry["message"] = record.getMessage()
        
        # Request context
        ctx = request_context.get()
        if self.config.include_request_id and ctx.get("request_id"):
            log_entry["request_id"] = ctx["request_id"]
        if self.config.include_user_id and ctx.get("user_id"):
            log_entry["user_id"] = ctx["user_id"]
        
        # Additional context from record
        if hasattr(record, "data") and record.data:
            log_entry["data"] = record.data
        
        # Location
        if self.config.include_location:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }
        
        # Exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info[0] else None,
            }
        
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """
    Human-readable text formatter with colors.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, config: Optional[LogConfig] = None, use_colors: bool = True):
        super().__init__()
        self.config = config or LogConfig()
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        parts = []
        
        # Timestamp
        if self.config.include_timestamp:
            parts.append(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        
        # Level with color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level = f"{color}{level:8}{self.RESET}"
        else:
            level = f"{level:8}"
        parts.append(level)
        
        # Request ID
        ctx = request_context.get()
        if self.config.include_request_id and ctx.get("request_id"):
            parts.append(f"[{ctx['request_id'][:8]}]")
        
        # Logger name
        if self.config.include_logger:
            parts.append(f"[{record.name}]")
        
        # Message
        parts.append(record.getMessage())
        
        # Additional data
        if hasattr(record, "data") and record.data:
            parts.append(f"| {json.dumps(record.data, default=str)}")
        
        result = " ".join(parts)
        
        # Exception
        if record.exc_info:
            result += "\n" + "".join(traceback.format_exception(*record.exc_info))
        
        return result


class StructuredLogger(logging.Logger):
    """
    Enhanced logger with structured logging support.
    
    Usage:
        logger = get_logger("my_service")
        
        # Basic logging
        logger.info("User logged in", user_id="123")
        
        # With context
        with logger.context(request_id="abc123"):
            logger.info("Processing request")
        
        # With extra data
        logger.info("API response", data={"status": 200, "latency_ms": 45})
    """
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
    
    def _log_with_data(
        self,
        level: int,
        msg: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log message with structured data."""
        extra = kwargs.pop("extra", {})
        extra["data"] = data or {}
        
        # Add any additional kwargs to data
        for key, value in kwargs.items():
            if key not in ("exc_info", "stack_info", "stacklevel"):
                extra["data"][key] = value
        
        super()._log(level, msg, (), extra=extra)
    
    def debug(self, msg: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_data(logging.DEBUG, msg, data, **kwargs)
    
    def info(self, msg: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_data(logging.INFO, msg, data, **kwargs)
    
    def warning(self, msg: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_data(logging.WARNING, msg, data, **kwargs)
    
    def error(self, msg: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_data(logging.ERROR, msg, data, **kwargs)
    
    def critical(self, msg: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_data(logging.CRITICAL, msg, data, **kwargs)
    
    def exception(self, msg: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        kwargs["exc_info"] = True
        self._log_with_data(logging.ERROR, msg, data, **kwargs)


class RequestContextManager:
    """Context manager for request-scoped logging context."""
    
    def __init__(self, **context):
        self.context = context
        self.token = None
    
    def __enter__(self):
        current = request_context.get().copy()
        current.update(self.context)
        self.token = request_context.set(current)
        return self
    
    def __exit__(self, *args):
        if self.token:
            request_context.reset(self.token)


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs
) -> None:
    """Set request context for logging."""
    ctx = request_context.get().copy()
    if request_id:
        ctx["request_id"] = request_id
    if user_id:
        ctx["user_id"] = user_id
    ctx.update(kwargs)
    request_context.set(ctx)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def setup_logging(config: Optional[LogConfig] = None) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        config: Logging configuration
    """
    config = config or LogConfig()
    
    # Set logging class
    logging.setLoggerClass(StructuredLogger)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    if config.output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif config.output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(config.output)
    
    # Set formatter
    if config.format == "json":
        handler.setFormatter(JSONFormatter(config))
    else:
        handler.setFormatter(TextFormatter(config))
    
    root_logger.addHandler(handler)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return logging.getLogger(name)


# FastAPI middleware for request logging
class PerformanceLogger:
    """
    Performance logging utility for tracking operation timing.
    
    Usage:
        perf = PerformanceLogger(logger)
        
        with perf.track("database_query"):
            result = db.query(...)
        
        # Or with decorator
        @perf.timed("api_call")
        async def call_api():
            ...
    """
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self._metrics: Dict[str, list] = {}
    
    def track(self, operation: str, threshold_ms: float = 100.0):
        """Context manager for tracking operation timing."""
        return _OperationTimer(self, operation, threshold_ms)
    
    def timed(self, operation: str, threshold_ms: float = 100.0):
        """Decorator for timing functions."""
        import functools
        
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with self.track(operation, threshold_ms):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.track(operation, threshold_ms):
                        return func(*args, **kwargs)
                return sync_wrapper
        return decorator
    
    def record(self, operation: str, duration_ms: float, threshold_ms: float):
        """Record an operation timing."""
        if operation not in self._metrics:
            self._metrics[operation] = []
        
        self._metrics[operation].append(duration_ms)
        
        # Keep only last 1000 measurements
        if len(self._metrics[operation]) > 1000:
            self._metrics[operation] = self._metrics[operation][-1000:]
        
        # Log if above threshold
        if duration_ms > threshold_ms:
            self.logger.warning(
                f"Slow operation: {operation}",
                data={"operation": operation, "duration_ms": round(duration_ms, 2), "threshold_ms": threshold_ms}
            )
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get timing statistics for an operation."""
        if operation not in self._metrics or not self._metrics[operation]:
            return {}
        
        measurements = self._metrics[operation]
        return {
            "count": len(measurements),
            "avg_ms": sum(measurements) / len(measurements),
            "min_ms": min(measurements),
            "max_ms": max(measurements),
            "p50_ms": sorted(measurements)[len(measurements) // 2],
            "p95_ms": sorted(measurements)[int(len(measurements) * 0.95)] if len(measurements) >= 20 else max(measurements),
        }


class _OperationTimer:
    """Internal timer context manager."""
    
    def __init__(self, perf_logger: PerformanceLogger, operation: str, threshold_ms: float):
        self.perf_logger = perf_logger
        self.operation = operation
        self.threshold_ms = threshold_ms
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        import time
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.perf_logger.record(self.operation, duration_ms, self.threshold_ms)


import asyncio


class LoggingMiddleware:
    """
    FastAPI middleware for request logging.
    
    Automatically:
    - Generates request IDs
    - Logs request start/end
    - Tracks response time
    - Adds request context
    """
    
    def __init__(self, app, config: Optional[LogConfig] = None):
        self.app = app
        self.config = config or LogConfig()
        self.logger = get_logger("http")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Skip excluded paths
        path = scope.get("path", "")
        if path in self.config.exclude_paths:
            await self.app(scope, receive, send)
            return
        
        # Generate request ID
        request_id = generate_request_id()
        
        # Set context
        set_request_context(request_id=request_id)
        
        # Log request start
        import time
        start_time = time.perf_counter()
        
        self.logger.info(
            "Request started",
            data={
                "method": scope.get("method"),
                "path": path,
                "query": scope.get("query_string", b"").decode(),
            }
        )
        
        # Track response status
        response_status = 500
        
        async def send_wrapper(message):
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = message.get("status", 500)
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Log request end
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            self.logger.info(
                "Request completed",
                data={
                    "method": scope.get("method"),
                    "path": path,
                    "status": response_status,
                    "duration_ms": round(duration_ms, 2),
                }
            )
