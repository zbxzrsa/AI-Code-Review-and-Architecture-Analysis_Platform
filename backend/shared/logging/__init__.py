"""
结构化日志模块 (Structured Logging Module)

模块功能描述:
    提供标准化的结构化日志功能。

主要功能:
    - JSON 和文本格式（基于 structlog）
    - 请求追踪 ID 和上下文变量
    - 上下文丰富
    - 多种传输方式
    - 敏感数据脱敏
    - FastAPI 中间件集成

主要组件:
    - StructuredLogger: 结构化日志器
    - RequestLoggingMiddleware: 请求日志中间件
    - LogContext: 日志上下文
    - JsonFormatter: JSON 格式化器

使用示例:
    from backend.shared.logging import init_logging, get_logger
    
    # 在应用启动时初始化
    init_logging(service_name="auth-service")
    
    # 获取日志器并使用
    logger = get_logger(__name__)
    logger.info("user_login", user_id="123", ip="192.168.1.1")

最后修改日期: 2024-12-07
"""
from .structured_logger import (
    # Core
    logger,
    get_logger as get_stdlib_logger,
    setup_logging,
    
    # Configuration
    LogConfig,
    StructuredFormatter,
    TextFormatter,
    
    # Context
    RequestContext,
    request_context,
    set_request_id,
    set_user_id,
    get_request_id,
    
    # Classes
    StructuredLogger,
    ContextualLogger,
)

# Import new structlog-based configuration
from .config import (
    # Configuration
    configure_logging,
    init_logging,
    get_logger,
    
    # Context management
    LogContext,
    set_request_context,
    clear_request_context,
    request_id_var,
    user_id_var,
    session_id_var,
    
    # Decorators
    log_function_call,
    
    # Middleware
    RequestLoggingMiddleware,
    
    # Formatters
    JsonFormatter,
    
    # Availability
    STRUCTLOG_AVAILABLE,
)

__all__ = [
    # Legacy exports
    "logger",
    "get_stdlib_logger",
    "setup_logging",
    "LogConfig",
    "StructuredFormatter",
    "TextFormatter",
    "RequestContext",
    "request_context",
    "set_request_id",
    "set_user_id",
    "get_request_id",
    "StructuredLogger",
    "ContextualLogger",
    
    # New structlog-based exports
    "configure_logging",
    "init_logging",
    "get_logger",
    "LogContext",
    "set_request_context",
    "clear_request_context",
    "request_id_var",
    "user_id_var",
    "session_id_var",
    "log_function_call",
    "RequestLoggingMiddleware",
    "JsonFormatter",
    "STRUCTLOG_AVAILABLE",
]
