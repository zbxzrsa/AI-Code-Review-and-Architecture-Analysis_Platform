"""
Structured Logging Module (TD-004)

Provides standardized logging with:
- JSON and text formats
- Request tracking IDs
- Context enrichment
- Multiple transports
"""
from .structured_logger import (
    # Core
    logger,
    get_logger,
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
