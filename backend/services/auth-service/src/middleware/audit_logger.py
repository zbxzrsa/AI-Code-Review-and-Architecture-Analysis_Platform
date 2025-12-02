"""
Audit logging middleware for auth service.
"""
import logging
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all authentication-related requests.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log request
        logger.info(
            f"Auth request: {request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration_ms:.2f}ms "
            f"ip={client_ip} "
            f"ua={user_agent[:50]}"
        )
        
        # Log failed auth attempts
        if response.status_code == 401:
            logger.warning(
                f"Failed auth attempt: {request.method} {request.url.path} "
                f"ip={client_ip}"
            )
        
        return response
