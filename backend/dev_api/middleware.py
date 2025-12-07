"""
Custom Middleware

Request/response middleware for the development API.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from .config import MAX_REQUEST_SIZE_BYTES, MAX_REQUEST_SIZE_MB


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""
    
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        
        if content_length and int(content_length) > MAX_REQUEST_SIZE_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"Request body too large. Maximum size is {MAX_REQUEST_SIZE_MB}MB"
                }
            )
        
        return await call_next(request)
