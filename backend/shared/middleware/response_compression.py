"""
Response Compression Middleware

Implements gzip/brotli compression for API responses to reduce bandwidth
and improve performance.
"""

import gzip
import io
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.datastructures import Headers, MutableHeaders


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that compresses response bodies using gzip.
    
    Supports:
    - gzip compression (widely supported)
    - Minimum size threshold
    - Content-type filtering
    """
    
    # Content types that should be compressed
    COMPRESSIBLE_TYPES = {
        "application/json",
        "text/html",
        "text/plain",
        "text/css",
        "text/javascript",
        "application/javascript",
        "application/xml",
        "text/xml",
        "application/ld+json",
    }
    
    def __init__(
        self,
        app,
        minimum_size: int = 500,  # Don't compress responses smaller than 500 bytes
        compression_level: int = 6,  # 1-9, 6 is default balance
    ):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if client accepts gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding.lower():
            return await call_next(request)
        
        # Get original response
        response = await call_next(request)
        
        # Skip if already encoded
        if response.headers.get("Content-Encoding"):
            return response
        
        # Check content type
        content_type = response.headers.get("Content-Type", "")
        base_type = content_type.split(";")[0].strip()
        if base_type not in self.COMPRESSIBLE_TYPES:
            return response
        
        # Handle streaming responses differently
        if isinstance(response, StreamingResponse):
            return await self._compress_streaming_response(response)
        
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Skip small responses
        if len(body) < self.minimum_size:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        
        # Compress body
        compressed_body = self._compress(body)
        
        # Only use compressed if it's actually smaller
        if len(compressed_body) >= len(body):
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        
        # Create compressed response
        headers = MutableHeaders(raw=list(response.headers.raw))
        headers["Content-Encoding"] = "gzip"
        headers["Content-Length"] = str(len(compressed_body))
        headers["Vary"] = "Accept-Encoding"
        
        return Response(
            content=compressed_body,
            status_code=response.status_code,
            headers=dict(headers),
            media_type=response.media_type,
        )
    
    def _compress(self, data: bytes) -> bytes:
        """Compress data using gzip"""
        buf = io.BytesIO()
        with gzip.GzipFile(
            fileobj=buf,
            mode="wb",
            compresslevel=self.compression_level
        ) as f:
            f.write(data)
        return buf.getvalue()
    
    async def _compress_streaming_response(
        self,
        response: StreamingResponse
    ) -> StreamingResponse:
        """Handle streaming response compression"""
        # For streaming, we'd need a more complex approach
        # For now, just pass through
        return response


# Usage:
# app.add_middleware(CompressionMiddleware, minimum_size=500, compression_level=6)
