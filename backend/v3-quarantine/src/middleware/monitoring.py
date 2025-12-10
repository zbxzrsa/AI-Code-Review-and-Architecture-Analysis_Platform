"""
Monitoring middleware for V3 quarantine.
"""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Version-specific metrics
request_count = Counter(
    "v3_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

request_duration = Histogram(
    "v3_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"],
)

active_requests = Gauge(
    "v3_http_active_requests",
    "Number of active HTTP requests",
)

# Unified cross-version metrics
version_request_count = Counter(
    "version_http_requests_total",
    "Total HTTP requests across versions",
    ["version", "method", "endpoint", "status"],
)

version_request_duration = Histogram(
    "version_http_request_duration_seconds",
    "HTTP request duration across versions",
    ["version", "method", "endpoint"],
)

version_active_requests = Gauge(
    "version_http_active_requests",
    "Number of active HTTP requests across versions",
    ["version"],
)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring requests."""

    async def dispatch(self, request: Request, call_next):
        """Process request and track metrics."""
        start_time = time.time()
        active_requests.inc()
        version_active_requests.labels(version="v3").inc()

        try:
            response = await call_next(request)
            return response
        finally:
            duration = time.time() - start_time
            active_requests.dec()
            version_active_requests.labels(version="v3").dec()

            # Record metrics
            endpoint = request.url.path
            method = request.method
            status_code = getattr(response, "status_code", 500)

            request_count.labels(
                method=method,
                endpoint=endpoint,
                status=status_code,
            ).inc()

            version_request_count.labels(
                version="v3",
                method=method,
                endpoint=endpoint,
                status=status_code,
            ).inc()

            request_duration.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            version_request_duration.labels(
                version="v3",
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            logger.info(
                "Request processed",
                method=method,
                path=endpoint,
                status=status_code,
                duration_ms=duration * 1000,
            )
