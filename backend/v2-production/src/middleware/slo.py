"""
SLO enforcement middleware for V2 production.
"""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from prometheus_client import Counter

logger = logging.getLogger(__name__)

# Metrics
slo_violations = Counter(
    "v2_slo_violations_total",
    "Total SLO violations",
    ["metric", "endpoint"],
)


class SLOMiddleware(BaseHTTPMiddleware):
    """Middleware for SLO enforcement and tracking."""

    # SLO thresholds
    RESPONSE_TIME_P95_MS = 3000
    ERROR_RATE_THRESHOLD = 0.02

    async def dispatch(self, request: Request, call_next):
        """Process request and enforce SLOs."""
        start_time = time.time()

        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Check response time SLO
            if duration_ms > self.RESPONSE_TIME_P95_MS:
                slo_violations.labels(
                    metric="response_time",
                    endpoint=request.url.path,
                ).inc()
                logger.warning(
                    "SLO violation: response time exceeded",
                    endpoint=request.url.path,
                    duration_ms=duration_ms,
                    threshold_ms=self.RESPONSE_TIME_P95_MS,
                )

            # Add SLO headers
            response.headers["X-Response-Time-Ms"] = str(duration_ms)
            response.headers["X-SLO-Compliant"] = (
                "true" if duration_ms <= self.RESPONSE_TIME_P95_MS else "false"
            )

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "Request failed",
                endpoint=request.url.path,
                error=str(e),
                duration_ms=duration_ms,
            )
            raise
