"""
Monitoring_V1 - Backend Integration Bridge

Integrates with backend/shared/monitoring implementations.
"""

import sys
from typing import Optional, Dict, Any, Callable
from pathlib import Path

# Add backend path for imports
_backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

# Import backend implementations
try:
    from shared.monitoring import (
        MetricsCollector as BackendMetricsCollector,
        PrometheusMiddleware as BackendPrometheusMiddleware,
        track_time as backend_track_time,
        count_calls as backend_count_calls,
        get_metrics as backend_get_metrics,
        PROMETHEUS_AVAILABLE,
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    PROMETHEUS_AVAILABLE = False
    BackendMetricsCollector = None
    BackendPrometheusMiddleware = None


class IntegratedMetricsCollector:
    """
    V1 Metrics Collector with backend Prometheus integration.
    """

    def __init__(self, prefix: str = "app", use_backend: bool = True):
        self.prefix = prefix
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if not self.use_backend:
            from .metrics_collector import MetricsCollector
            self._local = MetricsCollector(prefix=prefix)

    def record_counter(self, name: str, value: float = 1, labels: Optional[Dict] = None):
        """Record counter metric."""
        if self.use_backend:
            BackendMetricsCollector.increment(f"{self.prefix}_{name}", value, labels or {})
        else:
            self._local.inc(name, value)

    def record_gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """Record gauge metric."""
        if self.use_backend:
            BackendMetricsCollector.set_gauge(f"{self.prefix}_{name}", value, labels or {})
        else:
            self._local.set(name, value)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict] = None):
        """Record histogram metric."""
        if self.use_backend:
            BackendMetricsCollector.observe(f"{self.prefix}_{name}", value, labels or {})
        else:
            self._local.observe(name, value)

    def record_ai_request(
        self,
        provider: str,
        model: str,
        duration: float,
        status: str,
    ):
        """Record AI request metrics."""
        if self.use_backend:
            BackendMetricsCollector.record_ai_request(provider, model, duration, status)
        else:
            self.record_counter("ai_requests_total", labels={"provider": provider, "status": status})
            self.record_histogram("ai_request_duration", duration, {"provider": provider})

    def record_vulnerability(self, severity: str, category: str):
        """Record vulnerability found."""
        if self.use_backend:
            BackendMetricsCollector.record_vulnerability(severity, category)
        else:
            self.record_counter("vulnerabilities_total", labels={"severity": severity, "category": category})


class IntegratedTracing:
    """
    V1 Distributed Tracing with backend integration.
    """

    def __init__(self, service_name: str, use_backend: bool = True):
        self.service_name = service_name
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            try:
                from shared.monitoring.distributed_tracing import TracingManager
                self._backend = TracingManager(service_name)
            except ImportError:
                self.use_backend = False

        if not self.use_backend:
            from modules.Monitoring_V2.src.tracing_service import TracingService
            self._local = TracingService(service_name)

    def start_span(self, operation: str) -> Any:
        """Start a trace span."""
        if self.use_backend:
            return self._backend.start_span(operation)
        return self._local.start_span(operation)

    def finish_span(self, span: Any):
        """Finish a trace span."""
        if self.use_backend:
            self._backend.finish_span(span)
        else:
            self._local.finish_span(span)


def track_time(metric_name: str, **labels):
    """Decorator to track function execution time."""
    if BACKEND_AVAILABLE:
        return backend_track_time(metric_name, **labels)

    def decorator(func: Callable):
        import functools
        import time

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                _duration = time.time() - start  # Available for local logging
                # Log locally if backend unavailable
        return wrapper
    return decorator


def count_calls(metric_name: str, **labels):
    """Decorator to count function calls."""
    if BACKEND_AVAILABLE:
        return backend_count_calls(metric_name, **labels)

    def decorator(func: Callable):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_metrics() -> str:
    """Get Prometheus metrics output."""
    if BACKEND_AVAILABLE:
        return backend_get_metrics()
    return ""


# Factory
def get_metrics_collector(prefix: str = "app", use_backend: bool = True) -> IntegratedMetricsCollector:
    """Get integrated metrics collector."""
    return IntegratedMetricsCollector(prefix, use_backend)


def get_tracing(service_name: str, use_backend: bool = True) -> IntegratedTracing:
    """Get integrated tracing."""
    return IntegratedTracing(service_name, use_backend)


__all__ = [
    "BACKEND_AVAILABLE",
    "PROMETHEUS_AVAILABLE",
    "IntegratedMetricsCollector",
    "IntegratedTracing",
    "track_time",
    "count_calls",
    "get_metrics",
    "get_metrics_collector",
    "get_tracing",
]
