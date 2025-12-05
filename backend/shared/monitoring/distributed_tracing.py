"""
Distributed Tracing Module

Implements:
- OpenTelemetry integration
- Trace propagation between services
- AI model call tracing
- SLO-aware sampling
"""

import logging
import os
import functools
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available, tracing disabled")


class TracingConfig:
    """Tracing configuration."""
    
    def __init__(
        self,
        service_name: str = "code-review-platform",
        otlp_endpoint: Optional[str] = None,
        sampling_rate: float = 1.0,
        slo_sampling_rate: float = 1.0,  # Always trace SLO violations
        environment: str = "development",
    ):
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://localhost:4317"
        )
        self.sampling_rate = sampling_rate
        self.slo_sampling_rate = slo_sampling_rate
        self.environment = environment


class DistributedTracer:
    """
    Distributed tracing with OpenTelemetry.
    
    Features:
    - Automatic instrumentation for FastAPI/HTTPX
    - AI model call tracing
    - Custom span attributes
    - SLO violation tracking
    """
    
    def __init__(self, config: TracingConfig):
        self.config = config
        self._tracer = None
        self._propagator = None
        
        if OTEL_AVAILABLE:
            self._init_tracing()
    
    def _init_tracing(self):
        """Initialize OpenTelemetry tracing."""
        # Create resource
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.environment": self.config.environment,
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter
        exporter = OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint,
            insecure=True,  # Set False in production with TLS
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        
        # Set as global provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        self._tracer = trace.get_tracer(
            self.config.service_name,
            "1.0.0",
        )
        
        # Set up propagator
        self._propagator = TraceContextTextMapPropagator()
        
        logger.info(f"Tracing initialized for {self.config.service_name}")
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application."""
        if OTEL_AVAILABLE:
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI instrumented")
    
    def instrument_httpx(self):
        """Instrument HTTPX client."""
        if OTEL_AVAILABLE:
            HTTPXClientInstrumentor().instrument()
            logger.info("HTTPX instrumented")
    
    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Create trace span."""
        if not OTEL_AVAILABLE or not self._tracer:
            yield MockSpan()
            return
        
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def trace_ai_call(
        self,
        model: str,
        prompt_tokens: int,
        response_tokens: int,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ):
        """Record AI model call trace."""
        if not OTEL_AVAILABLE or not self._tracer:
            return
        
        current_span = trace.get_current_span()
        
        # Add AI-specific attributes
        current_span.set_attribute("ai.model", model)
        current_span.set_attribute("ai.prompt_tokens", prompt_tokens)
        current_span.set_attribute("ai.response_tokens", response_tokens)
        current_span.set_attribute("ai.total_tokens", prompt_tokens + response_tokens)
        current_span.set_attribute("ai.latency_ms", latency_ms)
        current_span.set_attribute("ai.success", success)
        
        if error:
            current_span.set_attribute("ai.error", error)
            current_span.set_status(Status(StatusCode.ERROR, error))
    
    def trace_slo_violation(
        self,
        slo_name: str,
        target: float,
        actual: float,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Record SLO violation trace."""
        if not OTEL_AVAILABLE or not self._tracer:
            return
        
        with self._tracer.start_as_current_span("slo_violation") as span:
            span.set_attribute("slo.name", slo_name)
            span.set_attribute("slo.target", target)
            span.set_attribute("slo.actual", actual)
            span.set_attribute("slo.violation", True)
            
            if details:
                for key, value in details.items():
                    span.set_attribute(f"slo.{key}", str(value))
            
            span.set_status(Status(StatusCode.ERROR, f"SLO {slo_name} violated"))
    
    def inject_context(self, headers: Dict[str, str]):
        """Inject trace context into headers for propagation."""
        if OTEL_AVAILABLE and self._propagator:
            self._propagator.inject(headers)
    
    def extract_context(self, headers: Dict[str, str]):
        """Extract trace context from headers."""
        if OTEL_AVAILABLE and self._propagator:
            return self._propagator.extract(headers)
        return None


class MockSpan:
    """Mock span for when tracing is disabled."""
    
    def set_attribute(self, key: str, value: Any):  # noqa: ARG002
        """No-op: tracing disabled."""
        pass  # Intentionally empty - mock implementation
    
    def set_status(self, status):  # noqa: ARG002
        """No-op: tracing disabled."""
        pass  # Intentionally empty - mock implementation
    
    def record_exception(self, exception):  # noqa: ARG002
        """No-op: tracing disabled."""
        pass  # Intentionally empty - mock implementation
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):  # noqa: ARG002
        """No-op: tracing disabled."""
        pass  # Intentionally empty - mock implementation


def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace function execution.
    
    Usage:
        @trace_function("my_operation", {"key": "value"})
        async def my_function():
            ...
    """
    def decorator(func: Callable):
        span_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            if tracer:
                with tracer.span(span_name, attributes):
                    return await func(*args, **kwargs)
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            if tracer:
                with tracer.span(span_name, attributes):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Singleton tracer instance
_tracer_instance: Optional[DistributedTracer] = None


def init_tracing(config: Optional[TracingConfig] = None) -> DistributedTracer:
    """Initialize global tracer."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = DistributedTracer(config or TracingConfig())
    return _tracer_instance


def get_tracer() -> Optional[DistributedTracer]:
    """Get global tracer instance."""
    return _tracer_instance


# Import asyncio for decorator
import asyncio
