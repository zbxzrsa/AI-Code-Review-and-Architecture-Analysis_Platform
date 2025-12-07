"""
Performance Monitoring Metrics Module

Comprehensive Prometheus metrics for the AI Code Review Platform.

Metrics Categories:
- Vulnerability Detection: Counts, severity, categories
- Code Analysis: Scan duration, throughput, queue depth
- AI Provider: Request latency, token usage, errors
- System Health: Memory, CPU, active connections
- Business Metrics: Reviews completed, fixes applied

Usage:
    from backend.shared.monitoring.metrics import MetricsCollector
    
    # Record vulnerability
    MetricsCollector.record_vulnerability("critical", "sql_injection")
    
    # Time a scan
    with MetricsCollector.scan_duration.time():
        perform_scan()
    
    # Update gauge
    MetricsCollector.active_fixes.set(5)
"""

import time
import functools
import logging
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import prometheus_client, provide fallback if not available
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        multiprocess,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning(
        "prometheus_client not installed. Metrics will be no-op. "
        "Install with: pip install prometheus-client"
    )
    PROMETHEUS_AVAILABLE = False


# =============================================================================
# Metric Label Constants
# =============================================================================

class Severity(str, Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityCategory(str, Enum):
    """Vulnerability categories."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    SSRF = "ssrf"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    SENSITIVE_DATA = "sensitive_data"
    BROKEN_AUTH = "broken_auth"
    SECURITY_MISCONFIG = "security_misconfig"
    DEPENDENCY = "dependency"
    OTHER = "other"


class AIProvider(str, Enum):
    """AI provider labels."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    MOCK = "mock"


class AnalysisType(str, Enum):
    """Analysis type labels."""
    CODE_REVIEW = "code_review"
    VULNERABILITY_SCAN = "vulnerability_scan"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"
    DEPENDENCY_CHECK = "dependency_check"
    QUALITY_SCORE = "quality_score"


# =============================================================================
# No-Op Fallback Classes (when prometheus_client not available)
# =============================================================================

if not PROMETHEUS_AVAILABLE:
    class NoOpMetric:
        """No-op metric that does nothing."""
        def __init__(self, *args, **kwargs):
            pass
        
        def inc(self, *args, **kwargs):
            pass
        
        def dec(self, *args, **kwargs):
            pass
        
        def set(self, *args, **kwargs):
            pass
        
        def observe(self, *args, **kwargs):
            pass
        
        def labels(self, *args, **kwargs):
            return self
        
        def time(self):
            return contextmanager(lambda: (yield))()
        
        def info(self, *args, **kwargs):
            pass
    
    Counter = Histogram = Gauge = Summary = Info = NoOpMetric


# =============================================================================
# Metrics Definitions
# =============================================================================

class MetricsCollector:
    """
    Central metrics collector for the AI Code Review Platform.
    
    All metrics are class-level singletons to ensure proper Prometheus registration.
    """
    
    # =========================================================================
    # Vulnerability Detection Metrics
    # =========================================================================
    
    # Total vulnerabilities detected (by severity and category)
    vulnerabilities_detected = Counter(
        'coderev_vulnerabilities_detected_total',
        'Total number of vulnerabilities detected',
        ['severity', 'category', 'project_id']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Vulnerabilities by status
    vulnerabilities_by_status = Gauge(
        'coderev_vulnerabilities_by_status',
        'Current count of vulnerabilities by status',
        ['status']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Auto-fix metrics
    auto_fixes_generated = Counter(
        'coderev_auto_fixes_generated_total',
        'Total auto-fix suggestions generated',
        ['vulnerability_type', 'confidence_level']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    auto_fixes_applied = Counter(
        'coderev_auto_fixes_applied_total',
        'Total auto-fixes applied',
        ['vulnerability_type', 'result']  # result: success, failed, rejected
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    active_fixes = Gauge(
        'coderev_active_fixes_count',
        'Number of fixes currently in progress'
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # =========================================================================
    # Code Analysis Metrics
    # =========================================================================
    
    # Scan duration histogram (seconds)
    scan_duration = Histogram(
        'coderev_scan_duration_seconds',
        'Time spent scanning codebase',
        ['scan_type', 'project_id'],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Analysis throughput
    analyses_completed = Counter(
        'coderev_analyses_completed_total',
        'Total analyses completed',
        ['analysis_type', 'status']  # status: success, failed, timeout
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Files analyzed
    files_analyzed = Counter(
        'coderev_files_analyzed_total',
        'Total files analyzed',
        ['language', 'project_id']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Lines of code analyzed
    lines_analyzed = Counter(
        'coderev_lines_analyzed_total',
        'Total lines of code analyzed',
        ['language']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Analysis queue depth
    analysis_queue_depth = Gauge(
        'coderev_analysis_queue_depth',
        'Number of analyses waiting in queue',
        ['priority']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Code quality score distribution
    code_quality_score = Histogram(
        'coderev_code_quality_score',
        'Distribution of code quality scores',
        ['project_id'],
        buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # =========================================================================
    # AI Provider Metrics
    # =========================================================================
    
    # AI request latency
    ai_request_duration = Histogram(
        'coderev_ai_request_duration_seconds',
        'AI provider request duration',
        ['provider', 'model', 'operation'],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # AI requests total
    ai_requests_total = Counter(
        'coderev_ai_requests_total',
        'Total AI provider requests',
        ['provider', 'model', 'status']  # status: success, error, timeout, rate_limited
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Token usage
    ai_tokens_used = Counter(
        'coderev_ai_tokens_used_total',
        'Total tokens consumed',
        ['provider', 'model', 'token_type']  # token_type: prompt, completion
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # AI cost tracking (in USD cents)
    ai_cost_cents = Counter(
        'coderev_ai_cost_cents_total',
        'Total AI cost in cents',
        ['provider', 'model']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Provider health status (1 = healthy, 0 = unhealthy)
    ai_provider_health = Gauge(
        'coderev_ai_provider_health',
        'AI provider health status',
        ['provider']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Active AI requests
    ai_requests_in_flight = Gauge(
        'coderev_ai_requests_in_flight',
        'Number of AI requests currently in progress',
        ['provider']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # =========================================================================
    # Three-Version System Metrics
    # =========================================================================
    
    # Version status
    version_status = Gauge(
        'coderev_version_status',
        'Current version status (1=active, 0=inactive)',
        ['version']  # v1, v2, v3
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Promotions/Degradations
    version_transitions = Counter(
        'coderev_version_transitions_total',
        'Total version transitions',
        ['from_version', 'to_version', 'reason']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Experiments
    experiments_total = Counter(
        'coderev_experiments_total',
        'Total experiments run',
        ['status']  # created, running, completed, failed, promoted, quarantined
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # =========================================================================
    # System Health Metrics
    # =========================================================================
    
    # Memory usage (bytes)
    memory_usage_bytes = Gauge(
        'coderev_memory_usage_bytes',
        'Current memory usage',
        ['type']  # rss, vms, shared
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # GPU memory (if available)
    gpu_memory_usage_bytes = Gauge(
        'coderev_gpu_memory_usage_bytes',
        'GPU memory usage',
        ['device']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Active connections
    active_connections = Gauge(
        'coderev_active_connections',
        'Number of active connections',
        ['type']  # http, websocket, database
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Background tasks
    background_tasks_active = Gauge(
        'coderev_background_tasks_active',
        'Number of active background tasks',
        ['task_type']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # Cache metrics
    cache_hits = Counter(
        'coderev_cache_hits_total',
        'Total cache hits',
        ['cache_type']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    cache_misses = Counter(
        'coderev_cache_misses_total',
        'Total cache misses',
        ['cache_type']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # =========================================================================
    # Business Metrics
    # =========================================================================
    
    # Reviews completed
    reviews_completed = Counter(
        'coderev_reviews_completed_total',
        'Total code reviews completed',
        ['project_id', 'review_type']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # User activity
    user_actions = Counter(
        'coderev_user_actions_total',
        'Total user actions',
        ['action_type']  # login, review_request, fix_apply, etc.
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # API endpoint latency
    http_request_duration = Histogram(
        'coderev_http_request_duration_seconds',
        'HTTP request duration',
        ['method', 'endpoint', 'status_code'],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # API requests total
    http_requests_total = Counter(
        'coderev_http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status_code']
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # =========================================================================
    # Application Info
    # =========================================================================
    
    app_info = Info(
        'coderev_app',
        'Application information'
    ) if PROMETHEUS_AVAILABLE else NoOpMetric()
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    @classmethod
    def record_vulnerability(
        cls,
        severity: str,
        category: str,
        project_id: str = "unknown"
    ):
        """Record a detected vulnerability."""
        cls.vulnerabilities_detected.labels(
            severity=severity,
            category=category,
            project_id=project_id
        ).inc()
    
    @classmethod
    def record_analysis(
        cls,
        analysis_type: str,
        duration_seconds: float,
        status: str = "success",
        project_id: str = "unknown"
    ):
        """Record a completed analysis."""
        cls.analyses_completed.labels(
            analysis_type=analysis_type,
            status=status
        ).inc()
        
        cls.scan_duration.labels(
            scan_type=analysis_type,
            project_id=project_id
        ).observe(duration_seconds)
    
    @classmethod
    def record_ai_request(
        cls,
        provider: str,
        model: str,
        duration_seconds: float,
        status: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_cents: float = 0
    ):
        """Record an AI provider request."""
        cls.ai_requests_total.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        cls.ai_request_duration.labels(
            provider=provider,
            model=model,
            operation="completion"
        ).observe(duration_seconds)
        
        if prompt_tokens > 0:
            cls.ai_tokens_used.labels(
                provider=provider,
                model=model,
                token_type="prompt"
            ).inc(prompt_tokens)
        
        if completion_tokens > 0:
            cls.ai_tokens_used.labels(
                provider=provider,
                model=model,
                token_type="completion"
            ).inc(completion_tokens)
        
        if cost_cents > 0:
            cls.ai_cost_cents.labels(
                provider=provider,
                model=model
            ).inc(cost_cents)
    
    @classmethod
    def record_http_request(
        cls,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ):
        """Record an HTTP request."""
        cls.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        cls.http_request_duration.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).observe(duration_seconds)
    
    @classmethod
    def update_system_metrics(cls):
        """Update system health metrics."""
        try:
            import psutil
            process = psutil.Process()
            memory = process.memory_info()
            
            cls.memory_usage_bytes.labels(type="rss").set(memory.rss)
            cls.memory_usage_bytes.labels(type="vms").set(memory.vms)
            
            if hasattr(memory, 'shared'):
                cls.memory_usage_bytes.labels(type="shared").set(memory.shared)
        except ImportError:
            pass
        
        # GPU memory if available
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i)
                    cls.gpu_memory_usage_bytes.labels(device=f"cuda:{i}").set(allocated)
        except ImportError:
            pass
    
    @classmethod
    def set_app_info(
        cls,
        version: str,
        environment: str,
        commit_hash: str = ""
    ):
        """Set application info."""
        if PROMETHEUS_AVAILABLE:
            cls.app_info.info({
                'version': version,
                'environment': environment,
                'commit_hash': commit_hash,
            })


# =============================================================================
# Decorators for Easy Instrumentation
# =============================================================================

def track_time(metric_name: str = "scan_duration", **labels):
    """
    Decorator to track execution time of a function.
    
    Usage:
        @track_time("scan_duration", scan_type="vulnerability")
        async def scan_codebase():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric = getattr(MetricsCollector, metric_name, None)
                if metric and hasattr(metric, 'labels'):
                    metric.labels(**labels).observe(duration)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric = getattr(MetricsCollector, metric_name, None)
                if metric and hasattr(metric, 'labels'):
                    metric.labels(**labels).observe(duration)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def count_calls(metric_name: str, **labels):
    """
    Decorator to count function calls.
    
    Usage:
        @count_calls("analyses_completed", analysis_type="code_review")
        def analyze_code():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                metric = getattr(MetricsCollector, metric_name, None)
                if metric:
                    metric.labels(**labels, status="success").inc()
                return result
            except Exception as e:
                metric = getattr(MetricsCollector, metric_name, None)
                if metric:
                    metric.labels(**labels, status="error").inc()
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                metric = getattr(MetricsCollector, metric_name, None)
                if metric:
                    metric.labels(**labels, status="success").inc()
                return result
            except Exception as e:
                metric = getattr(MetricsCollector, metric_name, None)
                if metric:
                    metric.labels(**labels, status="error").inc()
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


@contextmanager
def track_in_flight(gauge_metric, **labels):
    """
    Context manager to track in-flight operations.
    
    Usage:
        with track_in_flight(MetricsCollector.ai_requests_in_flight, provider="openai"):
            await make_ai_request()
    """
    gauge_metric.labels(**labels).inc()
    try:
        yield
    finally:
        gauge_metric.labels(**labels).dec()


# =============================================================================
# FastAPI Middleware for HTTP Metrics
# =============================================================================

class PrometheusMiddleware:
    """
    ASGI middleware for collecting HTTP metrics.
    
    Usage:
        from fastapi import FastAPI
        app = FastAPI()
        app.add_middleware(PrometheusMiddleware)
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        status_code = 500  # Default in case of error
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.time() - start_time
            method = scope.get("method", "UNKNOWN")
            path = scope.get("path", "/")
            
            # Normalize path (remove IDs) for cleaner metrics
            path = self._normalize_path(path)
            
            MetricsCollector.record_http_request(
                method=method,
                endpoint=path,
                status_code=status_code,
                duration_seconds=duration
            )
    
    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path by replacing IDs with placeholders."""
        import re
        # Replace UUIDs
        path = re.sub(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '{id}',
            path
        )
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        return path


# =============================================================================
# Metrics Endpoint
# =============================================================================

def get_metrics():
    """
    Generate Prometheus metrics output.
    
    Usage in FastAPI:
        @app.get("/metrics")
        def metrics():
            return Response(get_metrics(), media_type=CONTENT_TYPE_LATEST)
    """
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY)
    return b""


def get_metrics_content_type() -> str:
    """Get the content type for Prometheus metrics."""
    if PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return "text/plain"
