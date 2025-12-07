"""
Metrics Collection Module

Provides metrics collection for monitoring:
- Request counts and latencies
- Resource utilization
- Business metrics
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from collections import defaultdict
from functools import wraps
import threading

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsCollector:
    """
    Collects and exposes application metrics.
    
    Supports:
    - Counters (monotonically increasing)
    - Gauges (can go up or down)
    - Histograms (distribution of values)
    - Summaries (percentile calculations)
    
    Usage:
        metrics = MetricsCollector()
        
        # Counter
        metrics.increment("requests_total", labels={"path": "/api/users"})
        
        # Gauge
        metrics.set_gauge("active_connections", 42)
        
        # Histogram
        metrics.observe("request_duration_ms", 150, labels={"path": "/api/users"})
        
        # Get metrics
        all_metrics = metrics.get_all()
    """
    
    def __init__(self):
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._lock = threading.Lock()
        
        # Histogram buckets
        self._default_buckets = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    
    def _labels_to_key(self, labels: Optional[Dict[str, str]] = None) -> str:
        """Convert labels dict to a hashable key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    
    # ========================================
    # Counters
    # ========================================
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._counters[name][key] += value
    
    def get_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Get counter value."""
        key = self._labels_to_key(labels)
        with self._lock:
            return self._counters[name][key]
    
    # ========================================
    # Gauges
    # ========================================
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._gauges[name][key] = value
    
    def increment_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a gauge metric."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._gauges[name][key] += value
    
    def decrement_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Decrement a gauge metric."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._gauges[name][key] -= value
    
    def get_gauge(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Get gauge value."""
        key = self._labels_to_key(labels)
        with self._lock:
            return self._gauges[name][key]
    
    # ========================================
    # Histograms
    # ========================================
    
    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record an observation in a histogram."""
        key = self._labels_to_key(labels)
        with self._lock:
            self._histograms[name][key].append(value)
            # Keep only last 10000 observations per bucket
            if len(self._histograms[name][key]) > 10000:
                self._histograms[name][key] = self._histograms[name][key][-10000:]
    
    def get_histogram(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get histogram statistics."""
        key = self._labels_to_key(labels)
        with self._lock:
            values = self._histograms[name][key]
            
            if not values:
                return {"count": 0}
            
            sorted_values = sorted(values)
            count = len(values)
            
            return {
                "count": count,
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / count,
                "p50": sorted_values[int(count * 0.50)],
                "p90": sorted_values[int(count * 0.90)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)] if count > 100 else sorted_values[-1],
            }
    
    # ========================================
    # Export
    # ========================================
    
    def get_all(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            result = {
                "counters": {},
                "gauges": {},
                "histograms": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            # Counters
            for name, values in self._counters.items():
                result["counters"][name] = {
                    k if k else "default": v for k, v in values.items()
                }
            
            # Gauges
            for name, values in self._gauges.items():
                result["gauges"][name] = {
                    k if k else "default": v for k, v in values.items()
                }
            
            # Histograms
            for name, values in self._histograms.items():
                result["histograms"][name] = {}
                for k, v in values.items():
                    if v:
                        sorted_v = sorted(v)
                        count = len(v)
                        result["histograms"][name][k if k else "default"] = {
                            "count": count,
                            "sum": sum(v),
                            "avg": sum(v) / count,
                            "p50": sorted_v[int(count * 0.50)],
                            "p95": sorted_v[int(count * 0.95)],
                        }
            
            return result
    
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            # Counters
            for name, values in self._counters.items():
                for labels_key, value in values.items():
                    labels_str = f"{{{labels_key}}}" if labels_key else ""
                    lines.append(f"{name}{labels_str} {value}")
            
            # Gauges
            for name, values in self._gauges.items():
                for labels_key, value in values.items():
                    labels_str = f"{{{labels_key}}}" if labels_key else ""
                    lines.append(f"{name}{labels_str} {value}")
            
            # Histograms (simplified)
            for name, values in self._histograms.items():
                for labels_key, observations in values.items():
                    if observations:
                        labels_str = f"{{{labels_key}}}" if labels_key else ""
                        lines.append(f"{name}_count{labels_str} {len(observations)}")
                        lines.append(f"{name}_sum{labels_str} {sum(observations)}")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def track_request(
    method: str,
    path: str,
    status: int,
    duration_ms: float
) -> None:
    """Track HTTP request metrics."""
    metrics = get_metrics()
    
    labels = {"method": method, "path": path, "status": str(status)}
    
    metrics.increment("http_requests_total", labels=labels)
    metrics.observe("http_request_duration_ms", duration_ms, labels={"method": method, "path": path})


def track_latency(name: str):
    """
    Decorator to track function latency.
    
    Usage:
        @track_latency("database_query")
        async def query_database():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                get_metrics().observe(f"{name}_duration_ms", duration_ms)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                get_metrics().observe(f"{name}_duration_ms", duration_ms)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# AI-Specific Metrics (P1 Enhancement)
# =============================================================================

class AIMetrics:
    """
    AI-specific metrics for the three-version evolution system.
    
    Tracks:
    - Model inference latency and throughput
    - Version-specific performance (V1/V2/V3)
    - Learning cycle metrics
    - Error rates and recovery
    
    Usage:
        ai_metrics = AIMetrics()
        
        # Track inference
        with ai_metrics.track_inference("gpt-4", "v2"):
            result = await model.generate(...)
        
        # Track learning
        ai_metrics.record_learning_cycle("v1", items_learned=100, duration_ms=5000)
    """
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.metrics = collector or get_metrics()
    
    def track_inference(self, model: str, version: str):
        """Context manager for tracking model inference."""
        return _InferenceTimer(self.metrics, model, version)
    
    def record_inference(
        self,
        model: str,
        version: str,
        duration_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        success: bool = True,
    ) -> None:
        """Record a model inference."""
        labels = {"model": model, "version": version}
        
        self.metrics.increment("ai_inference_total", labels={**labels, "status": "success" if success else "error"})
        self.metrics.observe("ai_inference_duration_ms", duration_ms, labels=labels)
        
        if tokens_in > 0:
            self.metrics.increment("ai_tokens_input_total", tokens_in, labels=labels)
        if tokens_out > 0:
            self.metrics.increment("ai_tokens_output_total", tokens_out, labels=labels)
    
    def record_learning_cycle(
        self,
        version: str,
        items_learned: int,
        duration_ms: float,
        items_rejected: int = 0,
    ) -> None:
        """Record a learning cycle completion."""
        labels = {"version": version}
        
        self.metrics.increment("ai_learning_cycles_total", labels=labels)
        self.metrics.increment("ai_items_learned_total", items_learned, labels=labels)
        self.metrics.increment("ai_items_rejected_total", items_rejected, labels=labels)
        self.metrics.observe("ai_learning_duration_ms", duration_ms, labels=labels)
    
    def record_version_promotion(
        self,
        technology: str,
        from_version: str,
        to_version: str,
    ) -> None:
        """Record a technology promotion between versions."""
        self.metrics.increment(
            "ai_version_promotions_total",
            labels={"technology": technology, "from": from_version, "to": to_version}
        )
    
    def record_bug_fix(
        self,
        severity: str,
        auto_fixed: bool,
        duration_ms: float,
    ) -> None:
        """Record a bug fix event."""
        labels = {"severity": severity, "auto_fixed": str(auto_fixed).lower()}
        
        self.metrics.increment("ai_bug_fixes_total", labels=labels)
        self.metrics.observe("ai_bug_fix_duration_ms", duration_ms, labels={"severity": severity})
    
    def set_version_status(
        self,
        version: str,
        technologies_count: int,
        health_score: float,
    ) -> None:
        """Set current version status gauges."""
        self.metrics.set_gauge(
            "ai_version_technologies",
            technologies_count,
            labels={"version": version}
        )
        self.metrics.set_gauge(
            "ai_version_health_score",
            health_score,
            labels={"version": version}
        )
    
    def record_cache_operation(
        self,
        operation: str,  # hit, miss, set, evict
        cache_type: str,  # l1, l2, query
    ) -> None:
        """Record cache operation."""
        self.metrics.increment(
            "ai_cache_operations_total",
            labels={"operation": operation, "type": cache_type}
        )


class _InferenceTimer:
    """Context manager for timing model inference."""
    
    def __init__(self, metrics: MetricsCollector, model: str, version: str):
        self.metrics = metrics
        self.model = model
        self.version = version
        self.start_time = None
        self.success = True
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.success = exc_type is None
        
        labels = {"model": self.model, "version": self.version}
        status = "success" if self.success else "error"
        
        self.metrics.increment("ai_inference_total", labels={**labels, "status": status})
        self.metrics.observe("ai_inference_duration_ms", duration_ms, labels=labels)
        
        return False  # Don't suppress exceptions


# Global AI metrics instance
_ai_metrics: Optional[AIMetrics] = None


def get_ai_metrics() -> AIMetrics:
    """Get global AI metrics instance."""
    global _ai_metrics
    if _ai_metrics is None:
        _ai_metrics = AIMetrics()
    return _ai_metrics
