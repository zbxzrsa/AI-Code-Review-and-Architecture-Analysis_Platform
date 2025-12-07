"""
Monitoring_V2 - Metrics Collector

Production metrics collection with Prometheus integration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable
from enum import Enum


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Metric definition."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class MetricValue:
    """Recorded metric value."""
    value: float
    labels: Dict[str, str]
    timestamp: datetime


class MetricsCollector:
    """
    Production Metrics Collector.

    Features:
    - Multiple metric types (counter, gauge, histogram)
    - Label support
    - Prometheus-compatible output
    - Aggregations
    """

    def __init__(self, prefix: str = "app"):
        self.prefix = prefix
        self._definitions: Dict[str, MetricDefinition] = {}
        self._values: Dict[str, List[MetricValue]] = {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}

    def define_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        """Define a new metric."""
        full_name = f"{self.prefix}_{name}"
        self._definitions[full_name] = MetricDefinition(
            name=full_name,
            metric_type=metric_type,
            description=description,
            labels=labels or [],
            buckets=buckets,
        )
        self._values[full_name] = []

    def inc(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Increment counter."""
        full_name = f"{self.prefix}_{name}"
        key = self._make_key(full_name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
        self._record(full_name, self._counters[key], labels)

    def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value."""
        full_name = f"{self.prefix}_{name}"
        key = self._make_key(full_name, labels)
        self._gauges[key] = value
        self._record(full_name, value, labels)

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe value (for histograms/summaries)."""
        full_name = f"{self.prefix}_{name}"
        self._record(full_name, value, labels)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Make unique key for metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _record(self, name: str, value: float, labels: Optional[Dict[str, str]]):
        """Record metric value."""
        if name not in self._values:
            self._values[name] = []

        self._values[name].append(MetricValue(
            value=value,
            labels=labels or {},
            timestamp=datetime.now(timezone.utc),
        ))

        # Keep last 1000 values
        if len(self._values[name]) > 1000:
            self._values[name] = self._values[name][-1000:]

    def get_metric(self, name: str) -> Optional[List[MetricValue]]:
        """Get metric values."""
        full_name = f"{self.prefix}_{name}"
        return self._values.get(full_name)

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        full_name = f"{self.prefix}_{name}"
        key = self._make_key(full_name, labels)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        full_name = f"{self.prefix}_{name}"
        key = self._make_key(full_name, labels)
        return self._gauges.get(key, 0)

    def get_histogram_percentile(
        self,
        name: str,
        percentile: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Get histogram percentile value."""
        full_name = f"{self.prefix}_{name}"
        values = self._values.get(full_name, [])

        if labels:
            values = [v for v in values if v.labels == labels]

        if not values:
            return None

        sorted_values = sorted(v.value for v in values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, definition in self._definitions.items():
            lines.append(f"# HELP {name} {definition.description}")
            lines.append(f"# TYPE {name} {definition.metric_type.value}")

        for key, value in self._counters.items():
            lines.append(f"{key} {value}")

        for key, value in self._gauges.items():
            lines.append(f"{key} {value}")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "metrics_defined": len(self._definitions),
            "counters": len(self._counters),
            "gauges": len(self._gauges),
            "total_observations": sum(len(v) for v in self._values.values()),
        }


# Convenience decorators
def timed(collector: MetricsCollector, metric_name: str):
    """Decorator to time function execution."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            import time
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = (time.time() - start) * 1000
                collector.observe(metric_name, duration)

        def sync_wrapper(*args, **kwargs):
            import time
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.time() - start) * 1000
                collector.observe(metric_name, duration)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def counted(collector: MetricsCollector, metric_name: str):
    """Decorator to count function calls."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            collector.inc(metric_name)
            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            collector.inc(metric_name)
            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


__all__ = [
    "MetricType",
    "MetricDefinition",
    "MetricValue",
    "MetricsCollector",
    "timed",
    "counted",
]
