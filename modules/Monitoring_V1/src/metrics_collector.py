"""
Monitoring_V1 - Metrics Collector

Prometheus-compatible metrics collection.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Single metric value"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition"""
    name: str
    type: MetricType
    help: str
    labels: List[str] = field(default_factory=list)
    values: List[MetricValue] = field(default_factory=list)

    # For histograms
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])


class MetricsCollector:
    """
    Prometheus-compatible metrics collector.

    Metric Types:
    - Counter: Monotonically increasing
    - Gauge: Can go up or down
    - Histogram: Distribution buckets
    - Summary: Quantile distribution
    """

    def __init__(self, prefix: str = "app"):
        self.prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)

    def register_counter(self, name: str, help: str, labels: List[str] = None):
        """Register a counter metric"""
        full_name = f"{self.prefix}_{name}"
        self._metrics[full_name] = Metric(
            name=full_name,
            type=MetricType.COUNTER,
            help=help,
            labels=labels or [],
        )

    def register_gauge(self, name: str, help: str, labels: List[str] = None):
        """Register a gauge metric"""
        full_name = f"{self.prefix}_{name}"
        self._metrics[full_name] = Metric(
            name=full_name,
            type=MetricType.GAUGE,
            help=help,
            labels=labels or [],
        )

    def register_histogram(
        self,
        name: str,
        help: str,
        labels: List[str] = None,
        buckets: List[float] = None,
    ):
        """Register a histogram metric"""
        full_name = f"{self.prefix}_{name}"
        self._metrics[full_name] = Metric(
            name=full_name,
            type=MetricType.HISTOGRAM,
            help=help,
            labels=labels or [],
            buckets=buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
        )

    def inc(self, name: str, value: float = 1, labels: Dict[str, str] = None):
        """Increment counter"""
        key = self._make_key(name, labels)
        self._counters[key] += value

    def set(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge value"""
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def observe(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observe histogram value"""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)

    def timer(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing"""
        return Timer(self, name, labels)

    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create metric key with labels"""
        full_name = f"{self.prefix}_{name}"
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{full_name}{{{label_str}}}"
        return full_name

    def get_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get current metric value"""
        key = self._make_key(name, labels)

        if key in self._counters:
            return self._counters[key]
        if key in self._gauges:
            return self._gauges[key]

        return None

    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "sum": sum(values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(values) / n,
            "p50": sorted_values[int(n * 0.5)],
            "p90": sorted_values[int(n * 0.9)],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        for name, metric in self._metrics.items():
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} {metric.type.value}")

        # Export counters
        for key, value in self._counters.items():
            lines.append(f"{key} {value}")

        # Export gauges
        for key, value in self._gauges.items():
            lines.append(f"{key} {value}")

        # Export histogram summaries
        for key, values in self._histograms.items():
            if values:
                stats = self.get_histogram_stats(key.split("{")[0].replace(f"{self.prefix}_", ""))
                lines.append(f"{key}_count {stats['count']}")
                lines.append(f"{key}_sum {stats['sum']}")

        return "\n".join(lines)

    def reset(self):
        """Reset all metrics"""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


class Timer:
    """Context manager for timing operations"""

    def __init__(self, collector: MetricsCollector, name: str, labels: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.observe(self.name, duration, self.labels)
