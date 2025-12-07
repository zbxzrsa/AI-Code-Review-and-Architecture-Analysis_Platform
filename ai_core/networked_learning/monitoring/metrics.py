"""
Metrics Collection

Collects and exposes system metrics for monitoring.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """
    System metrics snapshot.

    Performance indicators:
    - Data processing latency < 500ms
    - System availability > 99.9%
    - Maximum daily processing capacity >= 1TB
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Collection metrics
    items_collected: int = 0
    items_processed: int = 0
    items_filtered: int = 0
    collection_errors: int = 0

    # Quality metrics
    avg_quality_score: float = 0.0
    quality_pass_rate: float = 0.0
    duplicate_rate: float = 0.0

    # Performance metrics
    avg_processing_latency_ms: float = 0.0
    max_processing_latency_ms: float = 0.0
    throughput_items_per_sec: float = 0.0

    # Storage metrics
    cache_hit_rate: float = 0.0
    storage_size_gb: float = 0.0
    memory_usage_percent: float = 0.0

    # System health
    uptime_seconds: float = 0.0
    availability_percent: float = 100.0
    active_collectors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "collection": {
                "items_collected": self.items_collected,
                "items_processed": self.items_processed,
                "items_filtered": self.items_filtered,
                "errors": self.collection_errors,
            },
            "quality": {
                "avg_score": round(self.avg_quality_score, 3),
                "pass_rate": round(self.quality_pass_rate * 100, 1),
                "duplicate_rate": round(self.duplicate_rate * 100, 1),
            },
            "performance": {
                "avg_latency_ms": round(self.avg_processing_latency_ms, 2),
                "max_latency_ms": round(self.max_processing_latency_ms, 2),
                "throughput_per_sec": round(self.throughput_items_per_sec, 2),
            },
            "storage": {
                "cache_hit_rate": round(self.cache_hit_rate * 100, 1),
                "storage_gb": round(self.storage_size_gb, 2),
                "memory_percent": round(self.memory_usage_percent, 1),
            },
            "health": {
                "uptime_seconds": round(self.uptime_seconds, 0),
                "availability": round(self.availability_percent, 2),
                "active_collectors": self.active_collectors,
            },
        }

    def to_prometheus(self) -> str:
        """Export as Prometheus format."""
        lines = [
            "# HELP networked_learning_items_collected Total items collected",
            "# TYPE networked_learning_items_collected counter",
            f"networked_learning_items_collected {self.items_collected}",
            "",
            "# HELP networked_learning_items_processed Total items processed",
            "# TYPE networked_learning_items_processed counter",
            f"networked_learning_items_processed {self.items_processed}",
            "",
            "# HELP networked_learning_quality_score Average quality score",
            "# TYPE networked_learning_quality_score gauge",
            f"networked_learning_quality_score {self.avg_quality_score}",
            "",
            "# HELP networked_learning_latency_ms Processing latency",
            "# TYPE networked_learning_latency_ms gauge",
            f"networked_learning_latency_ms {self.avg_processing_latency_ms}",
            "",
            "# HELP networked_learning_cache_hit_rate Cache hit rate",
            "# TYPE networked_learning_cache_hit_rate gauge",
            f"networked_learning_cache_hit_rate {self.cache_hit_rate}",
            "",
            "# HELP networked_learning_availability System availability",
            "# TYPE networked_learning_availability gauge",
            f"networked_learning_availability {self.availability_percent}",
        ]
        return "\n".join(lines)


class MetricsCollector:
    """
    Collects and aggregates system metrics.

    Features:
    - Real-time metric collection
    - Aggregation over time windows
    - Prometheus export
    - SLA tracking
    """

    def __init__(self, interval_seconds: int = 10):
        """
        Initialize metrics collector.

        Args:
            interval_seconds: Metrics collection interval
        """
        self.interval_seconds = interval_seconds
        self._start_time = time.time()

        # Counters
        self._items_collected = 0
        self._items_processed = 0
        self._items_filtered = 0
        self._collection_errors = 0

        # Latency tracking
        self._latencies: List[float] = []
        self._max_latency_window = 1000  # Keep last 1000 measurements

        # Quality tracking
        self._quality_scores: List[float] = []
        self._max_quality_window = 1000

        # Availability tracking
        self._health_checks: List[bool] = []
        self._max_health_window = 1000

    def record_collection(
        self,
        items_collected: int,
        items_filtered: int = 0,
        errors: int = 0,
    ):
        """Record collection cycle metrics."""
        self._items_collected += items_collected
        self._items_filtered += items_filtered
        self._collection_errors += errors

    def record_processing(
        self,
        items_processed: int,
        latency_ms: float,
        quality_scores: List[float] = None,
    ):
        """Record processing metrics."""
        self._items_processed += items_processed

        # Track latency
        self._latencies.append(latency_ms)
        if len(self._latencies) > self._max_latency_window:
            self._latencies.pop(0)

        # Track quality
        if quality_scores:
            self._quality_scores.extend(quality_scores)
            while len(self._quality_scores) > self._max_quality_window:
                self._quality_scores.pop(0)

    def record_health_check(self, healthy: bool):
        """Record health check result."""
        self._health_checks.append(healthy)
        if len(self._health_checks) > self._max_health_window:
            self._health_checks.pop(0)

    def get_metrics(self) -> SystemMetrics:
        """Get current metrics snapshot."""
        now = datetime.now(timezone.utc)
        uptime = time.time() - self._start_time

        # Calculate averages
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0
        max_latency = max(self._latencies) if self._latencies else 0
        avg_quality = sum(self._quality_scores) / len(self._quality_scores) if self._quality_scores else 0

        # Calculate rates
        total_input = self._items_collected + self._items_filtered
        quality_pass_rate = self._items_processed / total_input if total_input > 0 else 0
        duplicate_rate = self._items_filtered / total_input if total_input > 0 else 0

        # Calculate availability
        healthy_count = sum(self._health_checks)
        availability = healthy_count / len(self._health_checks) * 100 if self._health_checks else 100

        # Calculate throughput
        throughput = self._items_processed / uptime if uptime > 0 else 0

        return SystemMetrics(
            timestamp=now,
            items_collected=self._items_collected,
            items_processed=self._items_processed,
            items_filtered=self._items_filtered,
            collection_errors=self._collection_errors,
            avg_quality_score=avg_quality,
            quality_pass_rate=quality_pass_rate,
            duplicate_rate=duplicate_rate,
            avg_processing_latency_ms=avg_latency,
            max_processing_latency_ms=max_latency,
            throughput_items_per_sec=throughput,
            uptime_seconds=uptime,
            availability_percent=availability,
        )

    def reset(self):
        """Reset all counters."""
        self._items_collected = 0
        self._items_processed = 0
        self._items_filtered = 0
        self._collection_errors = 0
        self._latencies.clear()
        self._quality_scores.clear()
        self._health_checks.clear()
        self._start_time = time.time()
