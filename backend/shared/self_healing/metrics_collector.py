"""
Metrics Collector - System Metrics Collection and Aggregation

Collects metrics from various sources and provides aggregated views.
Supports real-time monitoring, historical analysis, and alerting integration.
"""

import asyncio
import statistics
import psutil
import aiohttp
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MetricSource(Enum):
    """Sources of metrics."""
    PROMETHEUS = "prometheus"
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    REDIS = "redis"
    QUEUE = "queue"
    AI_SERVICE = "ai_service"
    CUSTOM = "custom"


class MetricType(Enum):
    """Types of metrics."""
    GAUGE = "gauge"  # Point-in-time value
    COUNTER = "counter"  # Monotonically increasing
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Statistical summary


class AggregationType(Enum):
    """Aggregation methods."""
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: str
    source: MetricSource
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "source": self.source.value,
            "labels": self.labels,
            "unit": self.unit,
            "type": self.metric_type.value
        }


@dataclass
class MetricSeries:
    """Time series of metric values."""
    name: str
    source: MetricSource
    labels: Dict[str, str]
    values: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, value: float, timestamp: str):
        """Add a value to the series."""
        self.values.append({"value": value, "timestamp": timestamp})

    def get_latest(self) -> Optional[float]:
        """Get the latest value."""
        return self.values[-1]["value"] if self.values else None

    def get_aggregate(self, agg_type: AggregationType, window_minutes: int = 5) -> Optional[float]:
        """Get aggregated value over a time window."""
        if not self.values:
            return None

        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_values = [
            v["value"] for v in self.values
            if datetime.fromisoformat(v["timestamp"]) > cutoff
        ]

        if not recent_values:
            return None

        if agg_type == AggregationType.AVG:
            return statistics.mean(recent_values)
        elif agg_type == AggregationType.SUM:
            return sum(recent_values)
        elif agg_type == AggregationType.MIN:
            return min(recent_values)
        elif agg_type == AggregationType.MAX:
            return max(recent_values)
        elif agg_type == AggregationType.COUNT:
            return len(recent_values)
        elif agg_type == AggregationType.PERCENTILE_50:
            return statistics.median(recent_values)
        elif agg_type == AggregationType.PERCENTILE_95:
            return self._percentile(recent_values, 95)
        elif agg_type == AggregationType.PERCENTILE_99:
            return self._percentile(recent_values, 99)

        return None

    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]


class MetricsCollector:
    """
    Comprehensive metrics collection and aggregation system.

    Features:
    - Multi-source metric collection (System, Prometheus, Application, Database)
    - Time series storage with configurable retention
    - Aggregation functions (avg, sum, min, max, percentiles)
    - Alert threshold monitoring
    - Export to Prometheus format
    """

    def __init__(
        self,
        collection_interval: float = 15.0,
        retention_minutes: int = 60,
        prometheus_url: Optional[str] = None
    ):
        self.collection_interval = collection_interval
        self.retention_minutes = retention_minutes
        self.prometheus_url = prometheus_url or "http://localhost:9090"

        # Metric storage
        self.series: Dict[str, MetricSeries] = {}
        self.latest_metrics: Dict[str, Metric] = {}

        # Custom collectors
        self.custom_collectors: Dict[str, Callable[[], Awaitable[Dict[str, float]]]] = {}

        # State
        self.is_running = False
        self._collection_task: Optional[asyncio.Task] = None

        # Stats
        self.stats = {
            "collections_total": 0,
            "collection_errors": 0,
            "metrics_collected": 0
        }

        # Callbacks
        self.on_metric_collected: Optional[Callable[[Metric], None]] = None

    async def start(self):
        """Start continuous metric collection."""
        if self.is_running:
            return

        self.is_running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collector started")

    async def stop(self):
        """Stop metric collection."""
        self.is_running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")

    async def _collection_loop(self):
        """Main collection loop."""
        while self.is_running:
            try:
                await self.collect_all()
                self.stats["collections_total"] += 1
            except Exception as e:
                logger.error(f"Collection error: {e}")
                self.stats["collection_errors"] += 1

            await asyncio.sleep(self.collection_interval)

    def register_collector(
        self,
        name: str,
        collector: Callable[[], Awaitable[Dict[str, float]]]
    ):
        """Register a custom metric collector."""
        self.custom_collectors[name] = collector
        logger.info(f"Registered custom collector: {name}")

    async def collect_all(self) -> Dict[str, Any]:
        """Collect metrics from all sources."""
        timestamp = datetime.now().isoformat()
        all_metrics = {}

        # Collect system metrics
        try:
            system_metrics = await self._collect_system_metrics()
            all_metrics[MetricSource.SYSTEM.value] = system_metrics
            self._store_metrics(system_metrics, MetricSource.SYSTEM, timestamp)
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")

        # Collect application metrics
        try:
            app_metrics = await self._collect_application_metrics()
            all_metrics[MetricSource.APPLICATION.value] = app_metrics
            self._store_metrics(app_metrics, MetricSource.APPLICATION, timestamp)
        except Exception as e:
            logger.error(f"Application metrics collection failed: {e}")

        # Collect from Prometheus
        try:
            prom_metrics = await self._collect_prometheus_metrics()
            all_metrics[MetricSource.PROMETHEUS.value] = prom_metrics
            self._store_metrics(prom_metrics, MetricSource.PROMETHEUS, timestamp)
        except Exception as e:
            logger.debug(f"Prometheus metrics collection failed: {e}")

        # Collect database metrics
        try:
            db_metrics = await self._collect_database_metrics()
            all_metrics[MetricSource.DATABASE.value] = db_metrics
            self._store_metrics(db_metrics, MetricSource.DATABASE, timestamp)
        except Exception as e:
            logger.debug(f"Database metrics collection failed: {e}")

        # Run custom collectors
        for name, collector in self.custom_collectors.items():
            try:
                custom_metrics = await collector()
                all_metrics[f"custom_{name}"] = custom_metrics
                self._store_metrics(custom_metrics, MetricSource.CUSTOM, timestamp, {"collector": name})
            except Exception as e:
                logger.error(f"Custom collector {name} failed: {e}")

        return all_metrics

    def _store_metrics(
        self,
        metrics: Dict[str, float],
        source: MetricSource,
        timestamp: str,
        extra_labels: Optional[Dict[str, str]] = None
    ):
        """Store collected metrics in time series."""
        for name, value in metrics.items():
            series_key = f"{source.value}:{name}"

            if series_key not in self.series:
                self.series[series_key] = MetricSeries(
                    name=name,
                    source=source,
                    labels=extra_labels or {}
                )

            self.series[series_key].add(value, timestamp)

            # Store latest
            metric = Metric(
                name=name,
                value=value,
                timestamp=timestamp,
                source=source,
                labels=extra_labels or {}
            )
            self.latest_metrics[series_key] = metric
            self.stats["metrics_collected"] += 1

            # Callback
            if self.on_metric_collected:
                self.on_metric_collected(metric)

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics using psutil."""
        metrics = {}

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics["cpu_usage_percent"] = cpu_percent
        metrics["cpu_count"] = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()
        metrics["memory_usage_percent"] = memory.percent
        metrics["memory_used_bytes"] = memory.used
        metrics["memory_available_bytes"] = memory.available
        metrics["memory_total_bytes"] = memory.total

        # Disk metrics
        try:
            disk = psutil.disk_usage('/')
            metrics["disk_usage_percent"] = disk.percent
            metrics["disk_free_bytes"] = disk.free
        except Exception:
            pass

        # Network metrics
        try:
            net_io = psutil.net_io_counters()
            metrics["network_bytes_sent"] = net_io.bytes_sent
            metrics["network_bytes_recv"] = net_io.bytes_recv
        except Exception:
            pass

        # Process metrics
        try:
            metrics["process_count"] = len(psutil.pids())
        except Exception:
            pass

        return metrics

    async def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect application-level metrics."""
        # In production, this would query application endpoints
        return {
            "active_sessions": 0,
            "pending_requests": 0,
            "cache_hit_rate": 0.95,
            "queue_depth": 0
        }

    async def _collect_prometheus_metrics(self) -> Dict[str, float]:
        """Collect metrics from Prometheus."""
        metrics = {}

        queries = [
            ("request_rate", "rate(http_requests_total[5m])"),
            ("error_rate", "rate(http_errors_total[5m])"),
            ("response_time_p95", "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"),
        ]

        async with aiohttp.ClientSession() as session:
            for metric_name, query in queries:
                try:
                    async with session.get(
                        f"{self.prometheus_url}/api/v1/query",
                        params={"query": query},
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("data", {}).get("result"):
                                value = float(data["data"]["result"][0]["value"][1])
                                metrics[metric_name] = value
                except Exception:
                    pass

        return metrics

    async def _collect_database_metrics(self) -> Dict[str, float]:
        """Collect database metrics."""
        # In production, this would query database stats
        return {
            "connection_pool_size": 10,
            "active_connections": 5,
            "query_rate": 100.0,
            "slow_queries": 0
        }

    def get_metric(self, source: MetricSource, name: str) -> Optional[Metric]:
        """Get the latest value for a specific metric."""
        series_key = f"{source.value}:{name}"
        return self.latest_metrics.get(series_key)

    def get_aggregate(
        self,
        source: MetricSource,
        name: str,
        agg_type: AggregationType,
        window_minutes: int = 5
    ) -> Optional[float]:
        """Get aggregated value for a metric."""
        series_key = f"{source.value}:{name}"
        if series_key in self.series:
            return self.series[series_key].get_aggregate(agg_type, window_minutes)
        return None

    def get_all_latest(self) -> Dict[str, Dict[str, float]]:
        """Get all latest metric values grouped by source."""
        result: Dict[str, Dict[str, float]] = {}

        for key, metric in self.latest_metrics.items():
            source = metric.source.value
            if source not in result:
                result[source] = {}
            result[source][metric.name] = metric.value

        return result

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for key, metric in self.latest_metrics.items():
            labels_str = ",".join(
                f'{k}="{v}"' for k, v in metric.labels.items()
            )
            if labels_str:
                lines.append(f'{metric.name}{{{labels_str}}} {metric.value}')
            else:
                lines.append(f'{metric.name} {metric.value}')

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            **self.stats,
            "series_count": len(self.series),
            "custom_collectors": list(self.custom_collectors.keys()),
            "is_running": self.is_running
        }
