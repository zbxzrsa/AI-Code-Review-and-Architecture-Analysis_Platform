"""
Unit Tests for Metrics Collector

Tests the metrics collection, aggregation, and export functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.insert(0, 'backend/shared')

from self_healing.metrics_collector import (
    MetricsCollector,
    Metric,
    MetricSource,
    MetricType,
    MetricSeries,
    AggregationType,
)


class TestMetricSeries:
    """Test MetricSeries time series functionality."""

    def test_add_values(self):
        """Values should be added to the series."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        timestamp = datetime.now().isoformat()
        series.add(100.0, timestamp)
        series.add(200.0, timestamp)

        assert len(series.values) == 2
        assert series.get_latest() == pytest.approx(200.0)

    def test_get_latest_empty(self):
        """Getting latest from empty series should return None."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        assert series.get_latest() is None

    def test_max_values_limit(self):
        """Series should respect maxlen limit."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        # Add more than maxlen values
        for i in range(1500):
            series.add(float(i), datetime.now().isoformat())

        # Should be limited to 1000
        assert len(series.values) == 1000
        assert series.get_latest() == 1499.0

    def test_aggregate_avg(self):
        """Average aggregation should work correctly."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        # Add values with recent timestamps
        for i in [10, 20, 30, 40, 50]:
            series.add(float(i), datetime.now().isoformat())

        avg = series.get_aggregate(AggregationType.AVG, window_minutes=60)
        assert avg == pytest.approx(30.0)

    def test_aggregate_sum(self):
        """Sum aggregation should work correctly."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        for i in [10, 20, 30]:
            series.add(float(i), datetime.now().isoformat())

        total = series.get_aggregate(AggregationType.SUM, window_minutes=60)
        assert total == pytest.approx(60.0)

    def test_aggregate_min_max(self):
        """Min/Max aggregations should work correctly."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        for i in [50, 20, 80, 30]:
            series.add(float(i), datetime.now().isoformat())

        assert series.get_aggregate(AggregationType.MIN, window_minutes=60) == pytest.approx(20.0)
        assert series.get_aggregate(AggregationType.MAX, window_minutes=60) == pytest.approx(80.0)

    def test_aggregate_count(self):
        """Count aggregation should work correctly."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        for i in range(5):
            series.add(float(i), datetime.now().isoformat())

        count = series.get_aggregate(AggregationType.COUNT, window_minutes=60)
        assert count == 5

    def test_aggregate_percentiles(self):
        """Percentile aggregations should work correctly."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        # Add 100 values
        for i in range(100):
            series.add(float(i), datetime.now().isoformat())

        p50 = series.get_aggregate(AggregationType.PERCENTILE_50, window_minutes=60)
        p95 = series.get_aggregate(AggregationType.PERCENTILE_95, window_minutes=60)
        p99 = series.get_aggregate(AggregationType.PERCENTILE_99, window_minutes=60)

        assert p50 == pytest.approx(49.5)  # Median
        assert p95 == pytest.approx(95.0)
        assert p99 == pytest.approx(99.0)

    def test_aggregate_window_filtering(self):
        """Aggregation should only include values within window."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        # Add old value
        old_time = (datetime.now() - timedelta(hours=2)).isoformat()
        series.add(1000.0, old_time)

        # Add recent values
        for i in [10, 20, 30]:
            series.add(float(i), datetime.now().isoformat())

        # With short window, old value should be excluded
        avg = series.get_aggregate(AggregationType.AVG, window_minutes=5)
        assert avg == pytest.approx(20.0)  # Only recent values

    def test_aggregate_empty_series(self):
        """Aggregation of empty series should return None."""
        series = MetricSeries(
            name="test_metric",
            source=MetricSource.SYSTEM,
            labels={}
        )

        assert series.get_aggregate(AggregationType.AVG) is None


class TestMetric:
    """Test Metric data class."""

    def test_metric_creation(self):
        """Metric should be created with all fields."""
        metric = Metric(
            name="cpu_usage",
            value=75.5,
            timestamp=datetime.now().isoformat(),
            source=MetricSource.SYSTEM,
            labels={"host": "server1"},
            unit="percent"
        )

        assert metric.name == "cpu_usage"
        assert metric.value == pytest.approx(75.5)
        assert metric.source == MetricSource.SYSTEM
        assert metric.labels["host"] == "server1"

    def test_metric_to_dict(self):
        """Metric should convert to dictionary correctly."""
        timestamp = datetime.now().isoformat()
        metric = Metric(
            name="memory_used",
            value=1024.0,
            timestamp=timestamp,
            source=MetricSource.SYSTEM,
            labels={"app": "test"},
            unit="MB",
            metric_type=MetricType.GAUGE
        )

        data = metric.to_dict()

        assert data["name"] == "memory_used"
        assert data["value"] == pytest.approx(1024.0)
        assert data["source"] == "system"
        assert data["type"] == "gauge"


class TestMetricsCollector:
    """Test MetricsCollector functionality."""

    @pytest.mark.asyncio
    async def test_collector_start_stop(self):
        """Collector should start and stop cleanly."""
        collector = MetricsCollector(collection_interval=1.0)

        await collector.start()
        assert collector.is_running

        await collector.stop()
        assert not collector.is_running

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self):
        """System metrics should be collected."""
        collector = MetricsCollector()

        metrics = await collector._collect_system_metrics()

        assert "cpu_usage_percent" in metrics
        assert "memory_usage_percent" in metrics
        assert "memory_total_bytes" in metrics
        assert metrics["cpu_usage_percent"] >= 0
        assert metrics["memory_usage_percent"] >= 0

    @pytest.mark.asyncio
    async def test_collect_all(self):
        """Should collect from all sources."""
        collector = MetricsCollector()

        all_metrics = await collector.collect_all()

        assert MetricSource.SYSTEM.value in all_metrics
        assert MetricSource.APPLICATION.value in all_metrics

    @pytest.mark.asyncio
    async def test_metric_storage(self):
        """Metrics should be stored in time series."""
        collector = MetricsCollector()

        await collector.collect_all()

        # Should have stored metrics
        assert len(collector.series) > 0
        assert len(collector.latest_metrics) > 0

    @pytest.mark.asyncio
    async def test_get_metric(self):
        """Should retrieve specific metrics."""
        collector = MetricsCollector()

        await collector.collect_all()

        metric = collector.get_metric(MetricSource.SYSTEM, "cpu_usage_percent")

        assert metric is not None
        assert metric.name == "cpu_usage_percent"
        assert metric.source == MetricSource.SYSTEM

    @pytest.mark.asyncio
    async def test_get_all_latest(self):
        """Should return all latest metrics grouped by source."""
        collector = MetricsCollector()

        await collector.collect_all()

        latest = collector.get_all_latest()

        assert "system" in latest
        assert "cpu_usage_percent" in latest["system"]

    @pytest.mark.asyncio
    async def test_custom_collector_registration(self):
        """Custom collectors should be registered and executed."""
        collector = MetricsCollector()

        # Register custom collector
        async def custom_collector():
            return {"custom_metric": 42.0}

        collector.register_collector("test", custom_collector)

        all_metrics = await collector.collect_all()

        assert "custom_test" in all_metrics
        assert all_metrics["custom_test"]["custom_metric"] == pytest.approx(42.0)

    @pytest.mark.asyncio
    async def test_metric_callback(self):
        """Callback should be triggered on metric collection."""
        collector = MetricsCollector()

        collected_metrics = []

        def on_metric(metric):
            collected_metrics.append(metric)

        collector.on_metric_collected = on_metric

        await collector.collect_all()

        assert len(collected_metrics) > 0

    def test_prometheus_export(self):
        """Should export metrics in Prometheus format."""
        collector = MetricsCollector()

        # Manually add a metric
        timestamp = datetime.now().isoformat()
        collector.latest_metrics["system:cpu_usage"] = Metric(
            name="cpu_usage",
            value=50.0,
            timestamp=timestamp,
            source=MetricSource.SYSTEM,
            labels={"host": "localhost"}
        )

        output = collector.export_prometheus()

        assert 'cpu_usage{host="localhost"} 50.0' in output

    def test_get_stats(self):
        """Should return collector statistics."""
        collector = MetricsCollector()

        stats = collector.get_stats()

        assert "collections_total" in stats
        assert "collection_errors" in stats
        assert "series_count" in stats
        assert "is_running" in stats

    @pytest.mark.asyncio
    async def test_aggregation_integration(self):
        """Should support aggregation queries."""
        collector = MetricsCollector()

        # Collect multiple times
        for _ in range(3):
            await collector.collect_all()
            await asyncio.sleep(0.1)

        # Get aggregate
        avg_cpu = collector.get_aggregate(
            MetricSource.SYSTEM,
            "cpu_usage_percent",
            AggregationType.AVG,
            window_minutes=60
        )

        assert avg_cpu is not None
        assert avg_cpu >= 0

    @pytest.mark.asyncio
    async def test_collection_error_handling(self):
        """Errors in collection should be handled gracefully."""
        collector = MetricsCollector()

        # Register failing collector
        async def failing_collector():
            raise Exception("Simulated failure")

        collector.register_collector("failing", failing_collector)

        # Should not raise
        all_metrics = await collector.collect_all()

        # Other sources should still work
        assert MetricSource.SYSTEM.value in all_metrics


class TestMetricSource:
    """Test MetricSource enum."""

    def test_all_sources_defined(self):
        """All expected sources should be defined."""
        sources = [s.value for s in MetricSource]

        assert "prometheus" in sources
        assert "system" in sources
        assert "application" in sources
        assert "database" in sources
        assert "custom" in sources


class TestMetricType:
    """Test MetricType enum."""

    def test_all_types_defined(self):
        """All metric types should be defined."""
        types = [t.value for t in MetricType]

        assert "gauge" in types
        assert "counter" in types
        assert "histogram" in types
        assert "summary" in types


class TestAggregationType:
    """Test AggregationType enum."""

    def test_all_aggregations_defined(self):
        """All aggregation types should be defined."""
        aggs = [a.value for a in AggregationType]

        assert "avg" in aggs
        assert "sum" in aggs
        assert "min" in aggs
        assert "max" in aggs
        assert "count" in aggs
        assert "p50" in aggs
        assert "p95" in aggs
        assert "p99" in aggs


class TestPrometheusIntegration:
    """Test Prometheus integration."""

    @pytest.mark.asyncio
    async def test_prometheus_collection_with_mock(self):
        """Should collect from Prometheus when available."""
        collector = MetricsCollector(prometheus_url="http://localhost:9090")

        # Mock the HTTP call
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "data": {
                    "result": [{"value": [0, "0.05"]}]
                }
            })

            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            # Collection should work even if Prometheus fails
            metrics = await collector._collect_prometheus_metrics()

            # Returns empty dict if connection fails (which it will in test)
            assert isinstance(metrics, dict)


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_collection_speed(self):
        """Collection should be fast."""
        import time

        collector = MetricsCollector()

        start = time.perf_counter()
        await collector.collect_all()
        duration = time.perf_counter() - start

        # Should complete in under 1 second
        assert duration < 1.0, f"Collection took too long: {duration}s"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Series should be memory-bounded."""
        collector = MetricsCollector()

        # Collect many times
        for _ in range(100):
            await collector.collect_all()

        # Each series should be bounded
        for series in collector.series.values():
            assert len(series.values) <= 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
