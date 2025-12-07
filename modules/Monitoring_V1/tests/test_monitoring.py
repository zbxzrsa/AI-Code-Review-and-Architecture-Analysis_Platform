"""Tests for Monitoring_V1"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics_collector import MetricsCollector
from src.alert_manager import AlertManager, AlertRule, AlertSeverity
from src.dashboard_service import DashboardService


class TestMetricsCollector:
    @pytest.fixture
    def metrics(self):
        return MetricsCollector()

    def test_counter(self, metrics):
        metrics.register_counter("test_counter", "Test counter")
        metrics.inc("test_counter")
        metrics.inc("test_counter", 5)

        assert metrics.get_value("test_counter") == 6

    def test_gauge(self, metrics):
        metrics.register_gauge("test_gauge", "Test gauge")
        metrics.set("test_gauge", 42)

        assert metrics.get_value("test_gauge") == 42

    def test_histogram(self, metrics):
        metrics.register_histogram("test_histogram", "Test histogram")
        metrics.observe("test_histogram", 0.1)
        metrics.observe("test_histogram", 0.2)
        metrics.observe("test_histogram", 0.3)

        stats = metrics.get_histogram_stats("test_histogram")
        assert stats["count"] == 3
        assert stats["avg"] == pytest.approx(0.2, rel=0.01)

    def test_labels(self, metrics):
        metrics.register_counter("labeled_counter", "Counter with labels", ["method"])
        metrics.inc("labeled_counter", labels={"method": "GET"})
        metrics.inc("labeled_counter", labels={"method": "POST"})

        assert metrics.get_value("labeled_counter", {"method": "GET"}) == 1

    def test_export(self, metrics):
        metrics.register_counter("export_test", "Export test")
        metrics.inc("export_test")

        output = metrics.export_prometheus()
        assert "export_test" in output


class TestAlertManager:
    @pytest.fixture
    def alerts(self):
        return AlertManager()

    def test_add_rule(self, alerts):
        rule = AlertRule(
            name="test_alert",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
            message="Test alert"
        )
        alerts.add_rule(rule)

        assert "test_alert" in alerts._rules

    def test_fire_alert(self, alerts):
        rule = AlertRule(
            name="fire_test",
            condition=lambda: True,
            severity=AlertSeverity.ERROR,
            message="Fire test"
        )
        alerts.add_rule(rule)
        alerts.evaluate()

        active = alerts.get_active_alerts()
        assert len(active) == 1
        assert active[0].rule_name == "fire_test"

    def test_resolve_alert(self, alerts):
        condition_value = [True]
        rule = AlertRule(
            name="resolve_test",
            condition=lambda: condition_value[0],
            severity=AlertSeverity.WARNING,
            message="Resolve test"
        )
        alerts.add_rule(rule)

        alerts.evaluate()  # Fire
        assert len(alerts.get_active_alerts()) == 1

        condition_value[0] = False
        alerts.evaluate()  # Resolve
        assert len(alerts.get_active_alerts()) == 0


class TestDashboardService:
    @pytest.fixture
    def dashboards(self):
        return DashboardService()

    def test_create_dashboard(self, dashboards):
        dashboard = dashboards.create_dashboard("test-dash", "Test Dashboard")
        assert dashboard.dashboard_id == "test-dash"

    def test_add_panel(self, dashboards):
        dashboards.create_dashboard("panel-test", "Panel Test")
        dashboards.add_panel("panel-test", "panel-1", "CPU", "graph", "cpu_usage")

        dash = dashboards.get_dashboard("panel-test")
        assert len(dash["panels"]) == 1

    def test_record_and_query(self, dashboards):
        dashboards.record_metric("test_metric", 42)
        dashboards.record_metric("test_metric", 43)

        data = dashboards.query_metric("test_metric")
        assert len(data) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
