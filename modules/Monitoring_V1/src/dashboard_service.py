"""
Monitoring_V1 - Dashboard Service

Dashboard data aggregation and visualization support.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DashboardPanel:
    """Dashboard panel configuration"""
    panel_id: str
    title: str
    type: str  # graph, stat, table, gauge
    metric: str
    refresh_interval: int = 30
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    title: str
    panels: List[DashboardPanel] = field(default_factory=list)
    refresh_interval: int = 30
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dashboard_id": self.dashboard_id,
            "title": self.title,
            "panels": [
                {
                    "panel_id": p.panel_id,
                    "title": p.title,
                    "type": p.type,
                    "metric": p.metric,
                }
                for p in self.panels
            ],
            "refresh_interval": self.refresh_interval,
            "tags": self.tags,
        }


@dataclass
class TimeSeriesPoint:
    """Single time series data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class DashboardService:
    """
    Dashboard data service.

    Features:
    - Dashboard management
    - Time series data aggregation
    - Panel data queries
    """

    def __init__(self):
        self._dashboards: Dict[str, Dashboard] = {}
        self._time_series: Dict[str, List[TimeSeriesPoint]] = defaultdict(list)
        self._max_points = 1000

    def create_dashboard(
        self,
        dashboard_id: str,
        title: str,
        tags: List[str] = None,
    ) -> Dashboard:
        """Create a new dashboard"""
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            title=title,
            tags=tags or [],
        )

        self._dashboards[dashboard_id] = dashboard
        logger.info(f"Dashboard created: {dashboard_id}")

        return dashboard

    def add_panel(
        self,
        dashboard_id: str,
        panel_id: str,
        title: str,
        panel_type: str,
        metric: str,
        options: Dict[str, Any] = None,
    ):
        """Add panel to dashboard"""
        if dashboard_id not in self._dashboards:
            raise ValueError(f"Dashboard not found: {dashboard_id}")

        panel = DashboardPanel(
            panel_id=panel_id,
            title=title,
            type=panel_type,
            metric=metric,
            options=options or {},
        )

        self._dashboards[dashboard_id].panels.append(panel)

    def record_metric(
        self,
        metric: str,
        value: float,
        labels: Dict[str, str] = None,
    ):
        """Record metric data point"""
        point = TimeSeriesPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {},
        )

        series = self._time_series[metric]
        series.append(point)

        # Limit series length
        if len(series) > self._max_points:
            self._time_series[metric] = series[-self._max_points:]

    def query_metric(
        self,
        metric: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        labels: Dict[str, str] = None,
    ) -> List[Dict[str, Any]]:
        """Query metric time series"""
        series = self._time_series.get(metric, [])

        # Filter by time range
        if start:
            series = [p for p in series if p.timestamp >= start]
        if end:
            series = [p for p in series if p.timestamp <= end]

        # Filter by labels
        if labels:
            series = [
                p for p in series
                if all(p.labels.get(k) == v for k, v in labels.items())
            ]

        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "value": p.value,
                "labels": p.labels,
            }
            for p in series
        ]

    def get_panel_data(
        self,
        dashboard_id: str,
        panel_id: str,
        time_range_minutes: int = 60,
    ) -> Dict[str, Any]:
        """Get data for a specific panel"""
        if dashboard_id not in self._dashboards:
            return {}

        dashboard = self._dashboards[dashboard_id]
        panel = next((p for p in dashboard.panels if p.panel_id == panel_id), None)

        if not panel:
            return {}

        start = datetime.now(timezone.utc) - timedelta(minutes=time_range_minutes)
        data = self.query_metric(panel.metric, start=start)

        return {
            "panel": {
                "id": panel.panel_id,
                "title": panel.title,
                "type": panel.type,
            },
            "data": data,
        }

    def get_dashboard(self, dashboard_id: str) -> Optional[Dict]:
        """Get dashboard configuration"""
        dashboard = self._dashboards.get(dashboard_id)
        return dashboard.to_dict() if dashboard else None

    def list_dashboards(self) -> List[Dict]:
        """List all dashboards"""
        return [d.to_dict() for d in self._dashboards.values()]

    def delete_dashboard(self, dashboard_id: str):
        """Delete dashboard"""
        self._dashboards.pop(dashboard_id, None)

    def get_summary(self, metric: str, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get metric summary statistics"""
        start = datetime.now(timezone.utc) - timedelta(minutes=time_range_minutes)
        data = self.query_metric(metric, start=start)

        if not data:
            return {}

        values = [d["value"] for d in data]

        return {
            "metric": metric,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "last": values[-1] if values else None,
            "time_range_minutes": time_range_minutes,
        }
