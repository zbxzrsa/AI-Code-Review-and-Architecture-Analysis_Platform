"""
Data Analytics Business Logic Service

Handles analytics operations:
- Metrics aggregation
- Trend analysis
- Report generation
- Dashboard data

Module Size: ~180 lines (target < 2000)
"""

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
import random

from ..config import logger


class MetricType(str, Enum):
    """Metric types."""
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    TEST_COVERAGE = "test_coverage"


class TimeRange(str, Enum):
    """Time range for analytics."""
    DAY = "1d"
    WEEK = "7d"
    MONTH = "30d"
    QUARTER = "90d"
    YEAR = "365d"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: str
    value: float
    change_percent: Optional[float] = None


@dataclass
class TrendData:
    """Trend analysis data."""
    metric: MetricType
    current_value: float
    previous_value: float
    change_percent: float
    trend_direction: str  # up, down, stable
    data_points: List[MetricPoint]


class AnalyticsService:
    """
    Service for data analytics and reporting.
    
    Provides:
    - Metrics aggregation
    - Trend analysis
    - Report generation
    - Dashboard statistics
    """
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    async def get_dashboard_metrics(
        self,
        project_id: Optional[str] = None,
        time_range: TimeRange = TimeRange.WEEK,
    ) -> Dict[str, Any]:
        """
        Get dashboard metrics for a project or all projects.
        """
        # Generate mock metrics
        return {
            "summary": {
                "total_reviews": random.randint(100, 500),
                "total_issues": random.randint(50, 200),
                "resolved_issues": random.randint(30, 150),
                "avg_score": round(random.uniform(70, 95), 1),
                "security_score": round(random.uniform(80, 98), 1),
            },
            "trends": {
                "reviews_change": round(random.uniform(-10, 20), 1),
                "issues_change": round(random.uniform(-15, 10), 1),
                "score_change": round(random.uniform(-5, 10), 1),
            },
            "by_severity": {
                "critical": random.randint(0, 5),
                "high": random.randint(5, 20),
                "medium": random.randint(10, 50),
                "low": random.randint(20, 100),
            },
            "by_category": {
                "security": random.randint(10, 40),
                "performance": random.randint(15, 50),
                "maintainability": random.randint(20, 60),
                "style": random.randint(30, 80),
            },
            "time_range": time_range.value,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    async def get_trend_analysis(
        self,
        metric: MetricType,
        time_range: TimeRange = TimeRange.MONTH,
        project_id: Optional[str] = None,
    ) -> TrendData:
        """
        Get trend analysis for a specific metric.
        """
        # Generate mock trend data
        days = int(time_range.value.replace("d", ""))
        data_points = []
        base_value = random.uniform(60, 90)
        
        for i in range(min(days, 30)):
            date = datetime.now(timezone.utc) - timedelta(days=days - i)
            value = base_value + random.uniform(-5, 5)
            data_points.append(MetricPoint(
                timestamp=date.isoformat(),
                value=round(value, 1),
            ))
        
        current = data_points[-1].value if data_points else 0
        previous = data_points[0].value if data_points else 0
        change = ((current - previous) / previous * 100) if previous else 0
        
        return TrendData(
            metric=metric,
            current_value=current,
            previous_value=previous,
            change_percent=round(change, 1),
            trend_direction="up" if change > 2 else "down" if change < -2 else "stable",
            data_points=data_points,
        )
    
    async def generate_report(
        self,
        project_id: str,
        report_type: str = "summary",
        time_range: TimeRange = TimeRange.MONTH,
    ) -> Dict[str, Any]:
        """
        Generate an analytics report.
        """
        return {
            "report_id": f"report-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "project_id": project_id,
            "type": report_type,
            "time_range": time_range.value,
            "metrics": await self.get_dashboard_metrics(project_id, time_range),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
        }
    
    async def get_activity_feed(
        self,
        limit: int = 20,
        project_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent activity feed.
        """
        activities = []
        activity_types = [
            ("review", "Code review completed"),
            ("vulnerability", "Vulnerability detected"),
            ("fix", "Auto-fix applied"),
            ("deploy", "Deployment completed"),
            ("scan", "Security scan finished"),
        ]
        
        for i in range(min(limit, 20)):
            activity_type, description = random.choice(activity_types)
            activities.append({
                "id": f"activity-{i+1:03d}",
                "type": activity_type,
                "description": description,
                "timestamp": (datetime.now(timezone.utc) - timedelta(hours=i)).isoformat(),
                "user": f"user-{random.randint(1, 5):03d}",
                "project_id": project_id or f"project-{random.randint(1, 3):03d}",
            })
        
        return activities
    
    def clear_cache(self):
        """Clear analytics cache."""
        self._cache = {}
        logger.info("Analytics cache cleared")
