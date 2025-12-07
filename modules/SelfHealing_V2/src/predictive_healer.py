"""
SelfHealing_V2 - Predictive Healer

ML-based predictive failure detection and proactive healing.
"""

import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Time-series health metric"""
    timestamp: datetime
    value: float
    service: str
    metric_name: str


@dataclass
class PredictionResult:
    """Failure prediction result"""
    service: str
    risk_level: RiskLevel
    predicted_failure_time: Optional[datetime]
    confidence: float
    contributing_factors: List[str]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "risk_level": self.risk_level.value,
            "predicted_failure_time": self.predicted_failure_time.isoformat() if self.predicted_failure_time else None,
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors,
            "recommended_actions": self.recommended_actions,
            "timestamp": self.timestamp.isoformat(),
        }


class MetricWindow:
    """Sliding window for metric analysis"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._values: deque = deque(maxlen=window_size)

    def add(self, value: float):
        self._values.append(value)

    def mean(self) -> float:
        return statistics.mean(self._values) if self._values else 0

    def std(self) -> float:
        return statistics.stdev(self._values) if len(self._values) > 1 else 0

    def trend(self) -> float:
        """Calculate trend (positive = increasing)"""
        if len(self._values) < 2:
            return 0

        values = list(self._values)
        n = len(values)

        # Simple linear regression slope
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator else 0

    def is_anomaly(self, value: float, threshold: float = 3.0) -> bool:
        """Check if value is anomalous (z-score based)"""
        if len(self._values) < 10:
            return False

        mean = self.mean()
        std = self.std()

        if std == 0:
            return False

        z_score = abs(value - mean) / std
        return z_score > threshold


class PredictiveHealer:
    """
    Predictive failure detection and proactive healing.

    V2 Features:
    - Time-series analysis
    - Anomaly detection
    - Trend prediction
    - Proactive actions
    """

    def __init__(
        self,
        analysis_window: int = 100,
        prediction_horizon_minutes: int = 30,
    ):
        self.analysis_window = analysis_window
        self.prediction_horizon = timedelta(minutes=prediction_horizon_minutes)

        # Metric windows per service and metric
        self._metrics: Dict[str, Dict[str, MetricWindow]] = {}

        # Thresholds
        self._thresholds: Dict[str, Dict[str, float]] = {
            "cpu_usage": {"warning": 70, "critical": 90},
            "memory_usage": {"warning": 75, "critical": 95},
            "error_rate": {"warning": 0.02, "critical": 0.05},
            "latency_p99": {"warning": 2000, "critical": 5000},
            "queue_depth": {"warning": 100, "critical": 500},
        }

    async def record_metric(
        self,
        service: str,
        metric_name: str,
        value: float,
    ):
        """Record a health metric"""
        if service not in self._metrics:
            self._metrics[service] = {}

        if metric_name not in self._metrics[service]:
            self._metrics[service][metric_name] = MetricWindow(self.analysis_window)

        window = self._metrics[service][metric_name]

        # Check for anomaly before adding
        if window.is_anomaly(value):
            logger.warning(f"Anomaly detected: {service}.{metric_name}={value}")

        window.add(value)

    async def predict_failures(self, service: str) -> PredictionResult:
        """Predict potential failures for service"""
        if service not in self._metrics:
            return PredictionResult(
                service=service,
                risk_level=RiskLevel.LOW,
                predicted_failure_time=None,
                confidence=0.5,
                contributing_factors=[],
                recommended_actions=["No data available - monitor service"],
            )

        metrics = self._metrics[service]
        risk_factors = []
        actions = []
        max_risk = RiskLevel.LOW
        confidence = 0.0
        predicted_time = None

        for metric_name, window in metrics.items():
            if metric_name not in self._thresholds:
                continue

            thresholds = self._thresholds[metric_name]
            current = window.mean()
            trend = window.trend()
            std = window.std()

            # Check current level
            if current >= thresholds.get("critical", float("inf")):
                risk_factors.append(f"{metric_name} critical ({current:.2f})")
                max_risk = max(max_risk, RiskLevel.CRITICAL, key=lambda x: list(RiskLevel).index(x))
                confidence = max(confidence, 0.9)
                actions.append(f"Immediate action: reduce {metric_name}")

            elif current >= thresholds.get("warning", float("inf")):
                risk_factors.append(f"{metric_name} elevated ({current:.2f})")
                max_risk = max(max_risk, RiskLevel.MEDIUM, key=lambda x: list(RiskLevel).index(x))
                confidence = max(confidence, 0.7)
                actions.append(f"Monitor {metric_name} closely")

            # Check trend
            if trend > 0:
                warning = thresholds.get("warning", float("inf"))
                time_to_warning = (warning - current) / trend if trend > 0 else float("inf")

                if time_to_warning < 60:  # Less than 60 data points
                    risk_factors.append(f"{metric_name} trending up rapidly")
                    max_risk = max(max_risk, RiskLevel.HIGH, key=lambda x: list(RiskLevel).index(x))
                    confidence = max(confidence, 0.75)

                    # Estimate failure time
                    if predicted_time is None:
                        predicted_time = datetime.now(timezone.utc) + timedelta(minutes=time_to_warning)

                    actions.append(f"Scale up before {metric_name} threshold")

        # Add default actions if none
        if not actions:
            actions.append("Continue monitoring")

        return PredictionResult(
            service=service,
            risk_level=max_risk,
            predicted_failure_time=predicted_time,
            confidence=confidence,
            contributing_factors=risk_factors,
            recommended_actions=actions,
        )

    async def predict_all(self) -> Dict[str, PredictionResult]:
        """Predict failures for all services"""
        results = {}

        for service in self._metrics:
            results[service] = await self.predict_failures(service)

        return results

    async def get_proactive_actions(self) -> List[Dict[str, Any]]:
        """Get recommended proactive actions"""
        predictions = await self.predict_all()
        actions = []

        for service, prediction in predictions.items():
            if prediction.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                actions.append({
                    "service": service,
                    "risk_level": prediction.risk_level.value,
                    "actions": prediction.recommended_actions,
                    "urgency": "immediate" if prediction.risk_level == RiskLevel.CRITICAL else "soon",
                    "predicted_failure": prediction.predicted_failure_time.isoformat() if prediction.predicted_failure_time else None,
                })

        return sorted(actions, key=lambda x: list(RiskLevel).index(RiskLevel(x["risk_level"])), reverse=True)

    def set_threshold(
        self,
        metric_name: str,
        warning: Optional[float] = None,
        critical: Optional[float] = None,
    ):
        """Set custom thresholds for a metric"""
        if metric_name not in self._thresholds:
            self._thresholds[metric_name] = {}

        if warning is not None:
            self._thresholds[metric_name]["warning"] = warning
        if critical is not None:
            self._thresholds[metric_name]["critical"] = critical

    def get_metrics_summary(self, service: str) -> Dict[str, Any]:
        """Get metrics summary for service"""
        if service not in self._metrics:
            return {}

        summary = {}
        for metric_name, window in self._metrics[service].items():
            summary[metric_name] = {
                "mean": window.mean(),
                "std": window.std(),
                "trend": window.trend(),
                "samples": len(window._values),
            }

        return summary

    def get_stats(self) -> Dict[str, Any]:
        """Get healer statistics"""
        return {
            "monitored_services": len(self._metrics),
            "total_metrics": sum(len(m) for m in self._metrics.values()),
            "configured_thresholds": len(self._thresholds),
        }
