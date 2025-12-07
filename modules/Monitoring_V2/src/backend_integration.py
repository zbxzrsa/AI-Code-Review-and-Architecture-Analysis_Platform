"""
Monitoring_V2 - Production Backend Integration

Enhanced monitoring with SLO alerts, distributed tracing, and error budgets.
"""

import sys
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

# Add backend path
_backend_path = Path(__file__).parent.parent.parent.parent.parent / "backend"
if str(_backend_path) not in sys.path:
    sys.path.insert(0, str(_backend_path))

# Import backend implementations
try:
    from shared.monitoring import (
        MetricsCollector as BackendMetricsCollector,
        PrometheusMiddleware as BackendMiddleware,
        track_time as backend_track_time,
        count_calls as backend_count_calls,
        get_metrics as backend_get_metrics,
        PROMETHEUS_AVAILABLE,
    )
    from shared.monitoring.slo_alerts import SLOAlertManager
    from shared.monitoring.distributed_tracing import TracingManager
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    PROMETHEUS_AVAILABLE = False


@dataclass
class SLOAlert:
    """SLO violation alert."""
    slo_name: str
    current_value: float
    target_value: float
    severity: str
    timestamp: datetime
    message: str


class ProductionMetricsCollector:
    """
    V2 Production Metrics with SLO integration.
    """

    def __init__(self, prefix: str = "coderev", use_backend: bool = True):
        self.prefix = prefix
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if not self.use_backend:
            from .metrics_collector import MetricsCollector
            self._local = MetricsCollector(prefix=prefix)

        # SLO definitions
        self._slos: Dict[str, Dict] = {}
        self._slo_values: Dict[str, List[float]] = {}

    def define_slo(
        self,
        name: str,
        target: float,
        metric_name: str,
        comparison: str = ">=",  # >=, <=, ==
        window_minutes: int = 60,
    ):
        """Define an SLO for monitoring."""
        self._slos[name] = {
            "target": target,
            "metric": metric_name,
            "comparison": comparison,
            "window": window_minutes,
        }
        self._slo_values[name] = []

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record metric and check SLO."""
        full_name = f"{self.prefix}_{name}"

        if self.use_backend:
            BackendMetricsCollector.observe(full_name, value, labels or {})
        else:
            self._local.observe(name, value)

        # Track for SLO
        for slo_name, slo in self._slos.items():
            if slo["metric"] == name:
                self._slo_values[slo_name].append(value)
                # Keep only recent values
                max_values = slo["window"] * 60  # Assuming 1 value/second
                if len(self._slo_values[slo_name]) > max_values:
                    self._slo_values[slo_name] = self._slo_values[slo_name][-max_values:]

    def check_slo(self, name: str) -> Optional[SLOAlert]:
        """Check if SLO is being met."""
        if name not in self._slos:
            return None

        slo = self._slos[name]
        values = self._slo_values.get(name, [])

        if not values:
            return None

        current = sum(values) / len(values)
        target = slo["target"]
        comparison = slo["comparison"]

        violated = False
        if comparison == ">=" and current < target:
            violated = True
        elif comparison == "<=" and current > target:
            violated = True
        elif comparison == "==" and abs(current - target) > 0.01:
            violated = True

        if violated:
            return SLOAlert(
                slo_name=name,
                current_value=current,
                target_value=target,
                severity="warning" if abs(current - target) / target < 0.1 else "critical",
                timestamp=datetime.now(timezone.utc),
                message=f"SLO {name} violated: {current:.2f} vs target {target:.2f}",
            )

        return None

    def get_slo_status(self) -> Dict[str, Dict]:
        """Get status of all SLOs."""
        status = {}
        for name in self._slos:
            alert = self.check_slo(name)
            values = self._slo_values.get(name, [])
            current = sum(values) / len(values) if values else 0

            status[name] = {
                "target": self._slos[name]["target"],
                "current": current,
                "compliant": alert is None,
                "samples": len(values),
            }
        return status


class ProductionTracingService:
    """
    V2 Production Distributed Tracing.
    """

    def __init__(self, service_name: str, use_backend: bool = True):
        self.service_name = service_name
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            try:
                self._backend = TracingManager(service_name)
            except:
                self.use_backend = False

        if not self.use_backend:
            from .tracing_service import TracingService
            self._local = TracingService(service_name)

    def start_trace(self, operation: str) -> Any:
        """Start a new trace."""
        if self.use_backend:
            return self._backend.start_span(operation)
        return self._local.start_trace(operation)

    def start_span(self, operation: str, parent: Any = None) -> Any:
        """Start a child span."""
        if self.use_backend:
            return self._backend.start_span(operation, parent)
        return self._local.start_span(operation)

    def finish_span(self, span: Any, error: Optional[Exception] = None):
        """Finish a span."""
        if self.use_backend:
            self._backend.finish_span(span, error)
        else:
            if error:
                span.status = "error"
                span.tags["error"] = str(error)
            self._local.finish_span(span)

    def span(self, operation: str):
        """Context manager for span."""
        if self.use_backend:
            return self._backend.span(operation)
        return self._local.span(operation)

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get trace by ID."""
        if self.use_backend:
            return self._backend.get_trace(trace_id)
        return self._local.get_trace(trace_id)


class ProductionSLOTracker:
    """
    V2 Production SLO Tracker with error budgets.
    """

    def __init__(self, use_backend: bool = True):
        self.use_backend = use_backend and BACKEND_AVAILABLE

        if self.use_backend:
            try:
                self._backend = SLOAlertManager()
            except:
                self.use_backend = False

        if not self.use_backend:
            from .slo_tracker import SLOTracker
            self._local = SLOTracker()

    def define_slo(self, name: str, slo_type: str, target: float, **kwargs):
        """Define an SLO."""
        if self.use_backend:
            self._backend.define_slo(name, slo_type, target, **kwargs)
        else:
            from .slo_tracker import SLOType
            slo_type_enum = SLOType(slo_type) if isinstance(slo_type, str) else slo_type
            self._local.define_slo(name, slo_type_enum, target, **kwargs)

    def record_event(self, slo_name: str, is_good: bool, **kwargs):
        """Record SLO event."""
        if self.use_backend:
            self._backend.record_event(slo_name, is_good, **kwargs)
        else:
            self._local.record_event(slo_name, is_good, **kwargs)

    def get_status(self, slo_name: str) -> Dict[str, Any]:
        """Get SLO status."""
        if self.use_backend:
            status = self._backend.get_status(slo_name)
            return status.__dict__ if hasattr(status, '__dict__') else status

        status = self._local.get_status(slo_name)
        return {
            "current_value": status.current_value,
            "target": status.target,
            "is_meeting_target": status.is_meeting_target,
            "error_budget_remaining": status.error_budget_remaining,
            "budget_status": status.budget_status.value,
        }


# Factory functions
def get_metrics_collector(prefix: str = "coderev", use_backend: bool = True) -> ProductionMetricsCollector:
    """Get production metrics collector."""
    return ProductionMetricsCollector(prefix, use_backend)


def get_tracing_service(service_name: str, use_backend: bool = True) -> ProductionTracingService:
    """Get production tracing service."""
    return ProductionTracingService(service_name, use_backend)


def get_slo_tracker(use_backend: bool = True) -> ProductionSLOTracker:
    """Get production SLO tracker."""
    return ProductionSLOTracker(use_backend)


__all__ = [
    "BACKEND_AVAILABLE",
    "PROMETHEUS_AVAILABLE",
    "SLOAlert",
    "ProductionMetricsCollector",
    "ProductionTracingService",
    "ProductionSLOTracker",
    "get_metrics_collector",
    "get_tracing_service",
    "get_slo_tracker",
]
