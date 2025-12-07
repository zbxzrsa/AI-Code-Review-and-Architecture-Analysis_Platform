"""
Monitoring_V2 - SLO Tracker

Service Level Objective tracking with error budgets.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class SLOType(str, Enum):
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class BudgetStatus(str, Enum):
    HEALTHY = "healthy"       # > 50% budget remaining
    WARNING = "warning"       # 20-50% budget remaining
    CRITICAL = "critical"     # < 20% budget remaining
    EXHAUSTED = "exhausted"   # 0% budget remaining


@dataclass
class SLO:
    """Service Level Objective definition"""
    name: str
    slo_type: SLOType
    target: float  # Target percentage (e.g., 99.9 for 99.9%)
    window_days: int = 30
    description: str = ""

    # For latency SLOs
    latency_threshold_ms: Optional[float] = None
    percentile: Optional[float] = None  # e.g., 95 for p95


@dataclass
class SLOStatus:
    """Current SLO status"""
    slo_name: str
    current_value: float
    target: float
    is_meeting_target: bool
    error_budget_remaining: float
    budget_status: BudgetStatus
    window_start: datetime
    window_end: datetime
    total_requests: int = 0
    good_requests: int = 0
    bad_requests: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slo_name": self.slo_name,
            "current_value": round(self.current_value, 4),
            "target": self.target,
            "is_meeting_target": self.is_meeting_target,
            "error_budget_remaining": round(self.error_budget_remaining, 4),
            "budget_status": self.budget_status.value,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "total_requests": self.total_requests,
            "good_requests": self.good_requests,
            "bad_requests": self.bad_requests,
        }


class SLOTracker:
    """
    SLO tracking with error budgets.

    V2 Features:
    - Multiple SLO types
    - Rolling window calculation
    - Error budget tracking
    - Burn rate alerts
    """

    def __init__(self):
        self._slos: Dict[str, SLO] = {}
        self._events: Dict[str, List[Dict]] = defaultdict(list)
        self._max_events = 100000

    def define_slo(
        self,
        name: str,
        slo_type: SLOType,
        target: float,
        window_days: int = 30,
        **kwargs,
    ):
        """Define a new SLO"""
        self._slos[name] = SLO(
            name=name,
            slo_type=slo_type,
            target=target,
            window_days=window_days,
            **kwargs,
        )
        logger.info(f"SLO defined: {name} ({slo_type.value} >= {target}%)")

    def record_event(
        self,
        slo_name: str,
        is_good: bool,
        value: Optional[float] = None,
    ):
        """Record an event for SLO tracking"""
        if slo_name not in self._slos:
            return

        event = {
            "timestamp": datetime.now(timezone.utc),
            "is_good": is_good,
            "value": value,
        }

        self._events[slo_name].append(event)

        # Limit events
        if len(self._events[slo_name]) > self._max_events:
            self._events[slo_name] = self._events[slo_name][-self._max_events:]

    def record_request(
        self,
        slo_name: str,
        success: bool,
        latency_ms: Optional[float] = None,
    ):
        """Record a request (convenience method)"""
        slo = self._slos.get(slo_name)
        if not slo:
            return

        is_good = success

        # For latency SLOs, check threshold
        if slo.slo_type == SLOType.LATENCY and latency_ms is not None:
            if slo.latency_threshold_ms:
                is_good = latency_ms <= slo.latency_threshold_ms

        self.record_event(slo_name, is_good, latency_ms)

    def get_status(self, slo_name: str) -> Optional[SLOStatus]:
        """Get current SLO status"""
        slo = self._slos.get(slo_name)
        if not slo:
            return None

        # Calculate window
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(days=slo.window_days)

        # Filter events in window
        events = [
            e for e in self._events[slo_name]
            if e["timestamp"] >= window_start
        ]

        total = len(events)
        good = sum(1 for e in events if e["is_good"])
        bad = total - good

        # Calculate current value
        if total == 0:
            current_value = 100.0
        else:
            current_value = (good / total) * 100

        # Calculate error budget
        target_decimal = slo.target / 100
        allowed_bad_ratio = 1 - target_decimal

        if total > 0:
            actual_bad_ratio = bad / total
            if allowed_bad_ratio > 0:
                error_budget_remaining = max(0, 1 - (actual_bad_ratio / allowed_bad_ratio))
            else:
                error_budget_remaining = 1.0 if bad == 0 else 0.0
        else:
            error_budget_remaining = 1.0

        # Determine budget status
        if error_budget_remaining <= 0:
            budget_status = BudgetStatus.EXHAUSTED
        elif error_budget_remaining < 0.2:
            budget_status = BudgetStatus.CRITICAL
        elif error_budget_remaining < 0.5:
            budget_status = BudgetStatus.WARNING
        else:
            budget_status = BudgetStatus.HEALTHY

        return SLOStatus(
            slo_name=slo_name,
            current_value=current_value,
            target=slo.target,
            is_meeting_target=current_value >= slo.target,
            error_budget_remaining=error_budget_remaining * 100,
            budget_status=budget_status,
            window_start=window_start,
            window_end=now,
            total_requests=total,
            good_requests=good,
            bad_requests=bad,
        )

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status for all SLOs"""
        return {
            name: self.get_status(name).to_dict()
            for name in self._slos
        }

    def get_burn_rate(
        self,
        slo_name: str,
        window_hours: int = 1,
    ) -> Optional[float]:
        """
        Calculate error budget burn rate.

        Burn rate > 1 means consuming budget faster than sustainable.
        """
        slo = self._slos.get(slo_name)
        if not slo:
            return None

        # Get events in short window
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=window_hours)

        events = [
            e for e in self._events[slo_name]
            if e["timestamp"] >= window_start
        ]

        if not events:
            return 0.0

        # Calculate short-term error rate
        bad = sum(1 for e in events if not e["is_good"])
        error_rate = bad / len(events)

        # Calculate allowed error rate
        allowed_error_rate = (100 - slo.target) / 100

        if allowed_error_rate == 0:
            return float('inf') if error_rate > 0 else 0.0

        return error_rate / allowed_error_rate

    def clear_events(self, slo_name: Optional[str] = None):
        """Clear recorded events"""
        if slo_name:
            self._events.pop(slo_name, None)
        else:
            self._events.clear()
