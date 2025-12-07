"""Tests for Monitoring_V2 SLO and Tracing"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.slo_tracker import SLOTracker, SLOType, BudgetStatus
from src.tracing_service import TracingService


class TestSLOTracker:
    @pytest.fixture
    def tracker(self):
        t = SLOTracker()
        t.define_slo("availability", SLOType.AVAILABILITY, 99.9)
        t.define_slo("latency_p99", SLOType.LATENCY, 99.0, latency_threshold_ms=3000)
        return t

    def test_define_slo(self, tracker):
        assert "availability" in tracker._slos
        assert tracker._slos["availability"].target == 99.9

    def test_record_events(self, tracker):
        # Record 99 good, 1 bad = 99% (below 99.9% target)
        for _ in range(99):
            tracker.record_event("availability", is_good=True)
        tracker.record_event("availability", is_good=False)

        status = tracker.get_status("availability")
        assert status.current_value == 99.0
        assert not status.is_meeting_target

    def test_error_budget(self, tracker):
        # All good = full budget
        for _ in range(100):
            tracker.record_event("availability", is_good=True)

        status = tracker.get_status("availability")
        assert status.error_budget_remaining == 100.0
        assert status.budget_status == BudgetStatus.HEALTHY

    def test_budget_exhausted(self, tracker):
        # 10% bad requests exhausts budget for 99.9% SLO
        for _ in range(90):
            tracker.record_event("availability", is_good=True)
        for _ in range(10):
            tracker.record_event("availability", is_good=False)

        status = tracker.get_status("availability")
        assert status.budget_status == BudgetStatus.EXHAUSTED

    def test_latency_slo(self, tracker):
        # Record requests with latency
        tracker.record_request("latency_p99", success=True, latency_ms=1000)  # Good
        tracker.record_request("latency_p99", success=True, latency_ms=5000)  # Bad (>3000ms)

        status = tracker.get_status("latency_p99")
        assert status.total_requests == 2
        assert status.good_requests == 1


class TestTracingService:
    @pytest.fixture
    def tracer(self):
        return TracingService("test-service")

    def test_start_trace(self, tracer):
        span = tracer.start_trace("test-operation")

        assert span.trace_id is not None
        assert span.operation_name == "test-operation"
        assert span.parent_id is None

    def test_child_span(self, tracer):
        root = tracer.start_trace("root")
        child = tracer.start_span("child")

        assert child.parent_id == root.span_id
        assert child.trace_id == root.trace_id

    def test_span_context_manager(self, tracer):
        with tracer.span("context-op") as span:
            span.add_tag("key", "value")

        assert span.status == "ok"
        assert span.end_time is not None
        assert span.tags["key"] == "value"

    def test_span_error(self, tracer):
        try:
            with tracer.span("error-op") as span:
                raise ValueError("test error")
        except ValueError:
            pass

        assert span.status == "error"
        assert "error" in span.tags

    def test_get_trace(self, tracer):
        span = tracer.start_trace("get-test")
        tracer.finish_span(span)

        trace = tracer.get_trace(span.trace_id)
        assert trace is not None
        assert trace["trace_id"] == span.trace_id

    def test_stats(self, tracer):
        span = tracer.start_trace("stats-test")
        tracer.finish_span(span)

        stats = tracer.get_stats()
        assert stats["total_traces"] >= 1
        assert stats["service_name"] == "test-service"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
