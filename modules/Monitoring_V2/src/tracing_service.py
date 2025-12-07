"""
Monitoring_V2 - Tracing Service

Distributed tracing for request tracking.
"""

import logging
import secrets
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Trace span"""
    span_id: str
    trace_id: str
    parent_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "ok"
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def add_tag(self, key: str, value: str):
        self.tags[key] = value

    def add_log(self, message: str, **fields):
        self.logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            **fields,
        })

    def finish(self, status: str = "ok"):
        self.end_time = datetime.now(timezone.utc)
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs,
        }


@dataclass
class Trace:
    """Complete trace with spans"""
    trace_id: str
    spans: List[Span] = field(default_factory=list)

    @property
    def root_span(self) -> Optional[Span]:
        for span in self.spans:
            if span.parent_id is None:
                return span
        return self.spans[0] if self.spans else None

    @property
    def duration_ms(self) -> Optional[float]:
        root = self.root_span
        return root.duration_ms if root else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
        }


class TracingService:
    """
    Distributed tracing service.

    V2 Features:
    - Trace context propagation
    - Span relationships
    - Service dependency mapping
    - Performance analysis
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._traces: Dict[str, Trace] = {}
        self._current_span: Optional[Span] = None
        self._max_traces = 1000

    def start_trace(self, operation_name: str) -> Span:
        """Start a new trace"""
        trace_id = secrets.token_hex(16)
        span_id = secrets.token_hex(8)

        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=None,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.now(timezone.utc),
        )

        trace = Trace(trace_id=trace_id, spans=[span])
        self._traces[trace_id] = trace
        self._current_span = span

        # Limit stored traces
        if len(self._traces) > self._max_traces:
            oldest = min(self._traces.keys())
            del self._traces[oldest]

        return span

    def start_span(
        self,
        operation_name: str,
        parent_span: Optional[Span] = None,
    ) -> Span:
        """Start a child span"""
        parent = parent_span or self._current_span

        if not parent:
            return self.start_trace(operation_name)

        span_id = secrets.token_hex(8)

        span = Span(
            span_id=span_id,
            trace_id=parent.trace_id,
            parent_id=parent.span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.now(timezone.utc),
        )

        if parent.trace_id in self._traces:
            self._traces[parent.trace_id].spans.append(span)

        self._current_span = span
        return span

    def finish_span(self, span: Span, status: str = "ok"):
        """Finish a span"""
        span.finish(status)

        # Restore parent as current
        if span.parent_id and span.trace_id in self._traces:
            trace = self._traces[span.trace_id]
            for s in trace.spans:
                if s.span_id == span.parent_id:
                    self._current_span = s
                    break

    @contextmanager
    def span(self, operation_name: str):
        """Context manager for spans"""
        span = self.start_span(operation_name)
        try:
            yield span
            self.finish_span(span, "ok")
        except Exception as e:
            span.add_tag("error", str(e))
            self.finish_span(span, "error")
            raise

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get trace by ID"""
        trace = self._traces.get(trace_id)
        return trace.to_dict() if trace else None

    def get_recent_traces(self, limit: int = 100) -> List[Dict]:
        """Get recent traces"""
        traces = sorted(
            self._traces.values(),
            key=lambda t: t.root_span.start_time if t.root_span else datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return [t.to_dict() for t in traces[:limit]]

    def get_slow_traces(
        self,
        threshold_ms: float = 1000,
        limit: int = 50,
    ) -> List[Dict]:
        """Get traces slower than threshold"""
        slow = [
            t for t in self._traces.values()
            if t.duration_ms and t.duration_ms > threshold_ms
        ]

        slow.sort(key=lambda t: t.duration_ms or 0, reverse=True)
        return [t.to_dict() for t in slow[:limit]]

    def get_service_dependencies(self) -> Dict[str, List[str]]:
        """Get service dependency map"""
        deps: Dict[str, set] = {}

        for trace in self._traces.values():
            services_in_trace = set()
            for span in trace.spans:
                services_in_trace.add(span.service_name)

                # Find parent service
                if span.parent_id:
                    for parent in trace.spans:
                        if parent.span_id == span.parent_id:
                            if parent.service_name != span.service_name:
                                if parent.service_name not in deps:
                                    deps[parent.service_name] = set()
                                deps[parent.service_name].add(span.service_name)

        return {k: list(v) for k, v in deps.items()}

    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        durations = [
            t.duration_ms for t in self._traces.values()
            if t.duration_ms is not None
        ]

        return {
            "total_traces": len(self._traces),
            "total_spans": sum(len(t.spans) for t in self._traces.values()),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "service_name": self.service_name,
        }
