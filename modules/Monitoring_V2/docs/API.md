# Monitoring_V2 API Reference

## Overview

Production monitoring with SLO tracking and distributed tracing.

## Classes

### SLOTracker

Service Level Objective tracking with error budgets.

```python
from modules.Monitoring_V2.src.slo_tracker import SLOTracker, SLOType

tracker = SLOTracker()

# Define SLO
tracker.define_slo(
    name="api_availability",
    slo_type=SLOType.AVAILABILITY,
    target=99.9,
    window_days=30
)

# Record events
tracker.record_request("api_availability", success=True, latency_ms=150)

# Get status
status = tracker.get_status("api_availability")
print(f"Budget remaining: {status.error_budget_remaining}%")
```

#### SLO Types

- `AVAILABILITY` - Uptime percentage
- `LATENCY` - Response time thresholds
- `ERROR_RATE` - Error percentage
- `THROUGHPUT` - Requests per second

### TracingService

Distributed tracing for request tracking.

```python
from modules.Monitoring_V2.src.tracing_service import TracingService

tracer = TracingService("my-service")

# Manual spans
span = tracer.start_trace("handle-request")
child = tracer.start_span("database-query")
tracer.finish_span(child)
tracer.finish_span(span)

# Context manager
with tracer.span("process-data") as span:
    span.add_tag("user_id", "123")
    # ... processing
```

#### Methods

- `start_trace(operation)` - Start new trace
- `start_span(operation)` - Start child span
- `finish_span(span)` - Complete span
- `get_trace(trace_id)` - Get trace details
- `get_slow_traces(threshold_ms)` - Find slow traces

## Configuration

See `config/monitoring_config.yaml`
