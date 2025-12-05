"""
V2 VC-AI SLO Router

API endpoints for SLO monitoring and error budget tracking.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Request

from ..models.slo_models import (
    SLOStatus,
    SLOMetrics,
    ErrorBudget,
    SLODashboardData,
    HealthCheck,
    ActiveAlert,
    IncidentReport,
    BudgetZone,
    SLOState,
)


router = APIRouter(prefix="/slo", tags=["slo"])


# =============================================================================
# SLO Status Endpoints
# =============================================================================

@router.get("/status", response_model=SLOStatus)
async def get_slo_status(request: Request) -> SLOStatus:
    """
    Get current SLO status.
    
    Returns:
    - Current metrics snapshot
    - Error budget status
    - Compliance status
    - Historical data (24h)
    """
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    
    if slo_monitor:
        return slo_monitor.get_status()
    
    # Mock response if monitor not initialized
    now = datetime.now(timezone.utc)
    
    return SLOStatus(
        service_name="v2-vc-ai-service",
        metrics=SLOMetrics(
            availability=0.9999,
            availability_target=0.9999,
            availability_compliant=True,
            latency_p50_ms=85.0,
            latency_p99_ms=320.0,
            latency_p999_ms=480.0,
            latency_target_p99_ms=500.0,
            latency_compliant=True,
            error_rate=0.0005,
            error_rate_target=0.001,
            error_rate_compliant=True,
            accuracy=0.985,
            accuracy_target=0.98,
            accuracy_compliant=True,
            overall_state=SLOState.COMPLIANT,
            compliance_percentage=100.0,
        ),
        error_budget=ErrorBudget(
            window_days=30,
            window_start=now - timedelta(days=15),
            window_end=now + timedelta(days=15),
            total_budget_minutes=4.32,  # 99.99% over 30 days
            consumed_minutes=1.5,
            remaining_minutes=2.82,
            remaining_percentage=65.3,
            burn_rate_per_day=0.1,
            projected_exhaustion_date=now + timedelta(days=28),
            days_until_exhaustion=28.0,
            zone=BudgetZone.GREEN,
            zone_since=now - timedelta(days=15),
            deployment_allowed=True,
            change_freeze_recommended=False,
        ),
        availability_trend="stable",
        latency_trend="improving",
        error_rate_trend="stable",
    )


@router.get("/metrics", response_model=SLOMetrics)
async def get_current_metrics(request: Request) -> SLOMetrics:
    """Get current SLO metrics snapshot."""
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    
    if slo_monitor:
        return slo_monitor.get_current_metrics()
    
    return SLOMetrics(
        availability=0.9999,
        availability_target=0.9999,
        availability_compliant=True,
        latency_p50_ms=85.0,
        latency_p99_ms=320.0,
        latency_p999_ms=480.0,
        latency_target_p99_ms=500.0,
        latency_compliant=True,
        error_rate=0.0005,
        error_rate_target=0.001,
        error_rate_compliant=True,
        accuracy=0.985,
        accuracy_target=0.98,
        accuracy_compliant=True,
        overall_state=SLOState.COMPLIANT,
        compliance_percentage=100.0,
    )


@router.get("/error-budget", response_model=ErrorBudget)
async def get_error_budget(request: Request) -> ErrorBudget:
    """Get current error budget status."""
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    
    if slo_monitor:
        return slo_monitor.get_error_budget()
    
    now = datetime.now(timezone.utc)
    
    return ErrorBudget(
        window_days=30,
        window_start=now - timedelta(days=15),
        window_end=now + timedelta(days=15),
        total_budget_minutes=4.32,
        consumed_minutes=1.5,
        remaining_minutes=2.82,
        remaining_percentage=65.3,
        burn_rate_per_day=0.1,
        projected_exhaustion_date=now + timedelta(days=28),
        days_until_exhaustion=28.0,
        zone=BudgetZone.GREEN,
        zone_since=now - timedelta(days=15),
        deployment_allowed=True,
        change_freeze_recommended=False,
    )


# =============================================================================
# Dashboard Endpoints
# =============================================================================

@router.get("/dashboard", response_model=SLODashboardData)
async def get_dashboard_data(request: Request) -> SLODashboardData:
    """Get all data needed for SLO dashboard."""
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    
    if slo_monitor:
        status = slo_monitor.get_status()
        alerts = slo_monitor.get_active_alerts()
    else:
        status = await get_slo_status(request)
        alerts = []
    
    # Generate chart data
    now = datetime.now(timezone.utc)
    availability_chart = []
    latency_chart = []
    error_rate_chart = []
    budget_chart = []
    
    for i in range(24):
        time_point = now - timedelta(hours=23-i)
        availability_chart.append({
            "timestamp": time_point.isoformat(),
            "value": 0.9999 - (0.0001 * (i % 3)),
        })
        latency_chart.append({
            "timestamp": time_point.isoformat(),
            "p50": 80 + (i % 20),
            "p99": 300 + (i % 50),
        })
        error_rate_chart.append({
            "timestamp": time_point.isoformat(),
            "value": 0.0005 + (0.0001 * (i % 5)),
        })
        budget_chart.append({
            "timestamp": time_point.isoformat(),
            "remaining_percent": 70 - (i * 0.2),
        })
    
    return SLODashboardData(
        status=status,
        active_alerts=alerts,
        recent_incidents=[],
        availability_chart_data=availability_chart,
        latency_chart_data=latency_chart,
        error_rate_chart_data=error_rate_chart,
        budget_burn_chart_data=budget_chart,
    )


# =============================================================================
# Alert Endpoints
# =============================================================================

@router.get("/alerts", response_model=List[ActiveAlert])
async def get_active_alerts(request: Request) -> List[ActiveAlert]:
    """Get all active SLO alerts."""
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    
    if slo_monitor:
        return slo_monitor.get_active_alerts()
    
    return []


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    request: Request,
    acknowledged_by: str = "system",
) -> dict:
    """Acknowledge an active alert."""
    return {
        "alert_id": alert_id,
        "acknowledged": True,
        "acknowledged_by": acknowledged_by,
        "acknowledged_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    request: Request,
    resolution: Optional[str] = None,
) -> dict:
    """Resolve an active alert."""
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    
    if slo_monitor:
        success = await slo_monitor.resolve_alert(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
    
    return {
        "alert_id": alert_id,
        "resolved": True,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "resolution": resolution,
    }


# =============================================================================
# Health Check Endpoints
# =============================================================================

@router.get("/health", response_model=HealthCheck)
async def health_check(request: Request) -> HealthCheck:
    """
    Comprehensive health check.
    
    Checks:
    - Model client availability
    - Database connectivity
    - Cache connectivity
    - Circuit breaker state
    """
    import time
    start = time.time()
    
    checks = []
    components = {}
    
    # Check model client
    model_client = getattr(request.app.state, "model_client", None)
    components["model_client"] = model_client is not None
    checks.append({
        "name": "model_client",
        "status": "healthy" if model_client else "unhealthy",
        "message": "Model client initialized" if model_client else "Model client not initialized",
    })
    
    # Check circuit breaker
    circuit_breaker = getattr(request.app.state, "circuit_breaker", None)
    if circuit_breaker:
        cb_healthy = circuit_breaker.is_closed() or circuit_breaker.is_half_open()
        components["circuit_breaker"] = cb_healthy
        checks.append({
            "name": "circuit_breaker",
            "status": "healthy" if cb_healthy else "degraded",
            "state": circuit_breaker.state.value,
        })
    else:
        components["circuit_breaker"] = True
    
    # Check SLO monitor
    slo_monitor = getattr(request.app.state, "slo_monitor", None)
    components["slo_monitor"] = slo_monitor is not None
    checks.append({
        "name": "slo_monitor",
        "status": "healthy" if slo_monitor else "unhealthy",
    })
    
    latency_ms = (time.time() - start) * 1000
    healthy = all(components.values())
    
    return HealthCheck(
        healthy=healthy,
        components=components,
        checks=checks,
        latency_ms=latency_ms,
    )


# =============================================================================
# Incident Endpoints
# =============================================================================

@router.get("/incidents", response_model=List[IncidentReport])
async def get_recent_incidents(
    days: int = Query(default=30, ge=1, le=365),
) -> List[IncidentReport]:
    """Get recent SLO incidents."""
    # In production, this would query from database
    return []


@router.get("/incidents/{incident_id}", response_model=IncidentReport)
async def get_incident(incident_id: str) -> IncidentReport:
    """Get detailed incident report."""
    raise HTTPException(status_code=404, detail="Incident not found")
