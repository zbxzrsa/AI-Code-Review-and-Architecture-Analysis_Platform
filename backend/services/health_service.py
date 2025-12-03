"""
Health Monitoring Service API

FastAPI endpoints for system health monitoring:
- Service health checks
- Resource utilization
- AI cycle status
- Alert management
"""

import asyncio
import random
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/health", tags=["health"])


# =============================================================================
# Models
# =============================================================================

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceHealth(BaseModel):
    name: str
    status: ServiceStatus
    latency_ms: float
    uptime_percent: float
    last_check: datetime
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ResourceUsage(BaseModel):
    cpu_percent: float = Field(ge=0, le=100)
    memory_percent: float = Field(ge=0, le=100)
    disk_percent: float = Field(ge=0, le=100)
    network_in_mbps: float = Field(ge=0)
    network_out_mbps: float = Field(ge=0)
    gpu_percent: Optional[float] = Field(default=None, ge=0, le=100)
    gpu_memory_percent: Optional[float] = Field(default=None, ge=0, le=100)


class CycleHealth(BaseModel):
    name: str
    running: bool
    cycle_count: int
    last_cycle: Optional[datetime]
    success_rate: float = Field(ge=0, le=1)
    avg_duration_seconds: float
    errors_last_hour: int


class Alert(BaseModel):
    alert_id: str
    severity: str  # critical, high, medium, low
    source: str
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False


class SystemHealthSummary(BaseModel):
    overall_status: ServiceStatus
    services: List[ServiceHealth]
    resources: ResourceUsage
    cycles: List[CycleHealth]
    active_alerts: int
    uptime_seconds: int
    version: str


# =============================================================================
# Mock Data Generators
# =============================================================================

_start_time = datetime.now(timezone.utc)
_alerts: List[Alert] = []


def _generate_service_health() -> List[ServiceHealth]:
    """Generate mock service health data."""
    services = [
        ("API Gateway", ServiceStatus.HEALTHY, 45, 99.99),
        ("Auth Service", ServiceStatus.HEALTHY, 32, 99.98),
        ("Analysis Service", ServiceStatus.HEALTHY, 180, 99.95),
        ("AI Orchestrator", ServiceStatus.HEALTHY, 250, 99.90),
        ("PostgreSQL", ServiceStatus.HEALTHY, 5, 99.99),
        ("Redis", ServiceStatus.HEALTHY, 1, 99.99),
        ("Neo4j", ServiceStatus.DEGRADED if random.random() > 0.8 else ServiceStatus.HEALTHY, 85, 99.5),
        ("Kafka", ServiceStatus.HEALTHY, 15, 99.95),
    ]
    
    return [
        ServiceHealth(
            name=name,
            status=status,
            latency_ms=latency + random.uniform(-10, 10),
            uptime_percent=uptime,
            last_check=datetime.now(timezone.utc),
            error="High memory usage" if status == ServiceStatus.DEGRADED else None,
        )
        for name, status, latency, uptime in services
    ]


def _generate_resource_usage() -> ResourceUsage:
    """Generate mock resource usage data."""
    return ResourceUsage(
        cpu_percent=35 + random.uniform(-10, 20),
        memory_percent=60 + random.uniform(-10, 15),
        disk_percent=35 + random.uniform(-5, 5),
        network_in_mbps=100 + random.uniform(-30, 50),
        network_out_mbps=75 + random.uniform(-20, 30),
        gpu_percent=45 + random.uniform(-15, 25) if random.random() > 0.3 else None,
        gpu_memory_percent=55 + random.uniform(-10, 20) if random.random() > 0.3 else None,
    )


def _generate_cycle_health() -> List[CycleHealth]:
    """Generate mock cycle health data."""
    return [
        CycleHealth(
            name="Bug Fix Cycle",
            running=True,
            cycle_count=15 + random.randint(0, 5),
            last_cycle=datetime.now(timezone.utc),
            success_rate=0.88 + random.uniform(-0.05, 0.1),
            avg_duration_seconds=120 + random.uniform(-30, 60),
            errors_last_hour=random.randint(0, 3),
        ),
        CycleHealth(
            name="Evolution Cycle",
            running=True,
            cycle_count=24 + random.randint(0, 3),
            last_cycle=datetime.now(timezone.utc),
            success_rate=0.95 + random.uniform(-0.05, 0.05),
            avg_duration_seconds=300 + random.uniform(-60, 120),
            errors_last_hour=random.randint(0, 2),
        ),
        CycleHealth(
            name="Learning Cycle",
            running=True,
            cycle_count=120 + random.randint(0, 20),
            last_cycle=datetime.now(timezone.utc),
            success_rate=0.92 + random.uniform(-0.05, 0.08),
            avg_duration_seconds=60 + random.uniform(-20, 30),
            errors_last_hour=random.randint(0, 5),
        ),
    ]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/", response_model=SystemHealthSummary)
async def get_system_health():
    """Get comprehensive system health summary."""
    services = _generate_service_health()
    
    # Determine overall status
    if any(s.status == ServiceStatus.UNHEALTHY for s in services):
        overall_status = ServiceStatus.UNHEALTHY
    elif any(s.status == ServiceStatus.DEGRADED for s in services):
        overall_status = ServiceStatus.DEGRADED
    else:
        overall_status = ServiceStatus.HEALTHY
    
    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()
    
    return SystemHealthSummary(
        overall_status=overall_status,
        services=services,
        resources=_generate_resource_usage(),
        cycles=_generate_cycle_health(),
        active_alerts=len([a for a in _alerts if not a.resolved]),
        uptime_seconds=int(uptime),
        version="2.0.0",
    )


@router.get("/services", response_model=List[ServiceHealth])
async def get_services_health():
    """Get health status of all services."""
    return _generate_service_health()


@router.get("/services/{service_name}", response_model=ServiceHealth)
async def get_service_health(service_name: str):
    """Get health status of a specific service."""
    services = _generate_service_health()
    for service in services:
        if service.name.lower().replace(" ", "-") == service_name.lower():
            return service
    raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found")


@router.get("/resources", response_model=ResourceUsage)
async def get_resource_usage():
    """Get current resource utilization."""
    return _generate_resource_usage()


@router.get("/cycles", response_model=List[CycleHealth])
async def get_cycles_health():
    """Get health status of all AI cycles."""
    return _generate_cycle_health()


@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    severity: Optional[str] = None,
    resolved: Optional[bool] = None,
    limit: int = 50,
):
    """Get system alerts."""
    # Generate some sample alerts if empty
    if not _alerts:
        _alerts.extend([
            Alert(
                alert_id="alert-001",
                severity="medium",
                source="Neo4j",
                message="Memory usage above 80%",
                timestamp=datetime.now(timezone.utc),
            ),
            Alert(
                alert_id="alert-002",
                severity="low",
                source="Evolution Cycle",
                message="Cycle duration exceeded threshold",
                timestamp=datetime.now(timezone.utc),
            ),
        ])
    
    result = _alerts
    
    if severity:
        result = [a for a in result if a.severity == severity]
    if resolved is not None:
        result = [a for a in result if a.resolved == resolved]
    
    return result[:limit]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    for alert in _alerts:
        if alert.alert_id == alert_id:
            alert.acknowledged = True
            return {"success": True, "message": "Alert acknowledged"}
    raise HTTPException(status_code=404, detail="Alert not found")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert."""
    for alert in _alerts:
        if alert.alert_id == alert_id:
            alert.resolved = True
            return {"success": True, "message": "Alert resolved"}
    raise HTTPException(status_code=404, detail="Alert not found")


@router.get("/metrics")
async def get_metrics():
    """Get Prometheus-style metrics."""
    services = _generate_service_health()
    resources = _generate_resource_usage()
    cycles = _generate_cycle_health()
    
    metrics = []
    
    # Service metrics
    for service in services:
        status_value = {"healthy": 1, "degraded": 0.5, "unhealthy": 0}[service.status.value]
        metrics.append(f'service_status{{name="{service.name}"}} {status_value}')
        metrics.append(f'service_latency_ms{{name="{service.name}"}} {service.latency_ms:.2f}')
        metrics.append(f'service_uptime{{name="{service.name}"}} {service.uptime_percent}')
    
    # Resource metrics
    metrics.append(f'system_cpu_percent {resources.cpu_percent:.2f}')
    metrics.append(f'system_memory_percent {resources.memory_percent:.2f}')
    metrics.append(f'system_disk_percent {resources.disk_percent:.2f}')
    metrics.append(f'system_network_in_mbps {resources.network_in_mbps:.2f}')
    metrics.append(f'system_network_out_mbps {resources.network_out_mbps:.2f}')
    
    # Cycle metrics
    for cycle in cycles:
        name = cycle.name.lower().replace(" ", "_")
        metrics.append(f'cycle_running{{name="{name}"}} {1 if cycle.running else 0}')
        metrics.append(f'cycle_count{{name="{name}"}} {cycle.cycle_count}')
        metrics.append(f'cycle_success_rate{{name="{name}"}} {cycle.success_rate:.4f}')
        metrics.append(f'cycle_errors_last_hour{{name="{name}"}} {cycle.errors_last_hour}')
    
    return "\n".join(metrics)


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    services = _generate_service_health()
    critical_services = ["API Gateway", "PostgreSQL", "Redis"]
    
    for service in services:
        if service.name in critical_services and service.status == ServiceStatus.UNHEALTHY:
            raise HTTPException(status_code=503, detail=f"Critical service {service.name} is unhealthy")
    
    return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}
