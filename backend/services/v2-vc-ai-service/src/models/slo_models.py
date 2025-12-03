"""
V2 VC-AI SLO Models

Data models for SLO tracking and monitoring.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class SLOState(str, Enum):
    """SLO compliance state"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


class BudgetZone(str, Enum):
    """Error budget zone"""
    GREEN = "green"     # > 50% budget remaining
    YELLOW = "yellow"   # 25-50% budget remaining
    RED = "red"         # < 25% budget remaining


class AlertSeverity(str, Enum):
    """Alert severity level"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SLOMetrics(BaseModel):
    """Current SLO metrics snapshot"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Availability
    availability: float = Field(..., ge=0, le=1, description="Current availability (0-1)")
    availability_target: float = Field(default=0.9999)
    availability_compliant: bool = Field(default=True)
    
    # Latency
    latency_p50_ms: float = Field(..., ge=0)
    latency_p99_ms: float = Field(..., ge=0)
    latency_p999_ms: float = Field(..., ge=0)
    latency_target_p99_ms: float = Field(default=500)
    latency_compliant: bool = Field(default=True)
    
    # Error rate
    error_rate: float = Field(..., ge=0, le=1)
    error_rate_target: float = Field(default=0.001)
    error_rate_compliant: bool = Field(default=True)
    
    # Accuracy
    accuracy: float = Field(..., ge=0, le=1)
    accuracy_target: float = Field(default=0.98)
    accuracy_compliant: bool = Field(default=True)
    
    # Overall
    overall_state: SLOState = Field(default=SLOState.COMPLIANT)
    compliance_percentage: float = Field(..., ge=0, le=100)


class ErrorBudget(BaseModel):
    """Error budget status"""
    window_days: int = Field(default=30)
    window_start: datetime
    window_end: datetime
    
    # Budget calculation
    total_budget_minutes: float = Field(..., ge=0)
    consumed_minutes: float = Field(..., ge=0)
    remaining_minutes: float = Field(..., ge=0)
    remaining_percentage: float = Field(..., ge=0, le=100)
    
    # Projections
    burn_rate_per_day: float = Field(..., ge=0)
    projected_exhaustion_date: Optional[datetime] = None
    days_until_exhaustion: Optional[float] = None
    
    # Zone
    zone: BudgetZone
    zone_since: datetime
    
    # Policy
    deployment_allowed: bool = Field(default=True)
    change_freeze_recommended: bool = Field(default=False)


class SLOStatus(BaseModel):
    """Complete SLO status report"""
    service_name: str = Field(default="v2-vc-ai-service")
    report_time: datetime = Field(default_factory=datetime.utcnow)
    
    # Current metrics
    metrics: SLOMetrics
    
    # Error budget
    error_budget: ErrorBudget
    
    # Historical
    slo_history_24h: List[Dict[str, Any]] = Field(default_factory=list)
    violations_24h: int = Field(default=0, ge=0)
    violations_7d: int = Field(default=0, ge=0)
    violations_30d: int = Field(default=0, ge=0)
    
    # Trend
    availability_trend: str = Field(default="stable", description="improving, stable, degrading")
    latency_trend: str = Field(default="stable")
    error_rate_trend: str = Field(default="stable")


class AlertConfig(BaseModel):
    """Alert configuration"""
    name: str
    description: str
    metric: str
    condition: str
    threshold: float
    severity: AlertSeverity
    enabled: bool = Field(default=True)
    
    # Notification
    notification_channels: List[str] = Field(default_factory=list)
    cooldown_minutes: int = Field(default=5, ge=1)
    escalation_minutes: int = Field(default=30, ge=1)
    
    # Actions
    auto_rollback: bool = Field(default=False)
    page_on_call: bool = Field(default=True)


class ActiveAlert(BaseModel):
    """Currently active alert"""
    id: str
    config: AlertConfig
    triggered_at: datetime
    metric_value: float
    message: str
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = Field(default=False)
    resolved_at: Optional[datetime] = None
    duration_minutes: Optional[float] = None


class IncidentReport(BaseModel):
    """SLO incident report"""
    id: str
    title: str
    severity: AlertSeverity
    
    # Timeline
    started_at: datetime
    detected_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    
    # Impact
    affected_slos: List[str] = Field(default_factory=list)
    error_budget_consumed_minutes: float = Field(default=0, ge=0)
    requests_affected: int = Field(default=0, ge=0)
    users_affected: int = Field(default=0, ge=0)
    
    # Details
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Follow-up
    action_items: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
    post_mortem_url: Optional[str] = None


class SLODashboardData(BaseModel):
    """Data for SLO dashboard"""
    status: SLOStatus
    active_alerts: List[ActiveAlert] = Field(default_factory=list)
    recent_incidents: List[IncidentReport] = Field(default_factory=list)
    
    # Charts data
    availability_chart_data: List[Dict[str, Any]] = Field(default_factory=list)
    latency_chart_data: List[Dict[str, Any]] = Field(default_factory=list)
    error_rate_chart_data: List[Dict[str, Any]] = Field(default_factory=list)
    budget_burn_chart_data: List[Dict[str, Any]] = Field(default_factory=list)


class HealthCheck(BaseModel):
    """Service health check result"""
    healthy: bool
    components: Dict[str, bool] = Field(default_factory=dict)
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: float = Field(default=0, ge=0)
