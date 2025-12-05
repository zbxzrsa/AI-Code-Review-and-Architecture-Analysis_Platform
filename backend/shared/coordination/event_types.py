"""
Event Types for Three-Version Coordination

Defines all events used in the version lifecycle system.
"""

import uuid
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime, timezone import datetime, timezone


class EventType(str, Enum):
    """Event types for version lifecycle."""
    
    # Experiment Events
    EXPERIMENT_CREATED = "experiment.created"
    EXPERIMENT_STARTED = "experiment.started"
    EXPERIMENT_EVALUATION_STARTED = "experiment.evaluation.started"
    EXPERIMENT_EVALUATION_COMPLETED = "experiment.evaluation.completed"
    
    # Promotion Events
    PROMOTION_REQUESTED = "promotion.requested"
    PROMOTION_APPROVED = "promotion.approved"
    PROMOTION_REJECTED = "promotion.rejected"
    PROMOTION_PHASE_CHANGED = "promotion.phase.changed"
    PROMOTION_COMPLETED = "promotion.completed"
    PROMOTION_FAILED = "promotion.failed"
    PROMOTION_ROLLBACK = "promotion.rollback"
    
    # Quarantine Events
    QUARANTINE_REQUESTED = "quarantine.requested"
    QUARANTINE_COMPLETED = "quarantine.completed"
    QUARANTINE_REVIEWED = "quarantine.reviewed"
    QUARANTINE_RETRY_APPROVED = "quarantine.retry.approved"
    
    # Monitoring Events
    MONITORING_ALERT = "monitoring.alert"
    MONITORING_DEGRADATION = "monitoring.degradation"
    MONITORING_RECOVERY = "monitoring.recovery"
    
    # Cost Events
    COST_THRESHOLD_WARNING = "cost.threshold.warning"
    COST_THRESHOLD_EXCEEDED = "cost.threshold.exceeded"
    
    # System Events
    HEALTH_CHECK = "system.health_check"
    AUTO_REMEDIATION = "system.auto_remediation"
    ROLLBACK_TRIGGERED = "system.rollback_triggered"


class Version(str, Enum):
    """Platform versions."""
    V1_EXPERIMENTATION = "v1"
    V2_PRODUCTION = "v2"
    V3_QUARANTINE = "v3"


class PromotionPhase(str, Enum):
    """Canary deployment phases."""
    VALIDATION = "validation"
    PHASE_1_10_PERCENT = "phase_1_10_percent"
    PHASE_2_50_PERCENT = "phase_2_50_percent"
    PHASE_3_100_PERCENT = "phase_3_100_percent"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    PAGE = "page"


@dataclass
class VersionEvent:
    """Event in the version lifecycle system."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.HEALTH_CHECK
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: Version = Version.V1_EXPERIMENTATION
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    source: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionEvent":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=Version(data["version"]),
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
            source=data.get("source", "system"),
        )


@dataclass
class ExperimentProposal:
    """Proposal for new experiment."""
    
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    hypothesis: str = ""
    success_criteria: Dict[str, str] = field(default_factory=dict)
    evaluation_period_days: int = 14
    sample_size: int = 2000
    rollback_plan: str = ""
    estimated_cost_usd: float = 0.0
    source: str = "manual"  # manual, industry_research, user_feedback, competitor
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "proposed"  # proposed, approved, running, completed, rejected


@dataclass
class PromotionRequest:
    """Request to promote V1 experiment to V2."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    requester: str = "version-control-ai"
    decision: str = "PROMOTE"
    confidence_score: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    evaluation_summary: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    current_phase: PromotionPhase = PromotionPhase.VALIDATION


@dataclass
class QuarantineRecord:
    """Record of quarantined experiment."""
    
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    failure_category: str = ""  # technical, quality, operational
    root_cause: str = ""
    contributing_factors: list = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    remediation_steps: list = field(default_factory=list)
    blacklist_entry: Optional[str] = None
    quarantined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    review_scheduled_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    retry_approved: bool = False
