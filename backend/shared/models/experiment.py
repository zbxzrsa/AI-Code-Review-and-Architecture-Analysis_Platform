"""
Data models for experiments and analysis results.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import uuid4


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"


class PromotionStatus(str, Enum):
    """Promotion evaluation status."""
    PENDING_EVALUATION = "pending_evaluation"
    PASSED = "passed"
    FAILED = "failed"
    MANUAL_REVIEW = "manual_review"


@dataclass
class ExperimentMetrics:
    """Metrics for experiment evaluation."""
    accuracy: float  # 0-1, correctness of analysis
    latency_ms: float  # Response time in milliseconds
    cost: float  # API/compute cost
    error_rate: float  # 0-1, failure rate
    throughput: int = 0  # Requests per second
    user_satisfaction: float = 0.0  # 0-5 star rating
    false_positives: int = 0
    false_negatives: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def meets_v2_threshold(self) -> bool:
        """Check if metrics meet V2 production thresholds."""
        return (
            self.accuracy >= 0.95 and
            self.latency_ms <= 3000 and
            self.error_rate <= 0.02
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "error_rate": self.error_rate,
            "throughput": self.throughput,
            "user_satisfaction": self.user_satisfaction,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CodeReviewAnalysis:
    """Result of code review analysis."""
    id: str = field(default_factory=lambda: str(uuid4()))
    code_snippet: str = ""
    language: str = ""
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    architecture_insights: Dict[str, Any] = field(default_factory=dict)
    security_concerns: List[Dict[str, Any]] = field(default_factory=list)
    performance_notes: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    analysis_time_ms: float = 0.0
    model_used: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "code_snippet": self.code_snippet,
            "language": self.language,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "architecture_insights": self.architecture_insights,
            "security_concerns": self.security_concerns,
            "performance_notes": self.performance_notes,
            "confidence_score": self.confidence_score,
            "analysis_time_ms": self.analysis_time_ms,
            "model_used": self.model_used,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Experiment:
    """Represents a single experiment in V1."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    version: str = "v1"
    status: ExperimentStatus = ExperimentStatus.PENDING
    
    # Configuration
    primary_model: str = ""
    secondary_model: str = ""
    prompt_template: str = ""
    routing_strategy: str = ""  # "primary", "secondary", "ensemble", "adaptive"
    
    # Execution
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    metrics: Optional[ExperimentMetrics] = None
    analyses: List[CodeReviewAnalysis] = field(default_factory=list)
    
    # Promotion
    promotion_status: PromotionStatus = PromotionStatus.PENDING_EVALUATION
    promotion_decision_at: Optional[datetime] = None
    promotion_reason: str = ""
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    created_by: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "primary_model": self.primary_model,
            "secondary_model": self.secondary_model,
            "prompt_template": self.prompt_template,
            "routing_strategy": self.routing_strategy,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "analyses": [a.to_dict() for a in self.analyses],
            "promotion_status": self.promotion_status.value,
            "promotion_decision_at": self.promotion_decision_at.isoformat() if self.promotion_decision_at else None,
            "promotion_reason": self.promotion_reason,
            "tags": self.tags,
            "notes": self.notes,
            "created_by": self.created_by,
        }


@dataclass
class QuarantineRecord:
    """Record of a quarantined experiment or configuration."""
    id: str = field(default_factory=lambda: str(uuid4()))
    experiment_id: str = ""
    reason: str = ""
    failure_analysis: Dict[str, Any] = field(default_factory=dict)
    metrics_at_failure: Optional[ExperimentMetrics] = None
    quarantined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quarantined_by: str = ""
    
    # Re-evaluation
    can_re_evaluate: bool = True
    re_evaluation_notes: str = ""
    re_evaluation_requested_at: Optional[datetime] = None
    re_evaluation_requested_by: str = ""
    
    # Metadata
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    related_experiments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "reason": self.reason,
            "failure_analysis": self.failure_analysis,
            "metrics_at_failure": self.metrics_at_failure.to_dict() if self.metrics_at_failure else None,
            "quarantined_at": self.quarantined_at.isoformat(),
            "quarantined_by": self.quarantined_by,
            "can_re_evaluate": self.can_re_evaluate,
            "re_evaluation_notes": self.re_evaluation_notes,
            "re_evaluation_requested_at": self.re_evaluation_requested_at.isoformat() if self.re_evaluation_requested_at else None,
            "re_evaluation_requested_by": self.re_evaluation_requested_by,
            "impact_analysis": self.impact_analysis,
            "related_experiments": self.related_experiments,
        }
