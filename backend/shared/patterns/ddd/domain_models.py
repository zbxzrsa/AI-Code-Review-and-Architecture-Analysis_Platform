"""
Domain Models and Value Objects

Implements core domain models following DDD principles.

Key Concepts:
- Entities: Objects with identity that persists over time
- Value Objects: Immutable objects defined by their attributes
- Aggregates: Clusters of entities and value objects with a root entity
"""
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4


# =============================================================================
# Base Classes
# =============================================================================

class Entity(ABC):
    """
    Base class for entities.
    
    Entities have identity and are compared by their ID.
    """
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Get entity ID."""
        pass
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)


class ValueObject(ABC):
    """
    Base class for value objects.
    
    Value objects are immutable and compared by their attributes.
    """
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__
    
    def __hash__(self) -> int:
        return hash(tuple(sorted(self.__dict__.items())))


class AggregateRoot(Entity):
    """
    Base class for aggregate roots.
    
    Aggregate roots are the entry point for accessing an aggregate.
    All modifications to the aggregate must go through the root.
    """
    
    def __init__(self):
        self._domain_events: List['DomainEvent'] = []
        self._version: int = 0
    
    def add_domain_event(self, event: 'DomainEvent'):
        """Add a domain event to be published."""
        self._domain_events.append(event)
    
    def clear_domain_events(self) -> List['DomainEvent']:
        """Clear and return all domain events."""
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
    
    @property
    def version(self) -> int:
        return self._version
    
    def increment_version(self):
        self._version += 1


# =============================================================================
# Code Analysis Domain - Value Objects
# =============================================================================

@dataclass(frozen=True)
class CodeHash(ValueObject):
    """Hash of code content for deduplication."""
    value: str
    algorithm: str = "sha256"
    
    @classmethod
    def create(cls, code: str) -> 'CodeHash':
        hash_value = hashlib.sha256(code.encode()).hexdigest()
        return cls(value=hash_value)


class Severity(str, Enum):
    """Issue severity level."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(str, Enum):
    """Type of code issue."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    MAINTAINABILITY = "maintainability"
    BUG = "bug"
    STYLE = "style"


@dataclass(frozen=True)
class IssueLocation(ValueObject):
    """Location of an issue in code."""
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0


@dataclass(frozen=True)
class FixSuggestion(ValueObject):
    """Suggested fix for an issue."""
    description: str
    old_code: str
    new_code: str
    confidence: float  # 0.0 to 1.0
    
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8


@dataclass(frozen=True)
class AnalysisMetrics(ValueObject):
    """Metrics for a code analysis."""
    total_issues: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    analysis_time_ms: float
    lines_analyzed: int
    
    @property
    def issues_by_severity(self) -> Dict[str, int]:
        return {
            "critical": self.critical_count,
            "high": self.high_count,
            "medium": self.medium_count,
            "low": self.low_count,
        }


# =============================================================================
# Code Analysis Domain - Entities
# =============================================================================

@dataclass
class Issue(Entity):
    """A detected issue in code."""
    _id: str
    issue_type: IssueType
    severity: Severity
    message: str
    location: IssueLocation
    rule_id: str
    suggestion: Optional[FixSuggestion] = None
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    @property
    def id(self) -> str:
        return self._id
    
    @classmethod
    def create(
        cls,
        issue_type: IssueType,
        severity: Severity,
        message: str,
        location: IssueLocation,
        rule_id: str,
        suggestion: Optional[FixSuggestion] = None
    ) -> 'Issue':
        return cls(
            _id=str(uuid4()),
            issue_type=issue_type,
            severity=severity,
            message=message,
            location=location,
            rule_id=rule_id,
            suggestion=suggestion,
        )
    
    def resolve(self):
        """Mark issue as resolved."""
        self.is_resolved = True
        self.resolved_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "type": self.issue_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": {
                "file": self.location.file_path,
                "line_start": self.location.line_start,
                "line_end": self.location.line_end,
            },
            "rule_id": self.rule_id,
            "is_resolved": self.is_resolved,
        }


# =============================================================================
# Code Analysis Domain - Aggregate Root
# =============================================================================

class AnalysisStatus(str, Enum):
    """Status of an analysis."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Analysis(AggregateRoot):
    """
    Code analysis aggregate root.
    
    Invariants:
    - Must have at least one file to analyze
    - Issues can only be added in RUNNING status
    - Cannot modify after COMPLETED/FAILED
    """
    _id: str
    project_id: str
    code_hash: CodeHash
    language: str
    status: AnalysisStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Nested entities
    _issues: List[Issue] = field(default_factory=list)
    
    # Metrics (computed)
    _metrics: Optional[AnalysisMetrics] = None
    
    def __post_init__(self):
        super().__init__()
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def issues(self) -> List[Issue]:
        return self._issues.copy()
    
    @property
    def metrics(self) -> Optional[AnalysisMetrics]:
        return self._metrics
    
    @classmethod
    def create(
        cls,
        project_id: str,
        code: str,
        language: str
    ) -> 'Analysis':
        """Factory method to create a new analysis."""
        analysis = cls(
            _id=str(uuid4()),
            project_id=project_id,
            code_hash=CodeHash.create(code),
            language=language,
            status=AnalysisStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )
        
        # Emit creation event
        analysis.add_domain_event(AnalysisCreatedEvent(
            analysis_id=analysis._id,
            project_id=project_id,
            language=language,
        ))
        
        return analysis
    
    def start(self):
        """Start the analysis."""
        if self.status != AnalysisStatus.PENDING:
            raise InvalidOperationError(
                f"Cannot start analysis in {self.status} status"
            )
        
        self.status = AnalysisStatus.RUNNING
        self.increment_version()
    
    def add_issue(self, issue: Issue):
        """Add an issue to the analysis."""
        if self.status != AnalysisStatus.RUNNING:
            raise InvalidOperationError(
                f"Cannot add issues in {self.status} status"
            )
        
        self._issues.append(issue)
        
        # Emit event
        self.add_domain_event(IssueDetectedEvent(
            analysis_id=self._id,
            issue_id=issue.id,
            severity=issue.severity.value,
        ))
    
    def complete(self, lines_analyzed: int, analysis_time_ms: float):
        """Mark analysis as completed."""
        if self.status != AnalysisStatus.RUNNING:
            raise InvalidOperationError(
                f"Cannot complete analysis in {self.status} status"
            )
        
        self.status = AnalysisStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        
        # Calculate metrics
        self._metrics = AnalysisMetrics(
            total_issues=len(self._issues),
            critical_count=len([i for i in self._issues if i.severity == Severity.CRITICAL]),
            high_count=len([i for i in self._issues if i.severity == Severity.HIGH]),
            medium_count=len([i for i in self._issues if i.severity == Severity.MEDIUM]),
            low_count=len([i for i in self._issues if i.severity == Severity.LOW]),
            analysis_time_ms=analysis_time_ms,
            lines_analyzed=lines_analyzed,
        )
        
        self.increment_version()
        
        # Emit event
        self.add_domain_event(AnalysisCompletedEvent(
            analysis_id=self._id,
            total_issues=len(self._issues),
            analysis_time_ms=analysis_time_ms,
        ))
    
    def fail(self, error_message: str):
        """Mark analysis as failed."""
        if self.status not in [AnalysisStatus.PENDING, AnalysisStatus.RUNNING]:
            raise InvalidOperationError(
                f"Cannot fail analysis in {self.status} status"
            )
        
        self.status = AnalysisStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc)
        self.increment_version()
        
        # Emit event
        self.add_domain_event(AnalysisFailedEvent(
            analysis_id=self._id,
            error_message=error_message,
        ))
    
    def resolve_issue(self, issue_id: str):
        """Resolve a specific issue."""
        for issue in self._issues:
            if issue.id == issue_id:
                issue.resolve()
                self.increment_version()
                return
        
        raise EntityNotFoundError(f"Issue not found: {issue_id}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "project_id": self.project_id,
            "language": self.language,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "issues_count": len(self._issues),
            "metrics": {
                "total_issues": self._metrics.total_issues if self._metrics else 0,
                "analysis_time_ms": self._metrics.analysis_time_ms if self._metrics else 0,
            } if self._metrics else None,
            "version": self._version,
        }


# =============================================================================
# Version Management Domain - Value Objects
# =============================================================================

@dataclass(frozen=True)
class VersionId(ValueObject):
    """Identifier for a version."""
    zone: str  # v1, v2, v3
    version_number: int
    
    def __str__(self) -> str:
        return f"{self.zone}-{self.version_number}"


@dataclass(frozen=True)
class ModelConfiguration(ValueObject):
    """Configuration for an AI model."""
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationMetrics(ValueObject):
    """Metrics from an experiment evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_p95_ms: float
    cost_per_request: float
    error_rate: float
    
    def meets_promotion_criteria(
        self,
        min_accuracy: float = 0.85,
        max_error_rate: float = 0.05,
        max_latency_ms: float = 3000
    ) -> bool:
        """Check if metrics meet promotion criteria."""
        return (
            self.accuracy >= min_accuracy and
            self.error_rate <= max_error_rate and
            self.latency_p95_ms <= max_latency_ms
        )


# =============================================================================
# Version Management Domain - Entities
# =============================================================================

class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    CREATED = "created"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"


@dataclass
class Evaluation(Entity):
    """An evaluation of an experiment."""
    _id: str
    experiment_id: str
    metrics: EvaluationMetrics
    sample_size: int
    evaluated_at: datetime
    evaluator: str
    
    @property
    def id(self) -> str:
        return self._id


@dataclass
class Experiment(AggregateRoot):
    """
    Experiment aggregate root.
    
    Represents an experiment in V1 zone.
    """
    _id: str
    name: str
    configuration: ModelConfiguration
    status: ExperimentStatus
    created_at: datetime
    created_by: str
    
    # Evaluations
    _evaluations: List[Evaluation] = field(default_factory=list)
    
    # Tracking
    baseline_id: Optional[str] = None
    promoted_at: Optional[datetime] = None
    quarantined_at: Optional[datetime] = None
    quarantine_reason: Optional[str] = None
    
    def __post_init__(self):
        super().__init__()
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def evaluations(self) -> List[Evaluation]:
        return self._evaluations.copy()
    
    @property
    def latest_evaluation(self) -> Optional[Evaluation]:
        if not self._evaluations:
            return None
        return max(self._evaluations, key=lambda e: e.evaluated_at)
    
    @classmethod
    def create(
        cls,
        name: str,
        configuration: ModelConfiguration,
        created_by: str,
        baseline_id: Optional[str] = None
    ) -> 'Experiment':
        return cls(
            _id=str(uuid4()),
            name=name,
            configuration=configuration,
            status=ExperimentStatus.CREATED,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            baseline_id=baseline_id,
        )
    
    def start(self):
        """Start the experiment."""
        if self.status != ExperimentStatus.CREATED:
            raise InvalidOperationError(f"Cannot start in {self.status} status")
        self.status = ExperimentStatus.RUNNING
        self.increment_version()
    
    def add_evaluation(self, metrics: EvaluationMetrics, sample_size: int, evaluator: str):
        """Add an evaluation result."""
        if self.status not in [ExperimentStatus.RUNNING, ExperimentStatus.EVALUATING]:
            raise InvalidOperationError(f"Cannot evaluate in {self.status} status")
        
        evaluation = Evaluation(
            _id=str(uuid4()),
            experiment_id=self._id,
            metrics=metrics,
            sample_size=sample_size,
            evaluated_at=datetime.now(timezone.utc),
            evaluator=evaluator,
        )
        
        self._evaluations.append(evaluation)
        self.status = ExperimentStatus.EVALUATING
        self.increment_version()
    
    def complete(self):
        """Mark experiment as completed."""
        if not self._evaluations:
            raise InvalidOperationError("Cannot complete without evaluations")
        self.status = ExperimentStatus.COMPLETED
        self.increment_version()
    
    def promote(self):
        """Promote experiment to V2."""
        if self.status != ExperimentStatus.COMPLETED:
            raise InvalidOperationError(f"Cannot promote in {self.status} status")
        
        self.status = ExperimentStatus.PROMOTED
        self.promoted_at = datetime.now(timezone.utc)
        self.increment_version()
        
        self.add_domain_event(ExperimentPromotedEvent(
            experiment_id=self._id,
            configuration=self.configuration,
        ))
    
    def quarantine(self, reason: str):
        """Quarantine experiment to V3."""
        self.status = ExperimentStatus.QUARANTINED
        self.quarantined_at = datetime.now(timezone.utc)
        self.quarantine_reason = reason
        self.increment_version()
        
        self.add_domain_event(ExperimentQuarantinedEvent(
            experiment_id=self._id,
            reason=reason,
        ))


# =============================================================================
# Domain Events
# =============================================================================

@dataclass
class DomainEvent:
    """Base class for domain events."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def event_type(self) -> str:
        return self.__class__.__name__


@dataclass
class AnalysisCreatedEvent(DomainEvent):
    analysis_id: str = ""
    project_id: str = ""
    language: str = ""


@dataclass
class AnalysisCompletedEvent(DomainEvent):
    analysis_id: str = ""
    total_issues: int = 0
    analysis_time_ms: float = 0


@dataclass
class AnalysisFailedEvent(DomainEvent):
    analysis_id: str = ""
    error_message: str = ""


@dataclass
class IssueDetectedEvent(DomainEvent):
    analysis_id: str = ""
    issue_id: str = ""
    severity: str = ""


@dataclass
class ExperimentPromotedEvent(DomainEvent):
    experiment_id: str = ""
    configuration: Optional[ModelConfiguration] = None


@dataclass
class ExperimentQuarantinedEvent(DomainEvent):
    experiment_id: str = ""
    reason: str = ""


# =============================================================================
# Exceptions
# =============================================================================

class DomainError(Exception):
    """Base exception for domain errors."""
    pass


class InvalidOperationError(DomainError):
    """Raised when an invalid operation is attempted."""
    pass


class EntityNotFoundError(DomainError):
    """Raised when an entity is not found."""
    pass


class InvariantViolationError(DomainError):
    """Raised when an aggregate invariant is violated."""
    pass
