"""
Database models for version-control-service.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, Enum, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

Base = declarative_base()


class ExperimentStatus(str, enum.Enum):
    """Experiment status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EVALUATING = "evaluating"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"


class PromotionStatus(str, enum.Enum):
    """Promotion status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


class Experiment(Base):
    """Experiment model."""
    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(10), default="v1", nullable=False)  # v1
    config = Column(JSON, nullable=False)  # Model configuration
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.CREATED, nullable=False)
    created_by = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "config": self.config,
            "dataset_id": str(self.dataset_id),
            "status": self.status.value,
            "created_by": str(self.created_by),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Evaluation(Base):
    """Evaluation result model."""
    __tablename__ = "evaluations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False, index=True)
    metrics = Column(JSON, nullable=False)  # Evaluation metrics
    ai_verdict = Column(String(50), nullable=False)  # pass, fail, manual_review
    ai_confidence = Column(String(50), nullable=True)  # Confidence score
    human_override = Column(String(50), nullable=True)  # Human decision override
    override_reason = Column(Text, nullable=True)
    evaluated_by = Column(String(50), nullable=False)  # ai, human
    evaluated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "experiment_id": str(self.experiment_id),
            "metrics": self.metrics,
            "ai_verdict": self.ai_verdict,
            "ai_confidence": self.ai_confidence,
            "human_override": self.human_override,
            "override_reason": self.override_reason,
            "evaluated_by": self.evaluated_by,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


class Promotion(Base):
    """Promotion record model."""
    __tablename__ = "promotions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    from_version_id = Column(UUID(as_uuid=True), nullable=False)  # v1 experiment ID
    to_version_id = Column(UUID(as_uuid=True), nullable=False)  # v2 version ID
    status = Column(Enum(PromotionStatus), default=PromotionStatus.PENDING, nullable=False)
    reason = Column(Text, nullable=False)
    approver_id = Column(UUID(as_uuid=True), nullable=True)
    promoted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "from_version_id": str(self.from_version_id),
            "to_version_id": str(self.to_version_id),
            "status": self.status.value,
            "reason": self.reason,
            "approver_id": str(self.approver_id) if self.approver_id else None,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "created_at": self.created_at.isoformat(),
        }


class Blacklist(Base):
    """Blacklist entry model."""
    __tablename__ = "blacklist"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_hash = Column(String(64), unique=True, nullable=False, index=True)
    reason = Column(Text, nullable=False)
    evidence = Column(JSON, nullable=False)  # Evidence of failure
    quarantined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    review_status = Column(String(50), default="pending", nullable=False)  # pending, reviewed, appealed
    reviewed_by = Column(UUID(as_uuid=True), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "config_hash": self.config_hash,
            "reason": self.reason,
            "evidence": self.evidence,
            "quarantined_at": self.quarantined_at.isoformat(),
            "review_status": self.review_status,
            "reviewed_by": str(self.reviewed_by) if self.reviewed_by else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


class ComparisonReport(Base):
    """Version comparison report model."""
    __tablename__ = "comparison_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    v1_experiment_id = Column(UUID(as_uuid=True), nullable=False)
    v2_version_id = Column(UUID(as_uuid=True), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    metrics = Column(JSON, nullable=False)  # Comparison metrics
    recommendation = Column(String(255), nullable=False)
    confidence = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "v1_experiment_id": str(self.v1_experiment_id),
            "v2_version_id": str(self.v2_version_id),
            "dataset_id": str(self.dataset_id),
            "metrics": self.metrics,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }
