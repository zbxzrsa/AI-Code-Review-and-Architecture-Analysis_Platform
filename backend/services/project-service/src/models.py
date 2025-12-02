"""
Database models for project service.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, Enum, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

Base = declarative_base()


class VersionTag(str, enum.Enum):
    """Version tags."""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class Project(Base):
    """Project model."""
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    settings = Column(JSON, default={}, nullable=False)  # Project settings
    archived = Column(Boolean, default=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "owner_id": str(self.owner_id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "settings": self.settings,
            "archived": self.archived,
        }


class Version(Base):
    """Version model."""
    __tablename__ = "versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    tag = Column(Enum(VersionTag), nullable=False)  # v1, v2, v3
    model_config = Column(JSON, nullable=False)  # AI model configuration
    changelog = Column(Text, nullable=True)
    promoted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "tag": self.tag.value,
            "model_config": self.model_config,
            "changelog": self.changelog,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Baseline(Base):
    """Baseline metrics model."""
    __tablename__ = "baselines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    metric_key = Column(String(255), nullable=False)  # e.g., "accuracy", "latency_ms"
    threshold = Column(String(255), nullable=False)  # Numeric threshold
    operator = Column(String(10), nullable=False)  # >, <, >=, <=, ==, !=
    snapshot_id = Column(String(255), nullable=True)  # Reference to snapshot
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "metric_key": self.metric_key,
            "threshold": self.threshold,
            "operator": self.operator,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at.isoformat(),
        }


class Policy(Base):
    """OPA policy model."""
    __tablename__ = "policies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    rego_code = Column(Text, nullable=False)  # Rego policy code
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
        }


class ProjectPolicy(Base):
    """Project-Policy association model."""
    __tablename__ = "project_policies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    policy_id = Column(UUID(as_uuid=True), ForeignKey("policies.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "policy_id": str(self.policy_id),
            "created_at": self.created_at.isoformat(),
        }


class VersionHistory(Base):
    """Version change history model."""
    __tablename__ = "version_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version_id = Column(UUID(as_uuid=True), ForeignKey("versions.id"), nullable=False, index=True)
    changed_by = Column(UUID(as_uuid=True), nullable=False)
    action = Column(String(50), nullable=False)  # promoted, degraded, updated
    from_tag = Column(Enum(VersionTag), nullable=True)
    to_tag = Column(Enum(VersionTag), nullable=True)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "version_id": str(self.version_id),
            "changed_by": str(self.changed_by),
            "action": self.action,
            "from_tag": self.from_tag.value if self.from_tag else None,
            "to_tag": self.to_tag.value if self.to_tag else None,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }
