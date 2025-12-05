"""
Database models for analysis service.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, Enum, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

Base = declarative_base()


class SessionStatus(str, enum.Enum):
    """Analysis session status."""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, enum.Enum):
    """Analysis task type."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    GRAPH = "graph"


class TaskStatus(str, enum.Enum):
    """Analysis task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactType(str, enum.Enum):
    """Artifact type."""
    REPORT = "report"
    PATCH = "patch"
    LOG = "log"
    METRICS = "metrics"


class AnalysisSession(Base):
    """Analysis session model."""
    __tablename__ = "analysis_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    version = Column(String(10), nullable=False)  # v1, v2, v3
    status = Column(Enum(SessionStatus), default=SessionStatus.CREATED, nullable=False)
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    finished_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    metadata = Column(JSON, default={}, nullable=False)  # Custom metadata

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "project_id": str(self.project_id),
            "version": self.version,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class AnalysisTask(Base):
    """Analysis task model."""
    __tablename__ = "analysis_tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("analysis_sessions.id"), nullable=False, index=True)
    type = Column(Enum(TaskType), nullable=False)  # static, dynamic, graph
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False)
    result = Column(JSON, nullable=True)  # Task result
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "type": self.type.value,
            "status": self.status.value,
            "result": self.result,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat(),
        }


class Artifact(Base):
    """Analysis artifact model."""
    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("analysis_sessions.id"), nullable=False, index=True)
    type = Column(Enum(ArtifactType), nullable=False)  # report, patch, log, metrics
    s3_uri = Column(String(255), nullable=False)  # s3://bucket/path/to/artifact
    sha256 = Column(String(64), nullable=False)  # SHA256 checksum
    size_bytes = Column(Integer, nullable=False)
    metadata = Column(JSON, default={}, nullable=False)  # Custom metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "type": self.type.value,
            "s3_uri": self.s3_uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class AnalysisCache(Base):
    """Analysis result cache model."""
    __tablename__ = "analysis_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    code_hash = Column(String(64), nullable=False, unique=True)  # SHA256 of code
    language = Column(String(50), nullable=False)
    result = Column(JSON, nullable=False)  # Cached analysis result
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    expires_at = Column(DateTime, nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "code_hash": self.code_hash,
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }
