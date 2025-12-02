"""
Database models for repo service.
"""
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text, JSON, Enum, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

Base = declarative_base()


class ProviderType(str, enum.Enum):
    """Repository provider types."""
    GITHUB = "github"
    GITLAB = "gitlab"


class PRStatus(str, enum.Enum):
    """Pull request status."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class CommentAuthor(str, enum.Enum):
    """Comment author type."""
    AI = "ai"
    USER = "user"


class Repository(Base):
    """Repository model."""
    __tablename__ = "repositories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    provider = Column(Enum(ProviderType), nullable=False)  # github, gitlab
    repo_url = Column(String(255), nullable=False, unique=True)
    repo_name = Column(String(255), nullable=False)
    owner = Column(String(255), nullable=False)
    webhook_secret = Column(String(255), nullable=False)
    access_token = Column(String(255), nullable=False)  # Encrypted
    webhook_id = Column(String(255), nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "provider": self.provider.value,
            "repo_url": self.repo_url,
            "repo_name": self.repo_name,
            "owner": self.owner,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
        }


class PullRequest(Base):
    """Pull request model."""
    __tablename__ = "pull_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=False, index=True)
    pr_number = Column(Integer, nullable=False)
    pr_title = Column(String(255), nullable=False)
    pr_url = Column(String(255), nullable=False)
    status = Column(Enum(PRStatus), default=PRStatus.PENDING, nullable=False)
    analysis_id = Column(UUID(as_uuid=True), nullable=True)  # Reference to analysis session
    author = Column(String(255), nullable=False)
    branch = Column(String(255), nullable=False)
    base_branch = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    analyzed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "repo_id": str(self.repo_id),
            "pr_number": self.pr_number,
            "pr_title": self.pr_title,
            "pr_url": self.pr_url,
            "status": self.status.value,
            "analysis_id": str(self.analysis_id) if self.analysis_id else None,
            "author": self.author,
            "branch": self.branch,
            "base_branch": self.base_branch,
            "created_at": self.created_at.isoformat(),
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
        }


class PRComment(Base):
    """Pull request comment model."""
    __tablename__ = "pr_comments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pr_id = Column(UUID(as_uuid=True), ForeignKey("pull_requests.id"), nullable=False, index=True)
    file_path = Column(String(255), nullable=False)
    line_number = Column(Integer, nullable=False)
    comment = Column(Text, nullable=False)
    author = Column(Enum(CommentAuthor), default=CommentAuthor.AI, nullable=False)
    severity = Column(String(50), nullable=True)  # critical, high, medium, low, info
    category = Column(String(100), nullable=True)  # security, performance, style, etc.
    suggestion = Column(Text, nullable=True)  # Suggested fix
    external_id = Column(String(255), nullable=True)  # GitHub/GitLab comment ID
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "pr_id": str(self.pr_id),
            "file_path": self.file_path,
            "line_number": self.line_number,
            "comment": self.comment,
            "author": self.author.value,
            "severity": self.severity,
            "category": self.category,
            "suggestion": self.suggestion,
            "created_at": self.created_at.isoformat(),
        }


class FileCache(Base):
    """File tree cache model."""
    __tablename__ = "file_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id"), nullable=False, index=True)
    branch = Column(String(255), nullable=False)
    file_tree = Column(JSON, nullable=False)  # Cached file tree structure
    commit_sha = Column(String(40), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "repo_id": str(self.repo_id),
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }
