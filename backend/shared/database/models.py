"""
Shared SQLAlchemy Models

Core database models used across services.
"""

import uuid
from datetime import datetime
from typing import Optional, List
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Text, Boolean, DateTime, Integer, Float,
    ForeignKey, JSON, Enum, Index, UniqueConstraint, Table
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


# ============================================
# Enums
# ============================================

class UserRole(str, PyEnum):
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class UserStatus(str, PyEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class ProjectStatus(str, PyEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class AnalysisStatus(str, PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IssueSeverity(str, PyEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class OAuthProvider(str, PyEnum):
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    GOOGLE = "google"


# ============================================
# Association Tables
# ============================================

project_members = Table(
    "project_members",
    Base.metadata,
    Column("project_id", UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE")),
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")),
    Column("role", String(50), default="member"),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("project_id", "user_id", name="uq_project_member"),
)


# ============================================
# User Models
# ============================================

class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    avatar_url = Column(String(500))
    
    # Status and role
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    status = Column(Enum(UserStatus), default=UserStatus.PENDING, nullable=False)
    email_verified = Column(Boolean, default=False)
    
    # 2FA
    totp_secret = Column(String(32))
    totp_enabled = Column(Boolean, default=False)
    backup_codes = Column(ARRAY(String(20)))
    
    # Metadata
    last_login_at = Column(DateTime)
    last_login_ip = Column(String(45))
    login_count = Column(Integer, default=0)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    projects = relationship("Project", back_populates="owner")
    oauth_connections = relationship("OAuthConnection", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")
    
    __table_args__ = (
        Index("ix_users_email_status", "email", "status"),
    )


class UserSession(Base):
    """User session for tracking active logins."""
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Session info
    refresh_token_hash = Column(String(64), unique=True, nullable=False)
    device_info = Column(String(500))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Status
    is_active = Column(Boolean, default=True)
    last_activity_at = Column(DateTime, server_default=func.now())
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class OAuthConnection(Base):
    """OAuth provider connections."""
    __tablename__ = "oauth_connections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Provider info
    provider = Column(Enum(OAuthProvider), nullable=False)
    provider_user_id = Column(String(255), nullable=False)
    provider_username = Column(String(255))
    provider_email = Column(String(255))
    
    # Tokens (encrypted at rest)
    access_token_encrypted = Column(Text)
    refresh_token_encrypted = Column(Text)
    token_expires_at = Column(DateTime)
    
    # Scopes
    scopes = Column(ARRAY(String(100)))
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="oauth_connections")
    
    __table_args__ = (
        UniqueConstraint("provider", "provider_user_id", name="uq_oauth_provider_user"),
        Index("ix_oauth_user_provider", "user_id", "provider"),
    )


class APIKey(Base):
    """API keys for programmatic access."""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Key info
    name = Column(String(255), nullable=False)
    key_prefix = Column(String(10), nullable=False)  # First 8 chars for identification
    key_hash = Column(String(64), unique=True, nullable=False)
    
    # Permissions
    scopes = Column(ARRAY(String(100)), default=["read"])
    
    # Usage
    last_used_at = Column(DateTime)
    usage_count = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")


# ============================================
# Project Models
# ============================================

class Project(Base):
    """Project model."""
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Owner
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Repository info
    repository_url = Column(String(500))
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="SET NULL"))
    default_branch = Column(String(100), default="main")
    
    # Tech stack
    language = Column(String(50))
    framework = Column(String(100))
    
    # Settings
    settings = Column(JSONB, default={})
    
    # Status
    status = Column(Enum(ProjectStatus), default=ProjectStatus.ACTIVE, nullable=False)
    is_public = Column(Boolean, default=False)
    
    # Stats (denormalized for performance)
    total_issues = Column(Integer, default=0)
    open_issues = Column(Integer, default=0)
    last_analysis_at = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    repository = relationship("Repository", back_populates="projects")
    analyses = relationship("Analysis", back_populates="project")
    
    __table_args__ = (
        Index("ix_projects_owner_status", "owner_id", "status"),
        Index("ix_projects_language", "language"),
    )


class Repository(Base):
    """Git repository model."""
    __tablename__ = "repositories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Provider info
    provider = Column(Enum(OAuthProvider), nullable=False)
    provider_repo_id = Column(String(255), nullable=False)
    
    # Repo info
    full_name = Column(String(255), nullable=False)  # e.g., "owner/repo"
    name = Column(String(255), nullable=False)
    owner = Column(String(255), nullable=False)
    description = Column(Text)
    url = Column(String(500), nullable=False)
    clone_url = Column(String(500))
    ssh_url = Column(String(500))
    
    # Default branch
    default_branch = Column(String(100), default="main")
    
    # Stats
    stars = Column(Integer, default=0)
    forks = Column(Integer, default=0)
    open_issues = Column(Integer, default=0)
    
    # Settings
    is_private = Column(Boolean, default=False)
    is_fork = Column(Boolean, default=False)
    
    # Webhook
    webhook_id = Column(String(255))
    webhook_secret = Column(String(100))
    
    # Clone info
    local_path = Column(String(500))
    last_synced_at = Column(DateTime)
    last_commit_sha = Column(String(40))
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    projects = relationship("Project", back_populates="repository")
    
    __table_args__ = (
        UniqueConstraint("provider", "provider_repo_id", name="uq_repo_provider"),
        Index("ix_repos_full_name", "full_name"),
    )


# ============================================
# Analysis Models
# ============================================

class Analysis(Base):
    """Code analysis session."""
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    
    # Analysis info
    commit_sha = Column(String(40))
    branch = Column(String(100))
    trigger = Column(String(50))  # manual, push, pull_request, scheduled
    
    # Status
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False)
    progress = Column(Float, default=0.0)
    
    # Results (denormalized)
    total_files = Column(Integer, default=0)
    analyzed_files = Column(Integer, default=0)
    total_issues = Column(Integer, default=0)
    critical_issues = Column(Integer, default=0)
    high_issues = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    
    # Error info
    error_message = Column(Text)
    
    # AI model info
    ai_model_version = Column(String(50))
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="analyses")
    issues = relationship("Issue", back_populates="analysis")


class Issue(Base):
    """Code issue found during analysis."""
    __tablename__ = "issues"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    
    # Location
    file_path = Column(String(500), nullable=False)
    line_start = Column(Integer)
    line_end = Column(Integer)
    column_start = Column(Integer)
    column_end = Column(Integer)
    
    # Issue info
    rule_id = Column(String(100), nullable=False)
    category = Column(String(50))  # security, performance, quality, style
    severity = Column(Enum(IssueSeverity), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    
    # Code context
    code_snippet = Column(Text)
    suggested_fix = Column(Text)
    
    # Status
    is_resolved = Column(Boolean, default=False)
    is_false_positive = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    # AI confidence
    confidence = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="issues")
    
    __table_args__ = (
        Index("ix_issues_analysis_severity", "analysis_id", "severity"),
        Index("ix_issues_file", "file_path"),
    )


# ============================================
# Audit Log
# ============================================

class AuditLog(Base):
    """Audit log for tracking all changes."""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Actor
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    user_email = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Action
    action = Column(String(100), nullable=False)  # e.g., "user.login", "project.create"
    resource_type = Column(String(50))
    resource_id = Column(String(255))
    
    # Details
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    metadata = Column(JSONB)
    
    # Result
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index("ix_audit_user_action", "user_id", "action"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
        Index("ix_audit_created", "created_at"),
    )
