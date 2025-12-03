"""
V2 VC-AI Version Models

Data models for version management operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class VersionStatus(str, Enum):
    """Version status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ChangeEntry(BaseModel):
    """Individual change entry in a version"""
    type: str = Field(..., description="Type of change (feat, fix, docs, etc.)")
    scope: Optional[str] = Field(None, description="Scope of the change")
    description: str = Field(..., description="Description of the change")
    commit_hash: str = Field(..., description="Associated commit hash")
    author: str = Field(..., description="Author of the change")
    timestamp: datetime = Field(..., description="When the change was made")
    breaking: bool = Field(default=False, description="Whether this is a breaking change")


class VersionMetadata(BaseModel):
    """Version metadata"""
    id: str = Field(..., description="Unique version identifier")
    name: str = Field(..., description="Version name (semantic version)")
    description: Optional[str] = Field(None, description="Version description")
    status: VersionStatus = Field(default=VersionStatus.ACTIVE)
    release_date: Optional[datetime] = Field(None, description="Release date")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(..., description="User who created the version")
    tags: List[str] = Field(default_factory=list)


class Version(BaseModel):
    """Complete version information"""
    metadata: VersionMetadata
    changes: List[ChangeEntry] = Field(default_factory=list)
    affected_components: List[str] = Field(default_factory=list)
    stability_score: float = Field(default=100.0, ge=0, le=100)
    test_coverage: float = Field(default=0.0, ge=0, le=100)
    documentation_coverage: float = Field(default=0.0, ge=0, le=100)
    breaking_changes_count: int = Field(default=0, ge=0)
    dependencies: Dict[str, str] = Field(default_factory=dict)
    changelog_url: Optional[str] = None
    release_notes_url: Optional[str] = None


class VersionCreate(BaseModel):
    """Request to create a new version"""
    name: str = Field(..., description="Version name (semantic version)", pattern=r"^\d+\.\d+\.\d+.*$")
    description: Optional[str] = None
    changes: List[ChangeEntry] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    release_notes: Optional[str] = None
    pre_release: bool = Field(default=False)


class VersionUpdate(BaseModel):
    """Request to update a version"""
    description: Optional[str] = None
    status: Optional[VersionStatus] = None
    tags: Optional[List[str]] = None
    release_notes: Optional[str] = None


class VersionHistory(BaseModel):
    """Version history response"""
    versions: List[VersionMetadata]
    total_count: int
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    has_more: bool = False


class ReleaseRequest(BaseModel):
    """Request to create a new release"""
    version_name: str = Field(..., description="Semantic version name", pattern=r"^\d+\.\d+\.\d+.*$")
    release_notes: str = Field(..., description="Release notes content")
    changelog: List[ChangeEntry] = Field(default_factory=list)
    target_branch: str = Field(default="main")
    tag_message: Optional[str] = None
    pre_release: bool = Field(default=False)
    draft: bool = Field(default=False)
    notify_subscribers: bool = Field(default=True)


class ReleaseResponse(BaseModel):
    """Response after creating a release"""
    version_id: str
    release_url: str
    tag_name: str
    created_at: datetime
    release_notes: str
    assets: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = Field(default="published")


class VersionComparison(BaseModel):
    """Version comparison result"""
    from_version: str
    to_version: str
    changes_added: int
    changes_removed: int
    changes_modified: int
    breaking_changes: List[str]
    migration_required: bool
    compatibility: str = Field(description="backward_compatible, forward_compatible, or breaking")
    diff_url: Optional[str] = None


class VersionTimeline(BaseModel):
    """Visual timeline entry for versions"""
    version: str
    date: datetime
    type: str = Field(description="major, minor, patch, pre-release")
    highlights: List[str] = Field(default_factory=list)
    contributors: List[str] = Field(default_factory=list)
