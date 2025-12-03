"""
V2 VC-AI Version Management Router

API endpoints for version history and release management.
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from ..models.version_models import (
    Version,
    VersionCreate,
    VersionUpdate,
    VersionMetadata,
    VersionHistory,
    ReleaseRequest,
    ReleaseResponse,
    VersionStatus,
    VersionComparison,
    VersionTimeline,
)


router = APIRouter(prefix="/versions", tags=["versions"])


# =============================================================================
# In-memory storage (replace with database in production)
# =============================================================================

_versions: dict = {}
_version_counter = 0


def _generate_version_id() -> str:
    global _version_counter
    _version_counter += 1
    return f"ver_{_version_counter:08d}"


# =============================================================================
# Version Management Endpoints
# =============================================================================

@router.get("", response_model=VersionHistory)
async def list_versions(
    sort_by: str = Query(default="creation_date", enum=["creation_date", "name", "release_status"]),
    filter: str = Query(default="active", enum=["active", "archived", "deprecated", "all"]),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> VersionHistory:
    """
    List all versions with filtering and pagination.
    
    SLA: <= 100ms response time
    """
    versions = list(_versions.values())
    
    # Filter
    if filter != "all":
        status_map = {
            "active": VersionStatus.ACTIVE,
            "archived": VersionStatus.ARCHIVED,
            "deprecated": VersionStatus.DEPRECATED,
        }
        if filter in status_map:
            versions = [v for v in versions if v.metadata.status == status_map[filter]]
    
    # Sort
    if sort_by == "creation_date":
        versions.sort(key=lambda v: v.metadata.created_at, reverse=True)
    elif sort_by == "name":
        versions.sort(key=lambda v: v.metadata.name)
    elif sort_by == "release_status":
        versions.sort(key=lambda v: v.metadata.status.value)
    
    total = len(versions)
    versions = versions[offset:offset + limit]
    
    return VersionHistory(
        versions=[v.metadata for v in versions],
        total_count=total,
        page=(offset // limit) + 1,
        page_size=limit,
        has_more=offset + limit < total,
    )


@router.get("/{version_id}", response_model=Version)
async def get_version(version_id: str) -> Version:
    """
    Get detailed version information.
    
    Returns complete version data including changes, affected components,
    stability score, and test coverage.
    """
    if version_id not in _versions:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    
    return _versions[version_id]


@router.post("", response_model=Version, status_code=201)
async def create_version(request: VersionCreate) -> Version:
    """
    Create a new version.
    
    Validates semantic versioning and initializes version metadata.
    """
    version_id = _generate_version_id()
    
    metadata = VersionMetadata(
        id=version_id,
        name=request.name,
        description=request.description,
        status=VersionStatus.DRAFT if request.pre_release else VersionStatus.ACTIVE,
        created_by="system",  # Would come from auth in production
        tags=request.tags,
    )
    
    version = Version(
        metadata=metadata,
        changes=request.changes,
        affected_components=[],
        stability_score=100.0,
        test_coverage=0.0,
        documentation_coverage=0.0,
        breaking_changes_count=sum(1 for c in request.changes if c.breaking),
    )
    
    _versions[version_id] = version
    return version


@router.patch("/{version_id}", response_model=Version)
async def update_version(version_id: str, request: VersionUpdate) -> Version:
    """Update version metadata."""
    if version_id not in _versions:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    
    version = _versions[version_id]
    
    if request.description is not None:
        version.metadata.description = request.description
    if request.status is not None:
        version.metadata.status = request.status
    if request.tags is not None:
        version.metadata.tags = request.tags
    
    version.metadata.updated_at = datetime.utcnow()
    
    return version


@router.delete("/{version_id}", status_code=204)
async def delete_version(version_id: str) -> None:
    """
    Delete a version (archive it).
    
    Note: Versions are never truly deleted for audit purposes.
    """
    if version_id not in _versions:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    
    # Archive instead of delete
    _versions[version_id].metadata.status = VersionStatus.ARCHIVED
    _versions[version_id].metadata.updated_at = datetime.utcnow()


# =============================================================================
# Release Endpoints
# =============================================================================

@router.post("/release", response_model=ReleaseResponse, status_code=201)
async def create_release(request: ReleaseRequest) -> ReleaseResponse:
    """
    Create a new release.
    
    Generates release notes, creates tag, and publishes the release.
    """
    version_id = _generate_version_id()
    
    # Create version for the release
    metadata = VersionMetadata(
        id=version_id,
        name=request.version_name,
        description=request.release_notes[:200] if request.release_notes else None,
        status=VersionStatus.DRAFT if request.draft else VersionStatus.ACTIVE,
        release_date=None if request.draft else datetime.utcnow(),
        created_by="system",
    )
    
    version = Version(
        metadata=metadata,
        changes=request.changelog,
        release_notes_url=f"/releases/{version_id}/notes",
    )
    
    _versions[version_id] = version
    
    return ReleaseResponse(
        version_id=version_id,
        release_url=f"/releases/{version_id}",
        tag_name=f"v{request.version_name}",
        created_at=datetime.utcnow(),
        release_notes=request.release_notes,
        status="draft" if request.draft else "published",
    )


# =============================================================================
# Comparison Endpoints
# =============================================================================

@router.get("/compare/{from_version}/{to_version}", response_model=VersionComparison)
async def compare_versions(from_version: str, to_version: str) -> VersionComparison:
    """
    Compare two versions.
    
    Returns diff information including breaking changes and migration requirements.
    """
    if from_version not in _versions:
        raise HTTPException(status_code=404, detail=f"Version {from_version} not found")
    if to_version not in _versions:
        raise HTTPException(status_code=404, detail=f"Version {to_version} not found")
    
    v1 = _versions[from_version]
    v2 = _versions[to_version]
    
    # Simple comparison logic
    v1_changes = set(c.commit_hash for c in v1.changes)
    v2_changes = set(c.commit_hash for c in v2.changes)
    
    added = len(v2_changes - v1_changes)
    removed = len(v1_changes - v2_changes)
    
    breaking = [c.description for c in v2.changes if c.breaking]
    
    return VersionComparison(
        from_version=from_version,
        to_version=to_version,
        changes_added=added,
        changes_removed=removed,
        changes_modified=0,
        breaking_changes=breaking,
        migration_required=len(breaking) > 0,
        compatibility="breaking" if breaking else "backward_compatible",
        diff_url=f"/versions/diff/{from_version}...{to_version}",
    )


# =============================================================================
# Timeline Endpoints
# =============================================================================

@router.get("/timeline", response_model=List[VersionTimeline])
async def get_version_timeline(
    limit: int = Query(default=20, ge=1, le=100),
) -> List[VersionTimeline]:
    """
    Get visual timeline of releases.
    
    Returns versions ordered by release date for timeline visualization.
    """
    versions = [v for v in _versions.values() if v.metadata.release_date]
    versions.sort(key=lambda v: v.metadata.release_date, reverse=True)
    versions = versions[:limit]
    
    timeline = []
    for v in versions:
        # Determine version type from name
        name = v.metadata.name
        if "alpha" in name or "beta" in name or "rc" in name:
            v_type = "pre-release"
        elif name.split(".")[0] != "0":
            parts = name.split(".")
            if len(parts) >= 2 and parts[1] == "0":
                v_type = "major"
            elif len(parts) >= 3 and parts[2] == "0":
                v_type = "minor"
            else:
                v_type = "patch"
        else:
            v_type = "patch"
        
        timeline.append(VersionTimeline(
            version=v.metadata.name,
            date=v.metadata.release_date,
            type=v_type,
            highlights=[c.description[:50] for c in v.changes[:3]],
            contributors=list(set(c.author for c in v.changes)),
        ))
    
    return timeline
