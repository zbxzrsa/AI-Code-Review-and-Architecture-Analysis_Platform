"""
Projects Routes

Project management endpoints.
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from ..models import Project, ProjectSettings, CreateProjectRequest, UpdateProjectRequest
from ..mock_data import mock_projects
from ..config import Literals, logger

router = APIRouter(prefix="/api/projects", tags=["Projects"])


@router.get("")
async def list_projects(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = None
):
    """List projects."""
    projects = mock_projects
    
    if search:
        projects = [p for p in projects if search.lower() in p.name.lower()]
    
    return {
        "items": projects,
        "total": len(projects),
        "page": page,
        "limit": limit
    }


@router.get("/{project_id}")
async def get_project(project_id: str):
    """Get project by ID."""
    for project in mock_projects:
        if project.id == project_id:
            return project
    raise HTTPException(status_code=404, detail=Literals.PROJECT_NOT_FOUND)


@router.post("")
async def create_project(request: CreateProjectRequest):
    """Create project."""
    settings_data = request.settings or {}
    project_settings = ProjectSettings(
        auto_review=settings_data.get("auto_review", True),
        review_on_push=settings_data.get("review_on_push", True),
        review_on_pr=settings_data.get("review_on_pr", True),
        severity_threshold=settings_data.get("severity_threshold", "warning"),
        enabled_rules=settings_data.get("enabled_rules", []),
        ignored_paths=settings_data.get("ignored_paths", ["node_modules", ".git", "__pycache__", "dist", "build"])
    )
    
    project = Project(
        id=f"proj_{secrets.token_hex(4)}",
        name=request.name,
        description=request.description or "",
        language=request.language,
        framework=request.framework or "",
        repository_url=request.repository_url or "",
        status="active",
        issues_count=0,
        settings=project_settings,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    mock_projects.append(project)
    logger.info(f"Created project: {project.id} - {project.name}")
    return project


@router.put("/{project_id}")
async def update_project(project_id: str, request: UpdateProjectRequest):
    """Update project."""
    for project in mock_projects:
        if project.id == project_id:
            if request.name:
                project.name = request.name
            if request.description is not None:
                project.description = request.description
            if request.language:
                project.language = request.language
            if request.framework is not None:
                project.framework = request.framework
            if request.status:
                project.status = request.status
            project.updated_at = datetime.now()
            return project
    raise HTTPException(status_code=404, detail=Literals.PROJECT_NOT_FOUND)


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete project."""
    for i, project in enumerate(mock_projects):
        if project.id == project_id:
            mock_projects.pop(i)
            return {"message": "Project deleted", "id": project_id}
    raise HTTPException(status_code=404, detail=Literals.PROJECT_NOT_FOUND)


# ============================================
# Project Team
# ============================================

@router.get("/{project_id}/team")
async def get_project_team(project_id: str):
    """Get project team members."""
    return {
        "items": [
            {
                "id": "member_1",
                "user_id": "user_1",
                "name": Literals.JOHN_DOE,
                "email": Literals.JOHN_EMAIL,
                "role": "owner",
                "avatar": None,
                "joined_at": (datetime.now() - timedelta(days=30)).isoformat()
            },
            {
                "id": "member_2",
                "user_id": "user_2",
                "name": Literals.JANE_SMITH,
                "email": Literals.JANE_EMAIL,
                "role": "admin",
                "avatar": None,
                "joined_at": (datetime.now() - timedelta(days=15)).isoformat()
            }
        ]
    }


@router.post("/{project_id}/team")
async def invite_team_member(project_id: str):
    """Invite team member."""
    return {
        "id": f"member_{secrets.token_hex(4)}",
        "message": "Invitation sent"
    }


# ============================================
# Project Files
# ============================================

@router.get("/{project_id}/files")
async def get_project_files(project_id: str, path: Optional[str] = None):
    """Get project files."""
    return {
        "items": [
            {"name": "src", "type": "directory", "path": "src"},
            {"name": "tests", "type": "directory", "path": "tests"},
            {"name": "README.md", "type": "file", "path": "README.md", "size": 2048},
            {"name": "package.json", "type": "file", "path": "package.json", "size": 1024},
        ]
    }


@router.get("/{project_id}/files/content")
async def get_file_content(project_id: str, path: str):
    """Get file content."""
    return {
        "path": path,
        "content": f"# Sample content for {path}\n\nThis is mock content.",
        "language": "python" if path.endswith(".py") else "typescript",
    }


# ============================================
# Project Analysis History
# ============================================

@router.get("/{project_id}/analyses")
async def get_project_analyses(project_id: str):
    """Get project analysis history."""
    return {
        "items": [
            {
                "id": "analysis_1",
                "status": "completed",
                "issues_found": 5,
                "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
            },
            {
                "id": "analysis_2",
                "status": "completed",
                "issues_found": 3,
                "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
            },
        ],
        "total": 2
    }
