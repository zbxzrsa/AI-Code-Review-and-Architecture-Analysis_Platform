"""
Project Routes / 项目路由
"""

from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..models import Project, ProjectSettings, AnalyzeRequest

router = APIRouter(prefix="/api", tags=["Projects"])

# Mock data
mock_projects = [
    Project(
        id="proj_1",
        name="AI Code Review Platform",
        description="Main platform codebase",
        language="TypeScript",
        framework="React",
        repository_url="https://github.com/example/ai-code-review",
        status="active",
        issues_count=12,
        settings=ProjectSettings(auto_review=True, review_on_push=True, review_on_pr=True),
        created_at=datetime.now() - timedelta(days=30),
        updated_at=datetime.now() - timedelta(hours=2)
    ),
    Project(
        id="proj_2",
        name="Backend Services",
        description="FastAPI microservices",
        language="Python",
        framework="FastAPI",
        repository_url="https://github.com/example/backend",
        status="active",
        issues_count=5,
        settings=ProjectSettings(auto_review=True, review_on_pr=True),
        created_at=datetime.now() - timedelta(days=15),
        updated_at=datetime.now() - timedelta(days=1)
    ),
]


class CreateProjectRequest(BaseModel):
    name: str
    description: Optional[str] = None
    language: str
    framework: Optional[str] = None
    repository_url: Optional[str] = None


@router.get("/projects")
async def list_projects(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = None,
    search: Optional[str] = None
):
    """List all projects / 列出所有项目"""
    filtered = mock_projects
    
    if status:
        filtered = [p for p in filtered if p.status == status]
    if search:
        filtered = [p for p in filtered if search.lower() in p.name.lower()]
    
    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    
    return {
        "items": filtered[start:end],
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit
    }


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Get project by ID / 根据 ID 获取项目"""
    for project in mock_projects:
        if project.id == project_id:
            return project
    raise HTTPException(status_code=404, detail="Project not found")


@router.post("/projects")
async def create_project(request: CreateProjectRequest):
    """Create a new project / 创建新项目"""
    new_project = Project(
        id=f"proj_{len(mock_projects) + 1}",
        name=request.name,
        description=request.description,
        language=request.language,
        framework=request.framework,
        repository_url=request.repository_url,
        status="active",
        issues_count=0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    mock_projects.append(new_project)
    return new_project


@router.post("/projects/{project_id}/analyze")
async def analyze_project(project_id: str, request: AnalyzeRequest):
    """Trigger project analysis / 触发项目分析"""
    for project in mock_projects:
        if project.id == project_id:
            return {
                "session_id": f"session_{project_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "status": "started",
                "project_id": project_id,
                "files_to_analyze": request.files or ["all"],
                "full_analysis": request.full_analysis
            }
    raise HTTPException(status_code=404, detail="Project not found")


@router.get("/projects/{project_id}/files")
async def get_project_files(project_id: str, path: Optional[str] = None):
    """Get project files / 获取项目文件"""
    return {
        "path": path or "/",
        "files": [
            {"name": "src", "type": "directory", "children": 5},
            {"name": "package.json", "type": "file", "size": 1024},
            {"name": "README.md", "type": "file", "size": 2048},
        ]
    }
