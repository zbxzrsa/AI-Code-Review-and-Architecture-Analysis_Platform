"""
Project Service - Project management for code review platform.
"""
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Project Service",
    description="Project management service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    language: str
    framework: Optional[str] = None
    repository_url: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    language: str
    framework: Optional[str] = None
    repository_url: Optional[str] = None
    owner_id: str
    status: str = "active"
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(BaseModel):
    items: List[ProjectResponse]
    total: int
    page: int
    limit: int


# Health endpoints
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/health/live")
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    return {"status": "ready"}


# Project endpoints
@app.get("/api/projects", response_model=ProjectListResponse)
async def list_projects(page: int = 1, limit: int = 20, search: Optional[str] = None):
    """List all projects for current user."""
    # Mock response for development
    return ProjectListResponse(
        items=[
            ProjectResponse(
                id="proj_1",
                name="Sample Project",
                description="A sample code review project",
                language="python",
                framework="fastapi",
                owner_id="user_123",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
        ],
        total=1,
        page=page,
        limit=limit,
    )


@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(project: ProjectCreate):
    """Create a new project."""
    return ProjectResponse(
        id="proj_new",
        name=project.name,
        description=project.description,
        language=project.language,
        framework=project.framework,
        repository_url=project.repository_url,
        owner_id="user_123",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Get project by ID."""
    return ProjectResponse(
        id=project_id,
        name="Sample Project",
        description="A sample project",
        language="python",
        framework="fastapi",
        owner_id="user_123",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@app.put("/api/projects/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, project: ProjectCreate):
    """Update project."""
    return ProjectResponse(
        id=project_id,
        name=project.name,
        description=project.description,
        language=project.language,
        framework=project.framework,
        repository_url=project.repository_url,
        owner_id="user_123",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete project."""
    return {"message": "Project deleted", "id": project_id}


@app.get("/api/projects/{project_id}/files")
async def get_project_files(project_id: str, path: Optional[str] = None):
    """Get project file tree."""
    return {
        "path": path or "/",
        "files": [
            {"name": "main.py", "type": "file", "size": 1024},
            {"name": "src", "type": "directory", "items": 5},
            {"name": "tests", "type": "directory", "items": 3},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
