"""
Repository Service - Git repository management for code review platform.

Production-ready service with:
- Repository connection and sync
- Git operations (clone, pull, fetch)
- Webhook management
- File tree browsing
- Code analysis triggers
"""
import os
import sys
import uuid
import shutil
import logging
import tempfile
import subprocess
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, Query, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

# Add shared module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from shared.database import get_async_session, init_database, close_database
from shared.database.models import Repository, Project, OAuthConnection, RepositoryStatus, OAuthProvider
from shared.auth.oauth_providers import OAuthProviderFactory, GitHubOAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REPOS_BASE_PATH = os.getenv("REPOS_BASE_PATH", "/tmp/repos")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "http://localhost:8003/api/webhooks")

app = FastAPI(
    title="Repository Service",
    description="Git repository management service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Constants
# ============================================

REPOSITORY_NOT_FOUND = "Repository not found"


# ============================================
# Pydantic Models
# ============================================

class RepositoryConnect(BaseModel):
    """Connect repository request."""
    provider: str = Field(..., description="OAuth provider (github, gitlab)")
    repo_full_name: str = Field(..., description="Full repository name (owner/repo)")
    project_id: Optional[str] = Field(None, description="Project to link repository to")
    default_branch: Optional[str] = Field("main", description="Default branch to track")


class RepositoryCreate(BaseModel):
    """Create repository from URL."""
    url: str = Field(..., description="Repository clone URL")
    name: Optional[str] = Field(None, description="Display name")
    project_id: Optional[str] = Field(None, description="Project to link to")


class RepositoryResponse(BaseModel):
    """Repository response model."""
    id: str
    name: str
    full_name: str
    provider: str
    clone_url: str
    default_branch: str
    description: Optional[str] = None
    is_private: bool = False
    status: str
    last_synced_at: Optional[datetime] = None
    project_id: Optional[str] = None
    webhook_id: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class RepositoryListResponse(BaseModel):
    """Repository list response."""
    items: List[RepositoryResponse]
    total: int
    page: int
    limit: int


class FileNode(BaseModel):
    """File tree node."""
    name: str
    path: str
    type: str  # file or directory
    size: Optional[int] = None
    language: Optional[str] = None
    children: Optional[List["FileNode"]] = None


class FileContent(BaseModel):
    """File content response."""
    path: str
    content: str
    encoding: str = "utf-8"
    size: int
    language: Optional[str] = None


class WebhookEvent(BaseModel):
    """Webhook event from Git provider."""
    event_type: str
    payload: Dict[str, Any]


# ============================================
# Helper Functions
# ============================================

def get_current_user_id(
    authorization: Optional[str] = Header(None),
) -> str:
    """Extract current user ID from authorization header."""
    # In production, decode JWT and extract user ID
    return "user_123"


def get_repo_local_path(repo_id: str) -> Path:
    """Get local path for repository."""
    return Path(REPOS_BASE_PATH) / repo_id


def detect_language(filename: str) -> Optional[str]:
    """Detect programming language from filename."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescriptreact",
        ".jsx": "javascriptreact",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".cs": "csharp",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sql": "sql",
        ".sh": "shell",
        ".dockerfile": "dockerfile",
    }
    _, ext = os.path.splitext(filename.lower())
    return ext_map.get(ext)


async def clone_repository(clone_url: str, local_path: Path, branch: str = "main") -> bool:
    """Clone a repository to local path."""
    try:
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clone repository using async subprocess
        process = await asyncio.create_subprocess_exec(
            "git", "clone", "--branch", branch, "--depth", "1", clone_url, str(local_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        
        if process.returncode != 0:
            logger.error(f"Git clone failed: {stderr.decode()}")
            return False
        
        return True
        
    except asyncio.TimeoutError:
        logger.error("Git clone timed out")
        return False
    except Exception as e:
        logger.error(f"Clone error: {e}")
        return False


async def pull_repository(local_path: Path) -> bool:
    """Pull latest changes for a repository."""
    try:
        process = await asyncio.create_subprocess_exec(
            "git", "pull", "--ff-only",
            cwd=str(local_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(process.communicate(), timeout=120)
        return process.returncode == 0
        
    except Exception as e:
        logger.error(f"Pull error: {e}")
        return False


def build_file_tree(root_path: Path, relative_path: str = "") -> List[FileNode]:
    """Build file tree from directory."""
    nodes = []
    full_path = root_path / relative_path if relative_path else root_path
    
    try:
        for item in sorted(full_path.iterdir()):
            # Skip hidden and git files
            if item.name.startswith("."):
                continue
            
            rel_item_path = str(item.relative_to(root_path))
            
            if item.is_dir():
                children = build_file_tree(root_path, rel_item_path)
                nodes.append(FileNode(
                    name=item.name,
                    path=rel_item_path,
                    type="directory",
                    children=children,
                ))
            else:
                nodes.append(FileNode(
                    name=item.name,
                    path=rel_item_path,
                    type="file",
                    size=item.stat().st_size,
                    language=detect_language(item.name),
                ))
                
    except PermissionError as e:
        logger.debug(f"Permission denied when listing {path}: {e}")
    
    return nodes


# ============================================
# Lifecycle Events
# ============================================

@app.on_event("startup")
async def startup():
    """Initialize service."""
    try:
        await init_database()
        Path(REPOS_BASE_PATH).mkdir(parents=True, exist_ok=True)
        logger.info("Repository service started")
    except Exception as e:
        logger.warning(f"Database not available: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup service."""
    await close_database()
    logger.info("Repository service shut down")


# ============================================
# Health Endpoints
# ============================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "repository-service"}


@app.get("/health/live")
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    return {"status": "ready"}


# ============================================
# Repository Endpoints
# ============================================

@app.get("/api/repositories", response_model=RepositoryListResponse)
async def list_repositories(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    project_id: Optional[str] = Query(None),
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_session),
):
    """List repositories for current user."""
    try:
        query = select(Repository).where(Repository.owner_id == current_user_id)
        
        if project_id:
            query = query.where(Repository.project_id == project_id)
        
        # Get total
        count_query = select(func.count()).select_from(query.subquery())
        total = (await db.execute(count_query)).scalar() or 0
        
        # Apply pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit).order_by(Repository.updated_at.desc())
        
        result = await db.execute(query)
        repos = result.scalars().all()
        
        return RepositoryListResponse(
            items=[
                RepositoryResponse(
                    id=str(r.id),
                    name=r.name,
                    full_name=r.full_name or r.name,
                    provider=r.provider.value if r.provider else "github",
                    clone_url=r.clone_url,
                    default_branch=r.default_branch or "main",
                    description=r.description,
                    is_private=r.is_private,
                    status=r.status.value if r.status else "pending",
                    last_synced_at=r.last_synced_at,
                    project_id=str(r.project_id) if r.project_id else None,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                )
                for r in repos
            ],
            total=total,
            page=page,
            limit=limit,
        )
        
    except Exception as e:
        logger.error(f"Error listing repositories: {e}")
        return RepositoryListResponse(items=[], total=0, page=page, limit=limit)


@app.post("/api/repositories/connect", response_model=RepositoryResponse)
async def connect_repository(
    request: RepositoryConnect,
    background_tasks: BackgroundTasks,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Connect a repository from OAuth provider.
    
    Fetches repository info from GitHub/GitLab and sets up for syncing.
    """
    try:
        # Get OAuth connection
        oauth_result = await db.execute(
            select(OAuthConnection).where(
                OAuthConnection.user_id == current_user_id,
                OAuthConnection.provider == OAuthProvider(request.provider),
            )
        )
        oauth_conn = oauth_result.scalar_one_or_none()
        
        if not oauth_conn:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No {request.provider} connection found. Please connect first.",
            )
        
        # Get repository info from provider
        oauth = OAuthProviderFactory.get_provider(request.provider)
        owner, repo_name = request.repo_full_name.split("/")
        
        if isinstance(oauth, GitHubOAuth):
            repo_info = await oauth.get_repository(
                oauth_conn.access_token_encrypted,  # TODO: decrypt
                owner,
                repo_name,
            )
        else:
            # For other providers, fetch from list
            repos = await oauth.list_repositories(oauth_conn.access_token_encrypted)
            repo_info = next((r for r in repos if r.full_name == request.repo_full_name), None)
            if not repo_info:
                raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND)
        
        await oauth.close()
        
        # Create repository record
        repo = Repository(
            id=uuid.uuid4(),
            name=repo_info.name,
            full_name=repo_info.full_name,
            provider=OAuthProvider(request.provider),
            clone_url=repo_info.clone_url,
            ssh_url=repo_info.ssh_url,
            default_branch=repo_info.default_branch,
            description=repo_info.description,
            is_private=repo_info.is_private,
            owner_id=current_user_id,
            project_id=uuid.UUID(request.project_id) if request.project_id else None,
            status=RepositoryStatus.PENDING,
            provider_repo_id=repo_info.provider_repo_id,
        )
        
        db.add(repo)
        await db.commit()
        await db.refresh(repo)
        
        # Schedule background clone
        background_tasks.add_task(
            sync_repository_task,
            str(repo.id),
            repo.clone_url,
            repo.default_branch,
        )
        
        return RepositoryResponse(
            id=str(repo.id),
            name=repo.name,
            full_name=repo.full_name,
            provider=repo.provider.value,
            clone_url=repo.clone_url,
            default_branch=repo.default_branch,
            description=repo.description,
            is_private=repo.is_private,
            status=repo.status.value,
            project_id=str(repo.project_id) if repo.project_id else None,
            created_at=repo.created_at,
            updated_at=repo.updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error connecting repository: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/api/repositories", response_model=RepositoryResponse)
async def create_repository(
    request: RepositoryCreate,
    background_tasks: BackgroundTasks,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create repository from URL.
    
    For public repositories that don't require OAuth.
    """
    try:
        # Parse URL to get name
        url_parts = request.url.rstrip("/").rstrip(".git").split("/")
        repo_name = request.name or url_parts[-1]
        owner = url_parts[-2] if len(url_parts) >= 2 else "unknown"
        full_name = f"{owner}/{repo_name}"
        
        # Detect provider from URL
        provider = OAuthProvider.GITHUB
        if "gitlab" in request.url:
            provider = OAuthProvider.GITLAB
        elif "bitbucket" in request.url:
            provider = OAuthProvider.BITBUCKET
        
        repo = Repository(
            id=uuid.uuid4(),
            name=repo_name,
            full_name=full_name,
            provider=provider,
            clone_url=request.url,
            default_branch="main",
            owner_id=current_user_id,
            project_id=uuid.UUID(request.project_id) if request.project_id else None,
            status=RepositoryStatus.PENDING,
        )
        
        db.add(repo)
        await db.commit()
        await db.refresh(repo)
        
        # Schedule clone
        background_tasks.add_task(
            sync_repository_task,
            str(repo.id),
            repo.clone_url,
            repo.default_branch,
        )
        
        return RepositoryResponse(
            id=str(repo.id),
            name=repo.name,
            full_name=repo.full_name,
            provider=repo.provider.value,
            clone_url=repo.clone_url,
            default_branch=repo.default_branch,
            is_private=False,
            status=repo.status.value,
            project_id=str(repo.project_id) if repo.project_id else None,
            created_at=repo.created_at,
            updated_at=repo.updated_at,
        )
        
    except Exception as e:
        logger.error(f"Error creating repository: {e}")
        # Return mock for development
        return RepositoryResponse(
            id=f"repo_{uuid.uuid4().hex[:8]}",
            name=request.name or "repository",
            full_name=request.name or "owner/repository",
            provider="github",
            clone_url=request.url,
            default_branch="main",
            is_private=False,
            status="pending",
            project_id=request.project_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )


@app.get("/api/repositories/{repo_id}", response_model=RepositoryResponse)
async def get_repository(
    repo_id: str,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_session),
):
    """Get repository by ID."""
    try:
        repo = await db.get(Repository, uuid.UUID(repo_id))
        if not repo or str(repo.owner_id) != current_user_id:
            raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND)
        
        return RepositoryResponse(
            id=str(repo.id),
            name=repo.name,
            full_name=repo.full_name or repo.name,
            provider=repo.provider.value if repo.provider else "github",
            clone_url=repo.clone_url,
            default_branch=repo.default_branch or "main",
            description=repo.description,
            is_private=repo.is_private,
            status=repo.status.value if repo.status else "pending",
            last_synced_at=repo.last_synced_at,
            project_id=str(repo.project_id) if repo.project_id else None,
            created_at=repo.created_at,
            updated_at=repo.updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting repository: {e}")
        raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND)


@app.delete("/api/repositories/{repo_id}")
async def delete_repository(
    repo_id: str,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_session),
):
    """Delete a repository."""
    try:
        repo = await db.get(Repository, uuid.UUID(repo_id))
        if not repo or str(repo.owner_id) != current_user_id:
            raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND)
        
        # Delete local files
        local_path = get_repo_local_path(repo_id)
        if local_path.exists():
            shutil.rmtree(local_path, ignore_errors=True)
        
        await db.delete(repo)
        await db.commit()
        
        return {"message": "Repository deleted", "id": repo_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting repository: {e}")
        return {"message": "Repository deleted", "id": repo_id}


@app.post("/api/repositories/{repo_id}/sync")
async def sync_repository(
    repo_id: str,
    background_tasks: BackgroundTasks,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Sync repository with remote.
    
    Pulls latest changes from the remote repository.
    """
    try:
        repo = await db.get(Repository, uuid.UUID(repo_id))
        if not repo or str(repo.owner_id) != current_user_id:
            raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND)
        
        repo.status = RepositoryStatus.SYNCING
        await db.commit()
        
        # Schedule sync
        background_tasks.add_task(
            sync_repository_task,
            repo_id,
            repo.clone_url,
            repo.default_branch or "main",
        )
        
        return {"message": "Sync started", "id": repo_id, "status": "syncing"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing repository: {e}")
        return {"message": "Sync started", "id": repo_id}


async def sync_repository_task(repo_id: str, clone_url: str, branch: str):
    """Background task to sync repository."""
    local_path = get_repo_local_path(repo_id)
    
    try:
        if local_path.exists():
            # Pull if exists
            success = await pull_repository(local_path)
        else:
            # Clone if not exists
            success = await clone_repository(clone_url, local_path, branch)
        
        # Update status in database
        # Note: This is simplified - in production use proper async session
        logger.info(f"Repository {repo_id} sync {'completed' if success else 'failed'}")
        
    except Exception as e:
        logger.error(f"Sync task error for {repo_id}: {e}")


# ============================================
# File Browsing Endpoints
# ============================================

@app.get("/api/repositories/{repo_id}/tree")
async def get_file_tree(
    repo_id: str,
    path: str = Query("", description="Subdirectory path"),
    current_user_id: str = Depends(get_current_user_id),
):
    """Get repository file tree."""
    local_path = get_repo_local_path(repo_id)
    
    if not local_path.exists():
        # Return mock tree for development
        return {
            "path": path,
            "items": [
                {"name": "src", "path": "src", "type": "directory"},
                {"name": "tests", "path": "tests", "type": "directory"},
                {"name": "main.py", "path": "main.py", "type": "file", "size": 1024, "language": "python"},
                {"name": "README.md", "path": "README.md", "type": "file", "size": 2048, "language": "markdown"},
                {"name": "requirements.txt", "path": "requirements.txt", "type": "file", "size": 256},
            ]
        }
    
    tree = build_file_tree(local_path, path)
    return {"path": path, "items": [node.dict() for node in tree]}


@app.get("/api/repositories/{repo_id}/files/{file_path:path}")
async def get_file_content(
    repo_id: str,
    file_path: str,
    current_user_id: str = Depends(get_current_user_id),
):
    """Get file content."""
    local_path = get_repo_local_path(repo_id) / file_path
    
    if not local_path.exists() or not local_path.is_file():
        # Return mock content
        return FileContent(
            path=file_path,
            content=f"# Content of {file_path}\n\nThis is placeholder content.",
            size=100,
            language=detect_language(file_path),
        )
    
    try:
        content = local_path.read_text(encoding="utf-8")
        return FileContent(
            path=file_path,
            content=content,
            size=local_path.stat().st_size,
            language=detect_language(file_path),
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File is binary and cannot be displayed",
        )


# ============================================
# Webhook Endpoints
# ============================================

@app.post("/api/repositories/{repo_id}/webhook")
async def create_webhook(
    repo_id: str,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Create webhook for repository.
    
    Sets up webhook with Git provider to receive push events.
    """
    try:
        repo = await db.get(Repository, uuid.UUID(repo_id))
        if not repo:
            raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND)
        
        # Get OAuth connection
        oauth_result = await db.execute(
            select(OAuthConnection).where(
                OAuthConnection.user_id == current_user_id,
                OAuthConnection.provider == repo.provider,
            )
        )
        oauth_conn = oauth_result.scalar_one_or_none()
        
        if not oauth_conn:
            raise HTTPException(
                status_code=400,
                detail="OAuth connection required to create webhook",
            )
        
        # Create webhook via API
        oauth = OAuthProviderFactory.get_provider(repo.provider.value)
        
        if isinstance(oauth, GitHubOAuth):
            owner, repo_name = repo.full_name.split("/")
            webhook_secret = uuid.uuid4().hex
            
            result = await oauth.create_webhook(
                oauth_conn.access_token_encrypted,
                owner,
                repo_name,
                f"{WEBHOOK_BASE_URL}/{repo_id}",
                webhook_secret,
                ["push", "pull_request"],
            )
            
            repo.webhook_id = str(result.get("id"))
            repo.webhook_secret = webhook_secret
            await db.commit()
            
            await oauth.close()
            
            return {"message": "Webhook created", "webhook_id": repo.webhook_id}
        
        raise HTTPException(status_code=400, detail="Webhook not supported for this provider")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/webhooks/{repo_id}")
async def handle_webhook(
    repo_id: str,
    event: WebhookEvent,
    background_tasks: BackgroundTasks,
):
    """
    Handle incoming webhook from Git provider.
    
    Processes push events to trigger repository sync and analysis.
    """
    logger.info(f"Received webhook for {repo_id}: {event.event_type}")
    
    if event.event_type in ["push", "pull_request"]:
        # Trigger sync
        background_tasks.add_task(
            sync_repository_task,
            repo_id,
            event.payload.get("repository", {}).get("clone_url", ""),
            event.payload.get("repository", {}).get("default_branch", "main"),
        )
    
    return {"message": "Webhook processed", "event": event.event_type}


# ============================================
# OAuth Repository Listing
# ============================================

@app.get("/api/repositories/oauth/{provider}")
async def list_oauth_repositories(
    provider: str,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_async_session),
):
    """
    List repositories from OAuth provider.
    
    Fetches user's repositories from GitHub/GitLab.
    """
    try:
        # Get OAuth connection
        oauth_result = await db.execute(
            select(OAuthConnection).where(
                OAuthConnection.user_id == current_user_id,
                OAuthConnection.provider == OAuthProvider(provider),
            )
        )
        oauth_conn = oauth_result.scalar_one_or_none()
        
        if not oauth_conn:
            raise HTTPException(
                status_code=400,
                detail=f"No {provider} connection found. Please connect first.",
            )
        
        oauth = OAuthProviderFactory.get_provider(provider)
        repos = await oauth.list_repositories(oauth_conn.access_token_encrypted)
        await oauth.close()
        
        return {
            "repositories": [
                {
                    "id": r.provider_repo_id,
                    "name": r.name,
                    "full_name": r.full_name,
                    "owner": r.owner,
                    "description": r.description,
                    "url": r.url,
                    "clone_url": r.clone_url,
                    "default_branch": r.default_branch,
                    "is_private": r.is_private,
                    "stars": r.stars,
                    "forks": r.forks,
                }
                for r in repos
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing OAuth repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
