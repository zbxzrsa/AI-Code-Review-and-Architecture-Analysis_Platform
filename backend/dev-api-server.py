"""
Development API Server / 开发API服务器

Handles all non-auth API endpoints for frontend development.
处理所有非认证API端点用于前端开发。

Run with: python dev-api-server.py
运行命令: python dev-api-server.py
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import random

# ============================================
# OAuth Configuration / OAuth配置
# ============================================
GITHUB_CLIENT_ID = os.getenv('GITHUB_CLIENT_ID', '')
GITHUB_CLIENT_SECRET = os.getenv('GITHUB_CLIENT_SECRET', '')
GITLAB_CLIENT_ID = os.getenv('GITLAB_CLIENT_ID', '')
GITLAB_CLIENT_SECRET = os.getenv('GITLAB_CLIENT_SECRET', '')
# Bitbucket uses API Token instead of OAuth (since Sep 2025)
BITBUCKET_API_TOKEN = os.getenv('BITBUCKET_API_TOKEN', '')

# ============================================
# Models / 模型
# ============================================

class ProjectSettings(BaseModel):
    auto_review: bool = False
    review_on_push: bool = False
    review_on_pr: bool = True
    severity_threshold: str = "warning"


class Project(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    language: str
    framework: Optional[str] = None
    repository_url: Optional[str] = None
    status: str = "active"
    issues_count: int = 0
    settings: ProjectSettings = ProjectSettings()
    created_at: datetime
    updated_at: datetime


class DashboardMetrics(BaseModel):
    total_projects: int
    total_analyses: int
    issues_found: int
    issues_resolved: int
    resolution_rate: float


class Activity(BaseModel):
    id: str
    type: str
    message: str
    project_id: Optional[str] = None
    created_at: datetime


# ============================================
# Mock Data / 模拟数据
# ============================================

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
        settings=ProjectSettings(auto_review=False, review_on_push=False, review_on_pr=True),
        created_at=datetime.now() - timedelta(days=20),
        updated_at=datetime.now() - timedelta(hours=5)
    ),
    Project(
        id="proj_3",
        name="Mobile App",
        description="React Native mobile application",
        language="TypeScript",
        framework="React Native",
        status="active",
        issues_count=8,
        settings=ProjectSettings(auto_review=True, review_on_push=True, review_on_pr=False),
        created_at=datetime.now() - timedelta(days=15),
        updated_at=datetime.now() - timedelta(days=1)
    ),
]

mock_activities = [
    Activity(
        id="act_1",
        type="analysis_complete",
        message="Code analysis completed for AI Code Review Platform",
        project_id="proj_1",
        created_at=datetime.now() - timedelta(hours=1)
    ),
    Activity(
        id="act_2",
        type="issue_fixed",
        message="Fixed 3 security issues in Backend Services",
        project_id="proj_2",
        created_at=datetime.now() - timedelta(hours=3)
    ),
    Activity(
        id="act_3",
        type="project_created",
        message="New project Mobile App created",
        project_id="proj_3",
        created_at=datetime.now() - timedelta(days=1)
    ),
]

# ============================================
# FastAPI App / FastAPI 应用
# ============================================

app = FastAPI(
    title="Dev API Server / 开发API服务器",
    description="Development API server for frontend testing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Health / 健康检查
# ============================================

@app.get("/")
async def root():
    return {"service": "Dev API Server", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ============================================
# Dashboard / 仪表板
# ============================================

@app.get("/api/metrics/dashboard")
async def get_dashboard_metrics():
    """Get dashboard metrics / 获取仪表板指标"""
    return DashboardMetrics(
        total_projects=len(mock_projects),
        total_analyses=47,
        issues_found=156,
        issues_resolved=131,
        resolution_rate=0.84
    )


@app.get("/api/metrics/system")
async def get_system_metrics():
    """Get system metrics / 获取系统指标"""
    return {
        "cpu_usage": random.uniform(20, 60),
        "memory_usage": random.uniform(40, 70),
        "disk_usage": random.uniform(30, 50),
        "active_users": random.randint(5, 20)
    }


# ============================================
# Projects / 项目
# ============================================

@app.get("/api/projects")
async def list_projects(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = None
):
    """List projects / 列出项目"""
    projects = mock_projects
    
    if search:
        projects = [p for p in projects if search.lower() in p.name.lower()]
    
    return {
        "items": projects,
        "total": len(projects),
        "page": page,
        "limit": limit
    }


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project by ID / 通过ID获取项目"""
    for project in mock_projects:
        if project.id == project_id:
            return project
    raise HTTPException(status_code=404, detail="Project not found")


class CreateProjectRequest(BaseModel):
    name: str
    language: str
    description: Optional[str] = None
    framework: Optional[str] = None
    repository_url: Optional[str] = None


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    status: Optional[str] = None
    settings: Optional[dict] = None


@app.post("/api/projects")
async def create_project(request: CreateProjectRequest):
    """Create project / 创建项目"""
    project = Project(
        id=f"proj_{secrets.token_hex(4)}",
        name=request.name,
        description=request.description or "",
        language=request.language,
        framework=request.framework or "",
        repository_url=request.repository_url or "",
        status="active",
        issues_count=0,
        settings=ProjectSettings(
            auto_review=True,
            review_on_push=True,
            review_on_pr=True,
            severity_threshold="warning",
            enabled_rules=[],
            ignored_paths=["node_modules", ".git", "__pycache__", "dist", "build"]
        ),
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    mock_projects.append(project)
    return project


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, request: UpdateProjectRequest):
    """Update project / 更新项目"""
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
    raise HTTPException(status_code=404, detail="Project not found")


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete project / 删除项目"""
    for i, project in enumerate(mock_projects):
        if project.id == project_id:
            mock_projects.pop(i)
            return {"message": "Project deleted", "id": project_id}
    raise HTTPException(status_code=404, detail="Project not found")


# ============================================
# Analysis / 分析
# ============================================

class AnalyzeRequest(BaseModel):
    files: Optional[List[str]] = None
    version: Optional[str] = None


@app.post("/api/projects/{project_id}/analyze")
async def start_analysis(project_id: str, request: Optional[AnalyzeRequest] = None):
    """Start analysis / 开始分析"""
    session_id = f"session_{secrets.token_hex(8)}"
    return {
        "id": session_id,
        "session_id": session_id,
        "status": "started",
        "project_id": project_id,
        "files": request.files if request else None,
        "created_at": datetime.now().isoformat()
    }


@app.get("/api/analyze/{session_id}")
async def get_analysis_session(session_id: str):
    """Get analysis session / 获取分析会话"""
    return {
        "id": session_id,
        "status": "completed",
        "issues_found": random.randint(0, 15),
        "started_at": datetime.now() - timedelta(minutes=5),
        "completed_at": datetime.now()
    }


@app.get("/api/analyze/{session_id}/issues")
async def get_analysis_issues(session_id: str):
    """Get analysis issues / 获取分析问题"""
    return {
        "items": [
            {
                "id": f"issue_{i}",
                "type": random.choice(["security", "performance", "quality", "style"]),
                "severity": random.choice(["critical", "high", "medium", "low"]),
                "message": f"Sample issue {i}",
                "file": "src/example.ts",
                "line": random.randint(1, 100),
                "column": random.randint(1, 50),
                "has_fix": random.choice([True, False])
            }
            for i in range(random.randint(0, 10))
        ],
        "total": random.randint(0, 10)
    }


# ============================================
# Project Files / 项目文件
# ============================================

@app.get("/api/projects/{project_id}/files")
async def get_project_files(project_id: str, path: Optional[str] = None):
    """Get project file tree / 获取项目文件树"""
    return {
        "path": path or "",
        "items": [
            {"name": "src", "path": "src", "type": "directory"},
            {"name": "tests", "path": "tests", "type": "directory"},
            {"name": "main.py", "path": "main.py", "type": "file", "size": 1024, "language": "python"},
            {"name": "app.ts", "path": "app.ts", "type": "file", "size": 2048, "language": "typescript"},
            {"name": "README.md", "path": "README.md", "type": "file", "size": 512, "language": "markdown"},
        ]
    }


@app.get("/api/projects/{project_id}/files/{file_path:path}")
async def get_project_file(project_id: str, file_path: str):
    """Get file content / 获取文件内容"""
    # Sample code based on file type
    if file_path.endswith('.py'):
        content = '''# Sample Python Code

def process_user_input(input_data):
    """Process user input - potential SQL injection"""
    # WARNING: This is vulnerable to SQL injection
    query = f"SELECT * FROM users WHERE id = {input_data}"
    return execute_query(query)

async def fetch_data(url: str):
    """Fetch data from URL"""
    try:
        response = await http_client.get(url)
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        # TODO: Better error handling

class UserService:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_user(self, user_id: int):
        return self.db.query(User).filter_by(id=user_id).first()
'''
    elif file_path.endswith('.ts') or file_path.endswith('.tsx'):
        content = '''// Sample TypeScript Code

interface User {
  id: number;
  name: string;
  email: string;
}

async function fetchUserData(userId: string): Promise<User> {
  // TODO: Add input validation
  const response = await fetch(`/api/users/${userId}`);
  return response.json();
}

function processData(data: any) {
  // Warning: Using 'any' type defeats TypeScript benefits
  console.log(data.value);
  return data;
}
'''
    else:
        content = f"# Content of {file_path}\n\nSample file content for demonstration."
    
    return {
        "path": file_path,
        "content": content,
        "size": len(content),
        "language": file_path.split('.')[-1] if '.' in file_path else "text",
        "last_modified": datetime.now().isoformat()
    }


# ============================================
# OAuth / OAuth认证
# ============================================

mock_oauth_connections = []
mock_repositories = [
    {
        "id": "repo_1",
        "name": "ai-code-review",
        "full_name": "user/ai-code-review",
        "provider": "github",
        "clone_url": "https://github.com/user/ai-code-review.git",
        "default_branch": "main",
        "description": "AI-powered code review platform",
        "is_private": False,
        "status": "ready",
        "stars": 42,
        "forks": 12,
        "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
        "updated_at": datetime.now().isoformat(),
    },
    {
        "id": "repo_2",
        "name": "backend-services",
        "full_name": "user/backend-services",
        "provider": "github",
        "clone_url": "https://github.com/user/backend-services.git",
        "default_branch": "main",
        "description": "FastAPI microservices",
        "is_private": True,
        "status": "ready",
        "stars": 15,
        "forks": 3,
        "created_at": (datetime.now() - timedelta(days=20)).isoformat(),
        "updated_at": datetime.now().isoformat(),
    },
]


@app.get("/api/auth/oauth/providers")
async def get_oauth_providers():
    """Get available OAuth providers / 获取可用的OAuth提供商"""
    # Check which providers are configured
    github_configured = bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET)
    gitlab_configured = bool(GITLAB_CLIENT_ID and GITLAB_CLIENT_SECRET)
    bitbucket_configured = bool(BITBUCKET_API_TOKEN)  # Bitbucket uses API Token
    
    return {
        "providers": [
            {
                "name": "github",
                "display_name": "GitHub",
                "icon": "github",
                "connected": False,
                "configured": github_configured,
                "auth_type": "oauth",
                "message": "Ready to connect" if github_configured else "OAuth not configured - set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET"
            },
            {
                "name": "gitlab",
                "display_name": "GitLab",
                "icon": "gitlab",
                "connected": False,
                "configured": gitlab_configured,
                "auth_type": "oauth",
                "message": "Ready to connect" if gitlab_configured else "OAuth not configured - set GITLAB_CLIENT_ID and GITLAB_CLIENT_SECRET"
            },
            {
                "name": "bitbucket",
                "display_name": "Bitbucket",
                "icon": "bitbucket",
                "connected": bitbucket_configured,  # API Token = already connected
                "configured": bitbucket_configured,
                "auth_type": "api_token",
                "message": "Connected via API Token" if bitbucket_configured else "API Token not configured - set BITBUCKET_API_TOKEN"
            },
        ]
    }


@app.get("/api/auth/oauth/connect/{provider}")
async def initiate_oauth(provider: str, return_url: str = "/"):
    """Initiate OAuth flow / 启动OAuth流程"""
    state = secrets.token_urlsafe(32)
    # Use port 5173 for Vite dev server, 3000 for production
    callback_url = f"http://localhost:5173/oauth/callback/{provider}"
    
    # Get OAuth configuration based on provider
    if provider == "github":
        if not GITHUB_CLIENT_ID:
            raise HTTPException(
                status_code=400,
                detail="GitHub OAuth not configured. Please set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET environment variables."
            )
        auth_url = f"https://github.com/login/oauth/authorize?client_id={GITHUB_CLIENT_ID}&scope=repo,user:email&state={state}&redirect_uri={callback_url}"
    elif provider == "gitlab":
        if not GITLAB_CLIENT_ID:
            raise HTTPException(
                status_code=400,
                detail="GitLab OAuth not configured. Please set GITLAB_CLIENT_ID and GITLAB_CLIENT_SECRET environment variables."
            )
        auth_url = f"https://gitlab.com/oauth/authorize?client_id={GITLAB_CLIENT_ID}&scope=read_user+read_repository+api&response_type=code&state={state}&redirect_uri={callback_url}"
    elif provider == "bitbucket":
        # Bitbucket uses API Token instead of OAuth (since Sep 2025)
        if not BITBUCKET_API_TOKEN:
            raise HTTPException(
                status_code=400,
                detail="Bitbucket API Token not configured. Please set BITBUCKET_API_TOKEN environment variable."
            )
        # No OAuth flow needed - API Token is used directly
        return {
            "message": "Bitbucket uses API Token authentication. Already connected.",
            "connected": True,
            "auth_type": "api_token"
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    return {
        "authorization_url": auth_url,
        "state": state,
        "callback_url": callback_url
    }


@app.get("/api/auth/oauth/callback/{provider}")
async def oauth_callback(provider: str, code: str = "", state: str = ""):
    """Handle OAuth callback / 处理OAuth回调"""
    # Simulate successful OAuth
    mock_oauth_connections.append({
        "provider": provider,
        "username": f"user_{provider}",
        "email": f"user@{provider}.com",
        "connected_at": datetime.now().isoformat()
    })
    return {
        "success": True,
        "message": f"Connected to {provider}",
        "is_new_user": False
    }


@app.get("/api/auth/oauth/connections")
async def get_oauth_connections():
    """Get connected OAuth accounts / 获取已连接的OAuth账户"""
    return {
        "connections": mock_oauth_connections if mock_oauth_connections else [
            {"provider": "github", "username": "demo_user", "email": "demo@github.com", "connected_at": datetime.now().isoformat()}
        ]
    }


@app.delete("/api/auth/oauth/connections/{provider}")
async def disconnect_oauth(provider: str):
    """Disconnect OAuth provider / 断开OAuth提供商连接"""
    return {"message": f"Disconnected from {provider}"}


# ============================================
# Repositories / 仓库管理
# ============================================

@app.get("/api/repositories")
async def list_repositories(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    project_id: Optional[str] = None
):
    """List repositories / 列出仓库"""
    repos = mock_repositories
    if project_id:
        repos = [r for r in repos if r.get("project_id") == project_id]
    return {
        "items": repos,
        "total": len(repos),
        "page": page,
        "limit": limit
    }


@app.post("/api/repositories")
async def create_repository(
    url: str = "",
    name: Optional[str] = None,
    project_id: Optional[str] = None
):
    """Create repository from URL / 从URL创建仓库"""
    repo_id = f"repo_{secrets.token_hex(4)}"
    repo = {
        "id": repo_id,
        "name": name or url.split("/")[-1].replace(".git", ""),
        "full_name": "/".join(url.split("/")[-2:]).replace(".git", ""),
        "provider": "github" if "github" in url else "gitlab" if "gitlab" in url else "bitbucket",
        "clone_url": url,
        "default_branch": "main",
        "is_private": False,
        "status": "pending",
        "project_id": project_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    mock_repositories.append(repo)
    return repo


@app.post("/api/repositories/connect")
async def connect_repository(
    provider: str = "github",
    repo_full_name: str = "",
    project_id: Optional[str] = None
):
    """Connect repository from OAuth provider / 从OAuth提供商连接仓库"""
    repo_id = f"repo_{secrets.token_hex(4)}"
    owner, name = repo_full_name.split("/") if "/" in repo_full_name else ("user", repo_full_name)
    repo = {
        "id": repo_id,
        "name": name,
        "full_name": repo_full_name,
        "provider": provider,
        "clone_url": f"https://{provider}.com/{repo_full_name}.git",
        "default_branch": "main",
        "is_private": False,
        "status": "pending",
        "project_id": project_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    mock_repositories.append(repo)
    return repo


@app.get("/api/repositories/{repo_id}")
async def get_repository(repo_id: str):
    """Get repository by ID / 通过ID获取仓库"""
    for repo in mock_repositories:
        if repo["id"] == repo_id:
            return repo
    raise HTTPException(status_code=404, detail="Repository not found")


@app.delete("/api/repositories/{repo_id}")
async def delete_repository(repo_id: str):
    """Delete repository / 删除仓库"""
    for i, repo in enumerate(mock_repositories):
        if repo["id"] == repo_id:
            mock_repositories.pop(i)
            return {"message": "Repository deleted", "id": repo_id}
    raise HTTPException(status_code=404, detail="Repository not found")


@app.post("/api/repositories/{repo_id}/sync")
async def sync_repository(repo_id: str):
    """Sync repository with remote / 与远程同步仓库"""
    return {"message": "Sync started", "id": repo_id, "status": "syncing"}


@app.get("/api/repositories/{repo_id}/tree")
async def get_repository_tree(repo_id: str, path: str = ""):
    """Get repository file tree / 获取仓库文件树"""
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


@app.get("/api/repositories/{repo_id}/files/{file_path:path}")
async def get_repository_file(repo_id: str, file_path: str):
    """Get file content from repository / 从仓库获取文件内容"""
    content = f"# Content of {file_path}\n\ndef main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()"
    return {
        "path": file_path,
        "content": content,
        "size": len(content),
        "language": "python" if file_path.endswith(".py") else "text"
    }


@app.get("/api/repositories/oauth/{provider}")
async def list_oauth_repositories(provider: str):
    """List repositories from OAuth provider / 列出OAuth提供商的仓库"""
    # Simulated repositories from the provider
    return {
        "repositories": [
            {
                "id": "12345",
                "name": "my-project",
                "full_name": "user/my-project",
                "owner": "user",
                "description": "My awesome project",
                "url": f"https://{provider}.com/user/my-project",
                "clone_url": f"https://{provider}.com/user/my-project.git",
                "default_branch": "main",
                "is_private": False,
                "stars": 42,
                "forks": 12
            },
            {
                "id": "67890",
                "name": "another-repo",
                "full_name": "user/another-repo",
                "owner": "user",
                "description": "Another repository",
                "url": f"https://{provider}.com/user/another-repo",
                "clone_url": f"https://{provider}.com/user/another-repo.git",
                "default_branch": "main",
                "is_private": True,
                "stars": 10,
                "forks": 2
            }
        ]
    }


# ============================================
# Activity / 活动
# ============================================

@app.get("/api/activity")
async def get_activity(limit: int = Query(10, ge=1, le=50)):
    """Get recent activity / 获取最近活动"""
    return {
        "items": mock_activities[:limit],
        "total": len(mock_activities)
    }


# ============================================
# User Profile / 用户资料
# ============================================

@app.get("/api/user/profile")
async def get_user_profile():
    """Get user profile / 获取用户资料"""
    return {
        "id": "user_demo",
        "email": "demo@example.com",
        "name": "Demo User",
        "username": "demouser",
        "bio": "Software developer passionate about code quality",
        "avatar": None,
        "role": "user",
        "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
        "email_verified": True,
        "two_factor_enabled": False
    }


@app.get("/api/user/settings")
async def get_user_settings():
    """Get user settings / 获取用户设置"""
    return {
        "theme": "system",
        "language": "en",
        "notifications": {
            "email": True,
            "push": True,
            "weekly_digest": True
        },
        "privacy": {
            "profile_visibility": "public",
            "show_email": False,
            "show_activity": True
        }
    }


@app.get("/api/user/login-history")
async def get_login_history(page: int = 1, limit: int = 10):
    """Get login history / 获取登录历史"""
    return {
        "items": [
            {
                "id": f"login_{i}",
                "ip_address": f"192.168.1.{random.randint(1, 255)}",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
                "location": random.choice(["Ho Chi Minh City, VN", "Hanoi, VN", "Singapore, SG"]),
                "device_type": random.choice(["desktop", "mobile", "tablet"]),
                "status": "success",
                "created_at": (datetime.now() - timedelta(hours=i*24)).isoformat()
            }
            for i in range(min(limit, 5))
        ],
        "total": 25,
        "page": page,
        "limit": limit
    }


@app.get("/api/user/api-activity")
async def get_api_activity(page: int = 1, limit: int = 10):
    """Get API activity / 获取API活动"""
    return {
        "items": [
            {
                "id": f"api_{i}",
                "endpoint": random.choice(["/projects", "/analyze", "/metrics"]),
                "method": random.choice(["GET", "POST"]),
                "status_code": random.choice([200, 200, 200, 201, 400]),
                "response_time_ms": random.randint(50, 500),
                "created_at": (datetime.now() - timedelta(minutes=i*30)).isoformat()
            }
            for i in range(min(limit, 10))
        ],
        "total": 100,
        "page": page,
        "limit": limit
    }


@app.get("/api/user/sessions")
async def get_user_sessions():
    """Get active sessions / 获取活跃会话"""
    return {
        "items": [
            {
                "id": "session_current",
                "device": "Chrome on Windows",
                "ip_address": "192.168.1.100",
                "location": "Ho Chi Minh City, VN",
                "is_current": True,
                "last_active": datetime.now().isoformat(),
                "created_at": (datetime.now() - timedelta(hours=2)).isoformat()
            }
        ],
        "total": 1
    }


@app.get("/api/user/2fa/status")
async def get_2fa_status():
    """Get 2FA status / 获取2FA状态"""
    return {
        "enabled": False,
        "method": None,
        "backup_codes_remaining": 0
    }


# ============================================
# OAuth Connections / OAuth连接
# ============================================

@app.get("/api/user/oauth/connections")
async def get_oauth_connections():
    """Get OAuth connections / 获取OAuth连接"""
    return {
        "connections": [
            {"provider": "github", "connected": False, "username": None},
            {"provider": "gitlab", "connected": False, "username": None},
            {"provider": "google", "connected": False, "email": None},
            {"provider": "microsoft", "connected": False, "email": None},
        ]
    }


@app.post("/api/user/oauth/connect/{provider}")
async def connect_oauth(provider: str):
    """Connect OAuth provider / 连接OAuth提供商"""
    # In a real app, this would initiate OAuth flow
    return {
        "redirect_url": f"https://oauth.example.com/{provider}/authorize",
        "message": f"OAuth connection for {provider} initiated (mock)"
    }


@app.delete("/api/user/oauth/disconnect/{provider}")
async def disconnect_oauth(provider: str):
    """Disconnect OAuth provider / 断开OAuth提供商"""
    return {"message": f"Disconnected from {provider}"}


# ============================================
# User Settings / 用户设置
# ============================================

@app.get("/api/user/settings/privacy")
async def get_privacy_settings():
    """Get privacy settings / 获取隐私设置"""
    return {
        "profile_visibility": "public",
        "show_email": False,
        "show_activity": True,
        "allow_analytics": True
    }


@app.put("/api/user/settings/privacy")
async def update_privacy_settings():
    """Update privacy settings / 更新隐私设置"""
    return {
        "message": "Privacy settings updated",
        "profile_visibility": "public",
        "show_email": False,
        "show_activity": True,
        "allow_analytics": True
    }


@app.get("/api/user/settings/notifications")
async def get_notification_settings():
    """Get notification settings / 获取通知设置"""
    return {
        "email_notifications": True,
        "push_notifications": True,
        "weekly_digest": True,
        "marketing_emails": False
    }


@app.put("/api/user/settings/notifications")
async def update_notification_settings():
    """Update notification settings / 更新通知设置"""
    return {"message": "Notification settings updated"}


@app.put("/api/user/profile")
async def update_user_profile():
    """Update user profile / 更新用户资料"""
    return {
        "message": "Profile updated successfully",
        "id": "user_demo",
        "email": "demo@example.com",
        "name": "Demo User",
        "username": "demouser"
    }


@app.put("/api/user/password")
async def change_password():
    """Change password / 修改密码"""
    return {"message": "Password changed successfully"}


@app.delete("/api/user/account")
async def delete_account():
    """Delete account / 删除账户"""
    return {"message": "Account deletion requested"}


# ============================================
# Experiments (Admin) / 实验（管理员）
# ============================================

@app.get("/api/experiments")
async def list_experiments():
    """List experiments / 列出实验"""
    return {
        "items": [
            {
                "id": "exp_1",
                "name": "GPT-4 Turbo Test",
                "model": "gpt-4-turbo",
                "status": "running",
                "accuracy": 0.92,
                "error_rate": 0.03,
                "latency_p95": 2.5,
                "cost_per_analysis": 0.08,
                "created_at": datetime.now() - timedelta(days=2)
            },
            {
                "id": "exp_2",
                "name": "Claude 3 Opus",
                "model": "claude-3-opus",
                "status": "completed",
                "accuracy": 0.89,
                "error_rate": 0.05,
                "latency_p95": 3.2,
                "cost_per_analysis": 0.12,
                "created_at": datetime.now() - timedelta(days=5)
            }
        ],
        "total": 2
    }


# ============================================
# Admin - Users / 管理员 - 用户管理
# ============================================

@app.get("/api/admin/users")
async def list_users(page: int = 1, limit: int = 10, search: Optional[str] = None):
    """List all users (Admin) / 列出所有用户（管理员）"""
    users = [
        {
            "id": "user_1",
            "email": "admin@example.com",
            "name": "Admin User",
            "role": "admin",
            "status": "active",
            "created_at": (datetime.now() - timedelta(days=90)).isoformat(),
            "last_login": (datetime.now() - timedelta(hours=2)).isoformat()
        },
        {
            "id": "user_2",
            "email": "user@example.com",
            "name": "Regular User",
            "role": "user",
            "status": "active",
            "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
            "last_login": (datetime.now() - timedelta(days=1)).isoformat()
        },
        {
            "id": "user_3",
            "email": "viewer@example.com",
            "name": "Viewer User",
            "role": "viewer",
            "status": "inactive",
            "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
            "last_login": (datetime.now() - timedelta(days=30)).isoformat()
        }
    ]
    if search:
        users = [u for u in users if search.lower() in u['name'].lower() or search.lower() in u['email'].lower()]
    return {"items": users, "total": len(users), "page": page, "limit": limit}


@app.get("/api/admin/users/{user_id}")
async def get_user(user_id: str):
    """Get user details (Admin) / 获取用户详情（管理员）"""
    return {
        "id": user_id,
        "email": "user@example.com",
        "name": "Demo User",
        "role": "user",
        "status": "active",
        "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
        "last_login": datetime.now().isoformat(),
        "projects_count": 3,
        "analyses_count": 47
    }


@app.put("/api/admin/users/{user_id}")
async def update_user(user_id: str):
    """Update user (Admin) / 更新用户（管理员）"""
    return {"message": f"User {user_id} updated successfully"}


@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: str):
    """Delete user (Admin) / 删除用户（管理员）"""
    return {"message": f"User {user_id} deleted"}


@app.post("/api/admin/users/{user_id}/activate")
async def activate_user(user_id: str):
    """Activate user (Admin) / 激活用户（管理员）"""
    return {"message": f"User {user_id} activated"}


@app.post("/api/admin/users/{user_id}/deactivate")
async def deactivate_user(user_id: str):
    """Deactivate user (Admin) / 停用用户（管理员）"""
    return {"message": f"User {user_id} deactivated"}


# ============================================
# Admin - Projects / 管理员 - 项目管理
# ============================================

@app.get("/api/admin/projects")
async def admin_list_projects(page: int = 1, limit: int = 10, search: Optional[str] = None):
    """List all projects (Admin) / 列出所有项目（管理员）"""
    projects = [
        {
            "id": "proj_1",
            "name": "AI Code Review Platform",
            "owner": {"id": "user_1", "name": "Admin User", "email": "admin@example.com"},
            "language": "TypeScript",
            "status": "active",
            "issues_count": 12,
            "analyses_count": 35,
            "created_at": (datetime.now() - timedelta(days=30)).isoformat()
        },
        {
            "id": "proj_2",
            "name": "Backend Services",
            "owner": {"id": "user_2", "name": "Regular User", "email": "user@example.com"},
            "language": "Python",
            "status": "active",
            "issues_count": 5,
            "analyses_count": 20,
            "created_at": (datetime.now() - timedelta(days=20)).isoformat()
        }
    ]
    return {"items": projects, "total": len(projects), "page": page, "limit": limit}


@app.delete("/api/admin/projects/{project_id}")
async def admin_delete_project(project_id: str):
    """Delete project (Admin) / 删除项目（管理员）"""
    return {"message": f"Project {project_id} deleted"}


# ============================================
# Admin - Audit Logs / 管理员 - 审计日志
# ============================================

@app.get("/api/admin/audit")
async def get_audit_logs(page: int = 1, limit: int = 20):
    """Get audit logs (Admin) / 获取审计日志（管理员）"""
    logs = [
        {
            "id": f"audit_{i}",
            "action": random.choice(["login", "logout", "create_project", "delete_project", "update_settings", "analyze_code"]),
            "user": {"id": f"user_{random.randint(1, 3)}", "name": f"User {random.randint(1, 3)}"},
            "ip_address": f"192.168.1.{random.randint(1, 255)}",
            "details": "Action completed successfully",
            "created_at": (datetime.now() - timedelta(hours=i)).isoformat()
        }
        for i in range(min(limit, 20))
    ]
    return {"items": logs, "total": 100, "page": page, "limit": limit}


# ============================================
# Admin - AI Providers / 管理员 - AI提供商
# ============================================

@app.get("/api/admin/providers")
async def list_providers():
    """List AI providers (Admin) / 列出AI提供商（管理员）"""
    return {
        "items": [
            {
                "id": "openai",
                "name": "OpenAI",
                "status": "healthy",
                "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "default_model": "gpt-4-turbo",
                "rate_limit": 10000,
                "usage_today": 1250,
                "cost_today": 45.60
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "status": "healthy",
                "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "default_model": "claude-3-sonnet",
                "rate_limit": 5000,
                "usage_today": 830,
                "cost_today": 32.40
            }
        ]
    }


@app.put("/api/admin/providers/{provider_id}")
async def update_provider(provider_id: str):
    """Update AI provider settings (Admin) / 更新AI提供商设置（管理员）"""
    return {"message": f"Provider {provider_id} updated"}


@app.get("/api/admin/providers/{provider_id}/health")
async def check_provider_health(provider_id: str):
    """Check provider health (Admin) / 检查提供商健康状态（管理员）"""
    return {
        "provider_id": provider_id,
        "status": "healthy",
        "latency_ms": random.randint(100, 500),
        "last_check": datetime.now().isoformat()
    }


# ============================================
# Admin - System Stats / 管理员 - 系统统计
# ============================================

@app.get("/api/admin/stats")
async def get_admin_stats():
    """Get admin dashboard stats / 获取管理仪表板统计"""
    return {
        "users": {"total": 150, "active": 120, "new_today": 5},
        "projects": {"total": 340, "active": 280},
        "analyses": {"total": 15000, "today": 230, "this_week": 1450},
        "issues": {"found": 45000, "resolved": 38000, "resolution_rate": 0.84},
        "ai_usage": {
            "requests_today": 2080,
            "cost_today": 78.00,
            "cost_this_month": 1250.45
        }
    }


# ============================================
# Admin - Invitations / 管理员 - 邀请管理
# ============================================

@app.get("/api/admin/invitations")
async def list_invitations():
    """List invitation codes (Admin) / 列出邀请码（管理员）"""
    return {
        "items": [
            {
                "id": "inv_1",
                "code": "ZBXzbx123",
                "type": "admin",
                "uses": 5,
                "max_uses": 10,
                "expires_at": (datetime.now() + timedelta(days=30)).isoformat()
            },
            {
                "id": "inv_2",
                "code": "USER2024",
                "type": "user",
                "uses": 20,
                "max_uses": 100,
                "expires_at": (datetime.now() + timedelta(days=60)).isoformat()
            }
        ]
    }


@app.post("/api/admin/invitations")
async def create_invitation():
    """Create invitation code (Admin) / 创建邀请码（管理员）"""
    return {
        "id": f"inv_{secrets.token_hex(4)}",
        "code": secrets.token_urlsafe(8),
        "message": "Invitation created"
    }


@app.delete("/api/admin/invitations/{invitation_id}")
async def delete_invitation(invitation_id: str):
    """Delete invitation code (Admin) / 删除邀请码（管理员）"""
    return {"message": f"Invitation {invitation_id} deleted"}


# ============================================
# AI Interaction / AI交互
# ============================================

@app.post("/api/ai/analyze")
async def ai_analyze_code(code: str = "", language: str = "typescript", model: str = "gpt-4-turbo"):
    """Analyze code with AI / AI代码分析"""
    return {
        "session_id": f"analysis_{secrets.token_hex(8)}",
        "model": model,
        "language": language,
        "issues": [
            {
                "id": "1",
                "type": "error",
                "severity": "critical",
                "message": "Hardcoded secret key detected",
                "line": 7,
                "column": 21,
                "rule": "security/no-hardcoded-secrets",
                "suggestion": "Use environment variables",
                "autoFix": True
            },
            {
                "id": "2",
                "type": "error",
                "severity": "critical",
                "message": "SQL Injection vulnerability",
                "line": 14,
                "column": 5,
                "rule": "security/sql-injection",
                "autoFix": False
            },
            {
                "id": "3",
                "type": "warning",
                "severity": "high",
                "message": "Missing input validation",
                "line": 10,
                "column": 3,
                "rule": "security/input-validation"
            }
        ],
        "summary": {
            "total_issues": 3,
            "critical": 2,
            "high": 1,
            "medium": 0,
            "low": 0
        },
        "processing_time_ms": random.randint(500, 2000)
    }


@app.post("/api/ai/chat")
async def ai_chat(message: str = "", code: str = "", language: str = "typescript", model: str = "gpt-4-turbo"):
    """Chat with AI about code / 与AI聊天讨论代码"""
    responses = [
        "Based on my analysis, I found several issues in your code. The most critical one is the hardcoded secret key which should be moved to environment variables.",
        "I can help you fix these issues. The SQL injection vulnerability can be resolved by using parameterized queries instead of string concatenation.",
        "To improve your code security, consider implementing input validation using a schema validation library like Zod or Joi.",
    ]
    return {
        "response": random.choice(responses),
        "model": model,
        "tokens_used": random.randint(100, 500),
        "processing_time_ms": random.randint(500, 3000)
    }


@app.get("/api/ai/models")
async def list_ai_models():
    """List available AI models / 列出可用AI模型"""
    return {
        "items": [
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "provider": "OpenAI", "status": "active"},
            {"id": "gpt-4", "name": "GPT-4", "provider": "OpenAI", "status": "active"},
            {"id": "claude-3-opus", "name": "Claude 3 Opus", "provider": "Anthropic", "status": "active"},
            {"id": "claude-3-sonnet", "name": "Claude 3 Sonnet", "provider": "Anthropic", "status": "active"},
        ]
    }


# ============================================
# Admin - AI Models / 管理员 - AI模型管理
# ============================================

@app.get("/api/admin/ai-models")
async def admin_list_ai_models():
    """List all AI models (Admin) / 列出所有AI模型（管理员）"""
    return {
        "items": [
            {
                "id": "model_1",
                "name": "GPT-4 Turbo",
                "provider": "OpenAI",
                "model_id": "gpt-4-turbo",
                "version": "v2.1.0",
                "zone": "v2-production",
                "status": "active",
                "metrics": {
                    "accuracy": 0.94,
                    "latency_p95": 2.3,
                    "error_rate": 0.02,
                    "cost_per_request": 0.08,
                    "requests_today": 1250,
                    "success_rate": 0.98
                },
                "config": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.95},
                "created_at": (datetime.now() - timedelta(days=45)).isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            {
                "id": "model_2",
                "name": "Claude 3 Opus",
                "provider": "Anthropic",
                "model_id": "claude-3-opus",
                "version": "v2.0.0",
                "zone": "v2-production",
                "status": "active",
                "metrics": {
                    "accuracy": 0.92,
                    "latency_p95": 3.1,
                    "error_rate": 0.03,
                    "cost_per_request": 0.12,
                    "requests_today": 830,
                    "success_rate": 0.97
                },
                "config": {"max_tokens": 4096, "temperature": 0.5, "top_p": 0.9},
                "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            {
                "id": "model_3",
                "name": "GPT-4 Vision",
                "provider": "OpenAI",
                "model_id": "gpt-4-vision",
                "version": "v1.0.0",
                "zone": "v1-experimentation",
                "status": "testing",
                "metrics": {
                    "accuracy": 0.88,
                    "latency_p95": 4.5,
                    "error_rate": 0.05,
                    "cost_per_request": 0.15,
                    "requests_today": 150,
                    "success_rate": 0.95
                },
                "config": {"max_tokens": 2048, "temperature": 0.6, "top_p": 0.95},
                "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            {
                "id": "model_4",
                "name": "Claude 2.1",
                "provider": "Anthropic",
                "model_id": "claude-2.1",
                "version": "v1.5.0",
                "zone": "v3-quarantine",
                "status": "quarantined",
                "metrics": {
                    "accuracy": 0.78,
                    "latency_p95": 5.2,
                    "error_rate": 0.12,
                    "cost_per_request": 0.10,
                    "requests_today": 0,
                    "success_rate": 0.88
                },
                "config": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.9},
                "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
                "updated_at": (datetime.now() - timedelta(days=15)).isoformat()
            }
        ]
    }


@app.get("/api/admin/ai-models/{model_id}")
async def admin_get_ai_model(model_id: str):
    """Get AI model details (Admin) / 获取AI模型详情（管理员）"""
    return {
        "id": model_id,
        "name": "GPT-4 Turbo",
        "provider": "OpenAI",
        "model_id": "gpt-4-turbo",
        "version": "v2.1.0",
        "zone": "v2-production",
        "status": "active",
        "metrics": {
            "accuracy": 0.94,
            "latency_p95": 2.3,
            "error_rate": 0.02,
            "cost_per_request": 0.08,
            "requests_today": 1250,
            "success_rate": 0.98
        },
        "config": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.95}
    }


@app.put("/api/admin/ai-models/{model_id}/config")
async def admin_update_ai_model_config(model_id: str):
    """Update AI model config (Admin) / 更新AI模型配置（管理员）"""
    return {"message": f"Model {model_id} configuration updated"}


@app.post("/api/admin/ai-models/{model_id}/promote")
async def admin_promote_ai_model(model_id: str):
    """Promote AI model to V2 Production / 将AI模型提升到V2生产环境"""
    return {
        "message": f"Model {model_id} promoted to V2 Production",
        "previous_zone": "v1-experimentation",
        "new_zone": "v2-production",
        "promoted_at": datetime.now().isoformat()
    }


@app.post("/api/admin/ai-models/{model_id}/rollback")
async def admin_rollback_ai_model(model_id: str):
    """Rollback AI model to V1 Experimentation / 将AI模型回滚到V1实验区"""
    return {
        "message": f"Model {model_id} rolled back to V1 Experimentation",
        "previous_zone": "v2-production",
        "new_zone": "v1-experimentation",
        "rolled_back_at": datetime.now().isoformat()
    }


@app.post("/api/admin/ai-models/{model_id}/quarantine")
async def admin_quarantine_ai_model(model_id: str):
    """Move AI model to V3 Quarantine / 将AI模型移至V3隔离区"""
    return {
        "message": f"Model {model_id} moved to V3 Quarantine",
        "previous_zone": "v2-production",
        "new_zone": "v3-quarantine",
        "quarantined_at": datetime.now().isoformat(),
        "reason": "Performance degradation detected"
    }


@app.get("/api/admin/ai-models/{model_id}/versions")
async def admin_get_ai_model_versions(model_id: str):
    """Get AI model version history / 获取AI模型版本历史"""
    return {
        "items": [
            {
                "version": "v2.1.0",
                "zone": "v2-production",
                "status": "active",
                "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
                "metrics": {"accuracy": 0.94, "error_rate": 0.02}
            },
            {
                "version": "v2.0.0",
                "zone": "v2-production",
                "status": "deprecated",
                "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
                "metrics": {"accuracy": 0.91, "error_rate": 0.04}
            },
            {
                "version": "v1.0.0",
                "zone": "v3-quarantine",
                "status": "quarantined",
                "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
                "metrics": {"accuracy": 0.85, "error_rate": 0.08}
            }
        ]
    }


@app.get("/api/admin/ai-models/metrics/overview")
async def admin_ai_models_metrics_overview():
    """Get AI models metrics overview / 获取AI模型指标概览"""
    return {
        "total_requests_today": 2080,
        "total_cost_today": 78.50,
        "average_latency": 2.8,
        "average_accuracy": 0.93,
        "error_rate": 0.025,
        "models_by_zone": {
            "v1-experimentation": 1,
            "v2-production": 2,
            "v3-quarantine": 1
        },
        "hourly_requests": [
            {"hour": "00:00", "requests": 45},
            {"hour": "01:00", "requests": 32},
            {"hour": "02:00", "requests": 28},
            {"hour": "03:00", "requests": 25},
            {"hour": "04:00", "requests": 30},
            {"hour": "05:00", "requests": 42},
            {"hour": "06:00", "requests": 85},
            {"hour": "07:00", "requests": 120},
            {"hour": "08:00", "requests": 180},
            {"hour": "09:00", "requests": 220},
            {"hour": "10:00", "requests": 250},
            {"hour": "11:00", "requests": 280}
        ]
    }


# ============================================
# API Keys / API密钥
# ============================================

@app.get("/api/user/api-keys")
async def list_api_keys():
    """List user API keys / 列出用户API密钥"""
    return {
        "items": [
            {
                "id": "key_1",
                "name": "Production API Key",
                "keyPreview": "crai_prod_...3f8a",
                "scopes": ["read", "write", "analyze"],
                "status": "active",
                "createdAt": (datetime.now() - timedelta(days=45)).isoformat(),
                "lastUsed": datetime.now().isoformat(),
                "usageCount": 15420,
                "rateLimit": 10000,
                "rateLimitUsed": 2340
            },
            {
                "id": "key_2",
                "name": "Development Key",
                "keyPreview": "crai_test_...9b2c",
                "scopes": ["read", "analyze"],
                "status": "active",
                "createdAt": (datetime.now() - timedelta(days=30)).isoformat(),
                "lastUsed": (datetime.now() - timedelta(hours=2)).isoformat(),
                "usageCount": 3250,
                "rateLimit": 1000,
                "rateLimitUsed": 450
            }
        ]
    }


@app.post("/api/user/api-keys")
async def create_api_key():
    """Create API key / 创建API密钥"""
    return {
        "id": f"key_{secrets.token_hex(4)}",
        "key": f"crai_{secrets.token_hex(24)}",
        "message": "API key created"
    }


@app.delete("/api/user/api-keys/{key_id}")
async def revoke_api_key(key_id: str):
    """Revoke API key / 撤销API密钥"""
    return {"message": f"API key {key_id} revoked"}


# ============================================
# Integrations & Webhooks / 集成和Webhooks
# ============================================

@app.get("/api/user/integrations")
async def list_integrations():
    """List integrations / 列出集成"""
    return {
        "items": [
            {"id": "int_1", "name": "GitHub", "provider": "github", "status": "connected"},
            {"id": "int_2", "name": "Slack", "provider": "slack", "status": "connected"}
        ]
    }


@app.post("/api/user/integrations/{provider}/connect")
async def connect_integration(provider: str):
    """Connect integration / 连接集成"""
    return {"message": f"Connected to {provider}", "redirect_url": f"https://oauth.{provider}.com/authorize"}


@app.delete("/api/user/integrations/{integration_id}")
async def disconnect_integration(integration_id: str):
    """Disconnect integration / 断开集成"""
    return {"message": f"Integration {integration_id} disconnected"}


@app.get("/api/user/webhooks")
async def list_webhooks():
    """List webhooks / 列出Webhooks"""
    return {
        "items": [
            {
                "id": "wh_1",
                "name": "CI/CD Pipeline",
                "url": "https://ci.example.com/webhook",
                "events": ["analysis.completed", "issue.critical"],
                "status": "active",
                "failureCount": 0
            }
        ]
    }


@app.post("/api/user/webhooks")
async def create_webhook():
    """Create webhook / 创建Webhook"""
    return {"id": f"wh_{secrets.token_hex(4)}", "message": "Webhook created"}


@app.put("/api/user/webhooks/{webhook_id}")
async def update_webhook(webhook_id: str):
    """Update webhook / 更新Webhook"""
    return {"message": f"Webhook {webhook_id} updated"}


@app.delete("/api/user/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str):
    """Delete webhook / 删除Webhook"""
    return {"message": f"Webhook {webhook_id} deleted"}


@app.post("/api/user/webhooks/{webhook_id}/test")
async def test_webhook(webhook_id: str):
    """Test webhook / 测试Webhook"""
    return {"message": "Test event sent", "success": True}


# ============================================
# Teams / 团队
# ============================================

@app.get("/api/teams")
async def list_teams():
    """List teams / 列出团队"""
    return {
        "items": [
            {
                "id": "team_1",
                "name": "Frontend Team",
                "description": "React and TypeScript development",
                "memberCount": 8,
                "projectCount": 5,
                "createdAt": (datetime.now() - timedelta(days=60)).isoformat()
            },
            {
                "id": "team_2",
                "name": "Backend Team",
                "description": "Python and FastAPI development",
                "memberCount": 6,
                "projectCount": 4,
                "createdAt": (datetime.now() - timedelta(days=45)).isoformat()
            }
        ]
    }


@app.post("/api/teams")
async def create_team():
    """Create team / 创建团队"""
    return {"id": f"team_{secrets.token_hex(4)}", "message": "Team created"}


@app.get("/api/teams/{team_id}")
async def get_team(team_id: str):
    """Get team details / 获取团队详情"""
    return {
        "id": team_id,
        "name": "Frontend Team",
        "description": "React and TypeScript development",
        "memberCount": 8,
        "projectCount": 5,
        "members": [
            {"id": "user_1", "name": "John Doe", "email": "john@example.com", "role": "owner", "status": "active"},
            {"id": "user_2", "name": "Jane Smith", "email": "jane@example.com", "role": "admin", "status": "active"}
        ]
    }


@app.post("/api/teams/{team_id}/invite")
async def invite_team_member(team_id: str):
    """Invite team member / 邀请团队成员"""
    return {"message": "Invitation sent"}


@app.put("/api/teams/{team_id}/members/{member_id}")
async def update_team_member(team_id: str, member_id: str):
    """Update team member / 更新团队成员"""
    return {"message": f"Member {member_id} updated"}


@app.delete("/api/teams/{team_id}/members/{member_id}")
async def remove_team_member(team_id: str, member_id: str):
    """Remove team member / 移除团队成员"""
    return {"message": f"Member {member_id} removed"}


# ============================================
# Reports / 报告
# ============================================

@app.get("/api/reports")
async def list_reports():
    """List reports / 列出报告"""
    return {
        "items": [
            {
                "id": "report_1",
                "name": "Weekly Security Scan",
                "type": "security",
                "status": "completed",
                "format": "pdf",
                "createdAt": datetime.now().isoformat(),
                "size": "2.4 MB"
            }
        ]
    }


@app.post("/api/reports/generate")
async def generate_report():
    """Generate report / 生成报告"""
    return {"id": f"report_{secrets.token_hex(4)}", "message": "Report generation started", "status": "generating"}


@app.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    """Get report details / 获取报告详情"""
    return {"id": report_id, "status": "completed", "downloadUrl": f"/api/reports/{report_id}/download"}


@app.delete("/api/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete report / 删除报告"""
    return {"message": f"Report {report_id} deleted"}


@app.get("/api/reports/scheduled")
async def list_scheduled_reports():
    """List scheduled reports / 列出定时报告"""
    return {
        "items": [
            {
                "id": "sched_1",
                "name": "Weekly Security Summary",
                "type": "security",
                "frequency": "weekly",
                "recipients": ["security@example.com"],
                "enabled": True,
                "nextRun": (datetime.now() + timedelta(days=7)).isoformat()
            }
        ]
    }


@app.post("/api/reports/schedule")
async def create_scheduled_report():
    """Create scheduled report / 创建定时报告"""
    return {"id": f"sched_{secrets.token_hex(4)}", "message": "Scheduled report created"}


# ============================================
# Security / 安全
# ============================================

@app.get("/api/security/vulnerabilities")
async def list_vulnerabilities():
    """List vulnerabilities / 列出漏洞"""
    return {
        "items": [
            {
                "id": "vuln_1",
                "title": "SQL Injection in User Authentication",
                "severity": "critical",
                "category": "A03:2021 Injection",
                "cve": "CVE-2024-1234",
                "project": "Backend Services",
                "file": "src/auth/login.py",
                "line": 45,
                "status": "open"
            }
        ],
        "summary": {
            "critical": 2,
            "high": 5,
            "medium": 12,
            "low": 8,
            "total": 27
        }
    }


@app.get("/api/security/compliance")
async def get_compliance_status():
    """Get compliance status / 获取合规状态"""
    return {
        "items": [
            {"name": "OWASP Top 10", "status": "partial", "score": 72, "items": 10, "passed": 7},
            {"name": "PCI DSS", "status": "passing", "score": 95, "items": 12, "passed": 11},
            {"name": "SOC 2", "status": "passing", "score": 88, "items": 15, "passed": 13}
        ]
    }


# ============================================
# Analytics / 分析
# ============================================

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview / 获取分析概览"""
    return {
        "totalAnalyses": 1547,
        "issuesFound": 4832,
        "issuesResolved": 4156,
        "avgResponseTime": 2.3,
        "totalCost": 1250.45,
        "securityIssues": 127
    }


@app.get("/api/analytics/trends")
async def get_analytics_trends():
    """Get analytics trends / 获取分析趋势"""
    return {
        "weekly": [45, 52, 38, 65, 72, 58, 81],
        "monthly": [120, 135, 128, 145, 162, 158, 175, 190, 185, 210, 225, 240]
    }


# ============================================
# Activity Feed / 活动动态
# ============================================

@app.get("/api/activity")
async def list_activities():
    """List activities / 列出活动"""
    return {
        "items": [
            {
                "id": "act_1",
                "type": "analysis",
                "action": "completed",
                "description": "Code analysis completed for frontend/src/components",
                "user": {"name": "AI Assistant"},
                "project": "Frontend",
                "branch": "feature/new-dashboard",
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "act_2",
                "type": "security",
                "action": "detected",
                "description": "Critical vulnerability detected: SQL Injection",
                "user": {"name": "Security Scanner"},
                "project": "Backend",
                "file": "src/auth/login.py",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat()
            },
            {
                "id": "act_3",
                "type": "review",
                "action": "approved",
                "description": "Approved pull request: Add user authentication",
                "user": {"name": "John Doe"},
                "project": "Backend",
                "branch": "feature/auth",
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()
            }
        ]
    }


# ============================================
# Repositories / 仓库
# ============================================

@app.get("/api/repositories")
async def list_repositories():
    """List repositories / 列出仓库"""
    return {
        "items": [
            {
                "id": "repo_1",
                "name": "ai-code-review-platform",
                "fullName": "myorg/ai-code-review-platform",
                "description": "AI-powered code review platform",
                "provider": "github",
                "visibility": "private",
                "defaultBranch": "main",
                "language": "TypeScript",
                "languageColor": "#3178c6",
                "stars": 128,
                "forks": 24,
                "issues": 12,
                "analysisStatus": "passing",
                "healthScore": 92,
                "branches": 8,
                "updatedAt": datetime.now().isoformat()
            },
            {
                "id": "repo_2",
                "name": "backend-services",
                "fullName": "myorg/backend-services",
                "description": "FastAPI backend microservices",
                "provider": "github",
                "visibility": "private",
                "defaultBranch": "main",
                "language": "Python",
                "languageColor": "#3572A5",
                "stars": 56,
                "forks": 12,
                "issues": 5,
                "analysisStatus": "failing",
                "healthScore": 78,
                "branches": 5,
                "updatedAt": (datetime.now() - timedelta(hours=2)).isoformat()
            }
        ]
    }


@app.post("/api/repositories/connect")
async def connect_repository():
    """Connect repository / 连接仓库"""
    return {"id": f"repo_{secrets.token_hex(4)}", "message": "Repository connected"}


@app.post("/api/repositories/{repo_id}/analyze")
async def analyze_repository(repo_id: str):
    """Analyze repository / 分析仓库"""
    return {"message": f"Analysis started for {repo_id}", "jobId": f"job_{secrets.token_hex(4)}"}


@app.delete("/api/repositories/{repo_id}")
async def disconnect_repository(repo_id: str):
    """Disconnect repository / 断开仓库"""
    return {"message": f"Repository {repo_id} disconnected"}


@app.get("/api/repositories/{repo_id}/branches")
async def get_repository_branches(repo_id: str):
    """Get repository branches / 获取仓库分支"""
    return {
        "items": [
            {"name": "main", "isDefault": True, "lastCommit": datetime.now().isoformat()},
            {"name": "develop", "isDefault": False, "lastCommit": (datetime.now() - timedelta(hours=1)).isoformat()},
            {"name": "feature/auth", "isDefault": False, "lastCommit": (datetime.now() - timedelta(hours=3)).isoformat()}
        ]
    }


# ============================================
# Pull Requests / 拉取请求
# ============================================

@app.get("/api/pull-requests")
async def list_pull_requests():
    """List pull requests / 列出拉取请求"""
    return {
        "items": [
            {
                "id": "pr_1",
                "number": 142,
                "title": "Add user authentication with JWT tokens",
                "author": {"name": "John Doe"},
                "repository": "backend-services",
                "sourceBranch": "feature/auth",
                "targetBranch": "main",
                "status": "open",
                "reviewStatus": "approved",
                "aiScore": 92,
                "issuesFound": 2,
                "comments": 8,
                "commits": 5,
                "createdAt": (datetime.now() - timedelta(days=2)).isoformat()
            },
            {
                "id": "pr_2",
                "number": 141,
                "title": "Fix SQL injection vulnerability",
                "author": {"name": "AI Auto-Fix"},
                "repository": "backend-services",
                "sourceBranch": "auto-fix/sql-injection",
                "targetBranch": "main",
                "status": "open",
                "reviewStatus": "pending",
                "aiScore": 98,
                "issuesFound": 0,
                "comments": 2,
                "commits": 1,
                "createdAt": (datetime.now() - timedelta(hours=1)).isoformat()
            }
        ]
    }


@app.post("/api/pull-requests/{pr_id}/merge")
async def merge_pull_request(pr_id: str):
    """Merge pull request / 合并拉取请求"""
    return {"message": f"Pull request {pr_id} merged", "mergedAt": datetime.now().isoformat()}


@app.post("/api/pull-requests/{pr_id}/close")
async def close_pull_request(pr_id: str):
    """Close pull request / 关闭拉取请求"""
    return {"message": f"Pull request {pr_id} closed"}


@app.get("/api/pull-requests/{pr_id}/review")
async def get_pr_review(pr_id: str):
    """Get PR AI review / 获取PR的AI审查"""
    return {
        "prId": pr_id,
        "aiScore": 92,
        "issues": [
            {"type": "warning", "message": "Consider adding unit tests", "file": "src/auth.py", "line": 45},
            {"type": "info", "message": "Code complexity is high", "file": "src/auth.py", "line": 78}
        ],
        "suggestions": [
            "Add input validation for email field",
            "Consider using async/await for database operations"
        ]
    }


# ============================================
# AI Auto-Fix / AI自动修复
# ============================================

@app.get("/api/admin/auto-fix")
async def list_auto_fixes():
    """List auto-fixes / 列出自动修复"""
    return {
        "items": [
            {
                "id": "fix_1",
                "type": "vulnerability",
                "severity": "critical",
                "title": "SQL Injection in User Query",
                "description": "Unsanitized user input used directly in SQL query",
                "file": "src/api/users.py",
                "line": 45,
                "status": "review",
                "aiModel": "GPT-4 Turbo",
                "confidence": 0.95,
                "fix": {
                    "before": 'query = f"SELECT * FROM users WHERE id = {user_id}"',
                    "after": 'query = "SELECT * FROM users WHERE id = %s"\\ncursor.execute(query, (user_id,))',
                    "explanation": "Replaced string interpolation with parameterized query"
                },
                "createdAt": (datetime.now() - timedelta(minutes=30)).isoformat()
            },
            {
                "id": "fix_2",
                "type": "vulnerability",
                "severity": "critical",
                "title": "Hardcoded API Secret",
                "file": "src/config/settings.py",
                "line": 12,
                "status": "pending",
                "aiModel": "Claude 3 Opus",
                "confidence": 0.98,
                "createdAt": (datetime.now() - timedelta(minutes=45)).isoformat()
            }
        ],
        "stats": {
            "total": 5,
            "applied": 2,
            "pending": 2,
            "rejected": 1
        }
    }


@app.post("/api/admin/auto-fix/{fix_id}/approve")
async def approve_auto_fix(fix_id: str):
    """Approve auto-fix / 批准自动修复"""
    return {
        "message": f"Fix {fix_id} approved and applied",
        "appliedAt": datetime.now().isoformat()
    }


@app.post("/api/admin/auto-fix/{fix_id}/reject")
async def reject_auto_fix(fix_id: str):
    """Reject auto-fix / 拒绝自动修复"""
    return {"message": f"Fix {fix_id} rejected"}


@app.post("/api/admin/auto-fix/{fix_id}/rollback")
async def rollback_auto_fix(fix_id: str):
    """Rollback auto-fix / 回滚自动修复"""
    return {"message": f"Fix {fix_id} rolled back"}


@app.post("/api/admin/auto-fix/cycle/start")
async def start_fix_cycle():
    """Start auto-fix cycle / 开始自动修复周期"""
    return {
        "cycleId": f"cycle_{secrets.token_hex(4)}",
        "message": "Fix cycle started",
        "startedAt": datetime.now().isoformat()
    }


@app.get("/api/admin/auto-fix/cycles")
async def list_fix_cycles():
    """List fix cycles / 列出修复周期"""
    return {
        "items": [
            {
                "id": "cycle_1",
                "startedAt": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "status": "running",
                "issuesFound": 12,
                "issuesFixed": 8,
                "issuesPending": 4
            },
            {
                "id": "cycle_2",
                "startedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                "completedAt": (datetime.now() - timedelta(hours=23)).isoformat(),
                "status": "completed",
                "issuesFound": 25,
                "issuesFixed": 23,
                "issuesPending": 0
            }
        ]
    }


# ============================================
# Deployments / 部署
# ============================================

@app.get("/api/deployments")
async def list_deployments():
    """List deployments / 列出部署"""
    return {
        "items": [
            {
                "id": "deploy_1",
                "version": "v2.1.5",
                "environment": "production",
                "status": "success",
                "branch": "main",
                "commit": "a1b2c3d",
                "commitMessage": "Add user authentication with JWT tokens",
                "deployedBy": "CI/CD Pipeline",
                "startedAt": (datetime.now() - timedelta(hours=2)).isoformat(),
                "duration": 480,
                "stages": [
                    {"name": "Build", "status": "success", "duration": 120},
                    {"name": "Test", "status": "success", "duration": 180},
                    {"name": "Security Scan", "status": "success", "duration": 60},
                    {"name": "Deploy", "status": "success", "duration": 120}
                ]
            },
            {
                "id": "deploy_2",
                "version": "v2.1.4",
                "environment": "staging",
                "status": "in_progress",
                "branch": "develop",
                "commit": "e4f5g6h",
                "commitMessage": "Implement dashboard analytics",
                "deployedBy": "John Doe",
                "startedAt": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "stages": [
                    {"name": "Build", "status": "success", "duration": 115},
                    {"name": "Test", "status": "success", "duration": 165},
                    {"name": "Security Scan", "status": "in_progress"},
                    {"name": "Deploy", "status": "pending"}
                ]
            }
        ]
    }


@app.post("/api/deployments")
async def create_deployment():
    """Create deployment / 创建部署"""
    return {
        "id": f"deploy_{secrets.token_hex(4)}",
        "message": "Deployment started",
        "status": "pending"
    }


@app.post("/api/deployments/{deploy_id}/rollback")
async def rollback_deployment(deploy_id: str):
    """Rollback deployment / 回滚部署"""
    return {"message": f"Deployment {deploy_id} rolled back"}


@app.get("/api/deployments/{deploy_id}")
async def get_deployment(deploy_id: str):
    """Get deployment details / 获取部署详情"""
    return {
        "id": deploy_id,
        "version": "v2.1.5",
        "environment": "production",
        "status": "success",
        "stages": [
            {"name": "Build", "status": "success", "duration": 120},
            {"name": "Test", "status": "success", "duration": 180},
            {"name": "Security Scan", "status": "success", "duration": 60},
            {"name": "Deploy", "status": "success", "duration": 120}
        ]
    }


# ============================================
# Code Comparison / 代码比较
# ============================================

@app.get("/api/compare")
async def compare_code(base: str = "main", head: str = "develop"):
    """Compare code between branches / 比较分支代码"""
    return {
        "base": base,
        "head": head,
        "additions": 45,
        "deletions": 12,
        "files": [
            {
                "path": "src/auth/authentication.py",
                "additions": 20,
                "deletions": 8,
                "status": "modified"
            },
            {
                "path": "src/utils/validation.py",
                "additions": 15,
                "deletions": 0,
                "status": "added"
            },
            {
                "path": "src/config/old_settings.py",
                "additions": 0,
                "deletions": 4,
                "status": "deleted"
            }
        ],
        "aiAnalysis": {
            "title": "Security Fix: SQL Injection Prevention",
            "description": "This change addresses a critical SQL injection vulnerability",
            "riskLevel": "low",
            "confidence": 0.96
        }
    }


# ============================================
# Code Quality Rules / 代码质量规则
# ============================================

@app.get("/api/rules")
async def list_rules():
    """List code quality rules / 列出代码质量规则"""
    return {
        "items": [
            {
                "id": "rule_1",
                "name": "no-sql-injection",
                "description": "Prevent SQL injection",
                "category": "security",
                "severity": "error",
                "enabled": True,
                "autoFix": True
            },
            {
                "id": "rule_2",
                "name": "no-hardcoded-secrets",
                "description": "Detect hardcoded secrets",
                "category": "security",
                "severity": "error",
                "enabled": True,
                "autoFix": False
            },
            {
                "id": "rule_3",
                "name": "max-complexity",
                "description": "Enforce max complexity",
                "category": "quality",
                "severity": "warning",
                "enabled": True,
                "autoFix": False
            }
        ],
        "stats": {
            "total": 15,
            "enabled": 12,
            "errors": 5,
            "warnings": 7
        }
    }


@app.post("/api/rules")
async def create_rule():
    """Create custom rule / 创建自定义规则"""
    return {"id": f"rule_{secrets.token_hex(4)}", "message": "Rule created"}


@app.put("/api/rules/{rule_id}")
async def update_rule(rule_id: str):
    """Update rule / 更新规则"""
    return {"message": f"Rule {rule_id} updated"}


@app.delete("/api/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """Delete rule / 删除规则"""
    return {"message": f"Rule {rule_id} deleted"}


# ============================================
# Billing / 账单
# ============================================

@app.get("/api/billing/usage")
async def get_usage():
    """Get usage statistics / 获取使用统计"""
    return {
        "currentPlan": {
            "name": "Professional",
            "price": 99,
            "tokensIncluded": 1000000,
            "tokensUsed": 685000,
            "analysesIncluded": 500,
            "analysesUsed": 342,
            "nextBillingDate": "2024-04-01"
        },
        "usage": [
            {"date": "2024-03-01", "type": "analysis", "tokens": 45000, "cost": 0.45},
            {"date": "2024-03-01", "type": "chat", "tokens": 12000, "cost": 0.12},
            {"date": "2024-02-29", "type": "analysis", "tokens": 52000, "cost": 0.52}
        ],
        "monthlyTotal": 32.45
    }


@app.get("/api/billing/plans")
async def get_plans():
    """Get available plans / 获取可用计划"""
    return {
        "items": [
            {"name": "Free", "price": 0, "tokens": 50000, "analyses": 25},
            {"name": "Professional", "price": 99, "tokens": 1000000, "analyses": 500},
            {"name": "Enterprise", "price": 499, "tokens": 10000000, "analyses": -1}
        ]
    }


@app.post("/api/billing/upgrade")
async def upgrade_plan():
    """Upgrade plan / 升级计划"""
    return {"message": "Plan upgrade initiated"}


# ============================================
# Notifications / 通知
# ============================================

@app.get("/api/notifications")
async def list_notifications():
    """List notifications / 列出通知"""
    return {
        "items": [
            {
                "id": "n1",
                "type": "security",
                "priority": "high",
                "title": "Critical Vulnerability Detected",
                "message": "SQL injection vulnerability found",
                "read": False,
                "timestamp": datetime.now().isoformat()
            },
            {
                "id": "n2",
                "type": "ai",
                "priority": "medium",
                "title": "AI Auto-Fix Applied",
                "message": "3 security fixes have been applied",
                "read": False,
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()
            },
            {
                "id": "n3",
                "type": "deployment",
                "priority": "low",
                "title": "Deployment Successful",
                "message": "v2.1.5 deployed to production",
                "read": True,
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
            }
        ],
        "unreadCount": 2
    }


@app.put("/api/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """Mark notification as read / 标记通知已读"""
    return {"message": f"Notification {notification_id} marked as read"}


@app.delete("/api/notifications/{notification_id}")
async def delete_notification(notification_id: str):
    """Delete notification / 删除通知"""
    return {"message": f"Notification {notification_id} deleted"}


@app.post("/api/notifications/mark-all-read")
async def mark_all_notifications_read():
    """Mark all notifications as read / 全部标记已读"""
    return {"message": "All notifications marked as read"}


# ============================================
# System Status / 系统状态
# ============================================

@app.get("/api/status")
async def get_system_status():
    """Get system status / 获取系统状态"""
    return {
        "status": "operational",
        "services": [
            {"name": "API Gateway", "status": "operational", "uptime": 99.99, "responseTime": 45},
            {"name": "Web Application", "status": "operational", "uptime": 99.98, "responseTime": 120},
            {"name": "AI Analysis Engine", "status": "operational", "uptime": 99.95, "responseTime": 850},
            {"name": "Database", "status": "operational", "uptime": 99.999, "responseTime": 12},
            {"name": "Job Queue", "status": "operational", "uptime": 99.97, "responseTime": 25},
            {"name": "Authentication", "status": "operational", "uptime": 99.99, "responseTime": 35}
        ],
        "incidents": [],
        "overallUptime": 99.98
    }


# ============================================
# Search / 搜索
# ============================================

@app.get("/api/search")
async def search(q: str = ""):
    """Global search / 全局搜索"""
    return {
        "query": q,
        "results": [
            {
                "id": "1",
                "type": "file",
                "title": "authentication.py",
                "description": "User authentication module",
                "path": "src/api/authentication.py",
                "relevance": 0.95
            },
            {
                "id": "2",
                "type": "issue",
                "title": "SQL Injection Vulnerability",
                "description": "Critical security issue",
                "relevance": 0.92
            },
            {
                "id": "3",
                "type": "project",
                "title": "Backend Services",
                "description": "FastAPI backend microservices",
                "relevance": 0.88
            }
        ],
        "totalCount": 8
    }


# ============================================
# Changelog / 更新日志
# ============================================

@app.get("/api/changelog")
async def get_changelog():
    """Get changelog / 获取更新日志"""
    return {
        "versions": [
            {
                "version": "v2.2.0",
                "date": "2024-03-01",
                "type": "minor",
                "title": "AI Auto-Fix & Enhanced Security",
                "changes": [
                    {"type": "feature", "text": "AI Auto-Fix system"},
                    {"type": "feature", "text": "Real-time code comparison"},
                    {"type": "security", "text": "OWASP Top 10 coverage"}
                ]
            },
            {
                "version": "v2.1.5",
                "date": "2024-02-25",
                "type": "patch",
                "title": "Bug Fixes & Performance",
                "changes": [
                    {"type": "bugfix", "text": "Fixed null pointer exception"},
                    {"type": "improvement", "text": "Optimized API response times"}
                ]
            }
        ]
    }


# ============================================
# Onboarding / 入门引导
# ============================================

@app.get("/api/onboarding/status")
async def get_onboarding_status():
    """Get onboarding status / 获取入门状态"""
    return {
        "completed": False,
        "steps": [
            {"key": "connect", "title": "Connect Repository", "completed": True},
            {"key": "review", "title": "Run First Analysis", "completed": False},
            {"key": "rules", "title": "Configure Rules", "completed": False},
            {"key": "team", "title": "Invite Team", "completed": False},
            {"key": "integrate", "title": "Setup CI/CD", "completed": False}
        ],
        "progress": 20
    }


@app.post("/api/onboarding/complete-step")
async def complete_onboarding_step(step: str = ""):
    """Complete onboarding step / 完成入门步骤"""
    return {"message": f"Step {step} completed"}


@app.post("/api/onboarding/skip")
async def skip_onboarding():
    """Skip onboarding / 跳过入门"""
    return {"message": "Onboarding skipped"}


# ============================================
# AI Assistant / AI助手
# ============================================

@app.post("/api/ai/chat")
async def ai_chat():
    """AI chat / AI对话"""
    return {
        "id": f"msg_{secrets.token_hex(4)}",
        "role": "assistant",
        "content": "Based on my analysis, the code has a potential SQL injection vulnerability. Here's a secure alternative using parameterized queries.",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/ai/conversations")
async def list_conversations():
    """List AI conversations / 列出AI对话"""
    return {
        "items": [
            {"id": "conv_1", "title": "SQL Injection Fix", "timestamp": datetime.now().isoformat()},
            {"id": "conv_2", "title": "React Hook Optimization", "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()}
        ]
    }


# ============================================
# Webhooks / Webhook
# ============================================

@app.get("/api/webhooks")
async def list_webhooks():
    """List webhooks / 列出Webhook"""
    return {
        "items": [
            {
                "id": "wh_1",
                "name": "Slack Notifications",
                "url": "https://hooks.slack.com/services/xxx",
                "events": ["analysis.completed", "security.alert"],
                "status": "active",
                "successRate": 99.5
            },
            {
                "id": "wh_2",
                "name": "CI/CD Pipeline",
                "url": "https://api.github.com/repos/myorg/myrepo/dispatches",
                "events": ["deployment.success"],
                "status": "active",
                "successRate": 100
            }
        ]
    }


@app.post("/api/webhooks")
async def create_webhook():
    """Create webhook / 创建Webhook"""
    return {"id": f"wh_{secrets.token_hex(4)}", "message": "Webhook created"}


@app.post("/api/webhooks/{webhook_id}/test")
async def test_webhook(webhook_id: str):
    """Test webhook / 测试Webhook"""
    return {"message": "Test webhook delivered successfully", "status": 200}


@app.delete("/api/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str):
    """Delete webhook / 删除Webhook"""
    return {"message": f"Webhook {webhook_id} deleted"}


# ============================================
# Code Metrics / 代码度量
# ============================================

@app.get("/api/metrics")
async def get_code_metrics():
    """Get code metrics / 获取代码度量"""
    return {
        "overall": {
            "codeQuality": 78,
            "coverage": 72,
            "technicalDebt": 24,
            "complexity": 12.5,
            "duplications": 3.2,
            "issues": 45
        },
        "files": [
            {"path": "src/api/authentication.py", "lines": 245, "complexity": 8, "coverage": 92, "grade": "A"},
            {"path": "src/api/users.py", "lines": 312, "complexity": 15, "coverage": 78, "grade": "B"},
            {"path": "src/services/analysis.py", "lines": 528, "complexity": 22, "coverage": 65, "grade": "C"}
        ]
    }


@app.get("/api/metrics/trends")
async def get_metrics_trends():
    """Get metrics trends / 获取度量趋势"""
    return {
        "quality": [75, 76, 74, 77, 78, 78],
        "coverage": [68, 69, 70, 71, 72, 72],
        "issues": [52, 50, 48, 47, 46, 45]
    }


# ============================================
# Scheduled Jobs / 计划任务
# ============================================

@app.get("/api/jobs")
async def list_jobs():
    """List scheduled jobs / 列出计划任务"""
    return {
        "items": [
            {
                "id": "job_1",
                "name": "Daily Security Scan",
                "type": "security",
                "schedule": "0 2 * * *",
                "nextRun": (datetime.now() + timedelta(hours=8)).isoformat(),
                "lastStatus": "success",
                "enabled": True
            },
            {
                "id": "job_2",
                "name": "Code Analysis - Backend",
                "type": "analysis",
                "schedule": "0 */6 * * *",
                "nextRun": (datetime.now() + timedelta(hours=2)).isoformat(),
                "lastStatus": "success",
                "enabled": True
            }
        ]
    }


@app.post("/api/jobs")
async def create_job():
    """Create job / 创建任务"""
    return {"id": f"job_{secrets.token_hex(4)}", "message": "Job created"}


@app.post("/api/jobs/{job_id}/run")
async def run_job(job_id: str):
    """Run job now / 立即运行任务"""
    return {"message": f"Job {job_id} started", "executionId": f"exec_{secrets.token_hex(4)}"}


@app.put("/api/jobs/{job_id}/toggle")
async def toggle_job(job_id: str):
    """Toggle job enabled state / 切换任务状态"""
    return {"message": f"Job {job_id} toggled"}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job / 删除任务"""
    return {"message": f"Job {job_id} deleted"}


# ============================================
# Import/Export / 导入导出
# ============================================

@app.get("/api/export")
async def export_config():
    """Export configuration / 导出配置"""
    return {
        "rules": {"count": 15, "size": "12 KB"},
        "webhooks": {"count": 3, "size": "3 KB"},
        "security": {"count": 8, "size": "15 KB"}
    }


@app.post("/api/import")
async def import_config():
    """Import configuration / 导入配置"""
    return {"message": "Import completed", "imported": {"rules": 15, "webhooks": 3}}


@app.get("/api/backups")
async def list_backups():
    """List backups / 列出备份"""
    return {
        "items": [
            {"id": "1", "name": "Full Backup - March 2024", "createdAt": datetime.now().isoformat(), "size": "45.2 MB"},
            {"id": "2", "name": "Config Backup", "createdAt": (datetime.now() - timedelta(days=1)).isoformat(), "size": "128 KB"}
        ]
    }


@app.post("/api/backups")
async def create_backup():
    """Create backup / 创建备份"""
    return {"id": f"backup_{secrets.token_hex(4)}", "message": "Backup created"}


@app.post("/api/backups/{backup_id}/restore")
async def restore_backup(backup_id: str):
    """Restore backup / 恢复备份"""
    return {"message": f"Backup {backup_id} restored"}


# ============================================
# Audit Logs / 审计日志
# ============================================

@app.get("/api/audit-logs")
async def list_audit_logs():
    """List audit logs / 列出审计日志"""
    return {
        "items": [
            {
                "id": "evt_1",
                "timestamp": datetime.now().isoformat(),
                "entity": "analysis",
                "action": "create",
                "actor": {"id": "usr_1", "name": "John Doe", "email": "john@example.com"},
                "resource": "project/backend/analysis/a1b2c3",
                "status": "success",
                "ipAddress": "192.168.1.100",
                "verified": True
            },
            {
                "id": "evt_2",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "entity": "user",
                "action": "login",
                "actor": {"id": "usr_2", "name": "Jane Smith", "email": "jane@example.com"},
                "resource": "auth/session/s1d2f3",
                "status": "success",
                "ipAddress": "10.0.0.50",
                "verified": True
            }
        ],
        "total": 150,
        "verified": 150
    }


@app.post("/api/audit-logs/verify")
async def verify_audit_integrity():
    """Verify audit log integrity / 验证审计日志完整性"""
    return {
        "status": "verified",
        "totalEntries": 150,
        "verifiedEntries": 150,
        "tamperedEntries": 0,
        "message": "All audit logs verified - No tampering detected"
    }


@app.get("/api/audit-logs/export")
async def export_audit_logs():
    """Export audit logs / 导出审计日志"""
    return {"message": "Export initiated", "format": "json", "entries": 150}


# ============================================
# Three-Version Evolution / 三版本演化
# ============================================

@app.get("/api/v1/evolution/status")
async def get_evolution_status():
    """Get evolution cycle status / 获取演化周期状态"""
    return {
        "running": True,
        "currentPhase": "experimentation",
        "cycleId": "cycle_abc123",
        "metrics": {
            "experimentsRun": 45,
            "errorsFixed": 12,
            "promotionsMade": 8,
            "degradationsMade": 3,
        },
        "versions": {
            "v1": {"version": "v1", "status": "online", "model": "claude-3-opus", "latency": 380, "accuracy": 0.89},
            "v2": {"version": "v2", "status": "online", "model": "gpt-4-turbo", "latency": 450, "accuracy": 0.92},
            "v3": {"version": "v3", "status": "online", "model": "gpt-3.5-turbo", "latency": 280, "accuracy": 0.78},
        }
    }


@app.post("/api/v1/evolution/start")
async def start_evolution():
    """Start evolution cycle / 启动演化周期"""
    return {"success": True, "message": "Evolution cycle started"}


@app.post("/api/v1/evolution/stop")
async def stop_evolution():
    """Stop evolution cycle / 停止演化周期"""
    return {"success": True, "message": "Evolution cycle stopped"}


@app.get("/api/v1/evolution/technologies")
async def get_technologies():
    """Get all technologies / 获取所有技术"""
    return [
        {"id": "tech-gpt4-v2", "name": "GPT-4 Turbo", "version": "v2", "status": "active", "accuracy": 0.92, "errorRate": 0.02, "latency": 450, "samples": 15000, "lastUpdated": datetime.now().isoformat()},
        {"id": "tech-claude3-v1", "name": "Claude-3 Opus", "version": "v1", "status": "testing", "accuracy": 0.89, "errorRate": 0.04, "latency": 380, "samples": 2500, "lastUpdated": datetime.now().isoformat()},
        {"id": "tech-llama3-v1", "name": "Llama-3 70B", "version": "v1", "status": "testing", "accuracy": 0.85, "errorRate": 0.05, "latency": 320, "samples": 1800, "lastUpdated": datetime.now().isoformat()},
        {"id": "tech-gpt35-v3", "name": "GPT-3.5 Turbo", "version": "v3", "status": "deprecated", "accuracy": 0.78, "errorRate": 0.08, "latency": 280, "samples": 50000, "lastUpdated": datetime.now().isoformat()},
    ]


@app.post("/api/v1/evolution/promote")
async def promote_technology(tech_id: str = "", reason: str = ""):
    """Promote technology to V2 / 将技术提升到V2"""
    return {"success": True, "tech_id": tech_id, "message": "Technology promoted to V2"}


@app.post("/api/v1/evolution/degrade")
async def degrade_technology(tech_id: str = "", reason: str = ""):
    """Degrade technology to V3 / 将技术降级到V3"""
    return {"success": True, "tech_id": tech_id, "message": "Technology degraded to V3"}


@app.post("/api/v1/evolution/reeval")
async def request_reevaluation(tech_id: str = "", reason: str = ""):
    """Request re-evaluation / 请求重新评估"""
    return {"success": True, "tech_id": tech_id, "message": "Re-evaluation requested"}


# ============================================
# AI Chat & Analysis / AI聊天与分析
# ============================================

@app.post("/api/v1/ai/chat")
@app.post("/api/v2/ai/chat")
@app.post("/api/v3/ai/chat")
async def ai_chat(message: str = "", context: dict = None):
    """AI chat endpoint / AI聊天端点"""
    responses = {
        "security": "Based on your question about security, I recommend: 1) Input validation 2) Authentication 3) Encryption",
        "performance": "For performance optimization: 1) Caching 2) Database optimization 3) Async operations",
        "default": "I'm your AI assistant for code review. How can I help you today?"
    }
    msg_lower = message.lower() if message else ""
    response = responses.get("security") if "security" in msg_lower else responses.get("performance") if "performance" in msg_lower else responses.get("default")
    return {"response": response, "model": "gpt-4-turbo", "latency": random.randint(100, 500), "tokens": len(response.split()) * 2, "version": "v2", "response_id": secrets.token_hex(8)}


@app.post("/api/v1/ai/analyze")
@app.post("/api/v2/ai/analyze")
@app.post("/api/v3/ai/analyze")
async def ai_analyze(code: str = "", language: str = "python", review_types: list = None):
    """AI code analysis / AI代码分析"""
    issues = []
    if "eval(" in code:
        issues.append({"id": "sec-1", "type": "security", "severity": "critical", "title": "Code Injection", "description": "Use of eval() is dangerous", "line": 1, "suggestion": "Use ast.literal_eval()", "fixAvailable": True})
    if "password" in code.lower():
        issues.append({"id": "sec-2", "type": "security", "severity": "high", "title": "Hardcoded Credentials", "description": "Potential hardcoded password", "line": 1, "suggestion": "Use environment variables", "fixAvailable": True})
    if code.count("for") >= 2:
        issues.append({"id": "perf-1", "type": "performance", "severity": "medium", "title": "Nested Loop", "description": "Nested loops may cause O(n²) complexity", "line": 1, "suggestion": "Use hash maps", "fixAvailable": True})
    score = max(0, 100 - len(issues) * 15)
    return {"id": secrets.token_hex(8), "issues": issues, "score": score, "summary": f"Found {len(issues)} issues", "model": "v2-analyzer", "latency": random.randint(100, 300), "version": "v2"}


@app.post("/api/v2/ai/fix")
async def ai_fix(issue_id: str = "", code: str = ""):
    """Apply AI fix / 应用AI修复"""
    fixed = code.replace("eval(", "ast.literal_eval(").replace("except:", "except Exception as e:")
    return {"success": True, "fixed_code": fixed, "fix_id": secrets.token_hex(4), "changes_made": 1}


@app.post("/api/v2/ai/feedback")
async def ai_feedback(response_id: str = "", helpful: bool = True, comment: str = ""):
    """Submit AI feedback / 提交AI反馈"""
    return {"success": True, "message": "Thank you for your feedback!"}


# ============================================
# Security / 安全
# ============================================

@app.get("/api/security/vulnerabilities")
async def get_vulnerabilities(severity: str = None, status: str = None, project: str = None, page: int = 1, limit: int = 20):
    """Get security vulnerabilities / 获取安全漏洞"""
    vulns = [
        {"id": "vuln_1", "title": "SQL Injection in User Authentication", "severity": "critical", "category": "A03:2021 Injection", "cve": "CVE-2024-1234", "project": "Backend Services", "file": "src/auth/login.py", "line": 45, "status": "open", "discoveredAt": datetime.now().isoformat(), "assignee": "John Doe"},
        {"id": "vuln_2", "title": "Hardcoded API Key Exposure", "severity": "critical", "category": "A02:2021 Cryptographic Failures", "project": "AI Platform", "file": "src/services/ai.ts", "line": 12, "status": "in_progress", "discoveredAt": datetime.now().isoformat(), "assignee": "Jane Smith"},
        {"id": "vuln_3", "title": "Cross-Site Scripting (XSS)", "severity": "high", "category": "A03:2021 Injection", "project": "Frontend", "file": "src/components/Comment.tsx", "line": 78, "status": "open", "discoveredAt": datetime.now().isoformat()},
        {"id": "vuln_4", "title": "Insecure Direct Object Reference", "severity": "medium", "category": "A01:2021 Broken Access Control", "project": "Backend Services", "file": "src/api/users.py", "line": 156, "status": "resolved", "discoveredAt": datetime.now().isoformat()},
        {"id": "vuln_5", "title": "Outdated Dependency with Known CVE", "severity": "low", "category": "A06:2021 Vulnerable Components", "cve": "CVE-2023-9999", "project": "Frontend", "file": "package.json", "line": 1, "status": "open", "discoveredAt": datetime.now().isoformat()},
    ]
    filtered = vulns
    if severity:
        filtered = [v for v in filtered if v["severity"] == severity]
    if status:
        filtered = [v for v in filtered if v["status"] == status]
    if project:
        filtered = [v for v in filtered if project.lower() in v["project"].lower()]
    return {"items": filtered[(page-1)*limit:page*limit], "total": len(filtered), "page": page, "limit": limit}


@app.get("/api/security/metrics")
async def get_security_metrics():
    """Get security metrics / 获取安全指标"""
    return {
        "total_vulnerabilities": 25,
        "critical": 2,
        "high": 5,
        "medium": 10,
        "low": 8,
        "open": 12,
        "resolved": 10,
        "in_progress": 3,
        "resolution_rate": 0.72,
        "avg_resolution_time_days": 3.5,
        "trends": {
            "new_this_week": 3,
            "resolved_this_week": 5,
            "critical_open": 2
        }
    }


@app.get("/api/security/compliance")
async def get_compliance_status():
    """Get compliance status / 获取合规状态"""
    return {
        "checks": [
            {"name": "OWASP Top 10", "status": "partial", "score": 72, "items": 10, "passed": 7},
            {"name": "PCI DSS", "status": "passing", "score": 95, "items": 12, "passed": 11},
            {"name": "SOC 2", "status": "passing", "score": 88, "items": 15, "passed": 13},
            {"name": "GDPR", "status": "partial", "score": 78, "items": 8, "passed": 6},
            {"name": "HIPAA", "status": "failing", "score": 45, "items": 10, "passed": 4},
        ],
        "overall_score": 76,
        "last_audit": datetime.now().isoformat()
    }


@app.get("/api/admin/ai-models")
async def get_admin_ai_models():
    """Get AI models for admin / 获取AI模型(管理员)"""
    return {
        "items": [
            {"id": "model_1", "name": "GPT-4 Turbo", "provider": "OpenAI", "model_id": "gpt-4-turbo", "version": "v2.1.0", "zone": "v2-production", "status": "active", "metrics": {"accuracy": 0.94, "latency_p95": 1.8, "error_rate": 0.02, "cost_per_request": 0.03, "requests_today": 15420, "success_rate": 0.98}, "config": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.9}},
            {"id": "model_2", "name": "Claude 3 Opus", "provider": "Anthropic", "model_id": "claude-3-opus", "version": "v1.0.0", "zone": "v1-experimentation", "status": "testing", "metrics": {"accuracy": 0.91, "latency_p95": 2.2, "error_rate": 0.04, "cost_per_request": 0.05, "requests_today": 850, "success_rate": 0.96}, "config": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.9}},
            {"id": "model_3", "name": "Llama 3 70B", "provider": "Meta", "model_id": "llama-3-70b", "version": "v1.2.0", "zone": "v1-experimentation", "status": "testing", "metrics": {"accuracy": 0.88, "latency_p95": 1.5, "error_rate": 0.05, "cost_per_request": 0.01, "requests_today": 2100, "success_rate": 0.95}, "config": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.9}},
        ],
        "total": 3
    }


# ============================================
# Reports / 报告
# ============================================

@app.get("/api/reports")
async def get_reports(page: int = 1, limit: int = 20):
    """Get reports / 获取报告"""
    return {
        "items": [
            {"id": "report_1", "name": "Weekly Security Scan", "type": "security", "status": "completed", "format": "pdf", "project": "AI Platform", "createdAt": datetime.now().isoformat(), "completedAt": datetime.now().isoformat(), "size": "2.3 MB"},
            {"id": "report_2", "name": "Code Quality Report", "type": "code_review", "status": "completed", "format": "pdf", "project": "Backend Services", "createdAt": datetime.now().isoformat(), "completedAt": datetime.now().isoformat(), "size": "1.5 MB"},
            {"id": "report_3", "name": "Compliance Check Q1", "type": "compliance", "status": "generating", "format": "pdf", "createdAt": datetime.now().isoformat()},
            {"id": "report_4", "name": "Performance Analytics", "type": "analytics", "status": "completed", "format": "csv", "createdAt": datetime.now().isoformat(), "completedAt": datetime.now().isoformat(), "size": "856 KB"},
        ],
        "total": 4,
        "page": page,
        "limit": limit
    }


@app.post("/api/reports/generate")
async def generate_report(name: str = "", type: str = "code_review", format: str = "pdf", project: str = ""):
    """Generate report / 生成报告"""
    return {"success": True, "report_id": secrets.token_hex(8), "status": "generating", "estimated_time": 30}


@app.post("/api/reports/schedule")
async def schedule_report(name: str = "", type: str = "code_review", frequency: str = "weekly", recipients: list = None):
    """Schedule report / 计划报告"""
    return {"success": True, "schedule_id": secrets.token_hex(8), "next_run": (datetime.now() + timedelta(days=7)).isoformat()}


# ============================================
# Main / 主程序
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Dev API Server Starting...")
    print("=" * 50)
    print("🌐 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
