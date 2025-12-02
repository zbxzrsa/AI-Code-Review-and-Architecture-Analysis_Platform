"""
Development API Server / 开发API服务器

Handles all non-auth API endpoints for frontend development.
处理所有非认证API端点用于前端开发。

Run with: python dev-api-server.py
运行命令: python dev-api-server.py
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import random

# ============================================
# Models / 模型
# ============================================

class Project(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    language: str
    framework: Optional[str] = None
    repository_url: Optional[str] = None
    status: str = "active"
    issues_count: int = 0
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


@app.post("/api/projects")
async def create_project(
    name: str,
    language: str,
    description: Optional[str] = None,
    framework: Optional[str] = None,
    repository_url: Optional[str] = None
):
    """Create project / 创建项目"""
    project = Project(
        id=f"proj_{secrets.token_hex(4)}",
        name=name,
        description=description,
        language=language,
        framework=framework,
        repository_url=repository_url,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    mock_projects.append(project)
    return project


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete project / 删除项目"""
    for i, project in enumerate(mock_projects):
        if project.id == project_id:
            mock_projects.pop(i)
            return {"message": "Project deleted"}
    raise HTTPException(status_code=404, detail="Project not found")


# ============================================
# Analysis / 分析
# ============================================

@app.post("/api/projects/{project_id}/analyze")
async def start_analysis(project_id: str):
    """Start analysis / 开始分析"""
    return {
        "session_id": f"session_{secrets.token_hex(8)}",
        "status": "started",
        "project_id": project_id
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
