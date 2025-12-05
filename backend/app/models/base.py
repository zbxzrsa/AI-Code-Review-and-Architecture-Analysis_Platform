"""
Base Pydantic Models / 基础 Pydantic 模型
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class ProjectSettings(BaseModel):
    """Project settings model / 项目设置模型"""
    auto_review: bool = False
    review_on_push: bool = False
    review_on_pr: bool = True
    severity_threshold: str = "warning"


class Project(BaseModel):
    """Project model / 项目模型"""
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
    """Dashboard metrics model / 仪表板指标模型"""
    total_projects: int
    total_analyses: int
    issues_found: int
    issues_resolved: int
    resolution_rate: float


class Activity(BaseModel):
    """Activity model / 活动模型"""
    id: str
    type: str
    message: str
    project_id: Optional[str] = None
    created_at: datetime


class AnalyzeRequest(BaseModel):
    """Analysis request model / 分析请求模型"""
    files: Optional[List[str]] = None
    full_analysis: bool = True


class User(BaseModel):
    """User model / 用户模型"""
    id: str
    email: str
    name: str
    role: str = "user"
    status: str = "active"
    created_at: datetime
    last_login: Optional[datetime] = None


class Experiment(BaseModel):
    """Experiment model / 实验模型"""
    id: str
    name: str
    status: str
    model: str
    accuracy: float
    created_at: datetime


class AuditLog(BaseModel):
    """Audit log model / 审计日志模型"""
    id: str
    user: str
    action: str
    resource: str
    timestamp: datetime
    ip: Optional[str] = None


class Provider(BaseModel):
    """AI Provider model / AI 提供商模型"""
    id: str
    name: str
    type: str
    status: str
    api_key_configured: bool = False
    created_at: datetime


class Invitation(BaseModel):
    """Invitation model / 邀请模型"""
    id: str
    email: str
    role: str
    status: str
    created_at: datetime
    expires_at: datetime


class ApiKey(BaseModel):
    """API Key model / API 密钥模型"""
    id: str
    name: str
    key_preview: str
    scopes: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
