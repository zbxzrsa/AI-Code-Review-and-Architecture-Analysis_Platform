"""
Pydantic 数据模型 (Pydantic Models)

模块功能描述:
    开发 API 的所有请求/响应数据模型。

模型分类:
    - Project: 项目相关模型
    - Dashboard: 仪表板模型
    - User: 用户模型
    - Analysis: 分析结果模型
    - Security: 安全相关模型

最后修改日期: 2024-12-07
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# ============================================
# Project Models
# ============================================

class ProjectSettings(BaseModel):
    auto_review: bool = False
    review_on_push: bool = False
    review_on_pr: bool = True
    severity_threshold: str = "warning"
    enabled_rules: List[str] = []
    ignored_paths: List[str] = ["node_modules", ".git", "__pycache__", "dist", "build"]


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


class CreateProjectRequest(BaseModel):
    name: str
    language: str
    description: Optional[str] = None
    framework: Optional[str] = None
    repository_url: Optional[str] = None
    settings: Optional[dict] = None


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    framework: Optional[str] = None
    status: Optional[str] = None
    settings: Optional[dict] = None


# ============================================
# Dashboard Models
# ============================================

class DashboardMetrics(BaseModel):
    total_projects: int
    total_analyses: int
    issues_found: int
    issues_resolved: int
    resolution_rate: float


# ============================================
# Activity Models
# ============================================

class Activity(BaseModel):
    id: str
    type: str
    message: str
    project_id: Optional[str] = None
    created_at: datetime


# ============================================
# Analysis Models
# ============================================

class AnalyzeRequest(BaseModel):
    files: Optional[List[str]] = None
    full_scan: bool = False


class CodeAnalyzeRequest(BaseModel):
    code: str
    language: str = "python"
    rules: Optional[List[str]] = None


# ============================================
# Authentication Models
# ============================================

class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    invitation_code: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


# ============================================
# User Models
# ============================================

class UserProfile(BaseModel):
    id: str
    email: str
    name: str
    avatar: Optional[str] = None
    role: str = "user"
    created_at: datetime
    settings: Dict[str, Any] = {}


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    avatar: Optional[str] = None
    bio: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ============================================
# Admin Models
# ============================================

class CreateUserRequest(BaseModel):
    email: str
    name: str
    role: str = "user"


class UpdateUserRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None


class ProviderConfig(BaseModel):
    api_key: Optional[str] = None
    enabled: bool = True
    priority: int = 1
    rate_limit: int = 100


# ============================================
# Three-Version Models
# ============================================

class PromoteRequest(BaseModel):
    version: str
    reason: Optional[str] = None


class DemoteRequest(BaseModel):
    version: str
    reason: str


class ReEvaluateRequest(BaseModel):
    version: str


# ============================================
# Security Models
# ============================================

class VulnerabilityFilter(BaseModel):
    severity: Optional[str] = None
    status: Optional[str] = None
    type: Optional[str] = None


# ============================================
# Report Models
# ============================================

class GenerateReportRequest(BaseModel):
    type: str
    format: str = "pdf"
    date_range: Optional[Dict[str, str]] = None
    include_details: bool = True


class ScheduleReportRequest(BaseModel):
    type: str
    format: str = "pdf"
    schedule: str  # cron expression
    recipients: List[str]
