"""
Project Service / 项目服务

Business logic for project management operations.
项目管理操作的业务逻辑。
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from ..models import Project, ProjectSettings


class ProjectService:
    """Service for project management / 项目管理服务"""
    
    def __init__(self):
        self._projects: Dict[str, Project] = {}
        self._next_id = 1
    
    def create_project(
        self,
        name: str,
        language: str,
        description: Optional[str] = None,
        framework: Optional[str] = None,
        repository_url: Optional[str] = None
    ) -> Project:
        """Create a new project / 创建新项目"""
        project_id = f"proj_{self._next_id}"
        self._next_id += 1
        
        project = Project(
            id=project_id,
            name=name,
            description=description,
            language=language,
            framework=framework,
            repository_url=repository_url,
            status="active",
            issues_count=0,
            settings=ProjectSettings(),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self._projects[project_id] = project
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID / 根据 ID 获取项目"""
        return self._projects.get(project_id)
    
    def list_projects(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        limit: int = 10
    ) -> Dict[str, Any]:
        """List projects with filtering / 列出项目（带过滤）"""
        projects = list(self._projects.values())
        
        # Apply filters
        if status:
            projects = [p for p in projects if p.status == status]
        
        if search:
            search_lower = search.lower()
            projects = [
                p for p in projects
                if search_lower in p.name.lower()
                or (p.description and search_lower in p.description.lower())
            ]
        
        # Pagination
        total = len(projects)
        start = (page - 1) * limit
        end = start + limit
        
        return {
            "items": projects[start:end],
            "total": total,
            "page": page,
            "limit": limit,
            "pages": (total + limit - 1) // limit if total > 0 else 0
        }
    
    def update_project(
        self,
        project_id: str,
        **updates
    ) -> Optional[Project]:
        """Update project / 更新项目"""
        project = self._projects.get(project_id)
        if not project:
            return None
        
        # Update allowed fields
        allowed_fields = {"name", "description", "framework", "repository_url", "status"}
        for field, value in updates.items():
            if field in allowed_fields and value is not None:
                setattr(project, field, value)
        
        project.updated_at = datetime.now()
        return project
    
    def delete_project(self, project_id: str) -> bool:
        """Delete project / 删除项目"""
        if project_id in self._projects:
            del self._projects[project_id]
            return True
        return False
    
    def update_settings(
        self,
        project_id: str,
        settings: ProjectSettings
    ) -> Optional[Project]:
        """Update project settings / 更新项目设置"""
        project = self._projects.get(project_id)
        if not project:
            return None
        
        project.settings = settings
        project.updated_at = datetime.now()
        return project
    
    def increment_issues(self, project_id: str, count: int = 1) -> bool:
        """Increment project issues count / 增加项目问题计数"""
        project = self._projects.get(project_id)
        if project:
            project.issues_count += count
            project.updated_at = datetime.now()
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get project statistics / 获取项目统计"""
        projects = list(self._projects.values())
        return {
            "total": len(projects),
            "active": sum(1 for p in projects if p.status == "active"),
            "archived": sum(1 for p in projects if p.status == "archived"),
            "total_issues": sum(p.issues_count for p in projects),
            "by_language": self._count_by_field(projects, "language"),
            "by_framework": self._count_by_field(projects, "framework")
        }
    
    def _count_by_field(
        self,
        projects: List[Project],
        field: str
    ) -> Dict[str, int]:
        """Count projects by field value / 按字段值统计项目"""
        counts: Dict[str, int] = {}
        for project in projects:
            value = getattr(project, field, None) or "Unknown"
            counts[value] = counts.get(value, 0) + 1
        return counts
