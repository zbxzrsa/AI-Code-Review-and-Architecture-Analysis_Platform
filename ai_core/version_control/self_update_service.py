"""
Project Self-Update Service

As a service layer, integrates the project self-update engine and enhanced
version control AI to provide complete project self-update capabilities.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .project_self_update_engine import (
    ProjectSelfUpdateEngine,
    ImprovementCategory,
    ImprovementPriority,
)
from .enhanced_version_control_ai import (
    EnhancedVersionControlAI,
    VersionControlAIConfig,
)

logger = logging.getLogger(__name__)


class ProjectSelfUpdateService:
    """
    Project Self-Update Service
    
    Provides a unified interface to start and manage project self-update cycles.
    """
    
    def __init__(
        self,
        project_root: str,
        version_manager: Optional[Any] = None,
        code_analysis_engine: Optional[Any] = None,
        ai_model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize service
        
        Args:
            project_root: Project root directory
            version_manager: Three-version manager
            code_analysis_engine: Code analysis engine
            ai_model: AI model
            config: Configuration dictionary
        """
        self.project_root = Path(project_root)
        
        # 构建配置
        ai_config = VersionControlAIConfig(
            project_root=str(self.project_root),
            scan_interval_hours=config.get("scan_interval_hours", 24) if config else 24,
            auto_improve=config.get("auto_improve", False) if config else False,
            create_pr=config.get("create_pr", True) if config else True,
            max_patches_per_cycle=config.get("max_patches_per_cycle", 50) if config else 50,
            priority_filter=config.get("priority_filter", [
                ImprovementPriority.CRITICAL,
                ImprovementPriority.HIGH,
            ]) if config else [ImprovementPriority.CRITICAL, ImprovementPriority.HIGH],
            integration_with_v1=config.get("integration_with_v1", True) if config else True,
            integration_with_v2=config.get("integration_with_v2", True) if config else True,
            integration_with_v3=config.get("integration_with_v3", True) if config else True,
        )
        
        # 初始化增强版版本控制AI
        self.vc_ai = EnhancedVersionControlAI(
            config=ai_config,
            version_manager=version_manager,
            code_analysis_engine=code_analysis_engine,
            ai_model=ai_model,
        )
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info(f"项目自更新服务初始化: {project_root}")
    
    async def start(self):
        """启动持续改进循环"""
        if self._running:
            logger.warning("服务已在运行")
            return
        
        self._running = True
        self._task = asyncio.create_task(self.vc_ai.start_continuous_improvement())
        logger.info("项目自更新服务已启动")
    
    async def stop(self):
        """停止服务"""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("项目自更新服务已停止")
    
    async def run_once(self) -> Dict[str, Any]:
        """运行一次改进周期"""
        cycle = await self.vc_ai.run_improvement_cycle()
        return {
            "cycle_id": cycle.cycle_id,
            "status": cycle.status,
            "patches_generated": cycle.patches_generated,
            "patches_applied": cycle.patches_applied,
            "patches_verified": cycle.patches_verified,
        }
    
    async def scan_project(self) -> Dict[str, Any]:
        """扫描项目"""
        scan_result = await self.vc_ai.manual_trigger_scan()
        return {
            "scan_id": scan_result.scan_id,
            "total_files": scan_result.total_files_scanned,
            "total_issues": len(scan_result.issues_found),
            "critical_issues": len([
                i for i in scan_result.issues_found
                if i.priority == ImprovementPriority.CRITICAL
            ]),
            "issues_by_category": scan_result.issues_by_category,
            "issues_by_priority": scan_result.issues_by_priority,
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        health_report = await self.vc_ai.get_project_health_report()
        return {
            "running": self._running,
            "project_root": str(self.project_root),
            "health": health_report,
        }
    
    @property
    def is_running(self) -> bool:
        """检查服务是否运行中"""
        return self._running


# 便捷函数
async def create_self_update_service(
    project_root: str,
    version_manager: Optional[Any] = None,
    code_analysis_engine: Optional[Any] = None,
    ai_model: Optional[Any] = None,
    **config_kwargs,
) -> ProjectSelfUpdateService:
    """
    创建项目自更新服务
    
    Args:
        project_root: 项目根目录
        version_manager: 三版本管理器
        code_analysis_engine: 代码分析引擎
        ai_model: AI模型
        **config_kwargs: 配置参数
        
    Returns:
        项目自更新服务实例
    """
    service = ProjectSelfUpdateService(
        project_root=project_root,
        version_manager=version_manager,
        code_analysis_engine=code_analysis_engine,
        ai_model=ai_model,
        config=config_kwargs,
    )
    return service

