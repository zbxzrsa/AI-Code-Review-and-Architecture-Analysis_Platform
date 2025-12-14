"""
版本架构主入口 (Version Architecture Main Entry)

整合所有模块，提供统一的接口
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from .version_config import (
    VersionArchitecture,
    create_version_config,
    VersionType,
)
from .ai_subsystem import (
    AISubsystem,
    create_ai_subsystem,
)
from .secure_communication import (
    SecureChannel,
    create_secure_channel,
)
from .version_collaboration import (
    VersionCollaboration,
    create_collaboration_system,
)
from .tech_update_standards import (
    TechUpdateValidator,
    create_validator,
    UpdateCriteria,
)
from .monitoring_rollback import (
    VersionMonitor,
    RollbackManager,
    create_monitoring_system,
)
from .documentation_system import (
    TechUpdateLogger,
    EvaluationLogger,
    VersionSwitchManual,
    create_documentation_system,
)
from .api_compatibility import (
    APIVersionManager,
    create_api_manager,
)

logger = logging.getLogger(__name__)


class VersionArchitectureSystem:
    """
    版本架构系统
    
    整合所有模块，提供完整的版本架构管理
    """
    
    def __init__(self):
        # 版本配置
        self.architecture: Optional[VersionArchitecture] = None
        
        # AI子系统
        self.ai_subsystems: Dict[str, AISubsystem] = {}
        
        # 安全通信
        self.secure_channel: Optional[SecureChannel] = None
        
        # 版本协作
        self.collaboration: Optional[VersionCollaboration] = None
        
        # 技术更新验证
        self.validator: Optional[TechUpdateValidator] = None
        
        # 监控和回滚
        self.monitor: Optional[VersionMonitor] = None
        self.rollback_manager: Optional[RollbackManager] = None
        
        # 文档系统
        self.update_logger: Optional[TechUpdateLogger] = None
        self.evaluation_logger: Optional[EvaluationLogger] = None
        self.switch_manual: Optional[VersionSwitchManual] = None
        
        # API兼容性
        self.api_manager: Optional[APIVersionManager] = None
    
    async def initialize(self) -> bool:
        """初始化系统"""
        try:
            logger.info("初始化版本架构系统...")
            
            # 1. 创建版本配置
            self.architecture = create_version_config()
            logger.info("版本配置已创建")
            
            # 2. 创建AI子系统
            for version_type in [VersionType.V1_DEVELOPMENT, VersionType.V2_STABLE, VersionType.V3_BENCHMARK]:
                subsystem = create_ai_subsystem(version_type)
                await subsystem.initialize()
                self.ai_subsystems[version_type.value] = subsystem
            logger.info("AI子系统已创建")
            
            # 3. 创建安全通信通道
            self.secure_channel = create_secure_channel()
            logger.info("安全通信通道已创建")
            
            # 4. 创建版本协作系统
            self.collaboration = create_collaboration_system(self.secure_channel)
            await self.collaboration.initialize()
            logger.info("版本协作系统已创建")
            
            # 5. 创建技术更新验证器
            criteria = UpdateCriteria(
                min_development_cycles=3,
                min_performance_improvement_pct=15.0,
                require_all_issues_resolved=True,
                require_triple_ai_verification=True,
                require_stress_test=True,
                require_user_scenario_simulation=True,
            )
            self.validator = create_validator(criteria)
            logger.info("技术更新验证器已创建")
            
            # 6. 创建监控和回滚系统
            self.monitor, self.rollback_manager = create_monitoring_system()
            logger.info("监控和回滚系统已创建")
            
            # 7. 创建文档系统
            self.update_logger, self.evaluation_logger, self.switch_manual = create_documentation_system()
            logger.info("文档系统已创建")
            
            # 8. 创建API管理器
            self.api_manager = create_api_manager()
            logger.info("API管理器已创建")
            
            # 9. 确保v2有分钟级回滚能力
            self.rollback_manager.ensure_minute_level_rollback("v2")
            logger.info("已为v2创建分钟级回滚计划")
            
            logger.info("版本架构系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            return False
    
    async def shutdown(self) -> bool:
        """关闭系统"""
        try:
            logger.info("关闭版本架构系统...")
            
            # 关闭AI子系统
            for subsystem in self.ai_subsystems.values():
                await subsystem.shutdown()
            
            # 停止监控
            if self.monitor:
                self.monitor.stop_monitoring()
            
            logger.info("版本架构系统已关闭")
            return True
            
        except Exception as e:
            logger.error(f"关闭失败: {e}", exc_info=True)
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "architecture": {
                "v1_state": self.architecture.v1_config.state.value if self.architecture else None,
                "v2_state": self.architecture.v2_config.state.value if self.architecture else None,
                "v3_state": self.architecture.v3_config.state.value if self.architecture else None,
            },
            "ai_subsystems": {
                version: {
                    "enabled": subsystem.enabled,
                    "vc_ai_status": "active",
                    "uc_ai_status": "active",
                }
                for version, subsystem in self.ai_subsystems.items()
            },
            "collaboration": self.collaboration.get_status() if self.collaboration else None,
            "monitoring": {
                "enabled": self.monitor.is_monitoring if self.monitor else False,
            },
            "api_compatibility": self.api_manager.get_compatibility_report() if self.api_manager else None,
        }
        
        return status


# 全局系统实例
_system_instance: Optional[VersionArchitectureSystem] = None


async def get_system() -> VersionArchitectureSystem:
    """获取系统实例（单例）"""
    global _system_instance
    
    if _system_instance is None:
        _system_instance = VersionArchitectureSystem()
        await _system_instance.initialize()
    
    return _system_instance


async def main():
    """主函数（示例）"""
    system = await get_system()
    
    # 获取系统状态
    status = system.get_system_status()
    print("系统状态:", status)
    
    # 保持运行
    try:
        await asyncio.sleep(3600)  # 运行1小时
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
        await system.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

