"""
AI子系统模块 (AI Subsystem)

为每个版本独立配备Version Control AI和User Code AI
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncGenerator

from .version_config import VersionType, VersionConfig
from services.shared.ai_models.version_control_ai import VersionControlAI as BaseVersionControlAI
from services.shared.ai_models.code_ai import CodeAI as BaseCodeAI
from services.shared.ai_models.base_ai import AIConfig, ModelProvider

logger = logging.getLogger(__name__)


class AIType(str, Enum):
    """AI类型"""
    VERSION_CONTROL = "version_control"  # 版本控制AI
    USER_CODE = "user_code"              # 用户代码AI


@dataclass
class AICapabilities:
    """AI能力定义"""
    # Version Control AI 能力
    can_update_tech: bool = True          # 技术更新
    can_fix_errors: bool = True           # 错误修复
    can_optimize_compatibility: bool = True  # 兼容性优化
    can_add_features: bool = True         # 添加功能
    
    # User Code AI 能力
    can_review_code: bool = True         # 代码审查
    can_diagnose_errors: bool = True      # 错误诊断
    can_suggest_fixes: bool = True        # 修正建议
    can_provide_tutorials: bool = True    # 修复教程


@dataclass
class AISubsystemConfig:
    """AI子系统配置"""
    version_type: VersionType
    version_control_ai_config: AIConfig
    user_code_ai_config: AIConfig
    capabilities: AICapabilities = field(default_factory=AICapabilities)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class VersionControlAI:
    """
    版本控制AI (Version Control AI)
    
    负责：
    - 技术更新
    - 错误修复
    - 兼容性优化
    - 添加更多功能
    """
    
    def __init__(
        self,
        config: AIConfig,
        version_type: VersionType,
        capabilities: AICapabilities
    ):
        self.config = config
        self.version_type = version_type
        self.capabilities = capabilities
        self.base_ai = BaseVersionControlAI(config, version_type)
        self.operation_history: List[Dict[str, Any]] = []
    
    async def update_technology(
        self,
        tech_name: str,
        tech_config: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """
        更新技术
        
        Args:
            tech_name: 技术名称
            tech_config: 技术配置
            reason: 更新原因
            
        Returns:
            更新结果
        """
        if not self.capabilities.can_update_tech:
            raise PermissionError("此AI不具备技术更新能力")
        
        logger.info(f"[{self.version_type.value}] 更新技术: {tech_name}")
        
        result = {
            "tech_name": tech_name,
            "version": self.version_type.value,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "status": "pending",
            "config": tech_config,
        }
        
        # 记录操作历史
        self.operation_history.append({
            "operation": "update_technology",
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })
        
        return result
    
    async def fix_error(
        self,
        error_id: str,
        error_details: Dict[str, Any],
        priority: str = "medium"
    ) -> Dict[str, Any]:
        """
        修复错误
        
        Args:
            error_id: 错误ID
            error_details: 错误详情
            priority: 优先级
            
        Returns:
            修复结果
        """
        if not self.capabilities.can_fix_errors:
            raise PermissionError("此AI不具备错误修复能力")
        
        logger.info(f"[{self.version_type.value}] 修复错误: {error_id}")
        
        result = {
            "error_id": error_id,
            "version": self.version_type.value,
            "timestamp": datetime.now().isoformat(),
            "priority": priority,
            "status": "fixing",
            "fix_details": {},
        }
        
        self.operation_history.append({
            "operation": "fix_error",
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })
        
        return result
    
    async def optimize_compatibility(
        self,
        source_version: VersionType,
        target_version: VersionType,
        compatibility_issues: List[str]
    ) -> Dict[str, Any]:
        """
        优化兼容性
        
        Args:
            source_version: 源版本
            target_version: 目标版本
            compatibility_issues: 兼容性问题列表
            
        Returns:
            优化结果
        """
        if not self.capabilities.can_optimize_compatibility:
            raise PermissionError("此AI不具备兼容性优化能力")
        
        logger.info(
            f"[{self.version_type.value}] 优化兼容性: "
            f"{source_version.value} -> {target_version.value}"
        )
        
        result = {
            "source_version": source_version.value,
            "target_version": target_version.value,
            "timestamp": datetime.now().isoformat(),
            "issues": compatibility_issues,
            "optimizations": [],
            "status": "optimizing",
        }
        
        self.operation_history.append({
            "operation": "optimize_compatibility",
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })
        
        return result
    
    async def add_feature(
        self,
        feature_name: str,
        feature_spec: Dict[str, Any],
        target_version: Optional[VersionType] = None
    ) -> Dict[str, Any]:
        """
        添加功能
        
        Args:
            feature_name: 功能名称
            feature_spec: 功能规格
            target_version: 目标版本（None表示当前版本）
            
        Returns:
            添加结果
        """
        if not self.capabilities.can_add_features:
            raise PermissionError("此AI不具备添加功能能力")
        
        target = target_version or self.version_type
        
        logger.info(f"[{self.version_type.value}] 添加功能: {feature_name} -> {target.value}")
        
        result = {
            "feature_name": feature_name,
            "source_version": self.version_type.value,
            "target_version": target.value,
            "timestamp": datetime.now().isoformat(),
            "spec": feature_spec,
            "status": "adding",
        }
        
        self.operation_history.append({
            "operation": "add_feature",
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })
        
        return result
    
    async def get_status(self) -> Dict[str, Any]:
        """获取AI状态"""
        return {
            "version": self.version_type.value,
            "ai_type": "version_control",
            "capabilities": {
                "can_update_tech": self.capabilities.can_update_tech,
                "can_fix_errors": self.capabilities.can_fix_errors,
                "can_optimize_compatibility": self.capabilities.can_optimize_compatibility,
                "can_add_features": self.capabilities.can_add_features,
            },
            "operation_count": len(self.operation_history),
            "recent_operations": self.operation_history[-5:] if self.operation_history else [],
        }


class UserCodeAI:
    """
    用户代码AI (User Code AI)
    
    提供：
    - 实时代码审查
    - 错误诊断
    - 修正建议
    - 修复教程
    """
    
    def __init__(
        self,
        config: AIConfig,
        version_type: VersionType,
        capabilities: AICapabilities
    ):
        self.config = config
        self.version_type = version_type
        self.capabilities = capabilities
        self.base_ai = BaseCodeAI(config, version_type)
        self.review_history: List[Dict[str, Any]] = []
    
    async def review_code(
        self,
        code: str,
        language: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        实时代码审查
        
        Args:
            code: 代码内容
            language: 编程语言
            context: 上下文信息
            
        Returns:
            审查结果
        """
        if not self.capabilities.can_review_code:
            raise PermissionError("此AI不具备代码审查能力")
        
        logger.info(f"[{self.version_type.value}] 代码审查: {language}")
        
        # 调用基础AI进行审查
        result = await self.base_ai.review_code(code, language, context)
        
        # 添加版本信息
        result["version"] = self.version_type.value
        result["timestamp"] = datetime.now().isoformat()
        
        self.review_history.append(result)
        
        return result
    
    async def diagnose_error(
        self,
        error_message: str,
        code_snippet: str,
        language: str
    ) -> Dict[str, Any]:
        """
        错误诊断
        
        Args:
            error_message: 错误消息
            code_snippet: 代码片段
            language: 编程语言
            
        Returns:
            诊断结果
        """
        if not self.capabilities.can_diagnose_errors:
            raise PermissionError("此AI不具备错误诊断能力")
        
        logger.info(f"[{self.version_type.value}] 错误诊断")
        
        result = {
            "version": self.version_type.value,
            "timestamp": datetime.now().isoformat(),
            "error_message": error_message,
            "diagnosis": {
                "root_cause": "分析中...",
                "error_type": "unknown",
                "severity": "medium",
                "affected_lines": [],
            },
            "suggestions": [],
        }
        
        self.review_history.append({
            "type": "error_diagnosis",
            "result": result,
        })
        
        return result
    
    async def suggest_fixes(
        self,
        issue_id: str,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """
        修正建议
        
        Args:
            issue_id: 问题ID
            code: 代码内容
            language: 编程语言
            
        Returns:
            修正建议
        """
        if not self.capabilities.can_suggest_fixes:
            raise PermissionError("此AI不具备修正建议能力")
        
        logger.info(f"[{self.version_type.value}] 提供修正建议: {issue_id}")
        
        result = {
            "version": self.version_type.value,
            "timestamp": datetime.now().isoformat(),
            "issue_id": issue_id,
            "suggestions": [
                {
                    "description": "建议1",
                    "code_change": "代码修改示例",
                    "explanation": "说明",
                    "priority": "high",
                }
            ],
        }
        
        self.review_history.append({
            "type": "fix_suggestions",
            "result": result,
        })
        
        return result
    
    async def provide_tutorial(
        self,
        topic: str,
        difficulty: str = "intermediate"
    ) -> Dict[str, Any]:
        """
        修复教程
        
        Args:
            topic: 主题
            difficulty: 难度级别
            
        Returns:
            教程内容
        """
        if not self.capabilities.can_provide_tutorials:
            raise PermissionError("此AI不具备教程提供能力")
        
        logger.info(f"[{self.version_type.value}] 提供教程: {topic}")
        
        result = {
            "version": self.version_type.value,
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "difficulty": difficulty,
            "tutorial": {
                "sections": [],
                "examples": [],
                "exercises": [],
            },
        }
        
        self.review_history.append({
            "type": "tutorial",
            "result": result,
        })
        
        return result
    
    async def get_status(self) -> Dict[str, Any]:
        """获取AI状态"""
        return {
            "version": self.version_type.value,
            "ai_type": "user_code",
            "capabilities": {
                "can_review_code": self.capabilities.can_review_code,
                "can_diagnose_errors": self.capabilities.can_diagnose_errors,
                "can_suggest_fixes": self.capabilities.can_suggest_fixes,
                "can_provide_tutorials": self.capabilities.can_provide_tutorials,
            },
            "review_count": len(self.review_history),
            "recent_reviews": self.review_history[-5:] if self.review_history else [],
        }


@dataclass
class AISubsystem:
    """
    AI子系统
    
    包含Version Control AI和User Code AI
    """
    version_type: VersionType
    version_control_ai: VersionControlAI
    user_code_ai: UserCodeAI
    config: AISubsystemConfig
    enabled: bool = True
    
    async def initialize(self) -> bool:
        """初始化AI子系统"""
        if not self.enabled:
            logger.warning(f"[{self.version_type.value}] AI子系统已禁用")
            return False
        
        logger.info(f"[{self.version_type.value}] 初始化AI子系统")
        return True
    
    async def shutdown(self) -> bool:
        """关闭AI子系统"""
        logger.info(f"[{self.version_type.value}] 关闭AI子系统")
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """获取子系统状态"""
        vc_status = await self.version_control_ai.get_status()
        uc_status = await self.user_code_ai.get_status()
        
        return {
            "version": self.version_type.value,
            "enabled": self.enabled,
            "version_control_ai": vc_status,
            "user_code_ai": uc_status,
        }


def create_ai_subsystem(
    version_type: VersionType,
    vc_ai_config: Optional[AIConfig] = None,
    uc_ai_config: Optional[AIConfig] = None,
    capabilities: Optional[AICapabilities] = None
) -> AISubsystem:
    """
    创建AI子系统
    
    Args:
        version_type: 版本类型
        vc_ai_config: Version Control AI配置
        uc_ai_config: User Code AI配置
        capabilities: AI能力配置
        
    Returns:
        AISubsystem: 配置好的AI子系统
    """
    # 默认配置
    if vc_ai_config is None:
        vc_ai_config = AIConfig(
            model_id=f"vc-ai-{version_type.value}",
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.3,  # 较低温度保证稳定性
        )
    
    if uc_ai_config is None:
        uc_ai_config = AIConfig(
            model_id=f"uc-ai-{version_type.value}",
            provider=ModelProvider.OPENAI,
            model_name="gpt-4",
            temperature=0.5,
        )
    
    if capabilities is None:
        capabilities = AICapabilities()
    
    # 根据版本类型调整能力
    if version_type == VersionType.V2_STABLE:
        # V2稳定版：所有能力都启用，但更保守
        capabilities.can_update_tech = True
        capabilities.can_fix_errors = True
        capabilities.can_optimize_compatibility = True
        capabilities.can_add_features = True
    elif version_type == VersionType.V1_DEVELOPMENT:
        # V1开发版：允许实验性功能
        capabilities.can_update_tech = True
        capabilities.can_fix_errors = True
        capabilities.can_optimize_compatibility = True
        capabilities.can_add_features = True
    elif version_type == VersionType.V3_BENCHMARK:
        # V3基准版：主要用于对比，能力受限
        capabilities.can_update_tech = False
        capabilities.can_fix_errors = False
        capabilities.can_optimize_compatibility = True
        capabilities.can_add_features = False
    
    # 创建AI实例
    vc_ai = VersionControlAI(vc_ai_config, version_type, capabilities)
    uc_ai = UserCodeAI(uc_ai_config, version_type, capabilities)
    
    # 创建子系统配置
    subsystem_config = AISubsystemConfig(
        version_type=version_type,
        version_control_ai_config=vc_ai_config,
        user_code_ai_config=uc_ai_config,
        capabilities=capabilities,
    )
    
    return AISubsystem(
        version_type=version_type,
        version_control_ai=vc_ai,
        user_code_ai=uc_ai,
        config=subsystem_config,
        enabled=True,
    )

