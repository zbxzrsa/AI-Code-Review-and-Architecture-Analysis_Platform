"""
版本配置模块 (Version Configuration Module)

定义三版本架构的配置和隔离机制。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import timedelta


class VersionType(str, Enum):
    """版本类型枚举"""
    V1_DEVELOPMENT = "v1"  # 开发版：实验性技术集成
    V2_STABLE = "v2"       # 稳定版：100%稳定可靠的生产版本
    V3_BENCHMARK = "v3"    # 基准版：技术对比参考


class VersionRole(str, Enum):
    """版本角色"""
    EXPERIMENTAL = "experimental"  # 实验性
    PRODUCTION = "production"      # 生产环境
    BENCHMARK = "benchmark"        # 基准对比


@dataclass
class SandboxConfig:
    """
    沙箱环境配置
    
    为V1开发版提供完整的沙箱环境，确保实验性技术不会影响生产环境。
    """
    enabled: bool = True
    isolation_level: str = "strict"  # strict, moderate, loose
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "cpu": "2",
        "memory": "4Gi",
        "storage": "20Gi",
        "network_bandwidth": "100Mbps"
    })
    timeout_seconds: int = 300
    max_concurrent_experiments: int = 10
    auto_cleanup: bool = True
    cleanup_delay_seconds: int = 3600


@dataclass
class ErrorIsolationConfig:
    """
    错误隔离配置
    
    确保V1开发版的错误不会传播到V2稳定版。
    """
    enabled: bool = True
    error_containment: bool = True
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5  # 连续错误次数
    circuit_breaker_timeout_seconds: int = 60
    error_logging_level: str = "detailed"
    error_notification_enabled: bool = True
    auto_rollback_on_critical_error: bool = True


@dataclass
class VersionIsolationConfig:
    """
    版本隔离配置
    
    确保三个版本之间的完全隔离，同时允许必要的通信。
    """
    network_isolation: bool = True
    data_isolation: bool = True
    resource_isolation: bool = True
    security_channel_enabled: bool = True
    allowed_communication_channels: List[str] = field(default_factory=lambda: [
        "tech_comparison",
        "diagnosis_coordination",
        "promotion_pipeline"
    ])


@dataclass
class VersionConfig:
    """
    版本配置
    
    定义每个版本的完整配置信息。
    """
    version_type: VersionType
    role: VersionRole
    name: str
    description: str
    
    # 稳定性要求
    stability_requirement: float = 1.0  # V2必须为1.0（100%稳定）
    reliability_target: float = 0.999  # 可靠性目标（99.9%）
    
    # 沙箱和隔离配置（V1专用）
    sandbox: Optional[SandboxConfig] = None
    error_isolation: Optional[ErrorIsolationConfig] = None
    
    # 版本隔离配置
    isolation: VersionIsolationConfig = field(default_factory=VersionIsolationConfig)
    
    # 性能要求
    min_performance_improvement_pct: float = 15.0  # 性能提升阈值（≥15%）
    
    # 监控配置
    monitoring_enabled: bool = True
    metrics_collection_interval_seconds: int = 60
    health_check_interval_seconds: int = 30
    
    # 回滚配置
    rollback_enabled: bool = True
    rollback_timeout_seconds: int = 60  # 分钟级回滚（60秒）
    
    # API兼容性
    api_compatibility_required: bool = True
    backward_compatible: bool = True
    
    # 资源分配
    resource_priority: int = 5  # 1-10，数字越大优先级越高
    min_resources: Dict[str, Any] = field(default_factory=lambda: {
        "cpu": "1",
        "memory": "2Gi"
    })
    max_resources: Dict[str, Any] = field(default_factory=lambda: {
        "cpu": "8",
        "memory": "16Gi"
    })
    
    def __post_init__(self):
        """初始化后处理"""
        # V1开发版必须配置沙箱和错误隔离
        if self.version_type == VersionType.V1_DEVELOPMENT:
            if self.sandbox is None:
                self.sandbox = SandboxConfig()
            if self.error_isolation is None:
                self.error_isolation = ErrorIsolationConfig()
        
        # V2稳定版必须100%稳定
        if self.version_type == VersionType.V2_STABLE:
            self.stability_requirement = 1.0
            self.reliability_target = 0.999
        
        # V3基准版需要保留历史数据
        if self.version_type == VersionType.V3_BENCHMARK:
            self.isolation.data_isolation = False  # 允许访问历史数据


def create_v1_config() -> VersionConfig:
    """创建V1开发版配置"""
    return VersionConfig(
        version_type=VersionType.V1_DEVELOPMENT,
        role=VersionRole.EXPERIMENTAL,
        name="V1 Development Version",
        description="实验性技术集成版本，用于测试新技术和功能",
        stability_requirement=0.7,  # 允许实验性错误
        reliability_target=0.85,
        sandbox=SandboxConfig(),
        error_isolation=ErrorIsolationConfig(),
        resource_priority=3,  # 中等优先级
    )


def create_v2_config() -> VersionConfig:
    """创建V2稳定版配置"""
    return VersionConfig(
        version_type=VersionType.V2_STABLE,
        role=VersionRole.PRODUCTION,
        name="V2 Stable Version",
        description="100%稳定可靠的生产版本，面向所有用户",
        stability_requirement=1.0,  # 100%稳定
        reliability_target=0.999,
        resource_priority=10,  # 最高优先级
    )


def create_v3_config() -> VersionConfig:
    """创建V3基准版配置"""
    return VersionConfig(
        version_type=VersionType.V3_BENCHMARK,
        role=VersionRole.BENCHMARK,
        name="V3 Benchmark Version",
        description="技术对比基准版本，保留完整历史数据和性能参数",
        stability_requirement=0.8,
        reliability_target=0.9,
        resource_priority=2,  # 低优先级，作为备份计算资源
    )

