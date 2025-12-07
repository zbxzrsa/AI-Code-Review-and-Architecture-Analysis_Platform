"""
共享配置模块 (Shared Configuration Module)

模块功能描述:
    提供集中式配置管理。

主要功能:
    - 环境变量支持
    - 配置验证逻辑
    - 多环境部署支持

主要组件:
    - Config: 配置主类
    - Environment: 环境枚举
    - get_config(): 获取配置
    - reload_config(): 重新加载配置

辅助函数:
    - get_env(): 获取环境变量
    - get_env_bool(): 获取布尔环境变量
    - get_env_int(): 获取整数环境变量
    - get_env_list(): 获取列表环境变量

最后修改日期: 2024-12-07
"""
from .config_manager import (
    Config,
    config,
    get_config,
    reload_config,
    ConfigError,
    MissingConfigError,
    InvalidConfigError,
    Environment,
    get_env,
    get_env_bool,
    get_env_int,
    get_env_list,
)
