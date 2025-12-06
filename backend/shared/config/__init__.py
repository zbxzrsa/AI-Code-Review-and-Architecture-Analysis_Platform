"""
Shared Configuration Module (TD-002)

Provides centralized configuration management with:
- Environment variable support
- Validation logic
- Multi-environment deployment support
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
