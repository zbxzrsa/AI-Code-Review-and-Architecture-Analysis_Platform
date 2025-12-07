"""
Configuration Manager (TD-002)

Centralizes all configuration management with:
- Environment variable support
- Validation logic
- Multi-environment deployment support
- Immediate feedback for configuration errors

Usage:
    from backend.shared.config import config
    
    api_key = config.openai.api_key
    db_url = config.database.url
"""
import os
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum
from functools import cached_property
import json

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class ConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


class MissingConfigError(ConfigError):
    """Raised when required configuration is missing."""
    def __init__(self, key: str, description: str = ""):
        self.key = key
        self.description = description
        message = f"Missing required configuration: {key}"
        if description:
            message += f" - {description}"
        super().__init__(message)


class InvalidConfigError(ConfigError):
    """Raised when configuration value is invalid."""
    def __init__(self, key: str, value: Any, reason: str):
        self.key = key
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid configuration for {key}: {reason}")


def get_env(
    key: str,
    default: Optional[str] = None,
    required: bool = False,
    description: str = ""
) -> Optional[str]:
    """
    Get environment variable with validation.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        required: Whether the variable is required
        description: Description for error messages
        
    Returns:
        Environment variable value or default
        
    Raises:
        MissingConfigError: If required and not set
    """
    value = os.environ.get(key, default)
    
    if required and value is None:
        raise MissingConfigError(key, description)
    
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.environ.get(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Get integer environment variable with optional bounds validation."""
    try:
        value = int(os.environ.get(key, str(default)))
        
        if min_val is not None and value < min_val:
            raise InvalidConfigError(key, value, f"Must be >= {min_val}")
        if max_val is not None and value > max_val:
            raise InvalidConfigError(key, value, f"Must be <= {max_val}")
        
        return value
    except ValueError:
        raise InvalidConfigError(key, os.environ.get(key), "Must be an integer")


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        raise InvalidConfigError(key, os.environ.get(key), "Must be a number")


def get_env_list(key: str, default: Optional[List[str]] = None, separator: str = ",") -> List[str]:
    """Get list environment variable (comma-separated by default)."""
    value = os.environ.get(key)
    if value is None:
        return default or []
    return [item.strip() for item in value.split(separator) if item.strip()]


def get_env_json(key: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """Get JSON environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default or {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise InvalidConfigError(key, value[:50] + "...", f"Invalid JSON: {e}")


@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    @property
    def host(self) -> str:
        return get_env("DB_HOST", "localhost")
    
    @property
    def port(self) -> int:
        return get_env_int("DB_PORT", 5432, min_val=1, max_val=65535)
    
    @property
    def name(self) -> str:
        return get_env("DB_NAME", "coderev")
    
    @property
    def user(self) -> str:
        return get_env("DB_USER", "postgres")
    
    @property
    def password(self) -> str:
        return get_env("DB_PASSWORD", "", required=False)
    
    @property
    def url(self) -> str:
        """Get database connection URL."""
        url = get_env("DATABASE_URL")
        if url:
            return url
        
        auth = f"{self.user}:{self.password}@" if self.password else f"{self.user}@"
        return f"postgresql://{auth}{self.host}:{self.port}/{self.name}"
    
    @property
    def pool_size(self) -> int:
        return get_env_int("DB_POOL_SIZE", 10, min_val=1, max_val=100)
    
    @property
    def max_overflow(self) -> int:
        return get_env_int("DB_MAX_OVERFLOW", 20, min_val=0, max_val=100)
    
    @property
    def ssl_mode(self) -> str:
        return get_env("DB_SSL_MODE", "prefer")


@dataclass
class RedisConfig:
    """Redis configuration."""
    
    @property
    def host(self) -> str:
        return get_env("REDIS_HOST", "localhost")
    
    @property
    def port(self) -> int:
        return get_env_int("REDIS_PORT", 6379)
    
    @property
    def password(self) -> Optional[str]:
        return get_env("REDIS_PASSWORD")
    
    @property
    def db(self) -> int:
        return get_env_int("REDIS_DB", 0, min_val=0, max_val=15)
    
    @property
    def url(self) -> str:
        """Get Redis connection URL."""
        url = get_env("REDIS_URL")
        if url:
            return url
        
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"
    
    @property
    def ssl(self) -> bool:
        return get_env_bool("REDIS_SSL", False)


@dataclass
class OpenAIConfig:
    """OpenAI provider configuration."""
    
    @property
    def api_key(self) -> str:
        return get_env("OPENAI_API_KEY", "", 
                      required=False,
                      description="Required for OpenAI-powered features")
    
    @property
    def organization(self) -> Optional[str]:
        return get_env("OPENAI_ORGANIZATION")
    
    @property
    def model(self) -> str:
        return get_env("OPENAI_MODEL", "gpt-4")
    
    @property
    def temperature(self) -> float:
        return get_env_float("OPENAI_TEMPERATURE", 0.7)
    
    @property
    def max_tokens(self) -> int:
        return get_env_int("OPENAI_MAX_TOKENS", 4096)
    
    @property
    def timeout(self) -> int:
        return get_env_int("OPENAI_TIMEOUT", 30)
    
    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass
class AnthropicConfig:
    """Anthropic provider configuration."""
    
    @property
    def api_key(self) -> str:
        return get_env("ANTHROPIC_API_KEY", "")
    
    @property
    def model(self) -> str:
        return get_env("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    
    @property
    def max_tokens(self) -> int:
        return get_env_int("ANTHROPIC_MAX_TOKENS", 4096)
    
    @property
    def timeout(self) -> int:
        return get_env_int("ANTHROPIC_TIMEOUT", 45)
    
    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass
class OllamaConfig:
    """Ollama local provider configuration."""
    
    @property
    def base_url(self) -> str:
        return get_env("OLLAMA_BASE_URL", "http://localhost:11434")
    
    @property
    def model(self) -> str:
        return get_env("OLLAMA_MODEL", "codellama")
    
    @property
    def timeout(self) -> int:
        return get_env_int("OLLAMA_TIMEOUT", 60)
    
    @property
    def enabled(self) -> bool:
        return get_env_bool("OLLAMA_ENABLED", False)


@dataclass 
class AuthConfig:
    """Authentication configuration."""
    
    @property
    def jwt_secret(self) -> str:
        secret = get_env("JWT_SECRET")
        if not secret and self.environment != Environment.TEST:
            raise MissingConfigError(
                "JWT_SECRET",
                "Required for production/staging. Generate with: openssl rand -hex 32"
            )
        return secret or "test-secret-key-do-not-use-in-production"
    
    @property
    def jwt_algorithm(self) -> str:
        return get_env("JWT_ALGORITHM", "HS256")
    
    @property
    def access_token_expire_minutes(self) -> int:
        return get_env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 30, min_val=5, max_val=1440)
    
    @property
    def refresh_token_expire_days(self) -> int:
        return get_env_int("REFRESH_TOKEN_EXPIRE_DAYS", 7, min_val=1, max_val=30)
    
    @property
    def password_min_length(self) -> int:
        return get_env_int("PASSWORD_MIN_LENGTH", 8, min_val=8, max_val=128)
    
    @property
    def session_timeout_minutes(self) -> int:
        return get_env_int("SESSION_TIMEOUT_MINUTES", 60)
    
    @property
    def max_login_attempts(self) -> int:
        return get_env_int("MAX_LOGIN_ATTEMPTS", 5, min_val=3, max_val=10)
    
    @property
    def lockout_duration_minutes(self) -> int:
        return get_env_int("LOCKOUT_DURATION_MINUTES", 15)
    
    @property
    def environment(self) -> Environment:
        return config.environment


@dataclass
class ServerConfig:
    """Server configuration."""
    
    @property
    def host(self) -> str:
        return get_env("SERVER_HOST", "0.0.0.0")
    
    @property
    def port(self) -> int:
        return get_env_int("SERVER_PORT", 8000, min_val=1, max_val=65535)
    
    @property
    def workers(self) -> int:
        return get_env_int("SERVER_WORKERS", 4, min_val=1, max_val=32)
    
    @property
    def debug(self) -> bool:
        return get_env_bool("DEBUG", False)
    
    @property
    def cors_origins(self) -> List[str]:
        default = ["http://localhost:3000", "http://localhost:5173"]
        return get_env_list("CORS_ORIGINS", default)
    
    @property
    def allowed_hosts(self) -> List[str]:
        return get_env_list("ALLOWED_HOSTS", ["*"])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    @property
    def level(self) -> str:
        return get_env("LOG_LEVEL", "INFO").upper()
    
    @property
    def format(self) -> str:
        return get_env("LOG_FORMAT", "json")
    
    @property
    def file_path(self) -> Optional[str]:
        return get_env("LOG_FILE")
    
    @property
    def max_file_size_mb(self) -> int:
        return get_env_int("LOG_MAX_SIZE_MB", 100)
    
    @property
    def backup_count(self) -> int:
        return get_env_int("LOG_BACKUP_COUNT", 5)


@dataclass
class DataSourceConfig:
    """
    Data source API keys configuration.
    
    Manages API keys for external data sources with secure access patterns.
    All keys are loaded from environment variables or secure key management.
    
    Supported sources:
        - GitHub: Repository data, trending, releases
        - HuggingFace: ML models and datasets
        - StackOverflow: Developer Q&A
    
    Security:
        - Keys are never logged or exposed in config dumps
        - Access is permission-controlled via properties
        - Supports key rotation via environment reload
    """
    
    @property
    def github_token(self) -> Optional[str]:
        """Get GitHub API token for authenticated requests."""
        return get_env(
            "GITHUB_TOKEN",
            default=None,
            required=False,
            description="GitHub API token for higher rate limits and private repo access"
        )
    
    @property
    def github_enabled(self) -> bool:
        """Check if GitHub integration is configured."""
        return bool(self.github_token)
    
    @property
    def huggingface_token(self) -> Optional[str]:
        """Get HuggingFace API token."""
        return get_env(
            "HUGGINGFACE_TOKEN",
            default=None,
            required=False,
            description="HuggingFace API token for model/dataset access"
        )
    
    @property
    def huggingface_enabled(self) -> bool:
        """Check if HuggingFace integration is configured."""
        return bool(self.huggingface_token)
    
    @property
    def stackoverflow_key(self) -> Optional[str]:
        """Get StackOverflow API key."""
        return get_env(
            "STACKOVERFLOW_KEY",
            default=None,
            required=False,
            description="StackOverflow API key for higher quotas"
        )
    
    @property
    def arxiv_enabled(self) -> bool:
        """ArXiv doesn't require authentication."""
        return True
    
    @property
    def devto_api_key(self) -> Optional[str]:
        """Get Dev.to API key."""
        return get_env(
            "DEVTO_API_KEY",
            default=None,
            required=False,
            description="Dev.to API key for authenticated requests"
        )
    
    def get_api_key(self, source_name: str) -> Optional[str]:
        """
        Get API key for a specific data source.
        
        Args:
            source_name: Name of the data source (github, huggingface, stackoverflow, devto)
            
        Returns:
            API key if configured, None otherwise
        """
        key_map = {
            "github": self.github_token,
            "huggingface": self.huggingface_token,
            "stackoverflow": self.stackoverflow_key,
            "devto": self.devto_api_key,
        }
        return key_map.get(source_name.lower())
    
    def is_source_configured(self, source_name: str) -> bool:
        """
        Check if a data source has valid configuration.
        
        Args:
            source_name: Name of the data source
            
        Returns:
            True if source is configured and enabled
        """
        # Sources that don't require API keys
        no_auth_sources = {"arxiv", "hackernews", "medium"}
        if source_name.lower() in no_auth_sources:
            return True
        
        return bool(self.get_api_key(source_name))


@dataclass
class FeatureFlagsConfig:
    """Feature flags configuration."""
    
    @property
    def enable_experiments(self) -> bool:
        return get_env_bool("FF_ENABLE_EXPERIMENTS", True)
    
    @property
    def enable_three_version_cycle(self) -> bool:
        return get_env_bool("FF_ENABLE_THREE_VERSION", True)
    
    @property
    def enable_audit_logging(self) -> bool:
        return get_env_bool("FF_ENABLE_AUDIT", True)
    
    @property
    def enable_caching(self) -> bool:
        return get_env_bool("FF_ENABLE_CACHE", True)
    
    @property
    def mock_mode(self) -> bool:
        return get_env_bool("MOCK_MODE", False)


class Config:
    """
    Main configuration class.
    
    Provides access to all configuration sections with validation.
    """
    
    def __init__(self):
        self._validation_errors: List[str] = []
    
    @cached_property
    def environment(self) -> Environment:
        env_str = get_env("ENVIRONMENT", "development").lower()
        try:
            return Environment(env_str)
        except ValueError:
            valid = ", ".join(e.value for e in Environment)
            raise InvalidConfigError("ENVIRONMENT", env_str, f"Must be one of: {valid}")
    
    @cached_property
    def database(self) -> DatabaseConfig:
        return DatabaseConfig()
    
    @cached_property
    def redis(self) -> RedisConfig:
        return RedisConfig()
    
    @cached_property
    def openai(self) -> OpenAIConfig:
        return OpenAIConfig()
    
    @cached_property
    def anthropic(self) -> AnthropicConfig:
        return AnthropicConfig()
    
    @cached_property
    def ollama(self) -> OllamaConfig:
        return OllamaConfig()
    
    @cached_property
    def auth(self) -> AuthConfig:
        return AuthConfig()
    
    @cached_property
    def server(self) -> ServerConfig:
        return ServerConfig()
    
    @cached_property
    def logging(self) -> LoggingConfig:
        return LoggingConfig()
    
    @cached_property
    def features(self) -> FeatureFlagsConfig:
        return FeatureFlagsConfig()
    
    @cached_property
    def data_sources(self) -> DataSourceConfig:
        return DataSourceConfig()
    
    def validate(self) -> List[str]:
        """
        Validate all configuration and return list of errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate database URL
        try:
            _ = self.database.url
        except ConfigError as e:
            errors.append(str(e))
        
        # Validate auth in non-test environments
        if self.environment != Environment.TEST:
            try:
                _ = self.auth.jwt_secret
            except ConfigError as e:
                errors.append(str(e))
        
        # Validate at least one AI provider is configured
        if not any([
            self.openai.enabled,
            self.anthropic.enabled,
            self.ollama.enabled,
            self.features.mock_mode,
        ]):
            errors.append(
                "No AI provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "OLLAMA_ENABLED=true, or MOCK_MODE=true"
            )
        
        return errors
    
    def validate_or_raise(self):
        """Validate configuration and raise if invalid."""
        errors = self.validate()
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigError(error_msg)
    
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    def is_test(self) -> bool:
        return self.environment == Environment.TEST
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Args:
            include_secrets: Whether to include sensitive values
            
        Returns:
            Configuration dictionary
        """
        result = {
            "environment": self.environment.value,
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers,
                "debug": self.server.debug,
            },
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
                "pool_size": self.database.pool_size,
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
            },
            "ai_providers": {
                "openai_enabled": self.openai.enabled,
                "anthropic_enabled": self.anthropic.enabled,
                "ollama_enabled": self.ollama.enabled,
            },
            "features": {
                "experiments": self.features.enable_experiments,
                "three_version_cycle": self.features.enable_three_version_cycle,
                "audit_logging": self.features.enable_audit_logging,
                "caching": self.features.enable_caching,
                "mock_mode": self.features.mock_mode,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
            },
        }
        
        if include_secrets:
            result["secrets"] = {
                "jwt_secret": self.auth.jwt_secret[:8] + "..." if self.auth.jwt_secret else None,
                "db_password": "***" if self.database.password else None,
                "openai_key": self.openai.api_key[:8] + "..." if self.openai.api_key else None,
            }
        
        return result


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration (useful for testing)."""
    global config
    # Clear cached properties
    for attr in list(vars(config).keys()):
        if attr.startswith('_'):
            continue
        try:
            delattr(config, attr)
        except AttributeError:
            pass
    config = Config()
    return config


# =============================================================================
# Hot Reload Support (P2 Enhancement)
# =============================================================================

class ConfigWatcher:
    """
    Configuration hot reload watcher.
    
    Monitors .env file for changes and triggers reload callbacks.
    
    Usage:
        watcher = ConfigWatcher()
        watcher.add_listener(lambda: print("Config reloaded!"))
        await watcher.start()
    """
    
    def __init__(self, env_path: Optional[str] = None):
        self.env_path = env_path or os.getenv("ENV_FILE", ".env")
        self._listeners: List[Callable[[], None]] = []
        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._last_mtime: Optional[float] = None
        self._check_interval = 5.0  # seconds
    
    def add_listener(self, callback: Callable[[], None]) -> None:
        """Add a callback to be called when config changes."""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable[[], None]) -> None:
        """Remove a config change callback."""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    async def start(self) -> None:
        """Start watching for config changes."""
        if self._running:
            return
        
        self._running = True
        self._last_mtime = self._get_mtime()
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info(f"Config watcher started, monitoring: {self.env_path}")
    
    async def stop(self) -> None:
        """Stop watching for config changes."""
        self._running = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        logger.info("Config watcher stopped")
    
    def _get_mtime(self) -> Optional[float]:
        """Get modification time of env file."""
        try:
            if os.path.exists(self.env_path):
                return os.path.getmtime(self.env_path)
        except OSError:
            pass
        return None
    
    async def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                
                current_mtime = self._get_mtime()
                if current_mtime and current_mtime != self._last_mtime:
                    self._last_mtime = current_mtime
                    await self._handle_change()
                    
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Config watch error: {e}")
    
    async def _handle_change(self) -> None:
        """Handle config file change."""
        logger.info("Config file changed, reloading...")
        
        # Reload environment from .env file
        try:
            self._reload_env_file()
        except Exception as e:
            logger.error(f"Failed to reload .env: {e}")
            return
        
        # Reload config
        reload_config()
        
        # Notify listeners
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener()
                else:
                    listener()
            except Exception as e:
                logger.error(f"Config listener error: {e}")
        
        logger.info("Config reloaded successfully")
    
    def _reload_env_file(self) -> None:
        """Reload environment variables from .env file."""
        if not os.path.exists(self.env_path):
            return
        
        with open(self.env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value


# Global config watcher instance
_config_watcher: Optional[ConfigWatcher] = None


def get_config_watcher() -> ConfigWatcher:
    """Get or create the global config watcher."""
    global _config_watcher
    if _config_watcher is None:
        _config_watcher = ConfigWatcher()
    return _config_watcher


async def start_config_watch() -> None:
    """Start config hot reload watching."""
    watcher = get_config_watcher()
    await watcher.start()


async def stop_config_watch() -> None:
    """Stop config hot reload watching."""
    watcher = get_config_watcher()
    await watcher.stop()
