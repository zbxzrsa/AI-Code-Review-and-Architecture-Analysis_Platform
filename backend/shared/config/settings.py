"""
Shared configuration settings for all platform versions.
"""
import os
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class VersionType(str, Enum):
    """Platform version types."""
    V1_EXPERIMENTATION = "v1"
    V2_PRODUCTION = "v2"
    V3_QUARANTINE = "v3"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    username: str
    password: str
    database: str
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


@dataclass
class AIModelConfig:
    """AI Model configuration."""
    provider: str  # "openai", "anthropic", "custom"
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30


@dataclass
class KubernetesConfig:
    """Kubernetes configuration."""
    namespace: str
    cluster_name: str
    resource_limits: dict
    resource_requests: dict


class Settings:
    """Main settings class."""

    def __init__(self):
        self.environment = Environment(
            os.getenv("ENVIRONMENT", "development")
        )
        self.version = VersionType(
            os.getenv("PLATFORM_VERSION", "v2")
        )
        self.debug = self.environment == Environment.DEVELOPMENT

        # Database configuration
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            database=self._get_database_name(),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            echo=self.debug,
        )

        # AI Model configuration
        self.primary_ai_model = AIModelConfig(
            provider=os.getenv("PRIMARY_AI_PROVIDER", "openai"),
            model_name=os.getenv("PRIMARY_AI_MODEL", "gpt-4"),
            api_key=os.getenv("PRIMARY_AI_API_KEY", ""),
            temperature=float(os.getenv("AI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("AI_MAX_TOKENS", "2000")),
        )

        self.secondary_ai_model = AIModelConfig(
            provider=os.getenv("SECONDARY_AI_PROVIDER", "anthropic"),
            model_name=os.getenv("SECONDARY_AI_MODEL", "claude-3-opus"),
            api_key=os.getenv("SECONDARY_AI_API_KEY", ""),
            temperature=float(os.getenv("AI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("AI_MAX_TOKENS", "2000")),
        )

        # Kubernetes configuration
        self.kubernetes = KubernetesConfig(
            namespace=self._get_k8s_namespace(),
            cluster_name=os.getenv("K8S_CLUSTER", "default"),
            resource_limits=self._parse_resource_limits(),
            resource_requests=self._parse_resource_requests(),
        )

        # API configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.api_prefix = "/api/v1"

        # Monitoring configuration
        self.prometheus_enabled = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
        self.prometheus_port = int(os.getenv("PROMETHEUS_PORT", "9090"))

        # SLO configuration (for V2)
        self.slo_response_time_p95_ms = int(
            os.getenv("SLO_RESPONSE_TIME_P95_MS", "3000")
        )
        self.slo_error_rate_threshold = float(
            os.getenv("SLO_ERROR_RATE_THRESHOLD", "0.02")
        )

        # Evaluation thresholds (for V1)
        self.v1_accuracy_threshold = float(
            os.getenv("V1_ACCURACY_THRESHOLD", "0.95")
        )
        self.v1_latency_threshold_ms = int(
            os.getenv("V1_LATENCY_THRESHOLD_MS", "3000")
        )
        self.v1_error_rate_threshold = float(
            os.getenv("V1_ERROR_RATE_THRESHOLD", "0.02")
        )

    def _get_database_name(self) -> str:
        """Get database name based on version."""
        if self.version == VersionType.V1_EXPERIMENTATION:
            return os.getenv("DB_NAME", "experiments_v1")
        elif self.version == VersionType.V2_PRODUCTION:
            return os.getenv("DB_NAME", "production")
        else:  # V3
            return os.getenv("DB_NAME", "quarantine")

    def _get_k8s_namespace(self) -> str:
        """Get Kubernetes namespace based on version."""
        if self.version == VersionType.V1_EXPERIMENTATION:
            return "platform-v1-exp"
        elif self.version == VersionType.V2_PRODUCTION:
            return "platform-v2-stable"
        else:  # V3
            return "platform-v3-quarantine"

    def _parse_resource_limits(self) -> dict:
        """Parse resource limits from environment."""
        return {
            "cpu": os.getenv("K8S_CPU_LIMIT", "2"),
            "memory": os.getenv("K8S_MEMORY_LIMIT", "2Gi"),
        }

    def _parse_resource_requests(self) -> dict:
        """Parse resource requests from environment."""
        return {
            "cpu": os.getenv("K8S_CPU_REQUEST", "500m"),
            "memory": os.getenv("K8S_MEMORY_REQUEST", "512Mi"),
        }


# Global settings instance
settings = Settings()
