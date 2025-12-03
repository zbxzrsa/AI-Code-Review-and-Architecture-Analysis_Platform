"""
V2 CR-AI Settings Configuration

Production-grade configuration with strict SLO enforcement.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class V2CRSettings(BaseSettings):
    """V2 Code Review AI Service Settings"""
    
    # Service Info
    service_name: str = "v2-cr-ai-service"
    version: str = "2.0.0"
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Primary Model (Claude 3 Sonnet)
    primary_model: str = Field(default="claude-3-sonnet-20240229", env="PRIMARY_MODEL")
    primary_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    primary_api_base: str = Field(default="https://api.anthropic.com/v1", env="ANTHROPIC_API_BASE")
    primary_timeout: float = Field(default=5.0, env="PRIMARY_TIMEOUT")
    
    # Secondary Model (GPT-4 for consensus)
    secondary_model: str = Field(default="gpt-4-turbo-2024-04-09", env="SECONDARY_MODEL")
    secondary_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    secondary_api_base: str = Field(default="https://api.openai.com/v1", env="OPENAI_API_BASE")
    secondary_timeout: float = Field(default=5.0, env="SECONDARY_TIMEOUT")
    
    # Model Parameters (Locked for Consistency)
    primary_temperature: float = Field(default=0.3, const=True)
    secondary_temperature: float = Field(default=0.2, const=True)  # More conservative
    
    # Consensus Settings
    consensus_enabled: bool = Field(default=True, env="CONSENSUS_ENABLED")
    consensus_threshold: float = Field(default=0.98, env="CONSENSUS_THRESHOLD")
    
    # SLO Targets
    slo_availability: float = Field(default=0.9999, env="SLO_AVAILABILITY")
    slo_error_rate: float = Field(default=0.001, env="SLO_ERROR_RATE")
    slo_p99_latency_ms: int = Field(default=500, env="SLO_P99_LATENCY_MS")
    
    # Accuracy Targets
    false_positive_rate: float = Field(default=0.02, env="FALSE_POSITIVE_RATE")  # <= 2%
    false_negative_rate: float = Field(default=0.05, env="FALSE_NEGATIVE_RATE")  # <= 5%
    
    # Database
    database_url: str = Field(
        default="postgresql://user:pass@localhost:5432/v2_cr_ai",
        env="DATABASE_URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/3",
        env="REDIS_URL"
    )
    
    # Monitoring
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    
    # Security
    code_retention_enabled: bool = Field(default=False, env="CODE_RETENTION_ENABLED")
    ephemeral_processing: bool = Field(default=True, env="EPHEMERAL_PROCESSING")
    
    # Audit Logging
    audit_log_retention_days: int = Field(default=2555, env="AUDIT_LOG_RETENTION_DAYS")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = V2CRSettings()
