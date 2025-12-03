"""
V2 VC-AI Settings Configuration

Production-grade configuration with strict SLO enforcement.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class V2VCSettings(BaseSettings):
    """V2 Version Control AI Service Settings"""
    
    # Service Info
    service_name: str = "v2-vc-ai-service"
    version: str = "2.0.0"
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Primary Model (GPT-4 Turbo)
    primary_model: str = Field(default="gpt-4-turbo-2024-04-09", env="PRIMARY_MODEL")
    primary_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    primary_api_base: str = Field(default="https://api.openai.com/v1", env="OPENAI_API_BASE")
    primary_timeout: float = Field(default=5.0, env="PRIMARY_TIMEOUT")
    
    # Backup Model (Claude 3 Opus)
    backup_model: str = Field(default="claude-3-opus-20240229", env="BACKUP_MODEL")
    backup_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    backup_api_base: str = Field(default="https://api.anthropic.com/v1", env="ANTHROPIC_API_BASE")
    backup_timeout: float = Field(default=5.0, env="BACKUP_TIMEOUT")
    
    # Model Parameters (Locked for Consistency)
    temperature: float = Field(default=0.3, const=True)
    top_p: float = Field(default=0.9, const=True)
    top_k: int = Field(default=40, const=True)
    
    # API Version Pinning
    api_version: str = Field(default="2024-01-15", const=True)
    
    # SLO Targets
    slo_availability: float = Field(default=0.9999, env="SLO_AVAILABILITY")
    slo_error_rate: float = Field(default=0.001, env="SLO_ERROR_RATE")
    slo_p99_latency_ms: int = Field(default=500, env="SLO_P99_LATENCY_MS")
    slo_user_satisfaction: float = Field(default=4.7, env="SLO_USER_SATISFACTION")
    
    # Failover Settings
    failover_trigger_error_rate: float = Field(default=0.01, env="FAILOVER_ERROR_RATE")
    failover_consecutive_failures: int = Field(default=3, env="FAILOVER_CONSECUTIVE_FAILURES")
    
    # Database
    database_url: str = Field(
        default="postgresql://user:pass@localhost:5432/v2_vc_ai",
        env="DATABASE_URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/2",
        env="REDIS_URL"
    )
    
    # Monitoring
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    
    # V1 Integration
    v1_api_endpoint: str = Field(
        default="http://v1-vc-ai-service:8000/api/v1/vc-ai",
        env="V1_API_ENDPOINT"
    )
    
    # Audit Logging
    audit_log_retention_days: int = Field(default=2555, env="AUDIT_LOG_RETENTION_DAYS")  # 7 years
    
    # CORS
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = V2VCSettings()
