"""
Application Configuration / 应用配置

Enterprise-grade configuration with security validation.
企业级配置，包含安全验证。
"""

import os
import secrets
import logging
from typing import List

logger = logging.getLogger(__name__)

# ============================================
# Environment Configuration / 环境配置
# ============================================
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"
IS_DEVELOPMENT = ENVIRONMENT == "development"
DEBUG = os.getenv("DEBUG", "false").lower() == "true" and not IS_PRODUCTION

# Mode Configuration / 模式配置
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# ============================================
# Security Configuration / 安全配置
# ============================================
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Validate JWT secret in production
if IS_PRODUCTION:
    if not JWT_SECRET_KEY or len(JWT_SECRET_KEY) < 32:
        raise ValueError(
            "CRITICAL: JWT_SECRET_KEY must be set and at least 256 bits (32 characters) in production"
        )
    if "changeme" in JWT_SECRET_KEY.lower() or "your-" in JWT_SECRET_KEY.lower():
        raise ValueError(
            "CRITICAL: JWT_SECRET_KEY appears to be a placeholder. Generate a secure random key."
        )
elif not JWT_SECRET_KEY:
    # Generate random key for development only
    JWT_SECRET_KEY = secrets.token_urlsafe(64)
    logger.warning("JWT_SECRET_KEY not set, using random key (development only)")

# ============================================
# OAuth Configuration / OAuth配置
# ============================================
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITLAB_CLIENT_ID = os.getenv("GITLAB_CLIENT_ID", "")
GITLAB_CLIENT_SECRET = os.getenv("GITLAB_CLIENT_SECRET", "")
BITBUCKET_API_TOKEN = os.getenv("BITBUCKET_API_TOKEN", "")

# ============================================
# Server Configuration / 服务器配置
# ============================================
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# API Configuration
API_PREFIX = "/api"
API_VERSION = "v1"

# ============================================
# CORS Configuration / 跨域配置
# ============================================
def get_cors_origins() -> List[str]:
    """Get CORS allowed origins based on environment."""
    # Check for explicit configuration
    cors_env = os.getenv("CORS_ORIGINS", "")
    
    if cors_env:
        # Production: use explicit origins only
        origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
        if IS_PRODUCTION and not origins:
            raise ValueError("CORS_ORIGINS must be configured in production")
        return origins
    
    # Development: allow localhost
    if IS_DEVELOPMENT:
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    
    # Production without config: fail safe
    if IS_PRODUCTION:
        raise ValueError(
            "CORS_ORIGINS environment variable must be set in production. "
            "Example: CORS_ORIGINS=https://your-domain.com,https://app.your-domain.com"
        )
    
    return []

CORS_ORIGINS = get_cors_origins()

# CORS methods - restricted set
CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
CORS_HEADERS = ["Content-Type", "Authorization", "X-CSRF-Token", "X-Request-ID"]

# ============================================
# Database Configuration / 数据库配置
# ============================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://coderev:changeme@localhost:5432/code_review_platform")

# Validate database password in production
if IS_PRODUCTION and "changeme" in DATABASE_URL:
    raise ValueError("CRITICAL: Database password must be changed in production")

# Connection pool settings
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))

# ============================================
# Redis Configuration / Redis配置
# ============================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Validate Redis password in production
if IS_PRODUCTION:
    redis_password = os.getenv("REDIS_PASSWORD", "")
    if not redis_password or "changeme" in redis_password.lower():
        logger.warning("REDIS_PASSWORD should be set in production")

# ============================================
# Rate Limiting / 速率限制
# ============================================
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "10000"))

# ============================================
# Request Limits / 请求限制
# ============================================
MAX_REQUEST_SIZE_MB = int(os.getenv("MAX_REQUEST_SIZE_MB", "10"))
MAX_REQUEST_SIZE_BYTES = MAX_REQUEST_SIZE_MB * 1024 * 1024

# ============================================
# Logging Configuration / 日志配置
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if IS_DEVELOPMENT else "WARNING")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================
# Feature Flags / 功能开关
# ============================================
FEATURE_2FA_ENABLED = os.getenv("FEATURE_2FA_ENABLED", "true").lower() == "true"
FEATURE_API_KEYS_ENABLED = os.getenv("FEATURE_API_KEYS_ENABLED", "true").lower() == "true"
FEATURE_AUDIT_LOGGING = os.getenv("FEATURE_AUDIT_LOGGING", "true").lower() == "true"

# ============================================
# Validation Summary / 验证摘要
# ============================================
if IS_PRODUCTION:
    logger.info("=== Production Environment Validation ===")
    logger.info(f"CORS Origins: {CORS_ORIGINS}")
    logger.info(f"JWT Secret: {'configured' if JWT_SECRET_KEY else 'MISSING'}")
    logger.info(f"Database Pool: {DB_POOL_SIZE} connections")
    logger.info(f"Rate Limit: {RATE_LIMIT_PER_MINUTE}/min")
