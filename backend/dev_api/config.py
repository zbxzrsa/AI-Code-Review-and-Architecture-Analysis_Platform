"""
配置与常量 (Configuration and Constants)

模块功能描述:
    开发环境 API 服务器的集中配置管理。

配置分类:
    - 环境配置: MOCK_MODE, ENVIRONMENT, IS_PRODUCTION
    - 安全配置: MAX_REQUEST_SIZE_MB, CORS_ORIGINS
    - OAuth 配置: GITHUB_CLIENT_ID, GOOGLE_CLIENT_ID 等
    - 服务 URL: FRONTEND_URL 等

最后修改日期: 2024-12-07
"""

import os
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Environment Configuration
# ============================================
MOCK_MODE = os.getenv('MOCK_MODE', 'true').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_PRODUCTION = ENVIRONMENT == 'production'


# ============================================
# Security Configuration
# ============================================
MAX_REQUEST_SIZE_MB = int(os.getenv('MAX_REQUEST_SIZE_MB', '10'))
MAX_REQUEST_SIZE_BYTES = MAX_REQUEST_SIZE_MB * 1024 * 1024


def get_cors_origins() -> List[str]:
    """
    获取 CORS 允许的源
    
    功能描述:
        根据环境变量或默认配置获取 CORS 允许的源列表。
        生产环境必须显式配置 CORS_ORIGINS 环境变量。
    
    返回值:
        List[str]: 允许的源 URL 列表
    
    异常:
        ValueError: 生产环境未配置 CORS_ORIGINS 时抛出
    """
    cors_env = os.getenv("CORS_ORIGINS", "")
    
    if cors_env:
        return [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    
    if not IS_PRODUCTION:
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    
    raise ValueError(
        "CORS_ORIGINS must be set in production. "
        "Example: CORS_ORIGINS=https://your-domain.com"
    )


CORS_ORIGINS = get_cors_origins()
logger.info(f"CORS configured for: {CORS_ORIGINS}")


# ============================================
# OAuth Configuration
# ============================================
GITHUB_CLIENT_ID = os.getenv('GITHUB_CLIENT_ID', '')
GITHUB_CLIENT_SECRET = os.getenv('GITHUB_CLIENT_SECRET', '')
GITLAB_CLIENT_ID = os.getenv('GITLAB_CLIENT_ID', '')
GITLAB_CLIENT_SECRET = os.getenv('GITLAB_CLIENT_SECRET', '')
BITBUCKET_API_TOKEN = os.getenv('BITBUCKET_API_TOKEN', '')


# ============================================
# String Literals (avoid duplication)
# ============================================
class Literals:
    """String literals to avoid duplication."""
    # Common services
    BACKEND_SERVICES = "Backend Services"
    
    # Error messages
    PROJECT_NOT_FOUND = "Project not found"
    USER_NOT_FOUND = "User not found"
    ANALYSIS_NOT_FOUND = "Analysis not found"
    
    # File names
    FILE_MAIN_PY = "main.py"
    FILE_README_MD = "README.md"
    
    # Demo users
    DEMO_EMAIL = "demo@example.com"
    DEMO_USER = "Demo User"
    ADMIN_EMAIL = "admin@example.com"
    ADMIN_USER = "Admin User"
    USER_EMAIL = "user@example.com"
    
    # AI Models
    GPT4_TURBO = "GPT-4 Turbo"
    GPT35_TURBO = "gpt-3.5-turbo"
    GPT35_TURBO_DISPLAY = "GPT-3.5 Turbo"
    CLAUDE_3_OPUS = "Claude 3 Opus"
    
    # Versions
    VERSION_2_1_0 = "v2.1.0"
    VERSION_2_1_5 = "v2.1.5"
    VERSION_1_0_0 = "v1.0.0"
    
    # Names
    JOHN_DOE = "John Doe"
    JANE_SMITH = "Jane Smith"
    
    # Categories
    CI_CD_PIPELINE = "CI/CD Pipeline"
    SECURITY_SCAN = "Security Scan"
    
    # Security
    OWASP_INJECTION = "A03:2021 Injection"
    
    # Files
    SRC_AUTH_LOGIN_PY = "src/auth/login.py"
    SRC_API_USERS_PY = "src/api/users.py"
    
    # Branches
    FEATURE_AUTH = "feature/auth"
    
    # Emails
    JOHN_EMAIL = "john@example.com"
    JANE_EMAIL = "jane@example.com"


# Alias for backward compatibility
Constants = Literals
