"""
Backend Constants

Centralized constants for duplicate literals across backend services.
"""

# Model Names
MODEL_CODELLAMA_34B = "codellama:34b"
MODEL_CODELLAMA_13B = "codellama:13b"
MODEL_CODELLAMA_7B = "codellama:7b"
MODEL_LLAMA3_70B = "llama3:70b"
MODEL_MISTRAL_7B = "mistral:7b"
MODEL_MIXTRAL_8X7B = "mixtral:8x7b"

MODEL_GPT_35_TURBO = "gpt-3.5-turbo"
MODEL_GPT_35_TURBO_DISPLAY = "GPT-3.5 Turbo"
MODEL_CLAUDE_3_OPUS = "Claude 3 Opus"

# File Names
FILE_MAIN_PY = "main.py"
FILE_README_MD = "README.md"

# Versions
VERSION_V1_0_0 = "v1.0.0"
VERSION_V2_1_0 = "v2.1.0"
VERSION_V2_1_5 = "v2.1.5"

# Categories
CATEGORY_CI_CD_PIPELINE = "CI/CD Pipeline"
CATEGORY_SECURITY_SCAN = "Security Scan"

# Security
OWASP_INJECTION = "A03:2021 Injection"

# File Paths
PATH_SRC_AUTH_LOGIN = "src/auth/login.py"
PATH_SRC_API_USERS = "src/api/users.py"

# Branches
BRANCH_FEATURE_AUTH = "feature/auth"

# Services
SERVICE_BACKEND = "Backend Services"

# Emails
EMAIL_JOHN = "john@example.com"
EMAIL_JANE = "jane@example.com"

# Names
NAME_JOHN_DOE = "John Doe"
NAME_JANE_SMITH = "Jane Smith"

# API Paths
API_PATH_CR_AI_V1 = "/api/v1/cr-ai"
API_PATH_VC_AI_V1 = "/api/v1/vc-ai"

# Messages
MSG_PROJECT_NOT_FOUND = "Project not found"
MSG_ADDITIONAL_CONTEXT = "Additional context"

# Database
DB_USERS_ID = "users.id"
DB_SET_NULL = "SET NULL"

# SQL Queries
SQL_SELECT_AUDIT_LOG = "SELECT * FROM audits.audit_log WHERE 1=1"
SQL_AND_TS_GTE = " AND ts >= $"
SQL_AND_TS_LTE = " AND ts <= $"

# Patterns
PATTERN_EVAL = "eval("
PATTERN_EXCEPT = "except:"
PATTERN_CLASS_REGEX = r'class\s+(\w+)'
