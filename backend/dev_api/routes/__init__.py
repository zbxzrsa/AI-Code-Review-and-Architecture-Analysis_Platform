"""
API Routes

All route modules for the development API.

Structure:
    routes/
    ├── admin.py          - System administration endpoints
    ├── analysis.py       - Code analysis processing
    ├── auth.py           - Authentication and authorization
    ├── dashboard.py      - Dashboard metrics
    ├── oauth.py          - OAuth integration
    ├── projects.py       - Project management
    ├── reports.py        - Reports and backups
    ├── security.py       - Security endpoints
    ├── three_version.py  - Three-version evolution
    ├── users.py          - User management
    └── vulnerabilities.py - Vulnerability scanning
"""

from .admin import router as admin_router
from .analysis import router as analysis_router
from .auth import router as auth_router
from .dashboard import router as dashboard_router
from .oauth import router as oauth_router
from .projects import router as projects_router
from .reports import router as reports_router
from .security import router as security_router
from .three_version import router as three_version_router
from .users import router as users_router
from .vulnerabilities import router as vulnerabilities_router

__all__ = [
    "admin_router",
    "analysis_router",
    "auth_router",
    "dashboard_router",
    "oauth_router",
    "projects_router",
    "reports_router",
    "security_router",
    "three_version_router",
    "users_router",
    "vulnerabilities_router",
]
