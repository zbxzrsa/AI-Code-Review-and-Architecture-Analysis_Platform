"""
Development API Server - Modular Structure

Modular architecture for the AI Code Review Platform API.

Structure:
    dev_api/
    ├── __init__.py      - Package initialization (this file)
    ├── app.py           - FastAPI application factory
    ├── config.py        - Configuration and constants
    ├── models.py        - Pydantic models
    ├── mock_data.py     - Mock data for development
    ├── middleware.py    - Custom middleware
    ├── core/            - Core infrastructure
    │   ├── config.py        - System configuration (Pydantic Settings)
    │   ├── dependencies.py  - Dependency injection
    │   └── middleware.py    - Middleware implementations
    ├── routes/          - API route modules (all < 2000 lines)
    │   ├── admin.py         - Admin endpoints
    │   ├── analysis.py      - Code analysis
    │   ├── auth.py          - Authentication/authorization
    │   ├── dashboard.py     - Dashboard metrics
    │   ├── oauth.py         - OAuth integration
    │   ├── projects.py      - Project management
    │   ├── reports.py       - Reports and backups
    │   ├── security.py      - Security endpoints
    │   ├── three_version.py - Three-version evolution
    │   ├── users.py         - User management
    │   └── vulnerabilities.py - Vulnerability management
    └── services/        - Business logic services
        ├── code_review_service.py    - Code review logic
        ├── vulnerability_service.py  - Vulnerability handling
        └── analytics_service.py      - Analytics logic

Usage:
    # Import the app
    from dev_api import app
    
    # Or create a new app instance
    from dev_api.app import create_app
    app = create_app()
    
    # Use services
    from dev_api.services import CodeReviewService
    service = CodeReviewService()
    
    # Use dependencies
    from dev_api.core import get_current_user, require_admin
"""

from .app import app, create_app
from .config import logger, MOCK_MODE, IS_PRODUCTION

__all__ = [
    "app",
    "create_app",
    "logger",
    "MOCK_MODE",
    "IS_PRODUCTION",
]

__version__ = "2.1.0"
