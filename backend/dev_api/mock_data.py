"""
Mock Data

Development mock data for testing frontend features.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any
from .models import Project, ProjectSettings, Activity
from .config import Literals


# ============================================
# Projects Mock Data
# ============================================

def create_mock_projects() -> List[Project]:
    """Create mock projects list."""
    return [
        Project(
            id="proj_1",
            name="AI Code Review Platform",
            description="Main platform codebase",
            language="TypeScript",
            framework="React",
            repository_url="https://github.com/example/ai-code-review",
            status="active",
            issues_count=12,
            settings=ProjectSettings(auto_review=True, review_on_push=True, review_on_pr=True),
            created_at=datetime.now() - timedelta(days=30),
            updated_at=datetime.now() - timedelta(hours=2)
        ),
        Project(
            id="proj_2",
            name=Literals.BACKEND_SERVICES,
            description="FastAPI microservices",
            language="Python",
            framework="FastAPI",
            repository_url="https://github.com/example/backend",
            status="active",
            issues_count=5,
            settings=ProjectSettings(auto_review=False, review_on_push=False, review_on_pr=True),
            created_at=datetime.now() - timedelta(days=20),
            updated_at=datetime.now() - timedelta(hours=5)
        ),
        Project(
            id="proj_3",
            name="Mobile App",
            description="React Native mobile application",
            language="TypeScript",
            framework="React Native",
            status="active",
            issues_count=8,
            settings=ProjectSettings(auto_review=True, review_on_push=True, review_on_pr=False),
            created_at=datetime.now() - timedelta(days=15),
            updated_at=datetime.now() - timedelta(days=1)
        ),
    ]


# Global mock projects list
mock_projects: List[Project] = create_mock_projects()


# ============================================
# Activities Mock Data
# ============================================

def create_mock_activities() -> List[Activity]:
    """Create mock activities list."""
    return [
        Activity(
            id="act_1",
            type="analysis_complete",
            message="Code analysis completed for AI Code Review Platform",
            project_id="proj_1",
            created_at=datetime.now() - timedelta(hours=1)
        ),
        Activity(
            id="act_2",
            type="issue_fixed",
            message="Fixed 3 security issues in Backend Services",
            project_id="proj_2",
            created_at=datetime.now() - timedelta(hours=3)
        ),
        Activity(
            id="act_3",
            type="project_created",
            message="New project Mobile App created",
            project_id="proj_3",
            created_at=datetime.now() - timedelta(days=1)
        ),
    ]


mock_activities: List[Activity] = create_mock_activities()


# ============================================
# OAuth Mock Data
# ============================================

oauth_states: Dict[str, Any] = {}
user_oauth_tokens: Dict[str, Any] = {}
mock_oauth_connections: List[Dict[str, Any]] = []

mock_repositories = [
    {
        "id": "repo_1",
        "name": "ai-code-review",
        "full_name": "example/ai-code-review",
        "provider": "github",
        "url": "https://github.com/example/ai-code-review",
        "default_branch": "main",
        "language": "TypeScript",
        "private": False,
        "connected": True,
    },
    {
        "id": "repo_2",
        "name": "backend-services",
        "full_name": "example/backend-services",
        "provider": "github",
        "url": "https://github.com/example/backend-services",
        "default_branch": "main",
        "language": "Python",
        "private": True,
        "connected": True,
    },
]


# ============================================
# Users Mock Data
# ============================================

mock_users = [
    {
        "id": "user_1",
        "email": Literals.JOHN_EMAIL,
        "name": Literals.JOHN_DOE,
        "role": "admin",
        "status": "active",
        "created_at": (datetime.now() - timedelta(days=90)).isoformat(),
        "last_login": datetime.now().isoformat(),
    },
    {
        "id": "user_2",
        "email": Literals.JANE_EMAIL,
        "name": Literals.JANE_SMITH,
        "role": "user",
        "status": "active",
        "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
        "last_login": (datetime.now() - timedelta(hours=2)).isoformat(),
    },
    {
        "id": "user_3",
        "email": "bob@example.com",
        "name": "Bob Wilson",
        "role": "user",
        "status": "inactive",
        "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
        "last_login": (datetime.now() - timedelta(days=7)).isoformat(),
    },
]


# ============================================
# AI Providers Mock Data
# ============================================

mock_providers = [
    {
        "id": "provider_1",
        "name": "OpenAI",
        "type": "openai",
        "status": "healthy",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4-turbo",
        "rate_limit": 100,
        "requests_today": 45,
        "enabled": True,
    },
    {
        "id": "provider_2",
        "name": "Anthropic",
        "type": "anthropic",
        "status": "healthy",
        "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
        "default_model": "claude-3-opus",
        "rate_limit": 50,
        "requests_today": 12,
        "enabled": True,
    },
    {
        "id": "provider_3",
        "name": "Local LLM",
        "type": "local",
        "status": "degraded",
        "models": ["llama-2-70b", "codellama-34b"],
        "default_model": "codellama-34b",
        "rate_limit": 1000,
        "requests_today": 234,
        "enabled": True,
    },
]


# ============================================
# Experiments Mock Data
# ============================================

mock_experiments = [
    {
        "id": "exp_1",
        "name": "GPT-4 vs Claude-3 Code Review",
        "status": "running",
        "variant_a": {"model": "gpt-4-turbo", "accuracy": 0.89},
        "variant_b": {"model": "claude-3-opus", "accuracy": 0.91},
        "traffic_split": 50,
        "start_date": (datetime.now() - timedelta(days=7)).isoformat(),
        "end_date": (datetime.now() + timedelta(days=7)).isoformat(),
    },
    {
        "id": "exp_2",
        "name": "New Prompt Template",
        "status": "completed",
        "variant_a": {"template": "v1", "accuracy": 0.85},
        "variant_b": {"template": "v2", "accuracy": 0.92},
        "winner": "variant_b",
        "traffic_split": 50,
        "start_date": (datetime.now() - timedelta(days=14)).isoformat(),
        "end_date": (datetime.now() - timedelta(days=1)).isoformat(),
    },
]


# ============================================
# Three-Version Cycle Mock Data
# ============================================

mock_three_version_status = {
    "v1_experiment": {
        "version": "v1",
        "status": "active",
        "model": "gpt-4-turbo-exp",
        "traffic_percentage": 10,
        "metrics": {
            "accuracy": 0.92,
            "latency_p95": 2.3,
            "error_rate": 0.02,
        },
        "start_date": (datetime.now() - timedelta(days=3)).isoformat(),
    },
    "v2_production": {
        "version": "v2",
        "status": "active",
        "model": "gpt-4-turbo",
        "traffic_percentage": 85,
        "metrics": {
            "accuracy": 0.89,
            "latency_p95": 1.8,
            "error_rate": 0.01,
        },
        "promoted_date": (datetime.now() - timedelta(days=30)).isoformat(),
    },
    "v3_legacy": {
        "version": "v3",
        "status": "quarantined",
        "model": "gpt-3.5-turbo",
        "traffic_percentage": 5,
        "metrics": {
            "accuracy": 0.82,
            "latency_p95": 1.2,
            "error_rate": 0.03,
        },
        "demoted_date": (datetime.now() - timedelta(days=30)).isoformat(),
    },
}


# ============================================
# Security Mock Data
# ============================================

mock_vulnerabilities = [
    {
        "id": "vuln_1",
        "title": "SQL Injection in User Query",
        "severity": "critical",
        "status": "open",
        "file": "src/api/users.py",
        "line": 45,
        "description": "User input not sanitized before SQL query",
        "cwe": "CWE-89",
        "owasp": "A03:2021",
        "detected_at": (datetime.now() - timedelta(hours=2)).isoformat(),
    },
    {
        "id": "vuln_2",
        "title": "Hardcoded API Key",
        "severity": "high",
        "status": "open",
        "file": "src/config/settings.py",
        "line": 12,
        "description": "API key hardcoded in source code",
        "cwe": "CWE-798",
        "owasp": "A07:2021",
        "detected_at": (datetime.now() - timedelta(hours=5)).isoformat(),
    },
    {
        "id": "vuln_3",
        "title": "Cross-Site Scripting (XSS)",
        "severity": "medium",
        "status": "fixed",
        "file": "src/components/UserProfile.tsx",
        "line": 78,
        "description": "User input rendered without escaping",
        "cwe": "CWE-79",
        "owasp": "A03:2021",
        "detected_at": (datetime.now() - timedelta(days=1)).isoformat(),
        "fixed_at": (datetime.now() - timedelta(hours=3)).isoformat(),
    },
]
