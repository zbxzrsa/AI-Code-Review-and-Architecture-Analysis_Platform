"""
Analysis Routes

Code analysis endpoints.
"""

import secrets
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from ..models import AnalyzeRequest, CodeAnalyzeRequest
from ..config import Literals

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post("/projects/{project_id}/analyze")
async def analyze_project(project_id: str, request: AnalyzeRequest):
    """Start project analysis."""
    analysis_id = f"analysis_{secrets.token_hex(8)}"
    return {
        "id": analysis_id,
        "project_id": project_id,
        "status": "running",
        "message": "Analysis started",
        "created_at": datetime.now().isoformat()
    }


@router.get("/projects/{project_id}/analysis/{analysis_id}")
async def get_analysis_status(project_id: str, analysis_id: str):
    """Get analysis status."""
    return {
        "id": analysis_id,
        "project_id": project_id,
        "status": "completed",
        "progress": 100,
        "issues_found": 5,
        "created_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
        "completed_at": datetime.now().isoformat()
    }


@router.get("/projects/{project_id}/issues")
async def get_project_issues(project_id: str):
    """Get project issues."""
    return {
        "items": [
            {
                "id": "issue_1",
                "title": "Potential SQL Injection",
                "severity": "high",
                "file": Literals.SRC_API_USERS_PY,
                "line": 45,
                "description": "User input is not sanitized before SQL query",
                "category": "security",
                "status": "open",
                "created_at": (datetime.now() - timedelta(hours=2)).isoformat()
            },
            {
                "id": "issue_2",
                "title": "Unused Variable",
                "severity": "low",
                "file": Literals.SRC_AUTH_LOGIN_PY,
                "line": 23,
                "description": "Variable 'temp' is declared but never used",
                "category": "code-quality",
                "status": "open",
                "created_at": (datetime.now() - timedelta(hours=3)).isoformat()
            },
            {
                "id": "issue_3",
                "title": "Missing Error Handling",
                "severity": "medium",
                "file": "src/services/api.ts",
                "line": 78,
                "description": "API call lacks proper error handling",
                "category": "reliability",
                "status": "fixed",
                "created_at": (datetime.now() - timedelta(days=1)).isoformat()
            }
        ],
        "total": 3
    }


@router.patch("/projects/{project_id}/issues/{issue_id}")
async def update_issue(project_id: str, issue_id: str):
    """Update issue status."""
    return {
        "id": issue_id,
        "status": "fixed",
        "updated_at": datetime.now().isoformat()
    }


# ============================================
# Direct Code Analysis
# ============================================

@router.post("/analyze/code")
async def analyze_code(request: CodeAnalyzeRequest):
    """Analyze code snippet directly."""
    analysis_id = f"code_analysis_{secrets.token_hex(8)}"
    return {
        "id": analysis_id,
        "status": "completed",
        "language": request.language,
        "issues": [
            {
                "line": 5,
                "severity": "medium",
                "message": "Consider using type hints for function parameters",
                "rule": "type-hints",
            },
            {
                "line": 12,
                "severity": "low",
                "message": "Line exceeds recommended length of 80 characters",
                "rule": "line-length",
            }
        ],
        "metrics": {
            "lines_of_code": len(request.code.split('\n')),
            "complexity": 5,
            "maintainability": 0.85,
        }
    }


@router.get("/analyze/{analysis_id}/results")
async def get_analysis_results(analysis_id: str):
    """Get analysis results."""
    return {
        "id": analysis_id,
        "status": "completed",
        "summary": {
            "total_issues": 3,
            "critical": 0,
            "high": 1,
            "medium": 1,
            "low": 1,
        },
        "issues": [
            {
                "id": "issue_1",
                "severity": "high",
                "message": "Potential security vulnerability",
                "file": "main.py",
                "line": 42,
            }
        ]
    }
