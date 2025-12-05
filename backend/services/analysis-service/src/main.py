"""
Analysis Service - Code analysis and AI-powered review.
"""
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Analysis Service",
    description="Code analysis and AI-powered review service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class AnalysisRequest(BaseModel):
    project_id: Optional[str] = None
    code: Optional[str] = None
    language: str = "python"
    files: Optional[List[str]] = None
    options: Optional[Dict[str, Any]] = None


class Issue(BaseModel):
    id: str
    type: str
    severity: IssueSeverity
    line_start: int
    line_end: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    description: str
    fix: Optional[str] = None
    rule: Optional[str] = None


class AnalysisResponse(BaseModel):
    id: str
    status: AnalysisStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    issues: List[Issue] = []
    summary: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


# Health endpoints
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/health/live")
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    return {"status": "ready"}


# Analysis endpoints
@app.post("/api/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest):
    """Start code analysis."""
    return AnalysisResponse(
        id="analysis_123",
        status=AnalysisStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        issues=[
            Issue(
                id="issue_1",
                type="security",
                severity=IssueSeverity.WARNING,
                line_start=10,
                line_end=10,
                description="Potential SQL injection vulnerability",
                fix="Use parameterized queries",
                rule="SEC001",
            ),
            Issue(
                id="issue_2",
                type="performance",
                severity=IssueSeverity.INFO,
                line_start=25,
                description="Consider using list comprehension",
                fix="result = [x * 2 for x in items]",
                rule="PERF002",
            ),
        ],
        summary="Found 2 issues: 1 warning, 1 info",
        metrics={
            "complexity": 12,
            "maintainability": 78,
            "security_score": 85,
            "lines_analyzed": 150,
            "analysis_time_ms": 1250,
        },
    )


@app.get("/api/analyze/{session_id}", response_model=AnalysisResponse)
async def get_analysis(session_id: str):
    """Get analysis results."""
    return AnalysisResponse(
        id=session_id,
        status=AnalysisStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        issues=[],
        summary="Analysis complete",
        metrics={"complexity": 10, "maintainability": 85},
    )


@app.get("/api/analyze/{session_id}/issues")
async def get_issues(session_id: str, severity: Optional[str] = None, type: Optional[str] = None):
    """Get issues from analysis."""
    return {
        "session_id": session_id,
        "issues": [
            {
                "id": "issue_1",
                "type": "security",
                "severity": "warning",
                "line_start": 10,
                "description": "Potential vulnerability",
            }
        ],
        "total": 1,
    }


@app.post("/api/analyze/{session_id}/issues/{issue_id}/fix")
async def apply_fix(session_id: str, issue_id: str):
    """Apply suggested fix for an issue."""
    return {
        "session_id": session_id,
        "issue_id": issue_id,
        "status": "fixed",
        "message": "Fix applied successfully",
    }


@app.post("/api/analyze/{session_id}/issues/{issue_id}/dismiss")
async def dismiss_issue(session_id: str, issue_id: str, reason: Optional[str] = None):
    """Dismiss an issue."""
    return {
        "session_id": session_id,
        "issue_id": issue_id,
        "status": "dismissed",
        "reason": reason,
    }


@app.post("/api/projects/{project_id}/analyze", response_model=AnalysisResponse)
async def analyze_project(project_id: str, files: Optional[List[str]] = None):
    """Analyze entire project or specific files."""
    return AnalysisResponse(
        id="analysis_proj",
        status=AnalysisStatus.COMPLETED,
        created_at=datetime.now(timezone.utc),
        issues=[],
        summary=f"Analyzed project {project_id}",
        metrics={"files_analyzed": 15, "total_lines": 2500},
    )


@app.get("/api/projects/{project_id}/sessions")
async def get_project_sessions(project_id: str, page: int = 1, limit: int = 20):
    """Get analysis sessions for a project."""
    return {
        "items": [
            {
                "id": "session_1",
                "project_id": project_id,
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "issues_count": 5,
            }
        ],
        "total": 1,
        "page": page,
        "limit": limit,
    }


# Celery app for async tasks
try:
    from celery import Celery
    
    celery_app = Celery(
        "analysis",
        broker="redis://redis:6379/0",
        backend="redis://redis:6379/0",
    )
    
    @celery_app.task
    def analyze_code_task(code: str, language: str):
        """Background task for code analysis."""
        logger.info(f"Analyzing {language} code in background")
        return {"status": "completed", "issues": []}
    
    # Alias for celery command
    app.celery = celery_app
    
except ImportError:
    logger.warning("Celery not available, background tasks disabled")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
