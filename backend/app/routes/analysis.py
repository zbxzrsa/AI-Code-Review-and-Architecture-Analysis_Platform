"""
Analysis Routes / 分析路由
"""

from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel
import random

from ..config import MOCK_MODE

router = APIRouter(prefix="/api", tags=["Analysis"])


class AnalyzeCodeRequest(BaseModel):
    code: str
    language: str = "typescript"
    model: str = "gpt-4-turbo"
    context: Optional[str] = None


class Issue(BaseModel):
    id: str
    type: str  # error, warning, info, suggestion
    severity: str  # critical, high, medium, low
    message: str
    line: int
    column: int
    file: Optional[str] = None
    suggestion: Optional[str] = None


@router.post("/ai/analyze")
async def ai_analyze_code(request: AnalyzeCodeRequest):
    """Analyze code with AI / 使用 AI 分析代码"""
    if MOCK_MODE:
        # Return mock analysis results
        return {
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "completed",
            "model": request.model,
            "issues": [
                {
                    "id": "issue_1",
                    "type": "warning",
                    "severity": "medium",
                    "message": "Consider using const instead of let for variables that are not reassigned",
                    "line": 5,
                    "column": 1,
                    "suggestion": "Replace 'let' with 'const'"
                },
                {
                    "id": "issue_2",
                    "type": "suggestion",
                    "severity": "low",
                    "message": "Function could benefit from explicit return type",
                    "line": 10,
                    "column": 1,
                    "suggestion": "Add return type annotation"
                }
            ],
            "metrics": {
                "complexity": 12,
                "maintainability": 85,
                "test_coverage": 0,
                "lines_analyzed": len(request.code.split('\n'))
            },
            "summary": "Code analysis completed. Found 2 issues.",
            "processing_time_ms": random.randint(500, 2000)
        }
    else:
        # TODO: Integrate with real AI provider
        raise NotImplementedError("Real AI analysis not implemented yet")


@router.get("/analyze/{session_id}/issues")
async def get_analysis_issues(
    session_id: str,
    severity: Optional[str] = None,
    type: Optional[str] = None
):
    """Get analysis issues / 获取分析问题"""
    issues = [
        Issue(
            id=f"issue_{i}",
            type=["error", "warning", "info", "suggestion"][i % 4],
            severity=["critical", "high", "medium", "low"][i % 4],
            message=f"Sample issue {i}",
            line=i * 10,
            column=1,
            suggestion=f"Fix for issue {i}"
        )
        for i in range(1, 11)
    ]
    
    if severity:
        issues = [i for i in issues if i.severity == severity]
    if type:
        issues = [i for i in issues if i.type == type]
    
    return {
        "session_id": session_id,
        "issues": [i.dict() for i in issues],
        "total": len(issues)
    }


@router.get("/analyze/{session_id}/results")
async def get_analysis_results(session_id: str):
    """Get full analysis results / 获取完整分析结果"""
    return {
        "session_id": session_id,
        "status": "completed",
        "started_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
        "completed_at": datetime.now().isoformat(),
        "summary": {
            "total_issues": 5,
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 2
        },
        "metrics": {
            "complexity": 15,
            "maintainability": 82,
            "duplications": 3,
            "code_smells": 5
        },
        "files_analyzed": 12,
        "lines_analyzed": 1500
    }


@router.get("/security/vulnerabilities")
async def get_vulnerabilities(
    severity: Optional[str] = None,
    status: Optional[str] = None
):
    """Get security vulnerabilities / 获取安全漏洞"""
    vulnerabilities = [
        {
            "id": f"vuln_{i}",
            "title": f"Security Vulnerability {i}",
            "severity": ["critical", "high", "medium", "low"][i % 4],
            "status": ["open", "in_progress", "resolved"][i % 3],
            "cve": f"CVE-2024-{1000 + i}" if i % 2 == 0 else None,
            "affected_file": f"src/module{i}.ts",
            "description": f"Description of vulnerability {i}",
            "recommendation": f"Fix recommendation for vulnerability {i}",
            "discovered_at": (datetime.now() - timedelta(days=i)).isoformat()
        }
        for i in range(1, 11)
    ]
    
    if severity:
        vulnerabilities = [v for v in vulnerabilities if v["severity"] == severity]
    if status:
        vulnerabilities = [v for v in vulnerabilities if v["status"] == status]
    
    return {
        "vulnerabilities": vulnerabilities,
        "total": len(vulnerabilities),
        "summary": {
            "critical": sum(1 for v in vulnerabilities if v["severity"] == "critical"),
            "high": sum(1 for v in vulnerabilities if v["severity"] == "high"),
            "medium": sum(1 for v in vulnerabilities if v["severity"] == "medium"),
            "low": sum(1 for v in vulnerabilities if v["severity"] == "low")
        }
    }


@router.get("/security/metrics")
async def get_security_metrics():
    """Get security metrics / 获取安全指标"""
    return {
        "overall_score": 85,
        "vulnerability_count": 5,
        "last_scan": datetime.now().isoformat(),
        "compliance": {
            "owasp": 92,
            "sans": 88,
            "pci_dss": 95
        },
        "trends": {
            "vulnerabilities_30d": [5, 4, 6, 3, 5, 4, 3],
            "score_30d": [82, 83, 81, 84, 85, 85, 85]
        }
    }
