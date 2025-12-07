"""
Vulnerability Scanning and Management API Endpoints

Contains all API endpoints related to:
- Vulnerability scanning
- Vulnerability reporting
- Auto-fix suggestions
- Severity management
- Vulnerability statistics

Module Size: ~350 lines (target < 2000)
"""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from enum import Enum
import uuid

from ..config import logger

router = APIRouter(prefix="/api/vulnerabilities", tags=["Vulnerabilities"])


# =============================================================================
# Enums and Models
# =============================================================================

class SeverityLevel(str, Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityStatus(str, Enum):
    """Vulnerability status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ACCEPTED = "accepted"


class VulnerabilityType(str, Enum):
    """Common vulnerability types."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    SSRF = "ssrf"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    BROKEN_AUTH = "broken_authentication"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    DEPENDENCY = "dependency_vulnerability"
    OTHER = "other"


class Vulnerability(BaseModel):
    """Vulnerability model."""
    id: str
    type: VulnerabilityType
    severity: SeverityLevel
    status: VulnerabilityStatus
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    detected_at: str
    resolved_at: Optional[str] = None
    assigned_to: Optional[str] = None


class ScanRequest(BaseModel):
    """Vulnerability scan request."""
    project_id: str
    scan_type: str = "full"  # full, incremental, quick
    include_dependencies: bool = True
    severity_threshold: SeverityLevel = SeverityLevel.LOW


class AutoFixRequest(BaseModel):
    """Auto-fix request for a vulnerability."""
    vulnerability_id: str
    apply_fix: bool = False
    review_mode: bool = True


class AutoFix(BaseModel):
    """Auto-fix suggestion."""
    id: str
    vulnerability_id: str
    description: str
    original_code: str
    fixed_code: str
    confidence: float
    risk_level: str
    requires_review: bool


class StatusUpdateRequest(BaseModel):
    """Vulnerability status update request."""
    status: VulnerabilityStatus
    comment: Optional[str] = None
    assigned_to: Optional[str] = None


# =============================================================================
# Mock Data Store
# =============================================================================

MOCK_VULNERABILITIES: dict[str, Vulnerability] = {
    "vuln-001": Vulnerability(
        id="vuln-001",
        type=VulnerabilityType.SQL_INJECTION,
        severity=SeverityLevel.CRITICAL,
        status=VulnerabilityStatus.OPEN,
        title="SQL Injection in user login",
        description="User input is directly concatenated into SQL query without sanitization",
        file_path="src/auth/login.py",
        line_number=42,
        code_snippet='query = f"SELECT * FROM users WHERE email = \'{email}\'"',
        recommendation="Use parameterized queries or an ORM",
        cwe_id="CWE-89",
        cvss_score=9.8,
        detected_at=datetime.now(timezone.utc).isoformat(),
    ),
    "vuln-002": Vulnerability(
        id="vuln-002",
        type=VulnerabilityType.XSS,
        severity=SeverityLevel.HIGH,
        status=VulnerabilityStatus.OPEN,
        title="Cross-Site Scripting in comment display",
        description="User comments are rendered without HTML escaping",
        file_path="src/components/Comments.tsx",
        line_number=78,
        code_snippet='<div dangerouslySetInnerHTML={{__html: comment.body}} />',
        recommendation="Sanitize HTML or use text content instead",
        cwe_id="CWE-79",
        cvss_score=7.5,
        detected_at=datetime.now(timezone.utc).isoformat(),
    ),
    "vuln-003": Vulnerability(
        id="vuln-003",
        type=VulnerabilityType.SENSITIVE_DATA_EXPOSURE,
        severity=SeverityLevel.MEDIUM,
        status=VulnerabilityStatus.IN_PROGRESS,
        title="API key exposed in client-side code",
        description="Third-party API key is hardcoded in frontend JavaScript",
        file_path="src/services/api.js",
        line_number=15,
        code_snippet='const API_KEY = "sk-abc123xyz789";',
        recommendation="Move API key to environment variables on backend",
        cwe_id="CWE-798",
        cvss_score=6.5,
        detected_at=datetime.now(timezone.utc).isoformat(),
        assigned_to="user-002",
    ),
    "vuln-004": Vulnerability(
        id="vuln-004",
        type=VulnerabilityType.DEPENDENCY,
        severity=SeverityLevel.HIGH,
        status=VulnerabilityStatus.OPEN,
        title="Vulnerable dependency: lodash < 4.17.21",
        description="Known prototype pollution vulnerability in lodash",
        file_path="package.json",
        line_number=25,
        code_snippet='"lodash": "^4.17.15"',
        recommendation="Upgrade lodash to version 4.17.21 or later",
        cwe_id="CWE-1321",
        cvss_score=7.4,
        detected_at=datetime.now(timezone.utc).isoformat(),
    ),
}

MOCK_AUTO_FIXES: dict[str, AutoFix] = {
    "fix-001": AutoFix(
        id="fix-001",
        vulnerability_id="vuln-001",
        description="Replace string concatenation with parameterized query",
        original_code='query = f"SELECT * FROM users WHERE email = \'{email}\'"',
        fixed_code='query = "SELECT * FROM users WHERE email = %s"\ncursor.execute(query, (email,))',
        confidence=0.95,
        risk_level="low",
        requires_review=True,
    ),
    "fix-002": AutoFix(
        id="fix-002",
        vulnerability_id="vuln-002",
        description="Replace dangerouslySetInnerHTML with sanitized content",
        original_code='<div dangerouslySetInnerHTML={{__html: comment.body}} />',
        fixed_code='<div>{DOMPurify.sanitize(comment.body)}</div>',
        confidence=0.88,
        risk_level="low",
        requires_review=True,
    ),
}


# =============================================================================
# Vulnerability Endpoints
# =============================================================================

@router.get("", response_model=List[Vulnerability])
async def list_vulnerabilities(
    project_id: Optional[str] = None,
    severity: Optional[SeverityLevel] = None,
    status: Optional[VulnerabilityStatus] = None,
    type: Optional[VulnerabilityType] = None,
    limit: int = Query(default=50, le=100),
    offset: int = 0,
):
    """
    List vulnerabilities with optional filters.
    
    - **project_id**: Filter by project
    - **severity**: Filter by severity level
    - **status**: Filter by status
    - **type**: Filter by vulnerability type
    """
    results = list(MOCK_VULNERABILITIES.values())
    
    if severity:
        results = [v for v in results if v.severity == severity]
    if status:
        results = [v for v in results if v.status == status]
    if type:
        results = [v for v in results if v.type == type]
    
    return results[offset:offset + limit]


@router.get("/stats")
async def get_vulnerability_stats(project_id: Optional[str] = None):
    """
    Get vulnerability statistics and summary.
    """
    vulns = list(MOCK_VULNERABILITIES.values())
    
    by_severity = {}
    by_status = {}
    by_type = {}
    
    for v in vulns:
        by_severity[v.severity.value] = by_severity.get(v.severity.value, 0) + 1
        by_status[v.status.value] = by_status.get(v.status.value, 0) + 1
        by_type[v.type.value] = by_type.get(v.type.value, 0) + 1
    
    return {
        "total": len(vulns),
        "by_severity": by_severity,
        "by_status": by_status,
        "by_type": by_type,
        "critical_open": len([v for v in vulns if v.severity == SeverityLevel.CRITICAL and v.status == VulnerabilityStatus.OPEN]),
        "high_open": len([v for v in vulns if v.severity == SeverityLevel.HIGH and v.status == VulnerabilityStatus.OPEN]),
        "mean_time_to_fix_days": 3.5,  # Mock value
        "fix_rate_30d": 0.78,  # Mock value
    }


@router.get("/{vulnerability_id}", response_model=Vulnerability)
async def get_vulnerability(vulnerability_id: str):
    """
    Get details of a specific vulnerability.
    """
    vuln = MOCK_VULNERABILITIES.get(vulnerability_id)
    if not vuln:
        raise HTTPException(status_code=404, detail="Vulnerability not found")
    return vuln


@router.patch("/{vulnerability_id}/status")
async def update_vulnerability_status(
    vulnerability_id: str,
    request: StatusUpdateRequest,
):
    """
    Update vulnerability status.
    
    - **status**: New status (open, in_progress, resolved, false_positive, accepted)
    - **comment**: Optional comment explaining the change
    - **assigned_to**: Optional user to assign
    """
    vuln = MOCK_VULNERABILITIES.get(vulnerability_id)
    if not vuln:
        raise HTTPException(status_code=404, detail="Vulnerability not found")
    
    # Update in mock store (in reality, would update in database)
    old_status = vuln.status
    vuln.status = request.status
    
    if request.assigned_to:
        vuln.assigned_to = request.assigned_to
    
    if request.status == VulnerabilityStatus.RESOLVED:
        vuln.resolved_at = datetime.now(timezone.utc).isoformat()
    
    logger.info(f"Vulnerability {vulnerability_id} status: {old_status} -> {request.status}")
    
    return {
        "message": "Status updated successfully",
        "vulnerability_id": vulnerability_id,
        "old_status": old_status,
        "new_status": request.status,
    }


@router.post("/scan")
async def trigger_scan(request: ScanRequest):
    """
    Trigger a vulnerability scan for a project.
    
    - **project_id**: Project to scan
    - **scan_type**: Type of scan (full, incremental, quick)
    - **include_dependencies**: Include dependency scanning
    """
    scan_id = f"scan-{uuid.uuid4().hex[:8]}"
    
    logger.info(f"Vulnerability scan triggered: {scan_id} for project {request.project_id}")
    
    return {
        "scan_id": scan_id,
        "project_id": request.project_id,
        "status": "queued",
        "estimated_duration_minutes": 5 if request.scan_type == "quick" else 15,
        "message": "Scan queued successfully",
    }


@router.get("/scan/{scan_id}/status")
async def get_scan_status(scan_id: str):
    """
    Get status of a vulnerability scan.
    """
    # Mock scan progress
    return {
        "scan_id": scan_id,
        "status": "completed",
        "progress": 100,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "vulnerabilities_found": len(MOCK_VULNERABILITIES),
        "files_scanned": 245,
        "dependencies_scanned": 128,
    }


# =============================================================================
# Auto-Fix Endpoints
# =============================================================================

@router.get("/{vulnerability_id}/fixes", response_model=List[AutoFix])
async def get_auto_fixes(vulnerability_id: str):
    """
    Get available auto-fix suggestions for a vulnerability.
    """
    fixes = [f for f in MOCK_AUTO_FIXES.values() if f.vulnerability_id == vulnerability_id]
    return fixes


@router.post("/{vulnerability_id}/fixes/generate")
async def generate_auto_fix(vulnerability_id: str):
    """
    Generate AI-powered auto-fix suggestions for a vulnerability.
    """
    vuln = MOCK_VULNERABILITIES.get(vulnerability_id)
    if not vuln:
        raise HTTPException(status_code=404, detail="Vulnerability not found")
    
    fix_id = f"fix-{uuid.uuid4().hex[:8]}"
    
    # Mock AI-generated fix
    return {
        "fix_id": fix_id,
        "vulnerability_id": vulnerability_id,
        "status": "generating",
        "message": "AI is analyzing the vulnerability and generating fix suggestions",
        "estimated_time_seconds": 30,
    }


@router.post("/fixes/{fix_id}/apply")
async def apply_auto_fix(fix_id: str, dry_run: bool = True):
    """
    Apply an auto-fix to the codebase.
    
    - **fix_id**: ID of the fix to apply
    - **dry_run**: If true, only simulate the fix without applying
    """
    fix = MOCK_AUTO_FIXES.get(fix_id)
    if not fix:
        raise HTTPException(status_code=404, detail="Auto-fix not found")
    
    if dry_run:
        return {
            "status": "simulated",
            "fix_id": fix_id,
            "changes": [
                {
                    "file": MOCK_VULNERABILITIES[fix.vulnerability_id].file_path,
                    "line": MOCK_VULNERABILITIES[fix.vulnerability_id].line_number,
                    "original": fix.original_code,
                    "fixed": fix.fixed_code,
                }
            ],
            "message": "Dry run completed. Review changes before applying.",
        }
    
    logger.info(f"Auto-fix applied: {fix_id}")
    
    return {
        "status": "applied",
        "fix_id": fix_id,
        "message": "Fix applied successfully. Please run tests to verify.",
        "vulnerability_status": "resolved",
    }


@router.post("/fixes/{fix_id}/reject")
async def reject_auto_fix(fix_id: str, reason: Optional[str] = None):
    """
    Reject an auto-fix suggestion.
    """
    fix = MOCK_AUTO_FIXES.get(fix_id)
    if not fix:
        raise HTTPException(status_code=404, detail="Auto-fix not found")
    
    return {
        "status": "rejected",
        "fix_id": fix_id,
        "reason": reason or "Rejected by user",
        "message": "Fix suggestion rejected and recorded for AI improvement",
    }
