"""
Auto-Fix Service API

FastAPI endpoints for the automated vulnerability fix cycle.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/auto-fix", tags=["auto-fix"])


# =============================================================================
# Models
# =============================================================================

class FixPhase(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    APPLYING = "applying"
    VERIFYING = "verifying"
    LEARNING = "learning"


class FixStrategy(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityInfo(BaseModel):
    vuln_id: str
    pattern_id: str
    file_path: str
    line_number: int
    code_snippet: str
    severity: Severity
    category: str
    description: str
    detected_at: datetime
    fix_suggestion: str
    confidence: float


class FixInfo(BaseModel):
    fix_id: str
    vuln_id: str
    file_path: str
    original_code: str
    fixed_code: str
    fix_description: str
    status: str
    applied_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    confidence: float


class CycleConfig(BaseModel):
    scan_interval_seconds: int = 3600
    min_confidence: float = 0.85
    max_concurrent_fixes: int = 5
    auto_apply: bool = False
    strategy: FixStrategy = FixStrategy.BALANCED


class CycleMetrics(BaseModel):
    cycles_completed: int = 0
    total_scans: int = 0
    vulnerabilities_detected: int = 0
    fixes_generated: int = 0
    fixes_applied: int = 0
    fixes_verified: int = 0
    fixes_failed: int = 0
    fixes_rolled_back: int = 0


class CycleStatus(BaseModel):
    running: bool
    phase: FixPhase
    pending_fixes: int
    metrics: CycleMetrics
    last_cycle_at: Optional[datetime] = None


# =============================================================================
# Mock Data Store
# =============================================================================

_cycle_status = CycleStatus(
    running=True,
    phase=FixPhase.IDLE,
    pending_fixes=2,
    metrics=CycleMetrics(
        cycles_completed=15,
        total_scans=18,
        vulnerabilities_detected=25,
        fixes_generated=20,
        fixes_applied=15,
        fixes_verified=12,
        fixes_failed=3,
        fixes_rolled_back=2,
    ),
    last_cycle_at=datetime.now(timezone.utc),
)

_vulnerabilities: List[VulnerabilityInfo] = [
    VulnerabilityInfo(
        vuln_id="vuln-001",
        pattern_id="SEC-001",
        file_path="backend/shared/security/auth.py",
        line_number=19,
        code_snippet='SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default")',
        severity=Severity.CRITICAL,
        category="security",
        description="Hardcoded default secret key detected",
        detected_at=datetime.now(timezone.utc),
        fix_suggestion='Remove default fallback and raise error if not set',
        confidence=0.95,
    ),
    VulnerabilityInfo(
        vuln_id="vuln-002",
        pattern_id="REL-001",
        file_path="backend/shared/services/reliability.py",
        line_number=41,
        code_snippet='expire = datetime.utcnow() + expires_delta',
        severity=Severity.MEDIUM,
        category="reliability",
        description="Using deprecated datetime.utcnow()",
        detected_at=datetime.now(timezone.utc),
        fix_suggestion='Use datetime.now(timezone.utc) instead',
        confidence=0.90,
    ),
    VulnerabilityInfo(
        vuln_id="vuln-003",
        pattern_id="SEC-003",
        file_path="backend/shared/security/auth.py",
        line_number=97,
        code_snippet='payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])',
        severity=Severity.HIGH,
        category="security",
        description="JWT decode without proper validation options",
        detected_at=datetime.now(timezone.utc),
        fix_suggestion='Add audience, issuer, and required claims validation',
        confidence=0.88,
    ),
]

_fixes: List[FixInfo] = [
    FixInfo(
        fix_id="fix-001",
        vuln_id="vuln-001",
        file_path="backend/shared/security/auth.py",
        original_code='SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default")',
        fixed_code='SECRET_KEY = os.getenv("JWT_SECRET_KEY")\nif not SECRET_KEY:\n    raise ValueError("JWT_SECRET_KEY must be set")',
        fix_description="Remove default fallback for secret key",
        status="verified",
        applied_at=datetime.now(timezone.utc),
        verified_at=datetime.now(timezone.utc),
        confidence=0.95,
    ),
    FixInfo(
        fix_id="fix-002",
        vuln_id="vuln-002",
        file_path="backend/shared/services/reliability.py",
        original_code='datetime.utcnow()',
        fixed_code='datetime.now(timezone.utc)',
        fix_description="Replace deprecated datetime.utcnow()",
        status="applied",
        applied_at=datetime.now(timezone.utc),
        verified_at=None,
        confidence=0.90,
    ),
]

_pending_fixes: List[FixInfo] = [
    FixInfo(
        fix_id="fix-003",
        vuln_id="vuln-003",
        file_path="backend/shared/security/auth.py",
        original_code='payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])',
        fixed_code='payload = jwt.decode(\n    token, SECRET_KEY, algorithms=[ALGORITHM],\n    options={"require": ["exp", "sub"], "verify_aud": True, "verify_iss": True}\n)',
        fix_description="Add JWT validation options",
        status="pending",
        applied_at=None,
        verified_at=None,
        confidence=0.88,
    ),
]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/status", response_model=CycleStatus)
async def get_status():
    """Get current auto-fix cycle status."""
    return _cycle_status


@router.post("/start")
async def start_cycle(background_tasks: BackgroundTasks):
    """Start the auto-fix cycle."""
    global _cycle_status
    if _cycle_status.running:
        return {"success": False, "message": "Cycle already running"}
    
    _cycle_status.running = True
    _cycle_status.phase = FixPhase.SCANNING
    
    return {"success": True, "message": "Auto-fix cycle started"}


@router.post("/stop")
async def stop_cycle():
    """Stop the auto-fix cycle."""
    global _cycle_status
    if not _cycle_status.running:
        return {"success": False, "message": "Cycle not running"}
    
    _cycle_status.running = False
    _cycle_status.phase = FixPhase.IDLE
    
    return {"success": True, "message": "Auto-fix cycle stopped"}


@router.post("/scan")
async def trigger_scan(background_tasks: BackgroundTasks):
    """Trigger an immediate vulnerability scan."""
    global _cycle_status
    
    _cycle_status.phase = FixPhase.SCANNING
    _cycle_status.metrics.total_scans += 1
    
    # Simulated scan result
    new_vulns = 1
    _cycle_status.metrics.vulnerabilities_detected += new_vulns
    _cycle_status.phase = FixPhase.IDLE
    
    return {
        "success": True,
        "message": f"Scan complete. Found {new_vulns} new vulnerabilities.",
        "vulnerabilities_found": new_vulns,
    }


@router.get("/vulnerabilities", response_model=List[VulnerabilityInfo])
async def list_vulnerabilities(
    severity: Optional[Severity] = None,
    category: Optional[str] = None,
):
    """List detected vulnerabilities."""
    vulns = _vulnerabilities
    
    if severity:
        vulns = [v for v in vulns if v.severity == severity]
    if category:
        vulns = [v for v in vulns if v.category == category]
    
    return vulns


@router.get("/vulnerabilities/{vuln_id}", response_model=VulnerabilityInfo)
async def get_vulnerability(vuln_id: str):
    """Get a specific vulnerability."""
    for v in _vulnerabilities:
        if v.vuln_id == vuln_id:
            return v
    raise HTTPException(status_code=404, detail="Vulnerability not found")


@router.get("/fixes", response_model=List[FixInfo])
async def list_fixes(status: Optional[str] = None):
    """List all fixes."""
    all_fixes = _fixes + _pending_fixes
    
    if status:
        all_fixes = [f for f in all_fixes if f.status == status]
    
    return all_fixes


@router.get("/fixes/pending", response_model=List[FixInfo])
async def list_pending_fixes():
    """List pending fixes awaiting approval."""
    return _pending_fixes


@router.post("/fixes/{fix_id}/approve")
async def approve_fix(fix_id: str):
    """Approve a pending fix for application."""
    global _cycle_status
    
    for fix in _pending_fixes:
        if fix.fix_id == fix_id:
            fix.status = "approved"
            return {"success": True, "message": "Fix approved"}
    
    raise HTTPException(status_code=404, detail="Fix not found")


@router.post("/fixes/{fix_id}/reject")
async def reject_fix(fix_id: str):
    """Reject a pending fix."""
    global _pending_fixes
    
    for i, fix in enumerate(_pending_fixes):
        if fix.fix_id == fix_id:
            _pending_fixes.pop(i)
            return {"success": True, "message": "Fix rejected"}
    
    raise HTTPException(status_code=404, detail="Fix not found")


@router.post("/fixes/{fix_id}/apply")
async def apply_fix(fix_id: str):
    """Manually apply a fix."""
    global _cycle_status
    
    for fix in _pending_fixes:
        if fix.fix_id == fix_id:
            fix.status = "applied"
            fix.applied_at = datetime.now(timezone.utc)
            _fixes.append(fix)
            _pending_fixes.remove(fix)
            _cycle_status.metrics.fixes_applied += 1
            return {"success": True, "message": "Fix applied"}
    
    raise HTTPException(status_code=404, detail="Fix not found")


@router.post("/fixes/{fix_id}/rollback")
async def rollback_fix(fix_id: str):
    """Rollback an applied fix."""
    global _cycle_status
    
    for fix in _fixes:
        if fix.fix_id == fix_id:
            if fix.status != "applied" and fix.status != "verified":
                raise HTTPException(status_code=400, detail="Fix not applied")
            
            fix.status = "rolled_back"
            _cycle_status.metrics.fixes_rolled_back += 1
            return {"success": True, "message": "Fix rolled back"}
    
    raise HTTPException(status_code=404, detail="Fix not found")


@router.get("/config", response_model=CycleConfig)
async def get_config():
    """Get current cycle configuration."""
    return CycleConfig()


@router.put("/config")
async def update_config(config: CycleConfig):
    """Update cycle configuration."""
    # In production, persist to database
    return {"success": True, "message": "Configuration updated", "config": config}


@router.get("/summary")
async def get_summary():
    """Get summary of vulnerabilities and fixes."""
    severity_counts = {}
    for v in _vulnerabilities:
        sev = v.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    category_counts = {}
    for v in _vulnerabilities:
        cat = v.category
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    fix_status_counts = {}
    for f in _fixes + _pending_fixes:
        status = f.status
        fix_status_counts[status] = fix_status_counts.get(status, 0) + 1
    
    return {
        "total_vulnerabilities": len(_vulnerabilities),
        "by_severity": severity_counts,
        "by_category": category_counts,
        "total_fixes": len(_fixes) + len(_pending_fixes),
        "by_fix_status": fix_status_counts,
        "pending_fixes": len(_pending_fixes),
        "cycle_status": _cycle_status.phase.value,
    }


@router.get("/history")
async def get_cycle_history(limit: int = 10):
    """Get recent cycle history."""
    # Mock history
    return {
        "cycles": [
            {
                "cycle": 15,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": 45,
                "vulnerabilities_found": 3,
                "fixes_applied": 1,
                "fixes_verified": 1,
            },
            {
                "cycle": 14,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": 52,
                "vulnerabilities_found": 5,
                "fixes_applied": 2,
                "fixes_verified": 2,
            },
        ]
    }
