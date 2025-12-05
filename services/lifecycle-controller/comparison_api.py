"""
Comparison API Endpoints for Lifecycle Controller

Provides endpoints for the admin Version Comparison UI to fetch
shadow comparison data, trigger rollbacks, and view audit history.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comparison-requests", tags=["comparison"])


# Data Models
class IssueData(BaseModel):
    id: str
    type: str
    severity: str
    message: str
    file: str
    line: int
    suggestion: Optional[str] = None


class VersionOutput(BaseModel):
    version: str
    versionId: str
    modelVersion: str
    promptVersion: str
    timestamp: str
    latencyMs: int
    cost: float
    issues: List[IssueData]
    rawOutput: str
    confidence: float
    securityPassed: bool


class ComparisonRequest(BaseModel):
    requestId: str
    code: str
    language: str
    timestamp: str
    v1Output: Optional[VersionOutput] = None
    v2Output: Optional[VersionOutput] = None
    v3Output: Optional[VersionOutput] = None


class RollbackRequest(BaseModel):
    versionId: str
    reason: str
    notes: str


class RollbackResponse(BaseModel):
    success: bool
    message: str
    rollbackId: str


# In-memory storage for demo (replace with database in production)
comparison_store: Dict[str, ComparisonRequest] = {}
rollback_history: List[Dict[str, Any]] = []


@router.get("", response_model=Dict[str, Any])
async def get_comparison_requests(
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    language: Optional[str] = None,
    has_v1: Optional[bool] = None,
    has_v3: Optional[bool] = None,
):
    """
    Get list of comparison requests with shadow traffic results.
    
    Returns requests where V1 shadow and V2 production outputs
    can be compared side-by-side.
    """
    # Filter requests
    filtered = list(comparison_store.values())
    
    if language:
        filtered = [r for r in filtered if r.language == language]
    
    if has_v1 is not None:
        filtered = [r for r in filtered if (r.v1Output is not None) == has_v1]
    
    if has_v3 is not None:
        filtered = [r for r in filtered if (r.v3Output is not None) == has_v3]
    
    # Sort by timestamp descending
    filtered.sort(key=lambda r: r.timestamp, reverse=True)
    
    # Paginate
    total = len(filtered)
    paginated = filtered[offset:offset + limit]
    
    return {
        "requests": [r.dict() for r in paginated],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/{request_id}", response_model=ComparisonRequest)
async def get_comparison_request(request_id: str):
    """Get a specific comparison request by ID."""
    if request_id not in comparison_store:
        raise HTTPException(status_code=404, detail="Request not found")
    
    return comparison_store[request_id]


@router.post("", response_model=ComparisonRequest)
async def create_comparison_request(request: ComparisonRequest):
    """
    Store a new comparison request.
    
    Called by the shadow traffic handler when both V1 and V2
    responses are available for a request.
    """
    comparison_store[request.requestId] = request
    
    # Trim old entries if store gets too large
    if len(comparison_store) > 10000:
        oldest_keys = sorted(
            comparison_store.keys(),
            key=lambda k: comparison_store[k].timestamp
        )[:1000]
        for key in oldest_keys:
            del comparison_store[key]
    
    return request


@router.post("/{request_id}/v1", response_model=ComparisonRequest)
async def add_v1_output(request_id: str, output: VersionOutput):
    """Add V1 shadow output to an existing request."""
    if request_id not in comparison_store:
        raise HTTPException(status_code=404, detail="Request not found")
    
    comparison_store[request_id].v1Output = output
    return comparison_store[request_id]


@router.post("/{request_id}/v3", response_model=ComparisonRequest)
async def add_v3_output(request_id: str, output: VersionOutput):
    """Add V3 comparison output to an existing request."""
    if request_id not in comparison_store:
        raise HTTPException(status_code=404, detail="Request not found")
    
    comparison_store[request_id].v3Output = output
    return comparison_store[request_id]


# Rollback endpoints
rollback_router = APIRouter(prefix="/rollback", tags=["rollback"])


@rollback_router.post("", response_model=RollbackResponse)
async def initiate_rollback(request: RollbackRequest):
    """
    Initiate a rollback for a specific version.
    
    This will:
    1. Abort any ongoing Argo Rollout
    2. Move the version to V3 quarantine
    3. Log the rollback event for audit
    """
    rollback_id = str(uuid4())[:8]
    
    try:
        # Log the rollback
        rollback_event = {
            "rollbackId": rollback_id,
            "versionId": request.versionId,
            "reason": request.reason,
            "notes": request.notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "initiated"
        }
        rollback_history.append(rollback_event)
        
        # In production, this would call:
        # 1. kubectl argo rollouts abort
        # 2. lifecycle controller downgrade API
        # 3. Slack/PagerDuty notification
        
        logger.info(f"Rollback initiated: {rollback_id} for version {request.versionId}")
        
        return RollbackResponse(
            success=True,
            message=f"Rollback initiated for version {request.versionId}",
            rollbackId=rollback_id
        )
    
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@rollback_router.get("/history", response_model=List[Dict[str, Any]])
async def get_rollback_history(
    limit: int = Query(default=50, le=100),
    version_id: Optional[str] = None
):
    """Get rollback history, optionally filtered by version."""
    history = rollback_history
    
    if version_id:
        history = [r for r in history if r["versionId"] == version_id]
    
    # Sort by timestamp descending
    history.sort(key=lambda r: r["timestamp"], reverse=True)
    
    return history[:limit]


# Audit endpoints
audit_router = APIRouter(prefix="/audit", tags=["audit"])


@audit_router.post("/opa-decision")
async def log_opa_decision(
    version_id: str,
    decision: str,
    reason: str,
    github_run_id: str,
    triggered_by: str
):
    """Log OPA policy decision for audit trail."""
    event = {
        "type": "opa_decision",
        "version_id": version_id,
        "decision": decision,
        "reason": reason,
        "github_run_id": github_run_id,
        "triggered_by": triggered_by,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # In production, store in database
    logger.info(f"OPA Decision logged: {event}")
    
    return {"status": "logged", "event": event}


@audit_router.post("/rollback")
async def log_rollback_event(
    version_id: str,
    reason: str,
    notes: str,
    triggered_by: str
):
    """Log rollback event for audit trail."""
    event = {
        "type": "rollback",
        "version_id": version_id,
        "reason": reason,
        "notes": notes,
        "triggered_by": triggered_by,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    logger.info(f"Rollback logged: {event}")
    
    return {"status": "logged", "event": event}


# Statistics endpoints
stats_router = APIRouter(prefix="/stats", tags=["statistics"])


@stats_router.get("/comparison")
async def get_comparison_stats():
    """Get statistics about comparison requests."""
    total = len(comparison_store)
    
    with_v1 = sum(1 for r in comparison_store.values() if r.v1Output)
    with_v3 = sum(1 for r in comparison_store.values() if r.v3Output)
    
    languages = {}
    for r in comparison_store.values():
        languages[r.language] = languages.get(r.language, 0) + 1
    
    # Calculate agreement metrics if both V1 and V2 present
    agreement_samples = [
        r for r in comparison_store.values()
        if r.v1Output and r.v2Output
    ]
    
    if agreement_samples:
        # Simple issue count agreement
        issue_deltas = [
            abs(len(r.v1Output.issues) - len(r.v2Output.issues))
            for r in agreement_samples
        ]
        avg_issue_delta = sum(issue_deltas) / len(issue_deltas)
        
        # Latency comparison
        latency_deltas = [
            r.v1Output.latencyMs - r.v2Output.latencyMs
            for r in agreement_samples
        ]
        avg_latency_delta = sum(latency_deltas) / len(latency_deltas)
    else:
        avg_issue_delta = 0
        avg_latency_delta = 0
    
    return {
        "total_requests": total,
        "with_v1_output": with_v1,
        "with_v3_output": with_v3,
        "languages": languages,
        "comparison_metrics": {
            "samples": len(agreement_samples),
            "avg_issue_count_delta": avg_issue_delta,
            "avg_latency_delta_ms": avg_latency_delta
        }
    }


@stats_router.get("/rollbacks")
async def get_rollback_stats():
    """Get statistics about rollbacks."""
    total = len(rollback_history)
    
    # Group by reason
    by_reason = {}
    for r in rollback_history:
        reason = r["reason"]
        by_reason[reason] = by_reason.get(reason, 0) + 1
    
    # Recent rollbacks (last 7 days)
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    recent = sum(1 for r in rollback_history if r["timestamp"] > week_ago)
    
    return {
        "total_rollbacks": total,
        "last_7_days": recent,
        "by_reason": by_reason
    }
