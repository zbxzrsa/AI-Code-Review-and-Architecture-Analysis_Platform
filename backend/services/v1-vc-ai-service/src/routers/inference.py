"""
Inference API Endpoints

Handles commit analysis inference:
- Analyze commits with experimental model
- Batch processing
- Real-time analysis
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..tracking import CommitAnalyzer, ChangeType, ImpactLevel

router = APIRouter(prefix="/inference", tags=["inference"])


# =============================================================================
# Request/Response Models
# =============================================================================

class AnalyzeCommitRequest(BaseModel):
    """Request to analyze a single commit"""
    commit_hash: str = Field(..., min_length=7, max_length=40)
    message: str = Field(..., min_length=1)
    diff: str = Field(..., min_length=1)
    experiment_id: Optional[str] = None  # Use specific experiment model


class AnalyzeCommitResponse(BaseModel):
    """Response from commit analysis"""
    commit_hash: str
    change_type: str
    change_type_confidence: float
    impact_level: str
    impact_confidence: float
    affected_modules: List[str]
    affected_functions: List[str]
    affected_classes: List[str]
    explanation: str
    key_changes: List[str]
    risk_score: float
    risk_factors: List[str]
    total_additions: int
    total_deletions: int
    analysis_time_ms: float


class BatchAnalyzeRequest(BaseModel):
    """Request to analyze multiple commits"""
    commits: List[AnalyzeCommitRequest] = Field(..., max_items=100)
    experiment_id: Optional[str] = None


class BatchAnalyzeResponse(BaseModel):
    """Response from batch analysis"""
    total: int
    successful: int
    failed: int
    results: List[AnalyzeCommitResponse]
    errors: List[Dict[str, str]]
    total_time_ms: float


class GenerateMessageRequest(BaseModel):
    """Request to generate commit message from diff"""
    diff: str = Field(..., min_length=1)
    style: str = Field(default="conventional", pattern="^(conventional|descriptive|brief)$")
    max_length: int = Field(default=72, ge=20, le=200)


class GenerateMessageResponse(BaseModel):
    """Response with generated commit message"""
    message: str
    confidence: float
    alternatives: List[str]


# =============================================================================
# Analyzer Instance
# =============================================================================

# Global analyzer instance (in production, use dependency injection)
_analyzer: Optional[CommitAnalyzer] = None


def get_analyzer() -> CommitAnalyzer:
    """Get or create analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = CommitAnalyzer()
    return _analyzer


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/analyze-commit", response_model=AnalyzeCommitResponse)
async def analyze_commit(request: AnalyzeCommitRequest):
    """
    Analyze a single commit with the experimental model.
    
    Returns detailed analysis including:
    - Change type classification
    - Impact level prediction
    - Affected components
    - Risk assessment
    - Explanation
    """
    analyzer = get_analyzer()
    
    try:
        result = await analyzer.analyze_commit(
            commit_hash=request.commit_hash,
            message=request.message,
            diff=request.diff,
            use_model=request.experiment_id is not None,
        )
        
        return AnalyzeCommitResponse(
            commit_hash=result.commit_hash,
            change_type=result.change_type.value,
            change_type_confidence=result.change_type_confidence,
            impact_level=result.impact_level.value,
            impact_confidence=result.impact_confidence,
            affected_modules=result.modules_affected,
            affected_functions=result.functions_affected,
            affected_classes=result.classes_affected,
            explanation=result.explanation,
            key_changes=result.key_changes,
            risk_score=result.risk_score,
            risk_factors=result.risk_factors,
            total_additions=result.total_additions,
            total_deletions=result.total_deletions,
            analysis_time_ms=result.analysis_time_ms,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch-analyze", response_model=BatchAnalyzeResponse)
async def batch_analyze_commits(request: BatchAnalyzeRequest):
    """
    Analyze multiple commits in batch.
    
    More efficient than individual requests for large numbers of commits.
    """
    import time
    start_time = time.time()
    
    analyzer = get_analyzer()
    results = []
    errors = []
    
    for commit_req in request.commits:
        try:
            result = await analyzer.analyze_commit(
                commit_hash=commit_req.commit_hash,
                message=commit_req.message,
                diff=commit_req.diff,
                use_model=request.experiment_id is not None,
            )
            
            results.append(AnalyzeCommitResponse(
                commit_hash=result.commit_hash,
                change_type=result.change_type.value,
                change_type_confidence=result.change_type_confidence,
                impact_level=result.impact_level.value,
                impact_confidence=result.impact_confidence,
                affected_modules=result.modules_affected,
                affected_functions=result.functions_affected,
                affected_classes=result.classes_affected,
                explanation=result.explanation,
                key_changes=result.key_changes,
                risk_score=result.risk_score,
                risk_factors=result.risk_factors,
                total_additions=result.total_additions,
                total_deletions=result.total_deletions,
                analysis_time_ms=result.analysis_time_ms,
            ))
            
        except Exception as e:
            errors.append({
                "commit_hash": commit_req.commit_hash,
                "error": str(e),
            })
    
    total_time_ms = (time.time() - start_time) * 1000
    
    return BatchAnalyzeResponse(
        total=len(request.commits),
        successful=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
        total_time_ms=total_time_ms,
    )


@router.post("/generate-message", response_model=GenerateMessageResponse)
async def generate_commit_message(request: GenerateMessageRequest):
    """
    Generate a commit message from a diff.
    
    Uses the experimental model to generate a descriptive commit message.
    """
    # Placeholder implementation - would use the actual model
    # Analyze the diff to understand what changed
    _ = get_analyzer()  # noqa: F841 - analyzer reserved for future model integration
    
    # Simple extraction of changes for message generation
    lines = request.diff.split('\n')
    additions = [l for l in lines if l.startswith('+') and not l.startswith('+++')]
    deletions = [l for l in lines if l.startswith('-') and not l.startswith('---')]
    
    # Extract file names
    files = []
    for line in lines:
        if line.startswith('diff --git'):
            import re
            match = re.search(r'a/(.+?) b/', line)
            if match:
                files.append(match.group(1))
    
    # Generate message based on style
    if request.style == "conventional":
        # Try to detect type
        if any('fix' in l.lower() for l in additions):
            prefix = "fix"
        elif any('test' in f.lower() for f in files):
            prefix = "test"
        elif any('doc' in f.lower() or 'readme' in f.lower() for f in files):
            prefix = "docs"
        else:
            prefix = "feat"
        
        scope = files[0].split('/')[0] if files else "core"
        message = f"{prefix}({scope}): update {len(files)} file(s)"
        
    elif request.style == "descriptive":
        message = f"Update {len(files)} file(s) with {len(additions)} additions and {len(deletions)} deletions"
        
    else:  # brief
        message = f"Update {files[0] if files else 'files'}"
    
    # Truncate to max length
    if len(message) > request.max_length:
        message = message[:request.max_length - 3] + "..."
    
    return GenerateMessageResponse(
        message=message,
        confidence=0.75,
        alternatives=[
            f"Modify {', '.join(files[:3])}",
            f"Change implementation in {files[0].split('/')[0] if files else 'codebase'}",
        ],
    )


@router.get("/metrics")
async def get_inference_metrics():
    """
    Get inference performance metrics.
    """
    analyzer = get_analyzer()
    metrics = analyzer.get_metrics()
    
    return {
        "total_analyzed": metrics["total_analyzed"],
        "avg_analysis_time_ms": round(metrics["avg_analysis_time_ms"], 2),
        "change_type_distribution": metrics["change_type_distribution"],
        "impact_distribution": metrics["impact_distribution"],
    }


@router.post("/metrics/reset")
async def reset_inference_metrics():
    """
    Reset inference metrics.
    """
    analyzer = get_analyzer()
    analyzer.reset_metrics()
    
    return {"message": "Metrics reset successfully"}
