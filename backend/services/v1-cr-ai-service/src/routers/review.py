"""
Code Review API Endpoints

Main review endpoints:
- Request experimental code review
- Compare review strategies
- Detect hallucinations
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..review import ReviewEngine, ReviewResult, Finding
from ..hallucination import HallucinationDetector

router = APIRouter(prefix="/review", tags=["review"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ReviewRequest(BaseModel):
    """Request for code review"""
    code: str = Field(..., min_length=1, max_length=100000)
    language: str = Field(default="python", pattern="^(python|typescript|javascript|java|go|rust|cpp|c|csharp)$")
    context: Optional[str] = Field(default=None, max_length=50000)
    dimensions: List[str] = Field(
        default=["correctness", "security", "performance"],
        description="Review dimensions to check"
    )
    strategy: str = Field(
        default="chain_of_thought",
        pattern="^(baseline|chain_of_thought|few_shot|contrastive|ensemble)$"
    )
    model_variant: Optional[str] = None


class FindingResponse(BaseModel):
    """A single finding from review"""
    dimension: str
    issue: str
    line_numbers: List[int]
    severity: str
    confidence: float
    suggestion: str
    explanation: str
    cwe_id: Optional[str] = None
    code_snippet: Optional[str] = None
    fix_snippet: Optional[str] = None
    reasoning_steps: List[str] = []


class ReviewResponse(BaseModel):
    """Response from code review"""
    review_id: str
    findings: List[FindingResponse]
    overall_score: float
    dimension_scores: Dict[str, float]
    processing_time_ms: float
    model_version: str
    strategy_used: str
    avg_confidence: float
    status: str


class CompareStrategiesRequest(BaseModel):
    """Request to compare multiple strategies"""
    code: str = Field(..., min_length=1, max_length=100000)
    language: str = Field(default="python")
    strategies: List[str] = Field(
        default=["baseline", "chain_of_thought", "few_shot"],
        min_items=2,
        max_items=5
    )
    dimensions: List[str] = Field(default=["correctness", "security"])


class StrategyResult(BaseModel):
    """Result from a single strategy"""
    strategy: str
    findings_count: int
    overall_score: float
    processing_time_ms: float
    avg_confidence: float
    findings: List[FindingResponse]


class CompareStrategiesResponse(BaseModel):
    """Response comparing strategies"""
    results: Dict[str, StrategyResult]
    winner: str
    recommendation: str
    comparison_time_ms: float


class HallucinationCheckRequest(BaseModel):
    """Request to check for hallucinations"""
    code: str = Field(..., min_length=1)
    review_id: str
    run_consistency_check: bool = True


class HallucinationCheckResponse(BaseModel):
    """Response from hallucination check"""
    hallucination_detected: bool
    confidence: float
    problematic_findings: List[int]
    explanation: str
    consistency_score: float
    mitigations_applied: List[str]


# =============================================================================
# Global Engine Instance
# =============================================================================

_review_engine: Optional[ReviewEngine] = None
_hallucination_detector: Optional[HallucinationDetector] = None
_reviews_cache: Dict[str, ReviewResult] = {}


def get_review_engine() -> ReviewEngine:
    global _review_engine
    if _review_engine is None:
        _review_engine = ReviewEngine()
    return _review_engine


def get_hallucination_detector() -> HallucinationDetector:
    global _hallucination_detector
    if _hallucination_detector is None:
        _hallucination_detector = HallucinationDetector()
    return _hallucination_detector


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", response_model=ReviewResponse)
async def request_review(request: ReviewRequest):
    """
    Request experimental code review.
    
    Performs multi-dimensional code analysis using the specified strategy.
    Returns detailed findings with confidence scores and suggestions.
    """
    engine = get_review_engine()
    
    try:
        result = await engine.review(
            code=request.code,
            language=request.language,
            dimensions=request.dimensions,
            strategy=request.strategy,
            context=request.context,
        )
        
        # Cache for hallucination checking
        _reviews_cache[result.review_id] = result
        
        return ReviewResponse(
            review_id=result.review_id,
            findings=[
                FindingResponse(
                    dimension=f.dimension,
                    issue=f.issue,
                    line_numbers=f.line_numbers,
                    severity=f.severity,
                    confidence=f.confidence,
                    suggestion=f.suggestion,
                    explanation=f.explanation,
                    cwe_id=f.cwe_id,
                    code_snippet=f.code_snippet,
                    fix_snippet=f.fix_snippet,
                    reasoning_steps=f.reasoning_steps,
                )
                for f in result.findings
            ],
            overall_score=result.overall_score,
            dimension_scores=result.dimension_scores,
            processing_time_ms=result.processing_time_ms,
            model_version=result.model_version,
            strategy_used=result.strategy_used,
            avg_confidence=result.avg_confidence,
            status=result.status.value,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Review failed: {str(e)}")


@router.post("/compare-strategies", response_model=CompareStrategiesResponse)
async def compare_strategies(request: CompareStrategiesRequest):
    """
    Compare multiple review strategies on the same code.
    
    Runs each strategy and compares results to identify
    the best approach for the given code.
    """
    import time
    start_time = time.time()
    
    engine = get_review_engine()
    results: Dict[str, StrategyResult] = {}
    
    for strategy in request.strategies:
        result = await engine.review(
            code=request.code,
            language=request.language,
            dimensions=request.dimensions,
            strategy=strategy,
            use_cache=False,
        )
        
        results[strategy] = StrategyResult(
            strategy=strategy,
            findings_count=len(result.findings),
            overall_score=result.overall_score,
            processing_time_ms=result.processing_time_ms,
            avg_confidence=result.avg_confidence,
            findings=[
                FindingResponse(
                    dimension=f.dimension,
                    issue=f.issue,
                    line_numbers=f.line_numbers,
                    severity=f.severity,
                    confidence=f.confidence,
                    suggestion=f.suggestion,
                    explanation=f.explanation,
                    cwe_id=f.cwe_id,
                    code_snippet=f.code_snippet,
                    fix_snippet=f.fix_snippet,
                    reasoning_steps=f.reasoning_steps,
                )
                for f in result.findings
            ],
        )
    
    # Determine winner
    # Prefer: higher confidence, more findings, lower latency
    def score_strategy(s: StrategyResult) -> float:
        return (
            s.avg_confidence * 0.4 +
            min(s.findings_count / 10, 1.0) * 0.3 +
            (1 - min(s.processing_time_ms / 1000, 1.0)) * 0.1 +
            (100 - s.overall_score) / 100 * 0.2  # More issues found = better (for testing)
        )
    
    best_strategy = max(results.keys(), key=lambda k: score_strategy(results[k]))
    
    # Generate recommendation
    best = results[best_strategy]
    if best.avg_confidence > 0.85:
        recommendation = f"{best_strategy} provides high-confidence reviews."
    elif best.findings_count > len(results) * 2:
        recommendation = f"{best_strategy} finds the most issues."
    else:
        recommendation = f"{best_strategy} offers the best balance of accuracy and speed."
    
    return CompareStrategiesResponse(
        results=results,
        winner=best_strategy,
        recommendation=recommendation,
        comparison_time_ms=(time.time() - start_time) * 1000,
    )


@router.post("/detect-hallucination", response_model=HallucinationCheckResponse)
async def detect_hallucination(request: HallucinationCheckRequest):
    """
    Check if a review contains hallucinations.
    
    Verifies findings against the actual code and checks
    for consistency across multiple review runs.
    """
    detector = get_hallucination_detector()
    engine = get_review_engine() if request.run_consistency_check else None
    
    # Get cached review
    if request.review_id not in _reviews_cache:
        raise HTTPException(status_code=404, detail="Review not found")
    
    review_result = _reviews_cache[request.review_id]
    
    result = await detector.detect(
        code=request.code,
        review_result=review_result,
        review_engine=engine,
        run_consistency_check=request.run_consistency_check,
    )
    
    return HallucinationCheckResponse(
        hallucination_detected=result.hallucination_detected,
        confidence=result.confidence,
        problematic_findings=result.problematic_findings,
        explanation=result.explanation,
        consistency_score=result.consistency_score,
        mitigations_applied=result.mitigations_applied,
    )


@router.get("/{review_id}")
async def get_review(review_id: str):
    """Get a previously completed review."""
    if review_id not in _reviews_cache:
        raise HTTPException(status_code=404, detail="Review not found")
    
    result = _reviews_cache[review_id]
    return result.to_dict()


@router.get("/dimensions")
async def list_dimensions():
    """List available review dimensions."""
    return {
        "dimensions": [
            {
                "name": "correctness",
                "description": "Logic errors, boundary conditions, null safety",
            },
            {
                "name": "security",
                "description": "SQL injection, XSS, authentication flaws",
            },
            {
                "name": "performance",
                "description": "Algorithmic complexity, memory allocation",
            },
            {
                "name": "maintainability",
                "description": "Code complexity, naming, documentation",
            },
            {
                "name": "architecture",
                "description": "Design patterns, SOLID principles",
            },
            {
                "name": "testing",
                "description": "Test coverage, test quality",
            },
        ]
    }


@router.get("/strategies")
async def list_strategies():
    """List available review strategies."""
    return {
        "strategies": [
            {
                "name": "baseline",
                "description": "Direct instruction-tuned review",
                "latency": "fast",
            },
            {
                "name": "chain_of_thought",
                "description": "Step-by-step reasoning decomposition",
                "latency": "medium",
            },
            {
                "name": "few_shot",
                "description": "In-context learning with examples",
                "latency": "medium",
            },
            {
                "name": "contrastive",
                "description": "Compare correct vs buggy versions",
                "latency": "slow",
            },
            {
                "name": "ensemble",
                "description": "Combine multiple strategies with voting",
                "latency": "slow",
            },
        ]
    }
