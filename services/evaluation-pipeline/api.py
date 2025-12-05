"""
Evaluation Pipeline API

REST API for the evaluation pipeline service.
Provides endpoints for:
- Shadow traffic output recording
- Promotion recommendation
- Gold-set evaluation
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .shadow_comparator import (
    ShadowComparator,
    AnalysisOutput,
    PromotionRecommendation,
)
from .gold_set_evaluator import GoldSetEvaluator

logger = logging.getLogger(__name__)


# ==================== Request/Response Models ====================

class AnalysisOutputRequest(BaseModel):
    version: str
    version_id: str
    request_id: str
    code: str
    language: str
    issues: List[Dict[str, Any]]
    latency_ms: float
    cost: float
    confidence: float = 0.0
    metadata: Dict[str, Any] = {}


class GoldSetRequest(BaseModel):
    version_id: str
    evaluation_type: str = "promotion"
    include_categories: Optional[List[str]] = None


class PromotionResponse(BaseModel):
    recommend_promotion: bool
    confidence: float
    reasons: List[str]
    blockers: List[str]
    metrics: Dict[str, Any]
    next_evaluation_in_hours: Optional[int] = None


class EvaluationResponse(BaseModel):
    version_id: str
    evaluation_type: str
    passed: bool
    score: float
    metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: str


# ==================== Application Setup ====================

comparator: Optional[ShadowComparator] = None
gold_evaluator: Optional[GoldSetEvaluator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global comparator, gold_evaluator
    
    comparator = ShadowComparator()
    gold_evaluator = GoldSetEvaluator()
    
    await comparator.start()
    await gold_evaluator.start()
    
    logger.info("Evaluation Pipeline started")
    
    yield
    
    await comparator.stop()
    await gold_evaluator.stop()
    
    logger.info("Evaluation Pipeline stopped")


app = FastAPI(
    title="Evaluation Pipeline API",
    description="API for shadow comparison and gold-set evaluation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "comparator": comparator is not None,
        "gold_evaluator": gold_evaluator is not None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ==================== Shadow Comparison Endpoints ====================

@app.post("/shadow/record/v1")
async def record_v1_output(request: AnalysisOutputRequest):
    """Record V1 shadow traffic output"""
    if not comparator:
        raise HTTPException(status_code=503, detail="Comparator not initialized")
    
    import hashlib
    code_hash = hashlib.sha256(request.code.encode()).hexdigest()[:16]
    
    output = AnalysisOutput(
        version=request.version,
        version_id=request.version_id,
        request_id=request.request_id,
        code_hash=code_hash,
        language=request.language,
        issues=request.issues,
        latency_ms=request.latency_ms,
        cost=request.cost,
        timestamp=datetime.now(timezone.utc),
        confidence=request.confidence,
        metadata=request.metadata
    )
    
    comparator.record_v1_output(output)
    
    return {"success": True, "code_hash": code_hash}


@app.post("/shadow/record/v2")
async def record_v2_output(request: AnalysisOutputRequest):
    """Record V2 production output"""
    if not comparator:
        raise HTTPException(status_code=503, detail="Comparator not initialized")
    
    import hashlib
    code_hash = hashlib.sha256(request.code.encode()).hexdigest()[:16]
    
    output = AnalysisOutput(
        version=request.version,
        version_id=request.version_id,
        request_id=request.request_id,
        code_hash=code_hash,
        language=request.language,
        issues=request.issues,
        latency_ms=request.latency_ms,
        cost=request.cost,
        timestamp=datetime.now(timezone.utc),
        confidence=request.confidence,
        metadata=request.metadata
    )
    
    comparator.record_v2_output(output)
    
    return {"success": True, "code_hash": code_hash}


@app.get("/shadow/status")
async def get_shadow_status():
    """Get shadow comparison status"""
    if not comparator:
        raise HTTPException(status_code=503, detail="Comparator not initialized")
    
    return comparator.get_status()


@app.post("/shadow/start-evaluation/{version_id}")
async def start_evaluation(version_id: str):
    """Start evaluation tracking for a version"""
    if not comparator:
        raise HTTPException(status_code=503, detail="Comparator not initialized")
    
    comparator.start_evaluation(version_id)
    
    return {
        "success": True,
        "message": f"Evaluation started for {version_id}"
    }


@app.get("/shadow/recommendation/{version_id}", response_model=PromotionResponse)
async def get_promotion_recommendation(version_id: str):
    """Get promotion recommendation for a V1 version"""
    if not comparator:
        raise HTTPException(status_code=503, detail="Comparator not initialized")
    
    recommendation = comparator.evaluate_promotion(version_id)
    
    return PromotionResponse(
        recommend_promotion=recommendation.recommend_promotion,
        confidence=recommendation.confidence,
        reasons=recommendation.reasons,
        blockers=recommendation.blockers,
        metrics={
            "total_pairs": recommendation.metrics.total_pairs,
            "complete_pairs": recommendation.metrics.complete_pairs,
            "accuracy_delta": recommendation.metrics.accuracy_delta,
            "latency_improvement_pct": recommendation.metrics.latency_improvement_pct,
            "cost_delta_pct": recommendation.metrics.cost_delta_pct,
            "agreement_rate": recommendation.metrics.agreement_rate,
            "v1_p95_latency_ms": recommendation.metrics.v1_p95_latency_ms,
            "v2_p95_latency_ms": recommendation.metrics.v2_p95_latency_ms,
            "is_statistically_significant": recommendation.metrics.is_statistically_significant,
        },
        next_evaluation_in_hours=recommendation.next_evaluation_in_hours
    )


# ==================== Gold-Set Evaluation Endpoints ====================

@app.post("/evaluate/gold-set", response_model=EvaluationResponse)
async def run_gold_set_evaluation(request: GoldSetRequest):
    """Run gold-set evaluation for a version"""
    if not gold_evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    report = await gold_evaluator.evaluate(
        version_id=request.version_id,
        evaluation_type=request.evaluation_type,
        include_categories=request.include_categories
    )
    
    return EvaluationResponse(
        version_id=report.version_id,
        evaluation_type=report.evaluation_type,
        passed=report.passed,
        score=report.score,
        metrics=report.metrics,
        recommendations=report.recommendations,
        timestamp=report.timestamp
    )


@app.get("/evaluate/gold-set/categories")
async def get_gold_set_categories():
    """Get available gold-set test categories"""
    return {
        "categories": ["security", "quality", "performance", "false_positive"],
        "description": {
            "security": "Security vulnerability detection tests",
            "quality": "Code quality issue detection tests",
            "performance": "Performance issue detection tests",
            "false_positive": "Tests that should NOT trigger issues"
        }
    }


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
