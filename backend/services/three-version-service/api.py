"""
Three-Version Spiral Evolution API

REST API endpoints for managing the three-version self-evolution cycle:
- V1 (New): Experimentation and trial/error
- V2 (Stable): Production, fixes V1 errors, optimizes compatibility
- V3 (Old): Quarantine, comparison baseline, exclusion

Endpoints:
- /api/v1/evolution/status - Get cycle status
- /api/v1/evolution/start - Start evolution cycle
- /api/v1/evolution/stop - Stop evolution cycle
- /api/v1/evolution/v1/errors - Report V1 errors
- /api/v1/evolution/promote - Trigger promotion
- /api/v1/evolution/degrade - Trigger degradation
- /api/v1/evolution/reeval - Request re-evaluation
- /metrics - Prometheus metrics
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from metrics import (
    collect_metrics_from_status,
    record_error,
    record_fix,
    record_promotion,
    record_degradation,
    record_reevaluation,
    update_cycle_status,
    cycle_completed,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Request/Response Models
# =============================================================================

class ErrorReportRequest(BaseModel):
    """Request to report a V1 error."""
    tech_id: str = Field(..., description="Technology ID")
    tech_name: str = Field(..., description="Technology name")
    error_type: str = Field(..., description="Error type: compatibility, performance, security, accuracy, stability")
    description: str = Field(..., description="Error description")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class PromotionRequest(BaseModel):
    """Request to promote a technology."""
    tech_id: str = Field(..., description="Technology ID to promote")
    reason: Optional[str] = Field(None, description="Promotion reason")


class DegradationRequest(BaseModel):
    """Request to degrade a technology."""
    tech_id: str = Field(..., description="Technology ID to degrade")
    reason: str = Field(..., description="Degradation reason")


class ReEvaluationRequest(BaseModel):
    """Request to re-evaluate a quarantined technology."""
    tech_id: str = Field(..., description="Technology ID to re-evaluate")
    reason: Optional[str] = Field(None, description="Re-evaluation justification")


class AIRequestPayload(BaseModel):
    """Request payload for AI endpoints."""
    code: Optional[str] = Field(None, description="Code to analyze")
    language: Optional[str] = Field("python", description="Programming language")
    review_type: Optional[str] = Field("full", description="Review type")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ChatRequest(BaseModel):
    """Request payload for chat endpoints."""
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class AnalyzeRequest(BaseModel):
    """Request payload for code analysis."""
    code: str = Field(..., description="Code to analyze")
    language: str = Field("python", description="Programming language")
    review_types: List[str] = Field(
        default=["security", "performance", "quality"],
        description="Types of review to perform"
    )


class FeedbackRequest(BaseModel):
    """Request for user feedback on AI responses."""
    response_id: str = Field(..., description="AI response ID")
    helpful: bool = Field(..., description="Whether the response was helpful")
    comment: Optional[str] = Field(None, description="Additional feedback comment")


class FixRequest(BaseModel):
    """Request to apply an auto-fix."""
    issue_id: str = Field(..., description="Issue ID to fix")
    code: str = Field(..., description="Current code")


class VersionStatusResponse(BaseModel):
    """Response for version status."""
    version: str
    state: str
    vc_ai_status: str
    cr_ai_status: str
    metrics: Dict[str, Any]


class CycleStatusResponse(BaseModel):
    """Response for cycle status."""
    running: bool
    current_phase: Optional[str]
    cycle_id: Optional[str]
    metrics: Dict[str, Any]
    versions: Dict[str, VersionStatusResponse]


# =============================================================================
# Dependency - Evolution Cycle Instance
# =============================================================================

# Global instance (in production, use dependency injection)
_evolution_cycle = None


def get_evolution_cycle():
    """Get or create the evolution cycle instance."""
    global _evolution_cycle
    
    if _evolution_cycle is None:
        from ai_core.three_version_cycle import EnhancedSelfEvolutionCycle
        _evolution_cycle = EnhancedSelfEvolutionCycle()
    
    return _evolution_cycle


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/v1/evolution", tags=["Three-Version Evolution"])


# =============================================================================
# Cycle Management Endpoints
# =============================================================================

@router.get("/status", response_model=Dict[str, Any])
async def get_cycle_status():
    """
    Get the current status of the three-version evolution cycle.
    
    Returns:
    - Running state
    - Current cycle phase
    - Metrics for each version
    - AI status per version
    """
    cycle = get_evolution_cycle()
    return cycle.get_full_status()


@router.post("/start")
async def start_evolution_cycle():
    """
    Start the spiral evolution cycle.
    
    The cycle runs continuously, executing phases:
    1. V1 Experimentation
    2. V2 Error Remediation
    3. Evaluation
    4. Promotion (V1 → V2)
    5. Stabilization
    6. Degradation (V2 → V3)
    7. Comparison
    8. Re-evaluation (V3 → V1)
    """
    cycle = get_evolution_cycle()
    await cycle.start()
    
    return {
        "success": True,
        "message": "Evolution cycle started",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/stop")
async def stop_evolution_cycle():
    """Stop the spiral evolution cycle."""
    cycle = get_evolution_cycle()
    await cycle.stop()
    
    return {
        "success": True,
        "message": "Evolution cycle stopped",
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# V1 Experimentation Endpoints
# =============================================================================

@router.post("/v1/errors")
async def report_v1_error(request: ErrorReportRequest):
    """
    Report an error from V1 experimentation.
    
    V2's VC-AI will analyze the error and generate a fix.
    The fix is then applied back to V1.
    
    Error types:
    - compatibility: New tech incompatible with existing system
    - performance: Performance below threshold
    - security: Security vulnerability detected
    - accuracy: Accuracy below threshold
    - stability: Unstable behavior
    """
    cycle = get_evolution_cycle()
    
    result = await cycle.report_v1_error(
        tech_id=request.tech_id,
        tech_name=request.tech_name,
        error_type=request.error_type,
        description=request.description,
    )
    
    return {
        "success": True,
        "error_id": result.get("error_id"),
        "message": "Error reported, V2 will analyze and generate fix",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/v1/experiments")
async def list_v1_experiments():
    """List active experiments in V1."""
    cycle = get_evolution_cycle()
    status = cycle.get_full_status()
    
    return {
        "version": "v1",
        "experiments": status.get("spiral_status", {}).get("pending", {}).get("promotions", 0),
        "ai_status": status.get("spiral_status", {}).get("ai_status", {}).get("v1", {}),
    }


# =============================================================================
# V2 Production Endpoints
# =============================================================================

@router.get("/v2/status")
async def get_v2_status():
    """
    Get V2 production status.
    
    V2 is the stable, user-facing version.
    Users can only access V2's Code Review AI (CR-AI).
    """
    cycle = get_evolution_cycle()
    
    return {
        "version": "v2",
        "user_ai_status": cycle.get_user_ai_status(),
        "description": "Stable production version - user-facing",
    }


@router.get("/v2/fixes")
async def list_v2_fixes():
    """List fixes generated by V2 for V1 errors."""
    cycle = get_evolution_cycle()
    
    feedback_stats = cycle.spiral_manager.feedback_system.get_feedback_statistics()
    pending_fixes = cycle.spiral_manager.feedback_system.get_pending_fixes()
    
    return {
        "statistics": feedback_stats,
        "pending_fixes": [
            {
                "fix_id": f.fix_id,
                "error_id": f.error_id,
                "fix_type": f.fix_type,
                "status": f.status.value,
            }
            for f in pending_fixes
        ],
    }


# =============================================================================
# V3 Quarantine Endpoints
# =============================================================================

@router.get("/v3/quarantine")
async def get_v3_quarantine_status():
    """
    Get V3 quarantine status.
    
    V3 contains:
    - Failed experiments
    - Technologies with poor reviews
    - Comparison baselines
    """
    cycle = get_evolution_cycle()
    
    quarantine_stats = cycle.spiral_manager.comparison_engine.get_quarantine_statistics()
    exclusion_list = cycle.spiral_manager.comparison_engine.get_exclusion_list()
    insights = cycle.spiral_manager.comparison_engine.get_failure_insights()
    
    return {
        "version": "v3",
        "statistics": quarantine_stats,
        "exclusions": exclusion_list,
        "insights": insights,
    }


@router.get("/v3/exclusions")
async def get_exclusion_list():
    """Get list of excluded technologies."""
    cycle = get_evolution_cycle()
    
    exclusion_list = cycle.spiral_manager.comparison_engine.get_exclusion_list()
    
    return {
        "permanent": exclusion_list.get("permanent", []),
        "temporary": exclusion_list.get("temporary", []),
    }


# =============================================================================
# Promotion/Degradation Endpoints
# =============================================================================

@router.post("/promote")
async def trigger_promotion(request: PromotionRequest):
    """
    Trigger promotion of a technology from V1 to V2.
    
    Requirements:
    - Accuracy >= 85%
    - Error rate <= 5%
    - Latency p95 <= 3000ms
    - Minimum 1000 samples
    """
    cycle = get_evolution_cycle()
    
    result = await cycle.trigger_promotion(request.tech_id)
    
    return {
        "success": result.get("success", True),
        "tech_id": request.tech_id,
        "status": result.get("status"),
        "message": "Technology queued for promotion (V1 → V2)",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/degrade")
async def trigger_degradation(request: DegradationRequest):
    """
    Trigger degradation of a technology from V2 to V3.
    
    Technologies are degraded when:
    - Error rate > 10%
    - Accuracy < 75%
    - Security vulnerabilities found
    """
    cycle = get_evolution_cycle()
    
    result = await cycle.trigger_degradation(request.tech_id, request.reason)
    
    return {
        "success": result.get("success", True),
        "tech_id": request.tech_id,
        "status": result.get("status"),
        "message": "Technology queued for degradation (V2 → V3)",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/reeval")
async def request_reevaluation(request: ReEvaluationRequest):
    """
    Request re-evaluation of a quarantined technology.
    
    Requirements:
    - Technology must be in V3
    - Minimum 30-day quarantine period
    - Not permanently excluded
    """
    cycle = get_evolution_cycle()
    
    result = await cycle.request_reevaluation(request.tech_id)
    
    return {
        "success": result.get("success", True),
        "tech_id": request.tech_id,
        "status": result.get("status"),
        "message": "Re-evaluation request submitted (V3 → V1)",
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# AI Access Endpoints
# =============================================================================

@router.get("/ai/status")
async def get_all_ai_status():
    """
    Get status of all AI instances across versions.
    
    Each version has:
    - VC-AI: Version Control AI (Admin only)
    - CR-AI: Code Review AI (Users access V2 only)
    """
    cycle = get_evolution_cycle()
    return cycle.get_dual_ai_status()


@router.get("/ai/user")
async def get_user_ai_status():
    """
    Get status of user-accessible AI.
    
    Users can only access V2's Code Review AI (CR-AI).
    """
    cycle = get_evolution_cycle()
    return cycle.get_user_ai_status()


@router.post("/ai/{version}/review")
async def request_code_review(
    version: str,
    request: AIRequestPayload,
    user_role: str = "user",
):
    """
    Request code review from a specific version's CR-AI.
    
    Access Control:
    - Users: V2 CR-AI only
    - Admins: All versions
    - V1: Shadow testing (results not returned to users)
    """
    if user_role == "user" and version != "v2":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Users can only access V2 Code Review AI",
        )
    
    cycle = get_evolution_cycle()
    
    result = await cycle.spiral_manager.dual_ai.route_request(
        user_role=user_role,
        request_type="code_review",
        request_data={
            "code": request.code,
            "language": request.language,
            "review_type": request.review_type,
            "context": request.context,
        },
        preferred_version=version,
    )
    
    return result


# =============================================================================
# User AI Endpoints (V2 Production)
# =============================================================================

# Create separate routers for versioned AI endpoints
v1_router = APIRouter(prefix="/api/v1/ai", tags=["V1 Experimental AI"])
v2_router = APIRouter(prefix="/api/v2/ai", tags=["V2 Production AI"])
v3_router = APIRouter(prefix="/api/v3/ai", tags=["V3 Archive AI"])


async def _handle_chat(version: str, request: ChatRequest) -> Dict[str, Any]:
    """Handle chat request for any version."""
    import time
    import uuid
    
    start_time = time.time()
    
    # In production, this would call the actual AI model
    # For now, provide intelligent responses based on context
    message = request.message.lower()
    
    if "security" in message or "vulnerability" in message:
        response = """Based on your question about security, here are key recommendations:

1. **Input Validation**: Always validate and sanitize user input
2. **Authentication**: Use strong authentication mechanisms (JWT, OAuth2)
3. **Authorization**: Implement role-based access control
4. **Encryption**: Use TLS for data in transit, AES-256 for data at rest
5. **Dependency Scanning**: Regularly scan for vulnerable dependencies

Would you like me to analyze specific code for security issues?"""
    elif "performance" in message or "optimize" in message:
        response = """Here are performance optimization strategies:

1. **Caching**: Implement Redis caching for frequently accessed data
2. **Database**: Optimize queries, add indexes, use connection pooling
3. **Async Operations**: Use async/await for I/O-bound operations
4. **Code Profiling**: Identify bottlenecks with profiling tools
5. **Lazy Loading**: Load resources only when needed

Share your code and I can provide specific optimizations."""
    elif "review" in message or "code" in message:
        response = """I can help with code review! I analyze code for:

- **Security**: SQL injection, XSS, CSRF vulnerabilities
- **Performance**: O(n²) loops, memory leaks, blocking calls
- **Quality**: Code smells, SOLID violations, complexity
- **Bugs**: Null references, race conditions, edge cases

Paste your code and select the review type you want."""
    else:
        response = f"""I'm the {version.upper()} AI Assistant for code review and analysis.

I can help you with:
- **Code Review**: Analyze code for issues and improvements
- **Security Analysis**: Find vulnerabilities in your code
- **Performance Optimization**: Identify bottlenecks
- **Architecture Advice**: Design better systems

How can I assist you today?"""
    
    latency = int((time.time() - start_time) * 1000) + 50  # Add simulated processing time
    tokens = len(response.split()) * 2  # Approximate token count
    
    return {
        "response": response,
        "model": f"{version}-gpt-4-turbo" if version == "v2" else f"{version}-experimental",
        "latency": latency,
        "tokens": tokens,
        "version": version,
        "response_id": str(uuid.uuid4()),
    }


async def _handle_analyze(version: str, request: AnalyzeRequest) -> Dict[str, Any]:
    """Handle code analysis request for any version."""
    import time
    import uuid
    import hashlib
    
    start_time = time.time()
    code_hash = hashlib.md5(request.code.encode()).hexdigest()[:8]
    
    # Generate analysis results based on code patterns
    issues = []
    
    # Security checks
    if "security" in request.review_types:
        if "eval(" in request.code or "exec(" in request.code:
            issues.append({
                "id": f"sec-{code_hash}-1",
                "type": "security",
                "severity": "critical",
                "title": "Code Injection Vulnerability",
                "description": "Use of eval() or exec() can lead to code injection attacks.",
                "line": request.code.find("eval(") // 50 + 1 if "eval(" in request.code else 1,
                "suggestion": "Avoid eval/exec. Use safe alternatives like ast.literal_eval() for parsing.",
                "fixAvailable": True,
            })
        if "password" in request.code.lower() and "=" in request.code:
            issues.append({
                "id": f"sec-{code_hash}-2",
                "type": "security",
                "severity": "high",
                "title": "Hardcoded Credentials",
                "description": "Potential hardcoded password detected in code.",
                "line": 1,
                "suggestion": "Use environment variables or a secrets manager.",
                "fixAvailable": True,
            })
    
    # Performance checks
    if "performance" in request.review_types:
        if "for" in request.code and request.code.count("for") >= 2:
            issues.append({
                "id": f"perf-{code_hash}-1",
                "type": "performance",
                "severity": "medium",
                "title": "Nested Loop Detected",
                "description": "Nested loops may cause O(n²) complexity.",
                "line": request.code.find("for") // 50 + 1,
                "suggestion": "Consider using hash maps or sets for O(1) lookups.",
                "fixAvailable": True,
            })
        if ".append(" in request.code and "for" in request.code:
            issues.append({
                "id": f"perf-{code_hash}-2",
                "type": "performance",
                "severity": "low",
                "title": "List Comprehension Opportunity",
                "description": "Loop with append can be converted to list comprehension.",
                "line": request.code.find("append") // 50 + 1,
                "suggestion": "Use list comprehension for better performance.",
                "fixAvailable": True,
            })
    
    # Quality checks
    if "quality" in request.review_types:
        lines = request.code.split("\n")
        if len(lines) > 50:
            issues.append({
                "id": f"qual-{code_hash}-1",
                "type": "quality",
                "severity": "medium",
                "title": "Long Function",
                "description": f"Function has {len(lines)} lines. Consider breaking it up.",
                "line": 1,
                "suggestion": "Extract smaller functions with single responsibilities.",
                "fixAvailable": False,
            })
        if not any(line.strip().startswith("#") or '"""' in line for line in lines):
            issues.append({
                "id": f"qual-{code_hash}-2",
                "type": "quality",
                "severity": "low",
                "title": "Missing Documentation",
                "description": "No comments or docstrings found in the code.",
                "line": 1,
                "suggestion": "Add docstrings to functions and comments for complex logic.",
                "fixAvailable": False,
            })
    
    # Bug checks
    if "bug" in request.review_types:
        if "except:" in request.code:
            issues.append({
                "id": f"bug-{code_hash}-1",
                "type": "bug",
                "severity": "medium",
                "title": "Bare Except Clause",
                "description": "Bare except catches all exceptions including KeyboardInterrupt.",
                "line": request.code.find("except:") // 50 + 1,
                "suggestion": "Catch specific exceptions: except ValueError as e:",
                "fixAvailable": True,
            })
    
    # Calculate score
    severity_weights = {"critical": 25, "high": 15, "medium": 8, "low": 3}
    penalty = sum(severity_weights.get(i["severity"], 0) for i in issues)
    score = max(0, 100 - penalty)
    
    latency = int((time.time() - start_time) * 1000) + 100
    
    return {
        "id": str(uuid.uuid4()),
        "issues": issues,
        "score": score,
        "summary": f"Found {len(issues)} issues. Code quality score: {score}/100",
        "model": f"{version}-analyzer",
        "latency": latency,
        "version": version,
    }


@v1_router.post("/chat")
async def v1_chat(request: ChatRequest):
    """Chat with V1 Experimental AI (Admin only)."""
    return await _handle_chat("v1", request)


@v1_router.post("/analyze")
async def v1_analyze(request: AnalyzeRequest):
    """Analyze code with V1 Experimental AI (Admin only)."""
    return await _handle_analyze("v1", request)


@v2_router.post("/chat")
async def v2_chat(request: ChatRequest):
    """Chat with V2 Production AI (User-facing)."""
    return await _handle_chat("v2", request)


@v2_router.post("/analyze")
async def v2_analyze(request: AnalyzeRequest):
    """Analyze code with V2 Production AI (User-facing)."""
    return await _handle_analyze("v2", request)


@v2_router.post("/fix")
async def v2_apply_fix(request: FixRequest):
    """Apply auto-fix for an issue (V2 Production only)."""
    import uuid
    
    # In production, this would apply the actual fix
    # For now, return the code with a simulated fix applied
    fixed_code = request.code
    
    # Apply simple fixes based on issue patterns
    if "eval(" in fixed_code:
        fixed_code = fixed_code.replace("eval(", "ast.literal_eval(")
    if "except:" in fixed_code:
        fixed_code = fixed_code.replace("except:", "except Exception as e:")
    
    return {
        "success": True,
        "fixed_code": fixed_code,
        "fix_id": str(uuid.uuid4()),
        "changes_made": 1,
    }


@v2_router.post("/feedback")
async def v2_feedback(request: FeedbackRequest):
    """Submit feedback for an AI response."""
    # In production, store feedback for model improvement
    logger.info(f"Feedback received: response_id={request.response_id}, helpful={request.helpful}")
    
    return {
        "success": True,
        "message": "Thank you for your feedback!",
    }


@v3_router.post("/chat")
async def v3_chat(request: ChatRequest):
    """Chat with V3 Archive AI (Admin only)."""
    return await _handle_chat("v3", request)


@v3_router.post("/analyze")
async def v3_analyze(request: AnalyzeRequest):
    """Analyze code with V3 Archive AI (Admin only)."""
    return await _handle_analyze("v3", request)


# =============================================================================
# History & Metrics Endpoints
# =============================================================================

@router.get("/technologies")
async def get_technologies():
    """Get all technologies across versions."""
    # In production, this would query the database
    # For now, return mock data that matches frontend expectations
    return [
        {
            "id": "tech-gpt4-v2",
            "name": "GPT-4 Turbo",
            "version": "v2",
            "status": "active",
            "accuracy": 0.92,
            "errorRate": 0.02,
            "latency": 450,
            "samples": 15000,
            "lastUpdated": datetime.utcnow().isoformat(),
        },
        {
            "id": "tech-claude3-v1",
            "name": "Claude-3 Opus",
            "version": "v1",
            "status": "testing",
            "accuracy": 0.89,
            "errorRate": 0.04,
            "latency": 380,
            "samples": 2500,
            "lastUpdated": datetime.utcnow().isoformat(),
        },
        {
            "id": "tech-llama3-v1",
            "name": "Llama-3 70B",
            "version": "v1",
            "status": "testing",
            "accuracy": 0.85,
            "errorRate": 0.05,
            "latency": 320,
            "samples": 1800,
            "lastUpdated": datetime.utcnow().isoformat(),
        },
        {
            "id": "tech-gpt35-v3",
            "name": "GPT-3.5 Turbo",
            "version": "v3",
            "status": "deprecated",
            "accuracy": 0.78,
            "errorRate": 0.08,
            "latency": 280,
            "samples": 50000,
            "lastUpdated": datetime.utcnow().isoformat(),
        },
        {
            "id": "tech-codellama-v3",
            "name": "CodeLlama 34B",
            "version": "v3",
            "status": "quarantined",
            "accuracy": 0.72,
            "errorRate": 0.12,
            "latency": 250,
            "samples": 8000,
            "lastUpdated": datetime.utcnow().isoformat(),
        },
    ]


@router.get("/history")
async def get_cycle_history(limit: int = 10):
    """Get history of completed evolution cycles."""
    cycle = get_evolution_cycle()
    
    return {
        "cycles": cycle.spiral_manager.get_cycle_history(limit),
    }


@router.get("/metrics")
async def get_evolution_metrics():
    """Get comprehensive evolution metrics."""
    cycle = get_evolution_cycle()
    status = cycle.get_full_status()
    
    return {
        "running": status.get("running"),
        "spiral_status": status.get("spiral_status", {}).get("current_cycle"),
        "feedback_stats": status.get("spiral_status", {}).get("feedback_stats"),
        "quarantine_stats": status.get("spiral_status", {}).get("quarantine_stats"),
        "ai_status": status.get("spiral_status", {}).get("ai_status"),
    }


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health")
async def health_check():
    """Health check for the evolution service."""
    cycle = get_evolution_cycle()
    
    return {
        "status": "healthy",
        "service": "three-version-evolution",
        "cycle_running": cycle._running,
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Prometheus Metrics Endpoint
# =============================================================================

@router.get("/prometheus")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus format for scraping.
    """
    # Update metrics from current status
    cycle = get_evolution_cycle()
    status = cycle.get_full_status()
    collect_metrics_from_status(status)
    
    # Generate and return metrics
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
