"""
Analysis API Endpoints

Additional analysis endpoints:
- Synthetic bug injection
- Batch review
- Code quality scoring
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/analysis", tags=["analysis"])


# =============================================================================
# Request/Response Models
# =============================================================================

class InjectBugRequest(BaseModel):
    """Request to inject synthetic bugs"""
    code: str = Field(..., min_length=1)
    language: str = Field(default="python")
    bug_types: List[str] = Field(
        default=["off_by_one", "null_pointer", "sql_injection"],
        description="Types of bugs to inject"
    )
    num_bugs: int = Field(default=3, ge=1, le=10)


class InjectedBug(BaseModel):
    """A bug that was injected"""
    bug_type: str
    line_number: int
    original_code: str
    buggy_code: str
    description: str


class InjectBugResponse(BaseModel):
    """Response with injected bugs"""
    original_code: str
    buggy_code: str
    injected_bugs: List[InjectedBug]
    can_be_detected: bool


class BatchReviewRequest(BaseModel):
    """Request for batch review"""
    code_files: List[Dict[str, str]] = Field(
        ...,
        description="List of {filename, code} pairs",
        max_items=50
    )
    language: str = Field(default="python")
    dimensions: List[str] = Field(default=["correctness", "security"])


class FileReviewResult(BaseModel):
    """Review result for a single file"""
    filename: str
    findings_count: int
    overall_score: float
    critical_count: int
    high_count: int


class BatchReviewResponse(BaseModel):
    """Response from batch review"""
    total_files: int
    total_findings: int
    avg_score: float
    results: List[FileReviewResult]
    processing_time_ms: float


class QualityScoreRequest(BaseModel):
    """Request for quality scoring"""
    code: str = Field(..., min_length=1)
    language: str = Field(default="python")


class QualityScoreResponse(BaseModel):
    """Response with quality scores"""
    overall_score: float
    scores: Dict[str, float]
    grade: str  # A, B, C, D, F
    recommendations: List[str]


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/inject-bugs", response_model=InjectBugResponse)
async def inject_bugs(request: InjectBugRequest):
    """
    Inject synthetic bugs into code for testing.
    
    Creates a version of the code with known bugs that
    can be used to test the review engine's detection capabilities.
    """
    import re
    import random
    
    lines = request.code.split('\n')
    buggy_lines = lines.copy()
    injected_bugs = []
    
    # Bug injection patterns
    bug_patterns = {
        "off_by_one": [
            (r'range\(len\((\w+)\)\)', r'range(len(\1) - 1)', "Off-by-one: missing last element"),
            (r'\[(\w+)\]', r'[\1 + 1]', "Off-by-one: index shifted by 1"),
        ],
        "null_pointer": [
            (r'(\w+)\.(\w+)\(', r'\1.\2( # WARNING: no null check\n', "Potential null pointer access"),
        ],
        "sql_injection": [
            (r'cursor\.execute\("(.+?)"\s*%\s*\(', r'cursor.execute(f"\1{', "SQL injection vulnerability"),
        ],
        "string_concat": [
            (r'(\w+)\s*\+=\s*"', r'\1 = \1 + "', "Inefficient string concatenation"),
        ],
    }
    
    bugs_injected = 0
    
    for i, line in enumerate(buggy_lines):
        if bugs_injected >= request.num_bugs:
            break
        
        for bug_type in request.bug_types:
            if bug_type not in bug_patterns:
                continue
            
            for pattern, replacement, description in bug_patterns[bug_type]:
                if re.search(pattern, line):
                    new_line = re.sub(pattern, replacement, line, count=1)
                    
                    if new_line != line:
                        buggy_lines[i] = new_line
                        injected_bugs.append(InjectedBug(
                            bug_type=bug_type,
                            line_number=i + 1,
                            original_code=line.strip(),
                            buggy_code=new_line.strip(),
                            description=description,
                        ))
                        bugs_injected += 1
                        break
            
            if bugs_injected >= request.num_bugs:
                break
    
    return InjectBugResponse(
        original_code=request.code,
        buggy_code='\n'.join(buggy_lines),
        injected_bugs=injected_bugs,
        can_be_detected=len(injected_bugs) > 0,
    )


@router.post("/batch-review", response_model=BatchReviewResponse)
async def batch_review(request: BatchReviewRequest):
    """
    Review multiple code files in batch.
    
    Efficiently processes multiple files and returns
    aggregated results.
    """
    import time
    from ..review import ReviewEngine
    
    start_time = time.time()
    engine = ReviewEngine()
    
    results = []
    total_findings = 0
    total_score = 0.0
    
    for file_info in request.code_files:
        filename = file_info.get("filename", "unknown")
        code = file_info.get("code", "")
        
        if not code:
            continue
        
        result = await engine.review(
            code=code,
            language=request.language,
            dimensions=request.dimensions,
            strategy="baseline",  # Fast strategy for batch
        )
        
        critical_count = sum(1 for f in result.findings if f.severity == "critical")
        high_count = sum(1 for f in result.findings if f.severity == "high")
        
        results.append(FileReviewResult(
            filename=filename,
            findings_count=len(result.findings),
            overall_score=result.overall_score,
            critical_count=critical_count,
            high_count=high_count,
        ))
        
        total_findings += len(result.findings)
        total_score += result.overall_score
    
    avg_score = total_score / max(1, len(results))
    
    return BatchReviewResponse(
        total_files=len(results),
        total_findings=total_findings,
        avg_score=avg_score,
        results=results,
        processing_time_ms=(time.time() - start_time) * 1000,
    )


@router.post("/quality-score", response_model=QualityScoreResponse)
async def quality_score(request: QualityScoreRequest):
    """
    Calculate overall code quality score.
    
    Analyzes code across all dimensions and provides
    a letter grade with recommendations.
    """
    from ..review import ReviewEngine
    
    engine = ReviewEngine()
    
    result = await engine.review(
        code=request.code,
        language=request.language,
        dimensions=["correctness", "security", "performance", "maintainability"],
        strategy="ensemble",
    )
    
    # Calculate letter grade
    score = result.overall_score
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"
    
    # Generate recommendations
    recommendations = []
    
    for finding in result.findings:
        if finding.severity in ["critical", "high"]:
            recommendations.append(f"Fix {finding.severity} issue: {finding.issue}")
    
    # Add general recommendations
    if "security" in result.dimension_scores and result.dimension_scores["security"] < 80:
        recommendations.append("Improve security practices and input validation")
    
    if "maintainability" in result.dimension_scores and result.dimension_scores["maintainability"] < 80:
        recommendations.append("Refactor for better code organization and naming")
    
    return QualityScoreResponse(
        overall_score=score,
        scores=result.dimension_scores,
        grade=grade,
        recommendations=recommendations[:5],  # Top 5
    )


@router.get("/bug-patterns")
async def list_bug_patterns():
    """List available bug patterns for injection."""
    return {
        "patterns": [
            {
                "name": "off_by_one",
                "description": "Off-by-one errors in loops and arrays",
                "severity": "high",
            },
            {
                "name": "null_pointer",
                "description": "Null/undefined reference access",
                "severity": "high",
            },
            {
                "name": "sql_injection",
                "description": "SQL injection vulnerabilities",
                "severity": "critical",
            },
            {
                "name": "xss",
                "description": "Cross-site scripting vulnerabilities",
                "severity": "critical",
            },
            {
                "name": "command_injection",
                "description": "OS command injection",
                "severity": "critical",
            },
            {
                "name": "string_concat",
                "description": "Inefficient string concatenation",
                "severity": "medium",
            },
            {
                "name": "race_condition",
                "description": "Race condition in concurrent code",
                "severity": "high",
            },
        ]
    }
