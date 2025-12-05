"""
V2 CR-AI Review Models

Data models for code review operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class FindingSeverity(str, Enum):
    """Severity level of a finding"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class FindingCategory(str, Enum):
    """Category of finding"""
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


class CodeLocation(BaseModel):
    """Location in source code"""
    file: str = Field(..., description="File path")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    start_column: Optional[int] = Field(None, ge=1)
    end_column: Optional[int] = Field(None, ge=1)
    code_snippet: Optional[str] = Field(None, description="Relevant code snippet")


class FixSuggestion(BaseModel):
    """Suggested fix for a finding"""
    description: str = Field(..., description="Description of the fix")
    code_before: Optional[str] = Field(None, description="Original code")
    code_after: str = Field(..., description="Fixed code")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the fix")
    auto_applicable: bool = Field(default=False, description="Can be auto-applied")


class ReviewFinding(BaseModel):
    """Individual code review finding"""
    id: str = Field(..., description="Unique finding ID")
    title: str = Field(..., description="Short title")
    description: str = Field(..., description="Detailed description")
    category: FindingCategory
    severity: FindingSeverity
    location: CodeLocation
    
    # Analysis details
    rule_id: Optional[str] = Field(None, description="Rule that triggered this finding")
    cwe_id: Optional[str] = Field(None, description="CWE ID for security issues")
    owasp_id: Optional[str] = Field(None, description="OWASP category")
    
    # Confidence
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    consensus_verified: bool = Field(default=False, description="Verified by consensus")
    
    # Recommendations
    fix_suggestions: List[FixSuggestion] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list, description="Reference links")
    
    # Metadata
    detected_by: str = Field(..., description="Model that detected this")
    requires_manual_review: bool = Field(default=False)


class FileReview(BaseModel):
    """Review results for a single file"""
    file_path: str
    language: str
    lines_of_code: int
    findings: List[ReviewFinding] = Field(default_factory=list)
    review_time_ms: int = Field(default=0, ge=0)


class ReviewRequest(BaseModel):
    """Request to review code"""
    # Code content
    files: List[Dict[str, Any]] = Field(..., description="Files to review [{path, content, language}]")
    
    # Context
    repo: Optional[str] = Field(None, description="Repository name")
    branch: Optional[str] = Field(None, description="Branch name")
    commit: Optional[str] = Field(None, description="Commit hash")
    pr_number: Optional[int] = Field(None, description="Pull request number")
    
    # Review configuration
    dimensions: Optional[List[FindingCategory]] = Field(
        None,
        description="Dimensions to review (all if not specified)"
    )
    severity_threshold: Optional[FindingSeverity] = Field(
        None,
        description="Minimum severity to report"
    )
    
    # Options
    include_suggestions: bool = Field(default=True)
    include_references: bool = Field(default=True)
    max_findings_per_file: int = Field(default=50, ge=1, le=200)
    consensus_enabled: bool = Field(default=True)


class ReviewSummary(BaseModel):
    """Summary of review findings"""
    total_findings: int = Field(default=0, ge=0)
    critical_count: int = Field(default=0, ge=0)
    high_count: int = Field(default=0, ge=0)
    medium_count: int = Field(default=0, ge=0)
    low_count: int = Field(default=0, ge=0)
    info_count: int = Field(default=0, ge=0)
    
    by_category: Dict[str, int] = Field(default_factory=dict)
    consensus_verified_count: int = Field(default=0, ge=0)
    manual_review_needed_count: int = Field(default=0, ge=0)
    
    overall_quality_score: float = Field(..., ge=0, le=100)
    recommendation: str = Field(..., description="Overall recommendation")


class ReviewResponse(BaseModel):
    """Response from code review"""
    id: str = Field(..., description="Review ID")
    status: str = Field(default="completed", description="Review status")
    
    # Timing
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_time_ms: int = Field(default=0, ge=0)
    
    # Models used
    primary_model: str
    secondary_model: Optional[str] = None
    consensus_used: bool = Field(default=False)
    
    # Results
    files_reviewed: int = Field(default=0, ge=0)
    file_reviews: List[FileReview] = Field(default_factory=list)
    summary: ReviewSummary
    
    # SLO compliance
    slo_compliant: bool = Field(default=True)
    
    # Findings organized by confidence
    high_confidence_findings: List[ReviewFinding] = Field(default_factory=list)
    medium_confidence_findings: List[ReviewFinding] = Field(default_factory=list)
    low_confidence_findings: List[ReviewFinding] = Field(default_factory=list)
    manual_review_needed: List[ReviewFinding] = Field(default_factory=list)


class InlineComment(BaseModel):
    """Inline comment for CI/CD integration"""
    file: str
    line: int
    body: str
    severity: FindingSeverity
    suggestion: Optional[str] = None


class ReviewStatus(BaseModel):
    """Review approval status"""
    approved: bool = Field(default=False)
    requires_changes: bool = Field(default=False)
    blocking_issues: int = Field(default=0, ge=0)
    comment: str = Field(default="")
