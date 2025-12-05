"""
V2 VC-AI Analysis Models

Data models for commit analysis operations.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    """Type of code change"""
    FEAT = "feat"           # New feature
    FIX = "fix"            # Bug fix
    DOCS = "docs"          # Documentation
    STYLE = "style"        # Code style (formatting, etc.)
    REFACTOR = "refactor"  # Code refactoring
    PERF = "perf"          # Performance improvement
    TEST = "test"          # Tests
    BUILD = "build"        # Build system
    CI = "ci"              # CI configuration
    CHORE = "chore"        # Maintenance
    REVERT = "revert"      # Revert previous commit
    SECURITY = "security"  # Security fix
    DEPS = "deps"          # Dependency update


class ImpactLevel(str, Enum):
    """Impact level of a change"""
    LOW = "LOW"            # Minimal impact, isolated change
    MEDIUM = "MEDIUM"      # Moderate impact, affects some components
    HIGH = "HIGH"          # Significant impact, affects many components
    CRITICAL = "CRITICAL"  # Critical impact, system-wide effects


class RiskAssessment(str, Enum):
    """Risk assessment for deployment"""
    SAFE = "SAFE"          # Safe to deploy
    CAUTION = "CAUTION"    # Deploy with monitoring
    RISKY = "RISKY"        # Requires additional review


class AffectedComponent(BaseModel):
    """Component affected by a change"""
    name: str = Field(..., description="Component/service name")
    path: str = Field(..., description="File or directory path")
    change_type: str = Field(..., description="Type of change to component")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in detection")
    dependencies: List[str] = Field(default_factory=list)


class BreakingChange(BaseModel):
    """Breaking change detail"""
    type: str = Field(..., description="Type of breaking change")
    description: str = Field(..., description="Description of the breaking change")
    affected_apis: List[str] = Field(default_factory=list)
    migration_steps: List[str] = Field(default_factory=list)
    deprecation_notice: Optional[str] = None
    removal_version: Optional[str] = None


class RollbackPlan(BaseModel):
    """Rollback plan for a change"""
    steps: List[str] = Field(..., description="Steps to rollback")
    estimated_duration_minutes: int = Field(..., ge=0)
    data_migration_required: bool = Field(default=False)
    service_restart_required: bool = Field(default=False)
    verification_steps: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class CommitAnalysisRequest(BaseModel):
    """Request to analyze a commit"""
    commit_hash: str = Field(..., description="Git commit hash")
    repo: str = Field(..., description="Repository identifier")
    branch: Optional[str] = Field(default="main", description="Branch name")
    include_diff: bool = Field(default=True, description="Include full diff analysis")
    include_dependencies: bool = Field(default=True, description="Include dependency analysis")
    context_commits: int = Field(default=5, ge=0, le=20, description="Number of context commits")


class CommitAnalysisResponse(BaseModel):
    """Response from commit analysis"""
    commit_hash: str
    repo: str
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_used: str = Field(..., description="AI model that performed analysis")
    
    # Core analysis
    change_type: ChangeType
    impact_level: ImpactLevel
    risk_assessment: RiskAssessment
    
    # Detailed findings
    summary: str = Field(..., description="One-line summary of the commit")
    description: str = Field(..., description="Detailed description")
    affected_services: List[str] = Field(default_factory=list)
    affected_components: List[AffectedComponent] = Field(default_factory=list)
    
    # Breaking changes
    breaking_changes: List[BreakingChange] = Field(default_factory=list)
    is_breaking: bool = Field(default=False)
    migration_guide: Optional[str] = None
    
    # Rollback
    rollback_plan: Optional[RollbackPlan] = None
    rollback_safe: bool = Field(default=True)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    review_suggestions: List[str] = Field(default_factory=list)
    test_suggestions: List[str] = Field(default_factory=list)
    
    # Metrics
    confidence_score: float = Field(..., ge=0, le=1)
    complexity_score: float = Field(..., ge=0, le=1)
    risk_score: float = Field(..., ge=0, le=1)
    
    # SLO compliance
    analysis_latency_ms: int = Field(..., ge=0)
    slo_compliant: bool = Field(default=True)


class BatchAnalysisRequest(BaseModel):
    """Request to analyze multiple commits"""
    commits: List[CommitAnalysisRequest] = Field(..., min_length=1, max_length=100)
    parallel: bool = Field(default=True, description="Process commits in parallel")
    stop_on_critical: bool = Field(default=False, description="Stop if critical impact found")


class BatchAnalysisResponse(BaseModel):
    """Response from batch commit analysis"""
    total_commits: int
    successful: int
    failed: int
    results: List[CommitAnalysisResponse]
    errors: List[Dict[str, str]] = Field(default_factory=list)
    aggregate_impact: ImpactLevel
    aggregate_risk: RiskAssessment
    total_breaking_changes: int
    processing_time_ms: int


class DependencyImpact(BaseModel):
    """Impact analysis for dependencies"""
    dependency: str
    current_version: str
    new_version: Optional[str] = None
    impact: ImpactLevel
    breaking_changes: List[str] = Field(default_factory=list)
    security_advisories: List[Dict[str, Any]] = Field(default_factory=list)
    upgrade_path: Optional[str] = None


class CommitDiff(BaseModel):
    """Detailed commit diff information"""
    files_changed: int
    insertions: int
    deletions: int
    files: List[Dict[str, Any]] = Field(default_factory=list)
    binary_files: List[str] = Field(default_factory=list)
    renamed_files: List[Dict[str, str]] = Field(default_factory=list)
