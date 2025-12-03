"""
V2 CR-AI Consensus Models

Data models for multi-model consensus protocol.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ConsensusDecision(str, Enum):
    """Consensus decision outcome"""
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    UNCERTAIN = "UNCERTAIN"


class VerificationStatus(str, Enum):
    """Status of verification"""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


class ModelVerification(BaseModel):
    """Verification result from a model"""
    model: str = Field(..., description="Model that performed verification")
    provider: str = Field(..., description="Model provider")
    decision: ConsensusDecision
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str = Field(..., description="Explanation for the decision")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: int = Field(default=0, ge=0)


class ConsensusResult(BaseModel):
    """Result of consensus verification"""
    finding_id: str = Field(..., description="ID of the finding being verified")
    
    # Verification results
    primary_verification: ModelVerification
    secondary_verification: Optional[ModelVerification] = None
    
    # Consensus outcome
    consensus_reached: bool = Field(default=False)
    final_decision: ConsensusDecision
    final_confidence: float = Field(..., ge=0, le=1)
    
    # Status
    status: VerificationStatus
    requires_manual_review: bool = Field(default=False)
    manual_review_reason: Optional[str] = None
    
    # Timing
    total_time_ms: int = Field(default=0, ge=0)


class ConsensusConfig(BaseModel):
    """Configuration for consensus protocol"""
    enabled: bool = Field(default=True)
    critical_issues_require_both: bool = Field(default=True)
    high_priority_require_one: bool = Field(default=True)
    agreement_threshold: float = Field(default=0.98, ge=0, le=1)
    
    # Confidence scoring
    single_model_multiplier: float = Field(default=0.7)
    both_agree_multiplier: float = Field(default=1.0)
    disagree_multiplier: float = Field(default=0.3)


class ConsensusMetrics(BaseModel):
    """Metrics for consensus protocol"""
    total_verifications: int = Field(default=0, ge=0)
    consensus_reached: int = Field(default=0, ge=0)
    disagreements: int = Field(default=0, ge=0)
    manual_reviews_triggered: int = Field(default=0, ge=0)
    
    agreement_rate: float = Field(default=0.0, ge=0, le=1)
    avg_confidence_boost: float = Field(default=0.0)
    avg_verification_time_ms: float = Field(default=0.0, ge=0)


class FindingClassification(BaseModel):
    """Classification of a finding for consensus requirements"""
    finding_id: str
    severity_class: str = Field(description="critical, high, medium_low")
    requires_consensus: bool
    consensus_type: str = Field(description="both_required, one_required, optional")


class ConsensusWorkflow(BaseModel):
    """Workflow state for consensus process"""
    review_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Progress
    findings_to_verify: int = Field(default=0, ge=0)
    findings_verified: int = Field(default=0, ge=0)
    findings_pending: int = Field(default=0, ge=0)
    
    # Results
    results: List[ConsensusResult] = Field(default_factory=list)
    
    # Status
    completed: bool = Field(default=False)
    completed_at: Optional[datetime] = None
    
    # Summary
    total_agreed: int = Field(default=0, ge=0)
    total_disagreed: int = Field(default=0, ge=0)
    total_uncertain: int = Field(default=0, ge=0)
    manual_review_count: int = Field(default=0, ge=0)
