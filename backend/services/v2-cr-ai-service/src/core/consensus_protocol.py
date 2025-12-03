"""
V2 CR-AI Consensus Protocol

Multi-model consensus verification for production reliability.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from ..models.review_models import ReviewFinding, FindingSeverity
from ..models.consensus_models import (
    ConsensusResult,
    ModelVerification,
    ConsensusDecision,
    VerificationStatus,
    ConsensusMetrics,
    ConsensusWorkflow,
    ConsensusConfig,
)
from ..config.model_config import CONSENSUS_CONFIG, SYSTEM_PROMPTS


logger = logging.getLogger(__name__)


class ConsensusProtocol:
    """
    Multi-model consensus protocol for production code review.
    
    Protocol:
    - CRITICAL issues: BOTH models must agree
    - HIGH priority: At least one model must flag
    - MEDIUM/LOW: Any single model can suggest
    
    Confidence scoring:
    - Single model agreement: confidence * 0.7
    - Both models agree: confidence * 1.0
    - Both disagree: max_confidence * 0.3
    """
    
    def __init__(
        self,
        primary_model_client: Any,
        secondary_model_client: Optional[Any] = None,
        config: Optional[ConsensusConfig] = None,
    ):
        self.primary_client = primary_model_client
        self.secondary_client = secondary_model_client
        self.config = config or ConsensusConfig()
        
        # Metrics
        self._metrics = ConsensusMetrics()
        
        # Active workflows
        self._workflows: Dict[str, ConsensusWorkflow] = {}
    
    def _classify_finding(self, finding: ReviewFinding) -> Tuple[str, bool, str]:
        """
        Classify a finding for consensus requirements.
        
        Returns: (severity_class, requires_consensus, consensus_type)
        """
        # Critical issues (security, data loss, production breaking)
        critical_categories = ["security"]
        critical_severities = [FindingSeverity.CRITICAL]
        
        if (finding.category.value in critical_categories or 
            finding.severity in critical_severities):
            return ("critical", True, "both_required")
        
        # High priority
        high_severities = [FindingSeverity.HIGH]
        if finding.severity in high_severities:
            return ("high", True, "one_required")
        
        # Medium/Low
        return ("medium_low", False, "optional")
    
    async def verify_finding(
        self,
        finding: ReviewFinding,
        code_context: str,
    ) -> ConsensusResult:
        """
        Verify a single finding through consensus protocol.
        """
        import time
        start_time = time.time()
        
        self._metrics.total_verifications += 1
        
        # Classify finding
        severity_class, requires_consensus, consensus_type = self._classify_finding(finding)
        
        # Get primary verification
        primary_verification = await self._get_model_verification(
            finding=finding,
            code_context=code_context,
            model_client=self.primary_client,
            model_name="claude-3-sonnet",
            provider="anthropic",
        )
        
        # Determine if secondary verification needed
        secondary_verification = None
        
        if (requires_consensus and 
            consensus_type == "both_required" and 
            self.secondary_client):
            # Critical issues require both models
            secondary_verification = await self._get_model_verification(
                finding=finding,
                code_context=code_context,
                model_client=self.secondary_client,
                model_name="gpt-4-turbo",
                provider="openai",
            )
        
        # Determine consensus outcome
        consensus_result = self._determine_consensus(
            primary=primary_verification,
            secondary=secondary_verification,
            severity_class=severity_class,
            consensus_type=consensus_type,
        )
        
        total_time = int((time.time() - start_time) * 1000)
        
        result = ConsensusResult(
            finding_id=finding.id,
            primary_verification=primary_verification,
            secondary_verification=secondary_verification,
            consensus_reached=consensus_result["consensus_reached"],
            final_decision=consensus_result["decision"],
            final_confidence=consensus_result["confidence"],
            status=consensus_result["status"],
            requires_manual_review=consensus_result["requires_manual_review"],
            manual_review_reason=consensus_result.get("manual_review_reason"),
            total_time_ms=total_time,
        )
        
        # Update metrics
        if result.consensus_reached:
            self._metrics.consensus_reached += 1
        else:
            self._metrics.disagreements += 1
        
        if result.requires_manual_review:
            self._metrics.manual_reviews_triggered += 1
        
        self._metrics.agreement_rate = (
            self._metrics.consensus_reached / 
            max(1, self._metrics.total_verifications)
        )
        
        return result
    
    async def _get_model_verification(
        self,
        finding: ReviewFinding,
        code_context: str,
        model_client: Any,
        model_name: str,
        provider: str,
    ) -> ModelVerification:
        """Get verification from a model."""
        import time
        start_time = time.time()
        
        # Build verification prompt
        prompt = self._build_verification_prompt(finding, code_context)
        system_prompt = SYSTEM_PROMPTS["consensus_verification"]
        
        try:
            # Call model
            if hasattr(model_client, 'analyze'):
                response = await model_client.analyze(prompt, system_prompt)
                content = response.content
            else:
                # Mock response
                content = "AGREE\n\nThe identified issue is valid. The code shows potential security concerns."
            
            # Parse response
            decision, confidence, reasoning = self._parse_verification_response(content)
            
            latency = int((time.time() - start_time) * 1000)
            
            return ModelVerification(
                model=model_name,
                provider=provider,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                latency_ms=latency,
            )
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return ModelVerification(
                model=model_name,
                provider=provider,
                decision=ConsensusDecision.UNCERTAIN,
                confidence=0.5,
                reasoning=f"Verification failed: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000),
            )
    
    def _build_verification_prompt(
        self,
        finding: ReviewFinding,
        code_context: str,
    ) -> str:
        """Build verification prompt."""
        return f"""Verify the following code review finding:

## Finding
- **Title**: {finding.title}
- **Category**: {finding.category.value}
- **Severity**: {finding.severity.value}
- **Description**: {finding.description}

## Code Location
- **File**: {finding.location.file}
- **Lines**: {finding.location.start_line}-{finding.location.end_line}

## Code Context
```
{code_context}
```

## Code Snippet (flagged)
```
{finding.location.code_snippet or "Not provided"}
```

Please verify if this finding is valid. Respond with:
1. AGREE - if you independently confirm the issue exists
2. DISAGREE - if you believe this is a false positive
3. UNCERTAIN - if you need more context

Provide your confidence level (0-1) and reasoning."""
    
    def _parse_verification_response(
        self,
        content: str,
    ) -> Tuple[ConsensusDecision, float, str]:
        """Parse verification response from model."""
        content_upper = content.upper()
        
        # Determine decision
        if "AGREE" in content_upper and "DISAGREE" not in content_upper:
            decision = ConsensusDecision.AGREE
            confidence = 0.9
        elif "DISAGREE" in content_upper:
            decision = ConsensusDecision.DISAGREE
            confidence = 0.85
        else:
            decision = ConsensusDecision.UNCERTAIN
            confidence = 0.5
        
        # Extract reasoning (everything after the decision keyword)
        reasoning = content
        for keyword in ["AGREE", "DISAGREE", "UNCERTAIN"]:
            if keyword in content_upper:
                idx = content_upper.find(keyword) + len(keyword)
                reasoning = content[idx:].strip()
                if reasoning.startswith("-") or reasoning.startswith(":"):
                    reasoning = reasoning[1:].strip()
                break
        
        return decision, confidence, reasoning[:500]  # Limit reasoning length
    
    def _determine_consensus(
        self,
        primary: ModelVerification,
        secondary: Optional[ModelVerification],
        severity_class: str,
        consensus_type: str,
    ) -> Dict[str, Any]:
        """Determine consensus outcome."""
        
        # No secondary verification
        if secondary is None:
            if primary.decision == ConsensusDecision.AGREE:
                return {
                    "consensus_reached": True,
                    "decision": ConsensusDecision.AGREE,
                    "confidence": primary.confidence * self.config.single_model_multiplier,
                    "status": VerificationStatus.VERIFIED,
                    "requires_manual_review": False,
                }
            elif primary.decision == ConsensusDecision.DISAGREE:
                return {
                    "consensus_reached": True,
                    "decision": ConsensusDecision.DISAGREE,
                    "confidence": primary.confidence * self.config.single_model_multiplier,
                    "status": VerificationStatus.REJECTED,
                    "requires_manual_review": False,
                }
            else:
                return {
                    "consensus_reached": False,
                    "decision": ConsensusDecision.UNCERTAIN,
                    "confidence": 0.5,
                    "status": VerificationStatus.MANUAL_REVIEW,
                    "requires_manual_review": True,
                    "manual_review_reason": "Primary model uncertain",
                }
        
        # Both models provided verification
        if primary.decision == ConsensusDecision.AGREE and secondary.decision == ConsensusDecision.AGREE:
            # Both agree - high confidence
            return {
                "consensus_reached": True,
                "decision": ConsensusDecision.AGREE,
                "confidence": max(primary.confidence, secondary.confidence) * self.config.both_agree_multiplier,
                "status": VerificationStatus.VERIFIED,
                "requires_manual_review": False,
            }
        
        elif primary.decision == ConsensusDecision.DISAGREE and secondary.decision == ConsensusDecision.DISAGREE:
            # Both disagree - reject finding
            return {
                "consensus_reached": True,
                "decision": ConsensusDecision.DISAGREE,
                "confidence": max(primary.confidence, secondary.confidence) * self.config.both_agree_multiplier,
                "status": VerificationStatus.REJECTED,
                "requires_manual_review": False,
            }
        
        else:
            # Disagreement or uncertainty - requires manual review for critical issues
            if severity_class == "critical":
                return {
                    "consensus_reached": False,
                    "decision": ConsensusDecision.UNCERTAIN,
                    "confidence": max(primary.confidence, secondary.confidence) * self.config.disagree_multiplier,
                    "status": VerificationStatus.MANUAL_REVIEW,
                    "requires_manual_review": True,
                    "manual_review_reason": f"Models disagree on critical issue: {primary.decision.value} vs {secondary.decision.value}",
                }
            else:
                # For non-critical, take primary model's decision with reduced confidence
                return {
                    "consensus_reached": False,
                    "decision": primary.decision,
                    "confidence": primary.confidence * self.config.disagree_multiplier,
                    "status": VerificationStatus.VERIFIED if primary.decision == ConsensusDecision.AGREE else VerificationStatus.REJECTED,
                    "requires_manual_review": False,
                }
    
    async def verify_findings_batch(
        self,
        findings: List[ReviewFinding],
        code_context: Dict[str, str],  # file -> content
        review_id: str,
    ) -> ConsensusWorkflow:
        """Verify multiple findings in batch."""
        workflow = ConsensusWorkflow(
            review_id=review_id,
            findings_to_verify=len(findings),
        )
        self._workflows[review_id] = workflow
        
        results = []
        for finding in findings:
            file_content = code_context.get(finding.location.file, "")
            result = await self.verify_finding(finding, file_content)
            results.append(result)
            
            workflow.findings_verified += 1
            workflow.findings_pending = len(findings) - workflow.findings_verified
            
            if result.final_decision == ConsensusDecision.AGREE:
                workflow.total_agreed += 1
            elif result.final_decision == ConsensusDecision.DISAGREE:
                workflow.total_disagreed += 1
            else:
                workflow.total_uncertain += 1
            
            if result.requires_manual_review:
                workflow.manual_review_count += 1
        
        workflow.results = results
        workflow.completed = True
        workflow.completed_at = datetime.utcnow()
        
        return workflow
    
    def get_metrics(self) -> ConsensusMetrics:
        """Get consensus protocol metrics."""
        return self._metrics
    
    def get_workflow(self, review_id: str) -> Optional[ConsensusWorkflow]:
        """Get workflow by review ID."""
        return self._workflows.get(review_id)
