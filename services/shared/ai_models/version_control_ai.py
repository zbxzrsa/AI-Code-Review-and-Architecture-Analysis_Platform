"""
Version Control AI - Internal AI for Version Management

This AI is responsible for:
- Managing version transitions (V1 → V2, V2 → V3)
- Evaluating model performance
- Making promotion/demotion decisions
- Optimizing compatibility between versions
- Self-updating the project cycle
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import json

from .base_ai import BaseAI, AIConfig, VersionType, ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class VersionTransition:
    """Record of a version transition"""
    from_version: VersionType
    to_version: VersionType
    model_id: str
    timestamp: str
    reason: str
    metrics: Dict[str, float]
    approved: bool
    approver: str  # 'auto' or admin user_id


@dataclass
class EvaluationResult:
    """Result of model evaluation"""
    model_id: str
    version: VersionType
    score: float
    metrics: Dict[str, float]
    issues: List[str]
    recommendations: List[str]
    eligible_for_promotion: bool
    eligible_for_demotion: bool


class VersionControlAI(BaseAI):
    """
    Version Control AI - Internal model for version management
    
    Features:
    - Automatic version transition decisions
    - Performance evaluation across versions
    - Compatibility checking
    - Self-evolution management
    """
    
    # Thresholds for promotion/demotion
    PROMOTION_ACCURACY_THRESHOLD = 0.85
    PROMOTION_ERROR_RATE_THRESHOLD = 0.05
    DEMOTION_ERROR_RATE_THRESHOLD = 0.15
    DEMOTION_NEGATIVE_FEEDBACK_THRESHOLD = 0.3
    
    def __init__(self, config: AIConfig, version_type: VersionType):
        super().__init__(config, version_type)
        self.transitions: List[VersionTransition] = []
        self.evaluations: List[EvaluationResult] = []
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response for version control tasks"""
        # Implementation would connect to actual AI provider
        # This is a placeholder for the interface
        self.record_request(tokens_used=100, success=True)
        
        # In production, this would call the actual AI model
        return f"Version Control AI ({self.version_type.value}) response"
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Stream response for version control tasks"""
        response = await self.generate(prompt, system_prompt, **kwargs)
        for chunk in response.split():
            yield chunk + " "
            await asyncio.sleep(0.05)
    
    async def analyze_code(
        self,
        code: str,
        language: str,
        analysis_type: str = "version_check"
    ) -> Dict[str, Any]:
        """Analyze code for version compatibility"""
        return {
            'version_type': self.version_type.value,
            'analysis_type': analysis_type,
            'compatible': True,
            'recommendations': []
        }
    
    async def evaluate_model(
        self,
        model_id: str,
        metrics: Dict[str, float],
        user_feedback: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate a model for promotion/demotion
        
        Args:
            model_id: ID of the model to evaluate
            metrics: Performance metrics
            user_feedback: User feedback data
            
        Returns:
            EvaluationResult with recommendations
        """
        # Calculate scores
        accuracy = metrics.get('accuracy', 0)
        error_rate = metrics.get('error_rate', 1)
        latency_p95 = metrics.get('latency_p95', 10000)
        
        # Calculate feedback score
        positive_feedback = sum(1 for f in user_feedback if f.get('rating', 0) >= 4)
        total_feedback = len(user_feedback) or 1
        feedback_score = positive_feedback / total_feedback
        
        # Calculate overall score
        score = (
            accuracy * 0.4 +
            (1 - error_rate) * 0.3 +
            feedback_score * 0.2 +
            (1 - min(latency_p95 / 5000, 1)) * 0.1
        )
        
        # Determine eligibility
        issues = []
        recommendations = []
        
        eligible_for_promotion = False
        eligible_for_demotion = False
        
        if accuracy >= self.PROMOTION_ACCURACY_THRESHOLD:
            if error_rate <= self.PROMOTION_ERROR_RATE_THRESHOLD:
                eligible_for_promotion = True
                recommendations.append("Model meets promotion criteria")
        else:
            issues.append(f"Accuracy {accuracy:.2%} below threshold {self.PROMOTION_ACCURACY_THRESHOLD:.2%}")
        
        if error_rate >= self.DEMOTION_ERROR_RATE_THRESHOLD:
            eligible_for_demotion = True
            issues.append(f"Error rate {error_rate:.2%} exceeds threshold")
        
        if (1 - feedback_score) >= self.DEMOTION_NEGATIVE_FEEDBACK_THRESHOLD:
            eligible_for_demotion = True
            issues.append(f"Negative feedback rate {(1-feedback_score):.2%} too high")
        
        result = EvaluationResult(
            model_id=model_id,
            version=self.version_type,
            score=score,
            metrics={
                'accuracy': accuracy,
                'error_rate': error_rate,
                'feedback_score': feedback_score,
                'latency_p95': latency_p95
            },
            issues=issues,
            recommendations=recommendations,
            eligible_for_promotion=eligible_for_promotion,
            eligible_for_demotion=eligible_for_demotion
        )
        
        self.evaluations.append(result)
        return result
    
    async def decide_transition(
        self,
        evaluation: EvaluationResult
    ) -> Optional[VersionTransition]:
        """
        Decide whether to transition a model between versions
        
        V1 (Experimental) → V2 (Production): When model is stable and tested
        V2 (Production) → V3 (Quarantine): When model has issues
        V3 (Quarantine) → Archived: Permanent removal
        """
        transition = None
        
        if self.version_type == VersionType.V1_EXPERIMENTAL:
            if evaluation.eligible_for_promotion:
                transition = VersionTransition(
                    from_version=VersionType.V1_EXPERIMENTAL,
                    to_version=VersionType.V2_PRODUCTION,
                    model_id=evaluation.model_id,
                    timestamp=datetime.now().isoformat(),
                    reason="Model passed all promotion criteria",
                    metrics=evaluation.metrics,
                    approved=True,
                    approver='auto'
                )
                logger.info(f"Auto-promoting model {evaluation.model_id} to V2")
        
        elif self.version_type == VersionType.V2_PRODUCTION:
            if evaluation.eligible_for_demotion:
                transition = VersionTransition(
                    from_version=VersionType.V2_PRODUCTION,
                    to_version=VersionType.V3_QUARANTINE,
                    model_id=evaluation.model_id,
                    timestamp=datetime.now().isoformat(),
                    reason="; ".join(evaluation.issues),
                    metrics=evaluation.metrics,
                    approved=True,
                    approver='auto'
                )
                logger.warning(f"Demoting model {evaluation.model_id} to V3 quarantine")
        
        if transition:
            self.transitions.append(transition)
        
        return transition
    
    async def check_compatibility(
        self,
        source_version: VersionType,
        target_version: VersionType,
        model_config: AIConfig
    ) -> Dict[str, Any]:
        """Check if a model is compatible with target version"""
        compatibility = {
            'compatible': True,
            'warnings': [],
            'blockers': []
        }
        
        # V1 → V2: Strict requirements
        if source_version == VersionType.V1_EXPERIMENTAL and target_version == VersionType.V2_PRODUCTION:
            if model_config.temperature > 0.8:
                compatibility['warnings'].append(
                    "High temperature may cause inconsistent outputs in production"
                )
            if not model_config.supports_streaming:
                compatibility['warnings'].append(
                    "Streaming not supported - may affect user experience"
                )
        
        # V2 → V3: Always compatible (demotion)
        elif source_version == VersionType.V2_PRODUCTION and target_version == VersionType.V3_QUARANTINE:
            compatibility['compatible'] = True
        
        return compatibility
    
    async def get_version_status(self) -> Dict[str, Any]:
        """Get current status of this version"""
        recent_evals = self.evaluations[-10:] if self.evaluations else []
        
        avg_score = sum(e.score for e in recent_evals) / len(recent_evals) if recent_evals else 0
        
        return {
            'version_type': self.version_type.value,
            'model_id': self.config.model_id,
            'is_active': self.is_active,
            'total_transitions': len(self.transitions),
            'recent_evaluations': len(recent_evals),
            'average_score': avg_score,
            'metrics': self.get_metrics()
        }
    
    async def optimize_version(self) -> Dict[str, Any]:
        """
        Optimize the current version based on collected data
        
        This is part of the self-updating cycle
        """
        optimizations = {
            'version_type': self.version_type.value,
            'timestamp': datetime.now().isoformat(),
            'actions': []
        }
        
        if self.version_type == VersionType.V1_EXPERIMENTAL:
            # V1: Focus on testing new configurations
            optimizations['actions'].append({
                'type': 'experiment',
                'description': 'Testing new model configurations'
            })
        
        elif self.version_type == VersionType.V2_PRODUCTION:
            # V2: Maintain stability, minor optimizations only
            optimizations['actions'].append({
                'type': 'stability_check',
                'description': 'Verifying production stability'
            })
        
        elif self.version_type == VersionType.V3_QUARANTINE:
            # V3: Analyze failures, prepare for removal or recovery
            optimizations['actions'].append({
                'type': 'failure_analysis',
                'description': 'Analyzing quarantined model issues'
            })
        
        return optimizations
