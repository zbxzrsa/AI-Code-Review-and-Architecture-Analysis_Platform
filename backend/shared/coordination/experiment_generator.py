"""
Experiment Generator

Proactive experiment suggestion:
- Industry research scanning
- Internal signal analysis
- Competitor monitoring
- V3 re-evaluation opportunities
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .event_types import EventType, Version, VersionEvent, ExperimentProposal

logger = logging.getLogger(__name__)


class ExperimentSource(str, Enum):
    """Source of experiment proposal."""
    INDUSTRY_RESEARCH = "industry_research"
    INTERNAL_SIGNAL = "internal_signal"
    USER_FEEDBACK = "user_feedback"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    V3_REEVALUATION = "v3_reevaluation"
    MANUAL = "manual"


@dataclass
class ResearchSignal:
    """Signal from research sources."""
    signal_id: str
    source: ExperimentSource
    title: str
    description: str
    relevance_score: float  # 0-1
    detected_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperimentGenerator:
    """
    Proactive experiment suggestion engine.
    
    Sources:
    - Industry research (arXiv, model releases)
    - Internal signals (bottlenecks, V3 review)
    - User feedback (complaints, requests)
    - Competitor analysis (feature gaps)
    """
    
    def __init__(
        self,
        event_bus = None,
        quarantine_manager = None,
        health_monitor = None,
    ):
        self.event_bus = event_bus
        self.quarantine = quarantine_manager
        self.health = health_monitor
        
        # Proposals
        self._proposals: Dict[str, ExperimentProposal] = {}
        
        # Research signals
        self._signals: List[ResearchSignal] = []
        
        # Generation config
        self._model_releases_to_watch = [
            "gpt-4", "gpt-5", "claude-3", "claude-4",
            "gemini", "llama", "mistral",
        ]
        
        self._analysis_types_to_optimize = [
            "security", "performance", "architecture",
        ]
    
    async def generate_proposals(self) -> List[ExperimentProposal]:
        """Generate new experiment proposals from all sources."""
        proposals = []
        
        # Industry research
        proposals.extend(await self._scan_industry_research())
        
        # Internal signals
        proposals.extend(await self._analyze_internal_signals())
        
        # V3 re-evaluation
        proposals.extend(await self._review_quarantine())
        
        # User feedback
        proposals.extend(await self._analyze_user_feedback())
        
        # Store and emit events
        for proposal in proposals:
            self._proposals[proposal.experiment_id] = proposal
            
            await self._emit_event(
                EventType.EXPERIMENT_CREATED,
                {
                    "experiment_id": proposal.experiment_id,
                    "title": proposal.title,
                    "source": proposal.source,
                    "hypothesis": proposal.hypothesis,
                },
            )
        
        return proposals
    
    async def _scan_industry_research(self) -> List[ExperimentProposal]:
        """Scan industry research for new techniques."""
        proposals = []
        
        # Simulate checking for new model releases
        new_models = await self._check_model_releases()
        
        for model in new_models:
            proposal = ExperimentProposal(
                title=f"Evaluate {model['name']} for code analysis",
                hypothesis=f"{model['name']}'s {model['improvement']} may improve analysis quality",
                success_criteria={
                    "accuracy_improvement": "> 10%",
                    "latency_acceptable": "p95 < 4s",
                    "cost_acceptable": "< 50% increase",
                },
                evaluation_period_days=14,
                sample_size=2000,
                rollback_plan=f"Revert to current model if criteria not met",
                estimated_cost_usd=500.0,
                source=ExperimentSource.INDUSTRY_RESEARCH.value,
            )
            proposals.append(proposal)
        
        return proposals
    
    async def _check_model_releases(self) -> List[Dict[str, Any]]:
        """Check for new AI model releases."""
        # In production, scan OpenAI/Anthropic APIs, blogs, arXiv
        # Mock new releases
        return [
            # {
            #     "name": "gpt-4.5-turbo",
            #     "improvement": "improved reasoning capabilities",
            #     "released": datetime.utcnow(),
            # }
        ]
    
    async def _analyze_internal_signals(self) -> List[ExperimentProposal]:
        """Analyze internal performance signals."""
        proposals = []
        
        if not self.health:
            return proposals
        
        # Check for performance bottlenecks
        metrics = self.health.get_current_health()
        if not metrics:
            return proposals
        
        # High latency → optimize slow analysis types
        if metrics.latency_p95_ms > 2000:
            proposal = ExperimentProposal(
                title="Optimize high-latency analysis pipeline",
                hypothesis="Parallel processing and caching will reduce p95 latency by 30%",
                success_criteria={
                    "latency_reduction": "> 30%",
                    "accuracy_maintained": "> 95% of current",
                    "cost_impact": "< 10% increase",
                },
                evaluation_period_days=7,
                sample_size=1000,
                rollback_plan="Revert to sequential processing",
                estimated_cost_usd=200.0,
                source=ExperimentSource.INTERNAL_SIGNAL.value,
            )
            proposals.append(proposal)
        
        # High cost → optimize prompts
        if metrics.cost_per_review > 0.15:
            proposal = ExperimentProposal(
                title="Optimize prompt efficiency for cost reduction",
                hypothesis="Shorter, focused prompts will reduce cost by 25% with minimal quality impact",
                success_criteria={
                    "cost_reduction": "> 20%",
                    "accuracy_maintained": "> 98% of current",
                    "latency_impact": "< 10% increase",
                },
                evaluation_period_days=7,
                sample_size=1000,
                rollback_plan="Revert to original prompts",
                estimated_cost_usd=150.0,
                source=ExperimentSource.INTERNAL_SIGNAL.value,
            )
            proposals.append(proposal)
        
        return proposals
    
    async def _review_quarantine(self) -> List[ExperimentProposal]:
        """Review V3 quarantine for retry opportunities."""
        proposals = []
        
        if not self.quarantine:
            return proposals
        
        # Get records pending review
        pending = self.quarantine.get_quarantine_records(pending_review=True)
        
        for record in pending:
            # Check if context has changed (new models, fixes available)
            if await self._context_changed_for_retry(record):
                proposal = ExperimentProposal(
                    title=f"Retry quarantined experiment: {record.experiment_id}",
                    hypothesis=f"With {self._get_context_change(record)}, previously failed approach may now succeed",
                    success_criteria={
                        "accuracy": "> 85%",
                        "error_rate": "< 5%",
                        "latency_p95": "< 4s",
                    },
                    evaluation_period_days=7,
                    sample_size=1000,
                    rollback_plan="Re-quarantine if same issues persist",
                    estimated_cost_usd=300.0,
                    source=ExperimentSource.V3_REEVALUATION.value,
                )
                proposals.append(proposal)
        
        return proposals
    
    async def _context_changed_for_retry(self, record) -> bool:
        """Check if context has changed enough to retry."""
        # Check if:
        # - New models available that might work better
        # - Fixes implemented for root cause
        # - Enough time passed for external factors to change
        
        if not record.quarantined_at:
            return False
        
        days_since = (datetime.utcnow() - record.quarantined_at).days
        return days_since >= 90  # Quarterly review
    
    def _get_context_change(self, record) -> str:
        """Describe what context has changed."""
        return "new model capabilities and platform improvements"
    
    async def _analyze_user_feedback(self) -> List[ExperimentProposal]:
        """Analyze user feedback for improvement opportunities."""
        proposals = []
        
        # In production, aggregate user feedback from:
        # - Support tickets
        # - Feedback forms
        # - Feature requests
        # - Usage patterns
        
        # Mock feedback analysis
        common_complaints = await self._get_common_complaints()
        
        for complaint in common_complaints:
            if complaint["frequency"] > 10:  # More than 10 mentions
                proposal = ExperimentProposal(
                    title=f"Address user feedback: {complaint['category']}",
                    hypothesis=f"Improving {complaint['category']} will increase user satisfaction by 15%",
                    success_criteria={
                        "user_satisfaction": "> 4.5/5.0",
                        "complaint_reduction": "> 50%",
                        "no_regression": "all metrics maintained",
                    },
                    evaluation_period_days=14,
                    sample_size=2000,
                    rollback_plan="Revert if satisfaction decreases",
                    estimated_cost_usd=400.0,
                    source=ExperimentSource.USER_FEEDBACK.value,
                )
                proposals.append(proposal)
        
        return proposals
    
    async def _get_common_complaints(self) -> List[Dict[str, Any]]:
        """Get common user complaints."""
        # Mock data
        return [
            # {"category": "false_positives", "frequency": 15},
            # {"category": "slow_analysis", "frequency": 12},
        ]
    
    async def approve_proposal(
        self,
        experiment_id: str,
        approver: str,
    ) -> bool:
        """Approve experiment proposal."""
        proposal = self._proposals.get(experiment_id)
        if not proposal:
            return False
        
        proposal.status = "approved"
        
        await self._emit_event(
            EventType.EXPERIMENT_STARTED,
            {
                "experiment_id": experiment_id,
                "title": proposal.title,
                "approver": approver,
                "evaluation_period_days": proposal.evaluation_period_days,
            },
        )
        
        logger.info(f"Experiment {experiment_id} approved by {approver}")
        return True
    
    async def reject_proposal(
        self,
        experiment_id: str,
        reason: str,
    ) -> bool:
        """Reject experiment proposal."""
        proposal = self._proposals.get(experiment_id)
        if not proposal:
            return False
        
        proposal.status = "rejected"
        
        logger.info(f"Experiment {experiment_id} rejected: {reason}")
        return True
    
    def get_proposals(
        self,
        status: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[ExperimentProposal]:
        """Get experiment proposals."""
        proposals = list(self._proposals.values())
        
        if status:
            proposals = [p for p in proposals if p.status == status]
        
        if source:
            proposals = [p for p in proposals if p.source == source]
        
        return proposals
    
    async def _emit_event(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
    ):
        """Emit event to event bus."""
        event = VersionEvent(
            event_type=event_type,
            version=Version.V1_EXPERIMENTATION,
            payload=payload,
            source="experiment-generator",
        )
        
        if self.event_bus:
            await self.event_bus.publish(event_type.value, event.to_dict())
