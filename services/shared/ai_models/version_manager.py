"""
Version Manager - Orchestrates the Three-Version AI System

Manages the lifecycle:
- V1 (Experimental): New technologies, trial and error
- V2 (Production): Stable, user-facing, unchanged
- V3 (Quarantine): Deprecated, poor reviews elimination

Self-updating cycle:
V1 experiments → Success → Promote to V2 → V2 issues → Demote to V3 → V3 cleanup
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio

from .base_ai import AIConfig, VersionType, ModelProvider, DEFAULT_CONFIGS
from .version_control_ai import VersionControlAI, VersionTransition, EvaluationResult
from .code_ai import CodeAI, UserModelPreference
from .model_registry import UserModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class VersionInstance:
    """Instance of a version with both AI models"""
    version_type: VersionType
    version_control_ai: VersionControlAI
    code_ai: CodeAI
    is_active: bool = True
    created_at: str = ""
    last_updated: str = ""
    
    # Statistics
    total_requests: int = 0
    successful_promotions: int = 0
    failed_experiments: int = 0


@dataclass
class CycleStatus:
    """Status of the self-updating cycle"""
    last_cycle_time: str
    cycles_completed: int
    pending_promotions: List[str]
    pending_demotions: List[str]
    active_experiments: List[str]
    quarantined_models: List[str]


class VersionManager:
    """
    Three-Version AI System Manager
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Version Manager                       │
    ├─────────────────────────────────────────────────────────┤
    │  V1 (Experimental)     V2 (Production)    V3 (Quarantine)│
    │  ┌─────────────┐      ┌─────────────┐    ┌─────────────┐│
    │  │VC AI (Int.) │      │VC AI (Int.) │    │VC AI (Int.) ││
    │  │Code AI (Usr)│ ───► │Code AI (Usr)│ ──►│Code AI (Usr)││
    │  └─────────────┘      └─────────────┘    └─────────────┘│
    │       ↑                      │                  │       │
    │       └──────────────────────┴──────────────────┘       │
    │                    Self-Updating Cycle                   │
    └─────────────────────────────────────────────────────────┘
    
    Users can ONLY access Code AI in V2 (Production)
    """
    
    def __init__(self, model_registry: UserModelRegistry):
        self.model_registry = model_registry
        self.versions: Dict[VersionType, VersionInstance] = {}
        self.cycle_status = CycleStatus(
            last_cycle_time="",
            cycles_completed=0,
            pending_promotions=[],
            pending_demotions=[],
            active_experiments=[],
            quarantined_models=[]
        )
        
        self._initialize_versions()
    
    def _initialize_versions(self) -> None:
        """Initialize all three versions with their AI models"""
        for version_type in VersionType:
            configs = DEFAULT_CONFIGS[version_type]
            
            # Create Version Control AI (Internal - not user accessible)
            vc_ai = VersionControlAI(
                config=configs['version_control'],
                version_type=version_type
            )
            
            # Create Code AI (User-facing)
            code_ai = CodeAI(
                config=configs['code_ai'],
                version_type=version_type
            )
            
            self.versions[version_type] = VersionInstance(
                version_type=version_type,
                version_control_ai=vc_ai,
                code_ai=code_ai,
                is_active=True,
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
            
            logger.info(f"Initialized {version_type.value} with VC AI and Code AI")
    
    # User Access Methods (V2 Production Only)
    
    def get_user_code_ai(
        self,
        user_id: str,
        preference: Optional[UserModelPreference] = None
    ) -> CodeAI:
        """
        Get Code AI for user - ONLY V2 Production
        
        Users can only access the stable V2 Code AI
        """
        v2 = self.versions[VersionType.V2_PRODUCTION]
        
        if preference:
            v2.code_ai.set_user_preference(preference)
        
        return v2.code_ai
    
    async def analyze_code_for_user(
        self,
        user_id: str,
        code: str,
        language: str,
        analysis_type: str = "review"
    ) -> Dict[str, Any]:
        """
        Analyze code for user using V2 Production Code AI
        """
        code_ai = self.get_user_code_ai(user_id)
        result = await code_ai.analyze_code(code, language, analysis_type)
        
        # Track usage
        self.versions[VersionType.V2_PRODUCTION].total_requests += 1
        
        return result
    
    # Internal Methods (Version Control)
    
    def get_version_control_ai(self, version_type: VersionType) -> VersionControlAI:
        """Get Version Control AI - Internal use only"""
        return self.versions[version_type].version_control_ai
    
    async def run_experiment(
        self,
        experiment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run an experiment in V1 (Experimental)
        
        This is where new technologies are tested
        """
        v1 = self.versions[VersionType.V1_EXPERIMENTAL]
        
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        logger.info(f"Starting experiment {experiment_id} in V1")
        
        # Run experiment using V1 Code AI
        # In production, this would test new model configurations
        result = {
            'experiment_id': experiment_id,
            'version': 'v1',
            'status': 'running',
            'config': experiment_config,
            'started_at': datetime.now().isoformat()
        }
        
        self.cycle_status.active_experiments.append(experiment_id)
        
        return result
    
    async def evaluate_experiment(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        user_feedback: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate an experiment using V1 Version Control AI
        """
        v1_vc = self.get_version_control_ai(VersionType.V1_EXPERIMENTAL)
        
        evaluation = await v1_vc.evaluate_model(
            model_id=experiment_id,
            metrics=metrics,
            user_feedback=user_feedback
        )
        
        return evaluation
    
    async def promote_to_production(
        self,
        experiment_id: str,
        evaluation: EvaluationResult
    ) -> Optional[VersionTransition]:
        """
        Promote successful experiment from V1 to V2
        
        V1 (Experimental) → V2 (Production)
        """
        if not evaluation.eligible_for_promotion:
            logger.warning(f"Experiment {experiment_id} not eligible for promotion")
            return None
        
        v1_vc = self.get_version_control_ai(VersionType.V1_EXPERIMENTAL)
        
        transition = await v1_vc.decide_transition(evaluation)
        
        if transition and transition.approved:
            # Update V2 with the new model
            self.versions[VersionType.V2_PRODUCTION].last_updated = datetime.now().isoformat()
            self.versions[VersionType.V2_PRODUCTION].successful_promotions += 1
            
            # Remove from experiments
            if experiment_id in self.cycle_status.active_experiments:
                self.cycle_status.active_experiments.remove(experiment_id)
            
            self.cycle_status.pending_promotions.append(experiment_id)
            
            logger.info(f"Promoted {experiment_id} to V2 Production")
        
        return transition
    
    async def demote_to_quarantine(
        self,
        model_id: str,
        reason: str,
        metrics: Dict[str, float]
    ) -> Optional[VersionTransition]:
        """
        Demote problematic model from V2 to V3
        
        V2 (Production) → V3 (Quarantine)
        """
        v2_vc = self.get_version_control_ai(VersionType.V2_PRODUCTION)
        
        # Create evaluation for demotion
        evaluation = EvaluationResult(
            model_id=model_id,
            version=VersionType.V2_PRODUCTION,
            score=0.0,
            metrics=metrics,
            issues=[reason],
            recommendations=["Demote to quarantine"],
            eligible_for_promotion=False,
            eligible_for_demotion=True
        )
        
        transition = await v2_vc.decide_transition(evaluation)
        
        if transition and transition.approved:
            self.cycle_status.quarantined_models.append(model_id)
            self.versions[VersionType.V3_QUARANTINE].last_updated = datetime.now().isoformat()
            
            logger.warning(f"Demoted {model_id} to V3 Quarantine: {reason}")
        
        return transition
    
    async def cleanup_quarantine(self) -> List[str]:
        """
        Clean up V3 Quarantine - remove deprecated models
        
        Models with consistently poor reviews are eliminated
        """
        # Get V3 VC-AI for quarantine analysis (reserved for extended metrics check)
        _ = self.get_version_control_ai(VersionType.V3_QUARANTINE)
        
        removed = []
        
        for model_id in self.cycle_status.quarantined_models.copy():
            # Check if model should be permanently removed
            # In production, this would check extended metrics from v3_vc
            should_remove = self._should_remove_from_quarantine(model_id)
            
            if should_remove:
                removed.append(model_id)
                self.cycle_status.quarantined_models.remove(model_id)
                logger.info(f"Removed deprecated model {model_id} from V3")
    
        return removed
    
    def _should_remove_from_quarantine(self, model_id: str) -> bool:
        """Check if a quarantined model should be permanently removed.
        
        TODO: In production, implement extended metrics check with v3_vc
        """
        # Placeholder - always remove in dev mode
        # Production would check: age in quarantine, failure severity, recovery attempts
        _ = model_id  # Acknowledge parameter for future use
        return True
    
    async def run_self_update_cycle(self) -> Dict[str, Any]:
        """
        Run the complete self-updating cycle
        
        Cycle:
        1. V1 experiments with new technologies
        2. Successful experiments promoted to V2
        3. V2 issues demoted to V3
        4. V3 cleanup of deprecated models
        5. Repeat
        """
        cycle_start = datetime.now()
        
        cycle_result = {
            'cycle_id': f"cycle_{cycle_start.strftime('%Y%m%d%H%M%S')}",
            'started_at': cycle_start.isoformat(),
            'promotions': [],
            'demotions': [],
            'cleanups': [],
            'experiments_started': 0
        }
        
        logger.info("Starting self-update cycle")
        
        # Step 1: Evaluate active V1 experiments
        for exp_id in self.cycle_status.active_experiments.copy():
            # Get metrics (placeholder)
            metrics = {
                'accuracy': 0.90,
                'error_rate': 0.03,
                'latency_p95': 2000
            }
            
            evaluation = await self.evaluate_experiment(
                exp_id, metrics, []
            )
            
            if evaluation.eligible_for_promotion:
                transition = await self.promote_to_production(exp_id, evaluation)
                if transition:
                    cycle_result['promotions'].append(exp_id)
        
        # Step 2: Check V2 production health
        v2_status = await self.get_version_status(VersionType.V2_PRODUCTION)
        
        # Step 3: Demote problematic V2 models
        # (In production, this would check actual metrics)
        
        # Step 4: Cleanup V3 quarantine
        removed = await self.cleanup_quarantine()
        cycle_result['cleanups'] = removed
        
        # Update cycle status
        self.cycle_status.last_cycle_time = cycle_start.isoformat()
        self.cycle_status.cycles_completed += 1
        
        cycle_result['completed_at'] = datetime.now().isoformat()
        
        logger.info(f"Self-update cycle completed: {cycle_result}")
        
        return cycle_result
    
    # Status Methods
    
    async def get_version_status(
        self,
        version_type: VersionType
    ) -> Dict[str, Any]:
        """Get status of a specific version"""
        version = self.versions[version_type]
        vc_status = await version.version_control_ai.get_version_status()
        
        return {
            'version_type': version_type.value,
            'is_active': version.is_active,
            'created_at': version.created_at,
            'last_updated': version.last_updated,
            'total_requests': version.total_requests,
            'successful_promotions': version.successful_promotions,
            'failed_experiments': version.failed_experiments,
            'version_control_status': vc_status,
            'code_ai_metrics': version.code_ai.get_metrics()
        }
    
    async def get_all_versions_status(self) -> Dict[str, Any]:
        """Get status of all versions"""
        statuses = {}
        
        for version_type in VersionType:
            statuses[version_type.value] = await self.get_version_status(version_type)
        
        return {
            'versions': statuses,
            'cycle_status': {
                'last_cycle_time': self.cycle_status.last_cycle_time,
                'cycles_completed': self.cycle_status.cycles_completed,
                'active_experiments': len(self.cycle_status.active_experiments),
                'pending_promotions': len(self.cycle_status.pending_promotions),
                'quarantined_models': len(self.cycle_status.quarantined_models)
            }
        }
    
    def get_user_accessible_version(self) -> VersionType:
        """Get the version users can access (always V2)"""
        return VersionType.V2_PRODUCTION
