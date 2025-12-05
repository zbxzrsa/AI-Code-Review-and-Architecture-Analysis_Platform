"""
Self-Evolution Cycle - Orchestrates V1→V2→V3 flow.

This module provides the core self-evolution cycle that enables:
- V1 (New): Tests new technologies through trial and error
- V2 (Stable): Remains stable for users; fixes V1 errors; optimizes compatibility
- V3 (Old): Provides comparison baseline; excludes poor performers

The cycle creates a spiral of continuous improvement:
V1 experiments → V2 validates & fixes → promote to V2 → degrade to V3 → re-evaluate → V1

Each version has its own dedicated:
- Version Control AI (VC-AI): Admin-only, manages version decisions
- Code Review AI (CR-AI): User-facing (V2 only), performs code analysis
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .version_manager import Version, VersionManager
from .experiment_framework import ExperimentFramework, ExperimentStatus, PREDEFINED_TECHNOLOGIES
from .version_ai_engine import create_version_ai

logger = logging.getLogger(__name__)


class CyclePhase(str, Enum):
    IDLE = "idle"
    EXPERIMENTING = "experimenting"
    EVALUATING = "evaluating"
    PROMOTING = "promoting"
    DEGRADING = "degrading"


@dataclass
class PromotionCriteria:
    min_accuracy: float = 0.85
    max_error_rate: float = 0.05
    max_latency_p95_ms: float = 3000
    min_samples: int = 1000


@dataclass
class EvolutionMetrics:
    cycle_count: int = 0
    promotions: int = 0
    degradations: int = 0
    experiments_started: int = 0
    experiments_completed: int = 0
    last_cycle_at: Optional[datetime] = None


@dataclass
class CycleResult:
    cycle_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    promotions_made: int = 0
    degradations_made: int = 0
    actions: List[Dict[str, Any]] = field(default_factory=list)


class SelfEvolutionCycle:
    """
    Self-Evolution Cycle: V1 (experiment) → V2 (production) → V3 (quarantine)
    """
    
    def __init__(
        self,
        version_manager: Optional[VersionManager] = None,
        cycle_interval_hours: int = 6,
    ):
        self.version_manager = version_manager or VersionManager()
        self.experiment_framework = ExperimentFramework(self.version_manager)
        self.cycle_interval = cycle_interval_hours
        self.promotion_criteria = PromotionCriteria()
        
        # AI engines per version
        self.v1_ai = create_version_ai("v1", self.version_manager, self.experiment_framework)
        self.v2_ai = create_version_ai("v2", self.version_manager)
        self.v3_ai = create_version_ai("v3", self.version_manager)
        
        self._running = False
        self._metrics = EvolutionMetrics()
        self._history: List[CycleResult] = []
        self._loop_task: Optional[asyncio.Task] = None
    
    def start(self):
        """Start the evolution cycle."""
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("Self-evolution cycle started")
    
    def stop(self):
        """Stop the evolution cycle."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
        logger.info("Self-evolution cycle stopped")
    
    async def _run_loop(self):
        """Main loop."""
        while self._running:
            try:
                result = await self.execute_cycle()
                self._history.append(result)
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            await asyncio.sleep(self.cycle_interval * 3600)
    
    async def execute_cycle(self) -> CycleResult:
        """Execute one evolution cycle."""
        result = CycleResult(
            cycle_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
        )
        
        self._metrics.cycle_count += 1
        
        # 1. Start new experiments
        await self._start_experiments(result)
        
        # 2. Evaluate running experiments
        await self._evaluate_experiments(result)
        
        # 3. Promote successful technologies to V2
        await self._process_promotions(result)
        
        # 4. Degrade failing technologies to V3
        await self._check_degradations(result)
        
        result.completed_at = datetime.now(timezone.utc)
        self._metrics.last_cycle_at = result.completed_at
        
        logger.info(f"Cycle complete: {result.promotions_made} promotions, {result.degradations_made} degradations")
        return result
    
    async def _start_experiments(self, result: CycleResult):
        """Start experiments for new technologies."""
        active = await self.experiment_framework.get_active_experiments()
        if len(active) >= 3:
            return
        
        tested = {e.config.technology_type for e in active}
        for tech_name in list(PREDEFINED_TECHNOLOGIES.keys())[:3]:
            if tech_name not in tested:
                try:
                    exp = await self.experiment_framework.create_experiment(tech_name)
                    await self.experiment_framework.start_experiment(exp.experiment_id)
                    self._metrics.experiments_started += 1
                    result.actions.append({"action": "start", "tech": tech_name})
                except Exception as e:
                    logger.error(f"Start experiment error: {e}")
    
    async def _evaluate_experiments(self, result: CycleResult):
        """Evaluate experiments with enough samples."""
        experiments = await self.experiment_framework.list_experiments(ExperimentStatus.RUNNING)
        for exp in experiments:
            if len(exp.samples) >= exp.config.min_samples:
                try:
                    await self.experiment_framework.evaluate_experiment(exp.experiment_id)
                    self._metrics.experiments_completed += 1
                    result.actions.append({"action": "evaluate", "exp_id": exp.experiment_id})
                except Exception as e:
                    logger.error(f"Evaluate error: {e}")
    
    async def _process_promotions(self, result: CycleResult):
        """Promote successful experiments to V2."""
        experiments = await self.experiment_framework.list_experiments(ExperimentStatus.COMPLETED)
        for exp in experiments:
            if exp.result and exp.result.should_promote:
                try:
                    tech = await self.version_manager.register_technology(
                        name=exp.config.name,
                        category=exp.config.category.value,
                        description=exp.config.description,
                        config=exp.config.technology_config,
                        source="experiment",
                    )
                    await self.version_manager.promote_technology(tech.tech_id, "Passed evaluation")
                    await self.v2_ai.add_promoted_technology(exp.config.technology_type)
                    
                    result.promotions_made += 1
                    self._metrics.promotions += 1
                    result.actions.append({"action": "promote", "tech": exp.config.name})
                except Exception as e:
                    logger.error(f"Promotion error: {e}")
    
    def _check_degradations(self, result: CycleResult):
        """Degrade failing technologies to V3."""
        v2_metrics = self.v2_ai.get_metrics()
        if v2_metrics["error_rate"] > 0.10:
            logger.warning(f"V2 error rate high: {v2_metrics['error_rate']:.2%}")
            result.actions.append({"action": "alert", "reason": "high_error_rate"})
    
    def get_metrics(self) -> EvolutionMetrics:
        """Get evolution metrics."""
        return self._metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get cycle status."""
        return {
            "running": self._running,
            "metrics": {
                "cycle_count": self._metrics.cycle_count,
                "promotions": self._metrics.promotions,
                "degradations": self._metrics.degradations,
                "experiments_started": self._metrics.experiments_started,
                "experiments_completed": self._metrics.experiments_completed,
            },
            "last_cycle": self._metrics.last_cycle_at.isoformat() if self._metrics.last_cycle_at else None,
            "v1_metrics": self.v1_ai.get_metrics(),
            "v2_metrics": self.v2_ai.get_metrics(),
            "v3_metrics": self.v3_ai.get_metrics(),
        }


# =============================================================================
# Enhanced Self-Evolution with Spiral Manager
# =============================================================================

class EnhancedSelfEvolutionCycle:
    """
    Enhanced self-evolution cycle with full spiral management.
    
    Integrates:
    - SpiralEvolutionManager: Full cycle orchestration
    - DualAICoordinator: VCAI + CRAI per version
    - CrossVersionFeedbackSystem: V2 fixes V1 errors
    - V3ComparisonEngine: Baseline comparison and exclusion
    
    Usage:
        cycle = EnhancedSelfEvolutionCycle()
        await cycle.start()
        
        # Report V1 error for V2 to fix
        await cycle.report_v1_error("tech_123", "new_attention", "compatibility", "Error X")
        
        # Manual promotion
        await cycle.trigger_promotion("tech_123")
        
        # Check status
        status = cycle.get_full_status()
    """
    
    def __init__(
        self,
        version_manager: Optional[VersionManager] = None,
        event_bus=None,
    ):
        # Import here to avoid circular imports
        from .spiral_evolution_manager import SpiralEvolutionManager, SpiralCycleConfig
        
        self.version_manager = version_manager or VersionManager()
        self.event_bus = event_bus
        
        # Initialize spiral evolution manager
        config = SpiralCycleConfig(
            cycle_interval_hours=6,
            experiment_duration_hours=24,
            stabilization_duration_hours=48,
            quarantine_cooldown_days=30,
            min_accuracy=0.85,
            max_error_rate=0.05,
            max_latency_p95_ms=3000,
            min_samples=1000,
            auto_promote=False,
            auto_degrade=True,
            auto_reeval=False,
        )
        
        self.spiral_manager = SpiralEvolutionManager(
            version_manager=self.version_manager,
            event_bus=event_bus,
            config=config,
        )
        
        self._running = False
    
    async def start(self):
        """Start the enhanced evolution cycle."""
        if self._running:
            return
        
        self._running = True
        await self.spiral_manager.start()
        
        logger.info("Enhanced Self-Evolution Cycle started with full spiral management")
    
    async def stop(self):
        """Stop the enhanced evolution cycle."""
        self._running = False
        await self.spiral_manager.stop()
        
        logger.info("Enhanced Self-Evolution Cycle stopped")
    
    async def report_v1_error(
        self,
        tech_id: str,
        tech_name: str,
        error_type: str,
        description: str,
    ) -> Dict[str, Any]:
        """Report a V1 error for V2 to analyze and fix."""
        return await self.spiral_manager.report_v1_error(
            tech_id, tech_name, error_type, description
        )
    
    async def trigger_promotion(self, tech_id: str) -> Dict[str, Any]:
        """Manually trigger promotion of a technology."""
        return await self.spiral_manager.trigger_promotion(tech_id)
    
    async def trigger_degradation(self, tech_id: str, reason: str) -> Dict[str, Any]:
        """Manually trigger degradation of a technology."""
        return await self.spiral_manager.trigger_degradation(tech_id, reason)
    
    async def request_reevaluation(self, tech_id: str) -> Dict[str, Any]:
        """Request re-evaluation of a quarantined technology."""
        return await self.spiral_manager.request_reevaluation(tech_id)
    
    def get_full_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the evolution cycle."""
        return {
            "running": self._running,
            "spiral_status": self.spiral_manager.get_cycle_status(),
            "cycle_history": self.spiral_manager.get_cycle_history(5),
        }
    
    def get_dual_ai_status(self) -> Dict[str, Any]:
        """Get status of all AI instances per version."""
        return self.spiral_manager.dual_ai.get_all_status()
    
    def get_user_ai_status(self) -> Dict[str, Any]:
        """Get status of user-accessible AI (V2 CR-AI)."""
        return self.spiral_manager.dual_ai.get_user_ai_status()
