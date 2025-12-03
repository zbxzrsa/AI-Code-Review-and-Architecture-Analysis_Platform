"""
Self-Evolution Cycle - Orchestrates V1→V2→V3 flow.
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
    
    async def start(self):
        """Start the evolution cycle."""
        self._running = True
        asyncio.create_task(self._run_loop())
        logger.info("Self-evolution cycle started")
    
    async def stop(self):
        """Stop the evolution cycle."""
        self._running = False
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
    
    async def _check_degradations(self, result: CycleResult):
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
