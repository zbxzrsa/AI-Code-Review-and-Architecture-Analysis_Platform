"""
Spiral Evolution Cycle Manager

Orchestrates the complete self-evolution spiral cycle:

┌─────────────────────────────────────────────────────────────────────────────┐
│                        SPIRAL EVOLUTION CYCLE                               │
│                                                                             │
│    ┌─────────┐      Validate & Promote      ┌─────────┐                    │
│    │   V1    │─────────────────────────────▶│   V2    │                    │
│    │ (New)   │       ┌──────────────────────│(Stable) │                    │
│    └────┬────┘       │    Fix Errors        └────┬────┘                    │
│         │            │                           │                          │
│         │ Trial      │                           │ Degrade                  │
│         │ & Error    │                           │ (Poor Perf)              │
│         │            │                           │                          │
│         ▼            │                           ▼                          │
│    ┌─────────┐       │                      ┌─────────┐                    │
│    │ Compare │◀──────┴──────────────────────│   V3    │                    │
│    │ Against │                              │ (Old)   │                    │
│    │ V3 Base │◀─────────────────────────────│Quarantin│                    │
│    └─────────┘       Re-evaluate            └─────────┘                    │
│                      (After 30+ days)                                       │
│                                                                             │
│    CYCLE: V1 → V2 (promote) → V3 (degrade) → V1 (re-eval) → ...           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Key Principles:
1. V1 (New): Tests new technologies through trial and error
2. V2 (Stable): Remains unchanged for users; fixes V1 errors; optimizes compatibility
3. V3 (Old): Provides comparison baseline; excludes poor performers
4. Spiral: Continuous cycle of experimentation → validation → production → archive
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from .cross_version_feedback import CrossVersionFeedbackSystem, ErrorType
from .v3_comparison_engine import V3ComparisonEngine, ExclusionReason
from .dual_ai_coordinator import DualAICoordinator, AIType

logger = logging.getLogger(__name__)


class CyclePhase(str, Enum):
    """Current phase of the evolution cycle."""
    EXPERIMENTATION = "experimentation"     # V1 testing new tech
    ERROR_REMEDIATION = "error_remediation" # V2 fixing V1 errors
    EVALUATION = "evaluation"               # Evaluating for promotion
    PROMOTION = "promotion"                 # Promoting V1 → V2
    STABILIZATION = "stabilization"         # V2 stabilizing new tech
    DEGRADATION = "degradation"             # Degrading V2 → V3
    COMPARISON = "comparison"               # Comparing against V3
    RE_EVALUATION = "re_evaluation"         # Re-evaluating V3 → V1
    IDLE = "idle"                           # Waiting for next cycle


class EvolutionEvent(str, Enum):
    """Events in the evolution cycle."""
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    ERROR_DETECTED = "error_detected"
    ERROR_FIXED = "error_fixed"
    PROMOTION_REQUESTED = "promotion_requested"
    PROMOTION_APPROVED = "promotion_approved"
    PROMOTION_COMPLETED = "promotion_completed"
    DEGRADATION_TRIGGERED = "degradation_triggered"
    DEGRADATION_COMPLETED = "degradation_completed"
    REEVAL_REQUESTED = "reeval_requested"
    REEVAL_APPROVED = "reeval_approved"
    CYCLE_COMPLETED = "cycle_completed"


@dataclass
class EvolutionCycleState:
    """State of the evolution cycle."""
    cycle_id: str
    phase: CyclePhase
    started_at: datetime
    
    # Current technology being processed
    current_tech_id: Optional[str] = None
    current_tech_name: Optional[str] = None
    
    # Cycle metrics
    experiments_run: int = 0
    errors_fixed: int = 0
    promotions_made: int = 0
    degradations_made: int = 0
    
    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamps
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


@dataclass
class SpiralCycleConfig:
    """Configuration for the spiral evolution cycle."""
    # Timing
    cycle_interval_hours: int = 6
    experiment_duration_hours: int = 24
    stabilization_duration_hours: int = 48
    quarantine_cooldown_days: int = 30
    
    # Thresholds for promotion (V1 → V2)
    min_accuracy: float = 0.85
    max_error_rate: float = 0.05
    max_latency_p95_ms: float = 3000
    min_samples: int = 1000
    
    # Thresholds for degradation (V2 → V3)
    degradation_error_rate: float = 0.10
    degradation_accuracy: float = 0.75
    
    # Auto-evolution settings
    auto_promote: bool = False
    auto_degrade: bool = True
    auto_reeval: bool = False


class SpiralEvolutionManager:
    """
    Manages the complete spiral evolution cycle.
    
    Coordinates all components:
    - Dual-AI Coordinator (VC-AI + CR-AI per version)
    - Cross-Version Feedback (V2 fixes V1 errors)
    - V3 Comparison Engine (baseline and exclusion)
    - Version Manager (state management)
    """
    
    def __init__(
        self,
        version_manager=None,
        event_bus=None,
        config: Optional[SpiralCycleConfig] = None,
    ):
        self.version_manager = version_manager
        self.event_bus = event_bus
        self.config = config or SpiralCycleConfig()
        
        # Initialize components
        self.dual_ai = DualAICoordinator(event_bus, version_manager)
        self.feedback_system = CrossVersionFeedbackSystem(event_bus=event_bus)
        self.comparison_engine = V3ComparisonEngine(event_bus=event_bus)
        
        # Cycle state
        self._current_cycle: Optional[EvolutionCycleState] = None
        self._cycle_history: List[EvolutionCycleState] = []
        
        # Running state
        self._running = False
        self._cycle_task: Optional[asyncio.Task] = None
        
        # Technology tracking
        self._pending_promotions: List[str] = []
        self._pending_degradations: List[str] = []
        self._pending_reevaluations: List[str] = []
        
        self._lock = asyncio.Lock()
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def start(self):
        """Start the spiral evolution cycle."""
        if self._running:
            return
        
        self._running = True
        self._cycle_task = asyncio.create_task(self._run_cycle_loop())
        
        logger.info("=" * 60)
        logger.info("SPIRAL EVOLUTION CYCLE STARTED")
        logger.info("=" * 60)
        logger.info("V1 (New): Testing new technologies with trial and error")
        logger.info("V2 (Stable): User-facing, fixes V1 errors, optimizes compatibility")
        logger.info("V3 (Old): Comparison baseline, excludes poor performers")
        logger.info("=" * 60)
    
    async def stop(self):
        """Stop the spiral evolution cycle."""
        self._running = False
        
        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                raise  # Re-raise CancelledError after cleanup
        
        logger.info("Spiral evolution cycle stopped")
    
    async def _run_cycle_loop(self):
        """Main cycle loop."""
        while self._running:
            try:
                await self._execute_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")
            
            # Wait before next cycle
            await asyncio.sleep(self.config.cycle_interval_hours * 3600)
    
    # =========================================================================
    # Main Cycle Execution
    # =========================================================================
    
    async def _execute_cycle(self):
        """Execute one complete spiral evolution cycle."""
        cycle = EvolutionCycleState(
            cycle_id=str(uuid.uuid4()),
            phase=CyclePhase.EXPERIMENTATION,
            started_at=datetime.now(timezone.utc),
        )
        
        self._current_cycle = cycle
        
        logger.info(f"Starting cycle {cycle.cycle_id}")
        
        try:
            # Phase 1: V1 Experimentation
            await self._phase_experimentation(cycle)
            
            # Phase 2: Error Remediation (V2 fixes V1 errors)
            await self._phase_error_remediation(cycle)
            
            # Phase 3: Evaluation
            await self._phase_evaluation(cycle)
            
            # Phase 4: Promotion (V1 → V2)
            await self._phase_promotion(cycle)
            
            # Phase 5: V2 Stabilization
            await self._phase_stabilization(cycle)
            
            # Phase 6: Degradation (V2 → V3)
            await self._phase_degradation(cycle)
            
            # Phase 7: V3 Comparison
            await self._phase_comparison(cycle)
            
            # Phase 8: Re-evaluation (V3 → V1)
            await self._phase_reevaluation(cycle)
            
            # Complete
            cycle.phase = CyclePhase.IDLE
            cycle.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Cycle {cycle.cycle_id} completed")
            logger.info(f"  Experiments: {cycle.experiments_run}")
            logger.info(f"  Errors Fixed: {cycle.errors_fixed}")
            logger.info(f"  Promotions: {cycle.promotions_made}")
            logger.info(f"  Degradations: {cycle.degradations_made}")
            
        except Exception as e:
            logger.error(f"Cycle {cycle.cycle_id} failed: {e}")
        
        self._cycle_history.append(cycle)
    
    # =========================================================================
    # Phase 1: V1 Experimentation
    # =========================================================================
    
    async def _phase_experimentation(self, cycle: EvolutionCycleState):
        """Phase 1: V1 tests new technologies."""
        cycle.phase = CyclePhase.EXPERIMENTATION
        self._record_event(cycle, EvolutionEvent.EXPERIMENT_STARTED)
        
        logger.info("Phase 1: V1 Experimentation")
        
        # Get V1 VC-AI to manage experiments (reserved for future use)
        _ = self.dual_ai.get_ai("v1", AIType.VERSION_CONTROL)
        
        # Check V3 for exclusions before experimenting
        for tech_id in self._pending_promotions:
            exclusion_check = await self.comparison_engine.check_before_experiment(
                tech_name=tech_id,
                tech_category="unknown",
            )
            
            if not exclusion_check["proceed"]:
                logger.warning(f"V3 warns against {tech_id}: {exclusion_check['warnings']}")
        
        # Run experiments (simulated)
        cycle.experiments_run += 1
        self._record_event(cycle, EvolutionEvent.EXPERIMENT_COMPLETED)
    
    # =========================================================================
    # Phase 2: Error Remediation
    # =========================================================================
    
    async def _phase_error_remediation(self, cycle: EvolutionCycleState):
        """Phase 2: V2 fixes V1 errors."""
        cycle.phase = CyclePhase.ERROR_REMEDIATION
        
        logger.info("Phase 2: V2 Error Remediation")
        
        # Check for pending V1 errors
        pending_fixes = self.feedback_system.get_pending_fixes()
        
        for fix in pending_fixes:
            try:
                # Apply fix from V2 to V1
                success = await self.feedback_system.apply_fix_to_v1(fix.fix_id)
                
                if success:
                    cycle.errors_fixed += 1
                    self._record_event(cycle, EvolutionEvent.ERROR_FIXED, {
                        "fix_id": fix.fix_id,
                        "error_id": fix.error_id,
                    })
                    
            except Exception as e:
                logger.error(f"Error applying fix {fix.fix_id}: {e}")
        
        # V2 also optimizes compatibility of V1 technologies
        for tech_id in self._pending_promotions:
            optimization = await self.feedback_system.optimize_compatibility(tech_id)
            logger.info(f"Compatibility optimization for {tech_id}: {optimization}")
    
    # =========================================================================
    # Phase 3: Evaluation
    # =========================================================================
    
    async def _phase_evaluation(self, cycle: EvolutionCycleState):
        """Phase 3: Evaluate V1 technologies for promotion."""
        cycle.phase = CyclePhase.EVALUATION
        
        logger.info("Phase 3: Evaluation")
        
        # Get metrics from version manager
        if self.version_manager:
            v1_techs = await self.version_manager.get_version_technologies(
                self.version_manager.configs.get("v1")
            ) if hasattr(self.version_manager, 'get_version_technologies') else []
            
            for tech in v1_techs if v1_techs else []:
                evaluation = await self._evaluate_technology(tech)
                
                if evaluation["eligible"]:
                    self._pending_promotions.append(tech.tech_id)
                    self._record_event(cycle, EvolutionEvent.PROMOTION_REQUESTED, {
                        "tech_id": tech.tech_id,
                    })
    
    async def _evaluate_technology(self, tech) -> Dict[str, Any]:
        """Evaluate a technology against promotion criteria."""
        metrics = tech.metrics if hasattr(tech, 'metrics') else {}
        
        checks = {
            "accuracy": metrics.get("accuracy", 0) >= self.config.min_accuracy,
            "error_rate": metrics.get("error_rate", 1) <= self.config.max_error_rate,
            "latency": metrics.get("latency_p95_ms", float("inf")) <= self.config.max_latency_p95_ms,
            "samples": metrics.get("sample_count", 0) >= self.config.min_samples,
        }
        
        return {
            "eligible": all(checks.values()),
            "checks": checks,
            "metrics": metrics,
        }
    
    # =========================================================================
    # Phase 4: Promotion (V1 → V2)
    # =========================================================================
    
    async def _phase_promotion(self, cycle: EvolutionCycleState):
        """Phase 4: Promote validated technologies from V1 to V2."""
        cycle.phase = CyclePhase.PROMOTION
        
        logger.info("Phase 4: Promotion (V1 → V2)")
        
        for tech_id in self._pending_promotions[:]:
            try:
                # Coordinate AI handoff
                handoff = await self.dual_ai.coordinate_promotion(
                    tech_id, "v1", "v2"
                )
                
                if handoff["success"]:
                    # Promote via version manager
                    if self.version_manager:
                        await self.version_manager.promote_technology(
                            tech_id,
                            reason="Passed all evaluation criteria and error remediation"
                        )
                    
                    cycle.promotions_made += 1
                    self._pending_promotions.remove(tech_id)
                    
                    self._record_event(cycle, EvolutionEvent.PROMOTION_COMPLETED, {
                        "tech_id": tech_id,
                    })
                    
                    logger.info(f"Promoted {tech_id} from V1 to V2")
                    
            except Exception as e:
                logger.error(f"Promotion failed for {tech_id}: {e}")
    
    # =========================================================================
    # Phase 5: Stabilization
    # =========================================================================
    
    def _phase_stabilization(self, cycle: EvolutionCycleState):
        """Phase 5: V2 stabilizes newly promoted technologies."""
        cycle.phase = CyclePhase.STABILIZATION
        
        logger.info("Phase 5: V2 Stabilization")
        
        # V2 VC-AI monitors and stabilizes (reserved for future use)
        _ = self.dual_ai.get_ai("v2", AIType.VERSION_CONTROL)
        
        # Check for any stability issues
        # In production, would wait for stabilization_duration_hours
        
        logger.info("Stabilization period - monitoring V2 health")
    
    # =========================================================================
    # Phase 6: Degradation (V2 → V3)
    # =========================================================================
    
    async def _phase_degradation(self, cycle: EvolutionCycleState):
        """Phase 6: Degrade poor-performing technologies from V2 to V3."""
        cycle.phase = CyclePhase.DEGRADATION
        
        logger.info("Phase 6: Degradation (V2 → V3)")
        
        # Check V2 for poor performers
        v2_metrics = self.dual_ai.get_ai("v2", AIType.CODE_REVIEW)
        
        if v2_metrics:
            error_rate = v2_metrics.error_rate
            
            if error_rate > self.config.degradation_error_rate:
                logger.warning(f"V2 error rate {error_rate:.2%} exceeds threshold")
                
                if self.config.auto_degrade:
                    # Identify and degrade problematic technologies
                    self._record_event(cycle, EvolutionEvent.DEGRADATION_TRIGGERED)
        
        # Process pending degradations
        for tech_id in self._pending_degradations[:]:
            try:
                # Move to V3 quarantine
                profile = await self.comparison_engine.quarantine_technology(
                    tech_id=tech_id,
                    name=tech_id,
                    category="unknown",
                    source="v2",
                    metrics={"error_rate": 0.15, "accuracy": 0.70},
                    reason=ExclusionReason.POOR_PERFORMANCE,
                )
                
                cycle.degradations_made += 1
                self._pending_degradations.remove(tech_id)
                
                self._record_event(cycle, EvolutionEvent.DEGRADATION_COMPLETED, {
                    "tech_id": tech_id,
                    "exclusion_decision": profile.exclusion_decision.value,
                })
                
                logger.info(f"Degraded {tech_id} from V2 to V3")
                
            except Exception as e:
                logger.error(f"Degradation failed for {tech_id}: {e}")
    
    # =========================================================================
    # Phase 7: V3 Comparison
    # =========================================================================
    
    async def _phase_comparison(self, cycle: EvolutionCycleState):
        """Phase 7: V3 provides comparison baseline for V1 experiments."""
        cycle.phase = CyclePhase.COMPARISON
        
        logger.info("Phase 7: V3 Comparison")
        
        # Get failure insights from V3
        insights = self.comparison_engine.get_failure_insights()
        
        for insight in insights:
            logger.info(f"V3 Insight: {insight['category']} - {insight['insight']}")
        
        # Update V1 experiments with V3 comparison data
        statistics = self.comparison_engine.get_quarantine_statistics()
        logger.info(f"V3 Statistics: {statistics}")
    
    # =========================================================================
    # Phase 8: Re-evaluation (V3 → V1)
    # =========================================================================
    
    async def _phase_reevaluation(self, cycle: EvolutionCycleState):
        """Phase 8: Re-evaluate quarantined technologies for retry in V1."""
        cycle.phase = CyclePhase.RE_EVALUATION
        
        logger.info("Phase 8: Re-evaluation (V3 → V1)")
        
        # Check for technologies eligible for re-evaluation
        for tech_id in self._pending_reevaluations[:]:
            try:
                result = await self.comparison_engine.request_re_evaluation(
                    tech_id,
                    reason="Scheduled re-evaluation",
                )
                
                if result["success"]:
                    # Move back to V1 for fresh experiment
                    self._pending_reevaluations.remove(tech_id)
                    
                    self._record_event(cycle, EvolutionEvent.REEVAL_APPROVED, {
                        "tech_id": tech_id,
                    })
                    
                    logger.info(f"Re-evaluation approved for {tech_id}, moving to V1")
                    
            except Exception as e:
                logger.error(f"Re-evaluation failed for {tech_id}: {e}")
        
        self._record_event(cycle, EvolutionEvent.CYCLE_COMPLETED)
    
    # =========================================================================
    # Event Recording
    # =========================================================================
    
    def _record_event(
        self,
        cycle: EvolutionCycleState,
        event: EvolutionEvent,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Record an event in the cycle."""
        event_record = {
            "event": event.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }
        
        cycle.events.append(event_record)
        cycle.last_activity = datetime.now(timezone.utc)
        
        if self.event_bus:
            # Store task reference to prevent GC
            task = asyncio.create_task(
                self.event_bus.publish(f"evolution_{event.value}", event_record)
            )
            # Add callback to handle exceptions
            task.add_done_callback(lambda t: t.exception() if t.done() and not t.cancelled() else None)
    
    # =========================================================================
    # Manual Operations
    # =========================================================================
    
    def trigger_promotion(self, tech_id: str) -> Dict[str, Any]:
        """Manually trigger promotion of a technology."""
        if tech_id not in self._pending_promotions:
            self._pending_promotions.append(tech_id)
        
        return {"success": True, "tech_id": tech_id, "status": "queued_for_promotion"}
    
    def trigger_degradation(
        self,
        tech_id: str,
        reason: str = ""  # noqa: ARG002 - reserved for audit logging
    ) -> Dict[str, Any]:
        """Manually trigger degradation of a technology."""
        if tech_id not in self._pending_degradations:
            self._pending_degradations.append(tech_id)
        
        return {"success": True, "tech_id": tech_id, "status": "queued_for_degradation"}
    
    def request_reevaluation(self, tech_id: str) -> Dict[str, Any]:
        """Manually request re-evaluation of a quarantined technology."""
        if tech_id not in self._pending_reevaluations:
            self._pending_reevaluations.append(tech_id)
        
        return {"success": True, "tech_id": tech_id, "status": "queued_for_reevaluation"}
    
    async def report_v1_error(
        self,
        tech_id: str,
        tech_name: str,
        error_type: str,
        description: str,
    ) -> Dict[str, Any]:
        """Report an error from V1 for V2 to fix."""
        error = await self.feedback_system.report_v1_error(
            technology_id=tech_id,
            technology_name=tech_name,
            error_type=ErrorType(error_type),
            description=description,
        )
        
        return {"success": True, "error_id": error.error_id}
    
    # =========================================================================
    # Status & Reporting
    # =========================================================================
    
    def get_cycle_status(self) -> Dict[str, Any]:
        """Get current cycle status."""
        cycle = self._current_cycle
        
        if not cycle:
            return {
                "running": self._running,
                "current_cycle": None,
            }
        
        return {
            "running": self._running,
            "current_cycle": {
                "cycle_id": cycle.cycle_id,
                "phase": cycle.phase.value,
                "started_at": cycle.started_at.isoformat(),
                "experiments_run": cycle.experiments_run,
                "errors_fixed": cycle.errors_fixed,
                "promotions_made": cycle.promotions_made,
                "degradations_made": cycle.degradations_made,
                "last_activity": cycle.last_activity.isoformat(),
            },
            "pending": {
                "promotions": len(self._pending_promotions),
                "degradations": len(self._pending_degradations),
                "reevaluations": len(self._pending_reevaluations),
            },
            "ai_status": self.dual_ai.get_all_status(),
            "feedback_stats": self.feedback_system.get_feedback_statistics(),
            "quarantine_stats": self.comparison_engine.get_quarantine_statistics(),
        }
    
    def get_cycle_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get history of completed cycles."""
        return [
            {
                "cycle_id": c.cycle_id,
                "started_at": c.started_at.isoformat(),
                "completed_at": c.completed_at.isoformat() if c.completed_at else None,
                "experiments_run": c.experiments_run,
                "errors_fixed": c.errors_fixed,
                "promotions_made": c.promotions_made,
                "degradations_made": c.degradations_made,
            }
            for c in self._cycle_history[-limit:]
        ]
