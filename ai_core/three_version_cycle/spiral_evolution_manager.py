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
        elimination_config: Optional["TechEliminationConfig"] = None,
    ):
        self.version_manager = version_manager
        self.event_bus = event_bus
        self.config = config or SpiralCycleConfig()
        
        # Initialize components
        self.dual_ai = DualAICoordinator(event_bus, version_manager)
        self.feedback_system = CrossVersionFeedbackSystem(event_bus=event_bus)
        self.comparison_engine = V3ComparisonEngine(event_bus=event_bus)
        
        # Technology elimination manager (initialized later to avoid circular reference)
        self._elimination_config = elimination_config
        self._elimination_manager: Optional["TechEliminationManager"] = None
        
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
    
    @property
    def elimination_manager(self) -> "TechEliminationManager":
        """Get or create the elimination manager."""
        if self._elimination_manager is None:
            self._elimination_manager = TechEliminationManager(
                version_manager=self.version_manager,
                event_bus=self.event_bus,
                config=self._elimination_config,
            )
        return self._elimination_manager
    
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
                logger.info("Evolution cycle task cancelled")
                raise
        
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
            self._phase_stabilization(cycle)
            
            # Phase 6: Degradation (V2 → V3)
            await self._phase_degradation(cycle)
            
            # Phase 7: V3 Comparison
            self._phase_comparison(cycle)
            
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
            v1_techs = self.version_manager.get_version_technologies(
                self.version_manager.configs.get("v1")
            ) if hasattr(self.version_manager, 'get_version_technologies') else []
            
            for tech in v1_techs if v1_techs else []:
                evaluation = self._evaluate_technology(tech)
                
                if evaluation["eligible"]:
                    self._pending_promotions.append(tech.tech_id)
                    self._record_event(cycle, EvolutionEvent.PROMOTION_REQUESTED, {
                        "tech_id": tech.tech_id,
                    })
    
    def _evaluate_technology(self, tech) -> Dict[str, Any]:
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
                handoff = self.dual_ai.coordinate_promotion(
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
    
    def _phase_comparison(self, cycle: EvolutionCycleState):
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
    
    def get_elimination_status(self) -> Dict[str, Any]:
        """Get technology elimination status."""
        return self.elimination_manager.get_elimination_status()


# =============================================================================
# Technology Elimination System
# =============================================================================

@dataclass
class TechEliminationConfig:
    """
    技术淘汰配置 / Technology Elimination Configuration
    
    Controls when and how technologies are eliminated from the system.
    """
    # Performance thresholds
    min_accuracy_threshold: float = 0.75
    max_error_rate_threshold: float = 0.15
    max_latency_p95_ms: float = 5000
    
    # Failure tracking
    consecutive_failures_to_eliminate: int = 3
    evaluation_window_hours: int = 24
    min_evaluations_required: int = 5
    
    # Automation
    auto_eliminate: bool = True
    require_approval: bool = False
    
    # Archival
    archive_before_delete: bool = True
    archive_retention_days: int = 90
    
    # Notifications
    notify_on_at_risk: bool = True
    notify_on_elimination: bool = True


@dataclass
class EliminationRecord:
    """Record of an eliminated technology."""
    tech_id: str
    tech_name: str
    eliminated_at: datetime
    reasons: List[str]
    final_metrics: Dict[str, float]
    evaluation_history: List[Dict[str, Any]]
    archived_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tech_id": self.tech_id,
            "tech_name": self.tech_name,
            "eliminated_at": self.eliminated_at.isoformat(),
            "reasons": self.reasons,
            "final_metrics": self.final_metrics,
            "evaluation_count": len(self.evaluation_history),
        }


class TechEliminationManager:
    """
    技术淘汰管理器 / Technology Elimination Manager
    
    Automatically evaluates and eliminates underperforming technologies.
    
    Elimination Criteria:
    - Accuracy < 75%
    - Error rate > 15%
    - 3 consecutive evaluation failures
    
    Features:
    - Continuous performance monitoring
    - Configurable thresholds
    - Automatic elimination with optional approval
    - Archive before deletion
    - At-risk technology tracking
    """
    
    def __init__(
        self,
        version_manager=None,
        event_bus=None,
        config: Optional[TechEliminationConfig] = None,
    ):
        """
        Initialize the elimination manager.
        
        Args:
            version_manager: Version manager instance
            event_bus: Event bus for notifications
            config: Elimination configuration
        """
        self.version_manager = version_manager
        self.event_bus = event_bus
        self.config = config or TechEliminationConfig()
        
        # Failure tracking per technology
        self.failure_counts: Dict[str, int] = {}
        
        # Evaluation history per technology
        self.evaluation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Eliminated technologies archive
        self.eliminated_techs: List[EliminationRecord] = []
        
        # Pending eliminations (awaiting approval)
        self.pending_eliminations: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._total_evaluations = 0
        self._total_eliminations = 0
    
    async def evaluate_technology(self, tech_id: str) -> Dict[str, Any]:
        """
        评估技术 / Evaluate a technology for potential elimination.
        
        Args:
            tech_id: Technology identifier
            
        Returns:
            Evaluation result with recommendation
        """
        self._total_evaluations += 1
        
        # Get technology data
        tech = await self._get_technology(tech_id)
        if not tech:
            return {
                "tech_id": tech_id,
                "found": False,
                "should_eliminate": False,
            }
        
        tech_name = tech.get("name", tech_id)
        metrics = tech.get("metrics", {})
        
        # Extract metrics
        accuracy = metrics.get("accuracy", 1.0)
        error_rate = metrics.get("error_rate", 0.0)
        latency_p95 = metrics.get("latency_p95_ms", 0.0)
        
        # Evaluate against thresholds
        should_eliminate = False
        reasons = []
        
        if accuracy < self.config.min_accuracy_threshold:
            reasons.append(
                f"Accuracy {accuracy:.2%} < {self.config.min_accuracy_threshold:.2%}"
            )
            should_eliminate = True
        
        if error_rate > self.config.max_error_rate_threshold:
            reasons.append(
                f"Error rate {error_rate:.2%} > {self.config.max_error_rate_threshold:.2%}"
            )
            should_eliminate = True
        
        if latency_p95 > self.config.max_latency_p95_ms:
            reasons.append(
                f"Latency P95 {latency_p95:.0f}ms > {self.config.max_latency_p95_ms:.0f}ms"
            )
            should_eliminate = True
        
        # Record evaluation
        evaluation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accuracy": accuracy,
            "error_rate": error_rate,
            "latency_p95_ms": latency_p95,
            "should_eliminate": should_eliminate,
            "reasons": reasons,
        }
        
        if tech_id not in self.evaluation_history:
            self.evaluation_history[tech_id] = []
        
        self.evaluation_history[tech_id].append(evaluation)
        
        # Trim history
        max_history = 100
        if len(self.evaluation_history[tech_id]) > max_history:
            self.evaluation_history[tech_id] = self.evaluation_history[tech_id][-max_history:]
        
        # Update failure count
        if should_eliminate:
            self.failure_counts[tech_id] = self.failure_counts.get(tech_id, 0) + 1
            
            # Notify if at risk
            if self.config.notify_on_at_risk:
                await self._notify_at_risk(tech_id, tech_name, self.failure_counts[tech_id])
        else:
            # Reset on success
            self.failure_counts[tech_id] = 0
        
        # Check if elimination threshold reached
        elimination_triggered = False
        consecutive_failures = self.failure_counts.get(tech_id, 0)
        
        if consecutive_failures >= self.config.consecutive_failures_to_eliminate:
            # Check minimum evaluations
            if len(self.evaluation_history.get(tech_id, [])) >= self.config.min_evaluations_required:
                if self.config.auto_eliminate and not self.config.require_approval:
                    await self.eliminate_technology(tech_id, reasons)
                    elimination_triggered = True
                else:
                    # Queue for approval
                    self._queue_for_approval(tech_id, tech_name, reasons, metrics)
        
        return {
            "tech_id": tech_id,
            "tech_name": tech_name,
            "found": True,
            "should_eliminate": should_eliminate,
            "reasons": reasons,
            "consecutive_failures": consecutive_failures,
            "remaining_chances": max(
                0,
                self.config.consecutive_failures_to_eliminate - consecutive_failures
            ),
            "elimination_triggered": elimination_triggered,
            "metrics": {
                "accuracy": accuracy,
                "error_rate": error_rate,
                "latency_p95_ms": latency_p95,
            },
        }
    
    async def evaluate_for_elimination(self, tech_id: str) -> bool:
        """
        Convenience method to check if technology should be eliminated.
        
        Returns:
            True if elimination criteria met
        """
        result = await self.evaluate_technology(tech_id)
        return result.get("elimination_triggered", False)
    
    async def eliminate_technology(
        self,
        tech_id: str,
        reasons: List[str],
        force: bool = False,
    ) -> bool:
        """
        淘汰技术 / Eliminate a technology.
        
        Args:
            tech_id: Technology to eliminate
            reasons: Reasons for elimination
            force: Force elimination without checks
            
        Returns:
            True if elimination successful
        """
        tech = await self._get_technology(tech_id)
        if not tech and not force:
            logger.warning(f"Technology not found: {tech_id}")
            return False
        
        tech_name = tech.get("name", tech_id) if tech else tech_id
        
        # Create elimination record
        record = EliminationRecord(
            tech_id=tech_id,
            tech_name=tech_name,
            eliminated_at=datetime.now(timezone.utc),
            reasons=reasons,
            final_metrics=tech.get("metrics", {}) if tech else {},
            evaluation_history=self.evaluation_history.get(tech_id, []),
        )
        
        # Archive before deletion
        if self.config.archive_before_delete and tech:
            record.archived_data = tech
        
        self.eliminated_techs.append(record)
        
        # Remove from version manager
        if self.version_manager:
            try:
                await self._remove_technology(tech_id)
            except Exception as e:
                logger.error(f"Failed to remove technology {tech_id}: {e}")
        
        # Clean up tracking data
        self.failure_counts.pop(tech_id, None)
        self.evaluation_history.pop(tech_id, None)
        self.pending_eliminations.pop(tech_id, None)
        
        # Update statistics
        self._total_eliminations += 1
        
        # Notify
        if self.config.notify_on_elimination:
            await self._notify_elimination(tech_id, tech_name, reasons)
        
        logger.warning(f"Technology eliminated: {tech_id} ({tech_name})")
        logger.warning(f"  Reasons: {', '.join(reasons)}")
        
        return True
    
    def _queue_for_approval(
        self,
        tech_id: str,
        tech_name: str,
        reasons: List[str],
        metrics: Dict[str, float],
    ):
        """Queue technology for elimination approval."""
        self.pending_eliminations[tech_id] = {
            "tech_id": tech_id,
            "tech_name": tech_name,
            "reasons": reasons,
            "metrics": metrics,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_count": len(self.evaluation_history.get(tech_id, [])),
        }
        
        logger.info(f"Technology queued for elimination approval: {tech_id}")
    
    async def approve_elimination(self, tech_id: str, approver: str) -> bool:
        """
        Approve a pending elimination.
        
        Args:
            tech_id: Technology to approve elimination for
            approver: User approving the elimination
            
        Returns:
            True if approved and eliminated
        """
        pending = self.pending_eliminations.get(tech_id)
        if not pending:
            return False
        
        reasons = pending.get("reasons", [])
        reasons.append(f"Approved by {approver}")
        
        return await self.eliminate_technology(tech_id, reasons)
    
    def reject_elimination(self, tech_id: str, rejector: str) -> bool:
        """
        Reject a pending elimination.
        
        Args:
            tech_id: Technology to reject elimination for
            rejector: User rejecting the elimination
            
        Returns:
            True if rejection processed
        """
        if tech_id not in self.pending_eliminations:
            return False
        
        del self.pending_eliminations[tech_id]
        
        # Reset failure count to give another chance
        self.failure_counts[tech_id] = 0
        
        logger.info(f"Elimination rejected by {rejector}: {tech_id}")
        return True
    
    def get_at_risk_technologies(self) -> List[Dict[str, Any]]:
        """
        获取有淘汰风险的技术 / Get technologies at risk of elimination.
        
        Returns:
            List of at-risk technologies with details
        """
        at_risk = []
        
        for tech_id, failures in self.failure_counts.items():
            if failures > 0:
                remaining = self.config.consecutive_failures_to_eliminate - failures
                history = self.evaluation_history.get(tech_id, [])
                last_eval = history[-1] if history else {}
                
                at_risk.append({
                    "tech_id": tech_id,
                    "consecutive_failures": failures,
                    "remaining_chances": remaining,
                    "risk_level": "high" if remaining <= 1 else "medium" if remaining <= 2 else "low",
                    "last_evaluation": last_eval,
                    "evaluation_count": len(history),
                })
        
        # Sort by remaining chances (most at risk first)
        return sorted(at_risk, key=lambda x: x["remaining_chances"])
    
    def get_eliminated_technologies(self) -> List[Dict[str, Any]]:
        """获取已淘汰技术列表 / Get list of eliminated technologies."""
        return [record.to_dict() for record in self.eliminated_techs]
    
    def get_pending_eliminations(self) -> List[Dict[str, Any]]:
        """Get pending elimination requests."""
        return list(self.pending_eliminations.values())
    
    def get_elimination_status(self) -> Dict[str, Any]:
        """Get overall elimination system status."""
        return {
            "config": {
                "min_accuracy": self.config.min_accuracy_threshold,
                "max_error_rate": self.config.max_error_rate_threshold,
                "consecutive_failures_required": self.config.consecutive_failures_to_eliminate,
                "auto_eliminate": self.config.auto_eliminate,
            },
            "statistics": {
                "total_evaluations": self._total_evaluations,
                "total_eliminations": self._total_eliminations,
                "technologies_tracked": len(self.failure_counts),
            },
            "at_risk_count": len(self.get_at_risk_technologies()),
            "pending_approvals": len(self.pending_eliminations),
            "eliminated_count": len(self.eliminated_techs),
        }
    
    async def _get_technology(self, tech_id: str) -> Optional[Dict[str, Any]]:
        """Get technology from version manager."""
        if self.version_manager:
            if hasattr(self.version_manager, "get_technology"):
                return await self.version_manager.get_technology(tech_id)
        return None
    
    async def _remove_technology(self, tech_id: str):
        """Remove technology from version manager."""
        if self.version_manager:
            if hasattr(self.version_manager, "remove_technology"):
                await self.version_manager.remove_technology(tech_id)
    
    async def _notify_at_risk(self, tech_id: str, tech_name: str, failures: int):
        """Notify about at-risk technology."""
        if self.event_bus:
            await self.event_bus.emit(
                "tech.at_risk",
                {
                    "tech_id": tech_id,
                    "tech_name": tech_name,
                    "consecutive_failures": failures,
                    "remaining_chances": self.config.consecutive_failures_to_eliminate - failures,
                }
            )
    
    async def _notify_elimination(self, tech_id: str, tech_name: str, reasons: List[str]):
        """Notify about technology elimination."""
        if self.event_bus:
            await self.event_bus.emit(
                "tech.eliminated",
                {
                    "tech_id": tech_id,
                    "tech_name": tech_name,
                    "reasons": reasons,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
