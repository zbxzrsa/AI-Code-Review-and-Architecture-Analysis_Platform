"""
Three-Version Parallel Orchestrator

A unified orchestrator that runs all three versions (V1, V2, V3) in parallel
with their respective Version Control AI (VC-AI) and Code Review AI (CR-AI).

Architecture:
    ┌────────────────────────────────────────────────────────────────────────┐
    │                    THREE-VERSION PARALLEL SYSTEM                        │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │   V1 (EXPERIMENTAL)        V2 (STABLE)           V3 (QUARANTINE)       │
    │   ┌───────────────┐        ┌───────────────┐     ┌───────────────┐     │
    │   │   V1-VCAI     │        │   V2-VCAI     │     │   V3-VCAI     │     │
    │   │ • Experiments │───────►│ • Fix V1 bugs │────►│ • Compare     │     │
    │   │ • Trial/Error │        │ • Optimize    │     │ • Eliminate   │     │
    │   └───────────────┘        └───────────────┘     └───────────────┘     │
    │   ┌───────────────┐        ┌───────────────┐     ┌───────────────┐     │
    │   │   V1-CRAI     │        │   V2-CRAI     │     │   V3-CRAI     │     │
    │   │ • Shadow test │        │ • USER-FACING │     │ • Baseline    │     │
    │   └───────────────┘        └───────────────┘     └───────────────┘     │
    │                                                                         │
    │   SPIRAL CYCLE: V1 → V2 (promote) → V3 (degrade) → V1 (re-evaluate)   │
    └────────────────────────────────────────────────────────────────────────┘

Usage:
    orchestrator = ThreeVersionOrchestrator()
    await orchestrator.start()
    
    # User code review (goes to V2 CR-AI)
    result = await orchestrator.user_code_review(code, language)
    
    # Admin can test on V1
    result = await orchestrator.experimental_review(code, language)
    
    # Report V1 error for V2 to fix
    await orchestrator.report_error("tech_001", "new_attention", "compatibility", "Error X")
    
    # Get system status
    status = orchestrator.get_parallel_status()
    
    await orchestrator.stop()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

# Import all three-version components
from .version_manager import Version, VersionManager, VersionState
from .dual_ai_coordinator import DualAICoordinator, AIType, AIStatus, DualAIRequestHandler
from .version_ai_engine import create_version_ai, V1ExperimentalAI, V2ProductionAI, V3QuarantineAI
from .self_evolution_cycle import SelfEvolutionCycle, EnhancedSelfEvolutionCycle
from .cross_version_feedback import CrossVersionFeedbackSystem, ErrorType
from .spiral_evolution_manager import SpiralEvolutionManager, SpiralCycleConfig
from .v3_comparison_engine import V3ComparisonEngine, ExclusionReason

logger = logging.getLogger(__name__)


@dataclass
class ParallelVersionStatus:
    """Status of all three parallel versions."""
    v1_status: Dict[str, Any]
    v2_status: Dict[str, Any]
    v3_status: Dict[str, Any]
    all_running: bool
    cycle_phase: str
    last_promotion: Optional[str] = None
    last_degradation: Optional[str] = None
    pending_errors: int = 0
    pending_fixes: int = 0


class ThreeVersionOrchestrator:
    """
    Unified orchestrator for the three-version parallel AI system.
    
    Responsibilities:
    1. Run V1, V2, V3 in parallel with their VC-AI and CR-AI
    2. Route user requests to V2 (stable) CR-AI only
    3. Route admin/test requests to any version
    4. Manage the spiral evolution cycle:
       - V1 experiments with new technologies
       - V2 remains stable, fixes V1 errors
       - V3 provides comparison baseline, eliminates poor performers
    5. Handle promotions (V1→V2) and degradations (V2→V3)
    """
    
    def __init__(
        self,
        version_manager: Optional[VersionManager] = None,
        event_bus: Optional[Any] = None,
        cycle_config: Optional[SpiralCycleConfig] = None,
    ):
        """
        Initialize the three-version orchestrator.
        
        Args:
            version_manager: Version manager (created if not provided)
            event_bus: Event bus for notifications
            cycle_config: Configuration for spiral evolution cycle
        """
        self.version_manager = version_manager or VersionManager()
        self.event_bus = event_bus
        
        # Initialize dual-AI coordinator (VC-AI + CR-AI per version)
        self.dual_ai = DualAICoordinator(
            event_bus=event_bus,
            version_manager=self.version_manager,
        )
        
        # Initialize request handler
        self.request_handler = DualAIRequestHandler(self.dual_ai)
        
        # Initialize cross-version feedback (V2 fixes V1 errors)
        self.feedback_system = CrossVersionFeedbackSystem(
            v2_ai_engine=self.dual_ai.get_ai("v2", AIType.VERSION_CONTROL),
            version_manager=self.version_manager,
            event_bus=event_bus,
        )
        
        # Initialize V3 comparison engine
        self.comparison_engine = V3ComparisonEngine(
            version_manager=self.version_manager,
        )
        
        # Initialize spiral evolution manager
        self.spiral_config = cycle_config or SpiralCycleConfig(
            cycle_interval_hours=6,
            experiment_duration_hours=24,
            stabilization_duration_hours=48,
            quarantine_cooldown_days=30,
            min_accuracy=0.85,
            max_error_rate=0.05,
            max_latency_p95_ms=3000,
            min_samples=1000,
            auto_promote=False,  # V2 stays unchanged until confirmed
            auto_degrade=True,   # Auto-degrade failing technologies
            auto_reeval=False,   # Manual re-evaluation from V3
        )
        
        self.spiral_manager = SpiralEvolutionManager(
            version_manager=self.version_manager,
            event_bus=event_bus,
            config=self.spiral_config,
        )
        
        # State tracking
        self._running = False
        self._start_time: Optional[datetime] = None
        self._parallel_tasks: List[asyncio.Task] = []
        
        logger.info("Three-Version Orchestrator initialized")
    
    # =========================================================================
    # Lifecycle Management
    # =========================================================================
    
    async def start(self) -> None:
        """
        Start all three versions in parallel.
        
        This starts:
        1. V1 VC-AI + CR-AI (experimental mode)
        2. V2 VC-AI + CR-AI (stable, user-facing)
        3. V3 VC-AI + CR-AI (quarantine/comparison mode)
        4. Spiral evolution cycle
        5. Cross-version feedback loop
        """
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        self._running = True
        self._start_time = datetime.now(timezone.utc)
        
        logger.info("=" * 60)
        logger.info("Starting Three-Version Parallel System")
        logger.info("=" * 60)
        
        # Activate all versions
        for version in [Version.V1_EXPERIMENTAL, Version.V2_PRODUCTION, Version.V3_QUARANTINE]:
            await self.version_manager.set_state(version, VersionState.ACTIVE)
            logger.info(f"  ✓ {version.value} activated")
        
        # Activate all AI instances
        for version in ["v1", "v2", "v3"]:
            for ai_type in [AIType.VERSION_CONTROL, AIType.CODE_REVIEW]:
                await self.dual_ai.set_ai_status(version, ai_type, AIStatus.ACTIVE)
        
        logger.info("  ✓ All VC-AI and CR-AI instances activated")
        
        # Start spiral evolution cycle
        await self.spiral_manager.start()
        logger.info("  ✓ Spiral evolution cycle started")
        
        logger.info("=" * 60)
        logger.info("Three-Version Parallel System RUNNING")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Version Roles:")
        logger.info("  V1 (Experimental): Testing new technologies, trial-and-error")
        logger.info("  V2 (Stable): User-facing, fixes V1 errors, unchanged for users")
        logger.info("  V3 (Quarantine): Comparison baseline, eliminates poor performers")
        logger.info("")
        logger.info("Spiral Cycle: V1 → V2 (promote) → V3 (degrade) → V1 (re-evaluate)")
    
    async def stop(self) -> None:
        """Stop all three versions."""
        if not self._running:
            return
        
        logger.info("Stopping Three-Version Parallel System...")
        
        # Stop spiral evolution
        await self.spiral_manager.stop()
        
        # Deactivate versions
        for version in [Version.V1_EXPERIMENTAL, Version.V2_PRODUCTION, Version.V3_QUARANTINE]:
            await self.version_manager.set_state(version, VersionState.SUSPENDED)
        
        # Cancel parallel tasks
        for task in self._parallel_tasks:
            task.cancel()
        
        self._running = False
        logger.info("Three-Version Parallel System stopped")
    
    # =========================================================================
    # User-Facing Operations (V2 only)
    # =========================================================================
    
    async def user_code_review(
        self,
        code: str,
        language: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform code review for users.
        
        Always routes to V2 CR-AI (stable version).
        Users NEVER interact with V1 or V3.
        
        Args:
            code: Source code to review
            language: Programming language
            context: Optional context
            
        Returns:
            Review result from V2 CR-AI
        """
        if not self._running:
            return {"success": False, "error": "System not running"}
        
        return await self.request_handler.handle_user_request({
            "code": code,
            "language": language,
            "context": context,
            "request_type": "code_review",
        })
    
    def get_user_ai_status(self) -> Dict[str, Any]:
        """
        Get status of user-accessible AI (V2 CR-AI only).
        
        This is what users see - they don't know about V1 or V3.
        """
        return self.dual_ai.get_user_ai_status()
    
    # =========================================================================
    # Experimental Operations (V1)
    # =========================================================================
    
    async def experimental_review(
        self,
        code: str,
        language: str,
        technology: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform experimental code review on V1.
        
        This is for testing new technologies before they go to V2.
        Only accessible by admins/testers.
        
        Args:
            code: Source code
            language: Programming language
            technology: Specific technology to test (optional)
            
        Returns:
            Review result from V1 CR-AI
        """
        if not self._running:
            return {"success": False, "error": "System not running"}
        
        return await self.dual_ai.route_request(
            user_role="tester",
            request_type="code_review",
            request_data={
                "code": code,
                "language": language,
                "technology": technology,
            },
            preferred_version="v1",
        )
    
    async def run_experiment(
        self,
        tech_name: str,
        tech_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Start a new technology experiment in V1.
        
        The technology will be tested through trial-and-error.
        If successful, it can be promoted to V2.
        
        Args:
            tech_name: Name of technology to test
            tech_config: Configuration for the technology
            
        Returns:
            Experiment status
        """
        tech = self.version_manager.register_technology(
            name=tech_name,
            category="experimental",
            description=f"Experimental test of {tech_name}",
            config=tech_config,
            source="api",
            version=Version.V1_EXPERIMENTAL,
        )
        
        logger.info(f"Started V1 experiment: {tech_name}")
        
        return {
            "success": True,
            "tech_id": tech.tech_id,
            "name": tech_name,
            "version": "v1",
            "status": "experimental",
        }
    
    # =========================================================================
    # Cross-Version Operations
    # =========================================================================
    
    async def report_error(
        self,
        tech_id: str,
        tech_name: str,
        error_type: str,
        description: str,
    ) -> Dict[str, Any]:
        """
        Report a V1 error for V2 to analyze and fix.
        
        This implements the V1→V2 feedback loop where:
        1. V1 detects an error in experimental technology
        2. V2 analyzes the error
        3. V2 generates a fix
        4. Fix is applied back to V1
        
        Args:
            tech_id: Technology ID
            tech_name: Technology name
            error_type: Type of error (compatibility, performance, etc.)
            description: Error description
            
        Returns:
            Fix generation result
        """
        try:
            error_enum = ErrorType(error_type)
        except ValueError:
            error_enum = ErrorType.COMPATIBILITY
        
        error = await self.feedback_system.report_v1_error(
            technology_id=tech_id,
            technology_name=tech_name,
            error_type=error_enum,
            description=description,
        )
        
        logger.info(f"V1 error reported, V2 analyzing: {error.error_id}")
        
        return {
            "success": True,
            "error_id": error.error_id,
            "status": "v2_analyzing",
            "message": "V2 VC-AI is analyzing the error and generating a fix",
        }
    
    async def trigger_promotion(self, tech_id: str, reason: str = "") -> Dict[str, Any]:
        """
        Manually trigger promotion of a technology from V1 to V2.
        
        This updates the stable version with the new technology.
        
        Args:
            tech_id: Technology to promote
            reason: Reason for promotion
            
        Returns:
            Promotion result
        """
        return await self.spiral_manager.trigger_promotion(tech_id)
    
    async def trigger_degradation(self, tech_id: str, reason: str) -> Dict[str, Any]:
        """
        Manually degrade a technology from V2 to V3.
        
        This removes underperforming technology from the stable version
        and moves it to comparison/quarantine.
        
        Args:
            tech_id: Technology to degrade
            reason: Reason for degradation
            
        Returns:
            Degradation result
        """
        return await self.spiral_manager.trigger_degradation(tech_id, reason)
    
    async def request_reevaluation(self, tech_id: str) -> Dict[str, Any]:
        """
        Request re-evaluation of a quarantined technology.
        
        Technologies in V3 can be re-evaluated and moved back to V1
        for another round of experimentation.
        
        Args:
            tech_id: Technology to re-evaluate
            
        Returns:
            Re-evaluation result
        """
        return await self.spiral_manager.request_reevaluation(tech_id)
    
    # =========================================================================
    # V3 Comparison Operations
    # =========================================================================
    
    async def compare_technology(
        self,
        tech_id: str,
    ) -> Dict[str, Any]:
        """
        Compare a technology against V3 baseline.
        
        V3 provides comparison data to help evaluate V1 technologies.
        
        Args:
            tech_id: Technology to compare
            
        Returns:
            Comparison result
        """
        return self.comparison_engine.compare_with_baseline(tech_id)
    
    async def get_elimination_candidates(self) -> List[Dict[str, Any]]:
        """
        Get technologies that should be eliminated based on V3 analysis.
        
        V3 identifies poor performers that should be removed.
        """
        return self.spiral_manager.get_elimination_status()
    
    # =========================================================================
    # Status & Monitoring
    # =========================================================================
    
    def get_parallel_status(self) -> ParallelVersionStatus:
        """Get comprehensive status of all three parallel versions."""
        dual_ai_status = self.dual_ai.get_all_status()
        cycle_status = self.spiral_manager.get_cycle_status()
        
        return ParallelVersionStatus(
            v1_status={
                "state": self.version_manager.get_state(Version.V1_EXPERIMENTAL).value,
                "ai_status": dual_ai_status.get("v1", {}),
                "metrics": self.version_manager.get_metrics(Version.V1_EXPERIMENTAL).__dict__,
                "technologies": len(self.version_manager.get_version_technologies(Version.V1_EXPERIMENTAL)),
            },
            v2_status={
                "state": self.version_manager.get_state(Version.V2_PRODUCTION).value,
                "ai_status": dual_ai_status.get("v2", {}),
                "metrics": self.version_manager.get_metrics(Version.V2_PRODUCTION).__dict__,
                "technologies": len(self.version_manager.get_version_technologies(Version.V2_PRODUCTION)),
            },
            v3_status={
                "state": self.version_manager.get_state(Version.V3_QUARANTINE).value,
                "ai_status": dual_ai_status.get("v3", {}),
                "metrics": self.version_manager.get_metrics(Version.V3_QUARANTINE).__dict__,
                "technologies": len(self.version_manager.get_version_technologies(Version.V3_QUARANTINE)),
            },
            all_running=all(
                self.version_manager.get_state(v) == VersionState.ACTIVE
                for v in Version
            ),
            cycle_phase=cycle_status.get("current_phase", "idle"),
            pending_errors=len(self.feedback_system.get_pending_fixes()),
            pending_fixes=len(self.feedback_system.get_pending_fixes()),
        )
    
    def get_version_report(self) -> Dict[str, Any]:
        """Get detailed report of all versions."""
        return self.version_manager.get_status_report()
    
    def get_spiral_cycle_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get history of spiral evolution cycles."""
        return self.spiral_manager.get_cycle_history(limit)
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics on V2 fixing V1 errors."""
        return self.feedback_system.get_feedback_statistics()
    
    @property
    def is_running(self) -> bool:
        """Check if the orchestrator is running."""
        return self._running
    
    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        if not self._start_time:
            return 0
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()


# =============================================================================
# Factory Functions
# =============================================================================

def create_three_version_system(
    config: Optional[SpiralCycleConfig] = None,
    event_bus: Optional[Any] = None,
) -> ThreeVersionOrchestrator:
    """
    Create a fully configured three-version parallel system.
    
    This is the recommended way to instantiate the system.
    
    Args:
        config: Spiral cycle configuration (optional)
        event_bus: Event bus for notifications (optional)
        
    Returns:
        Configured ThreeVersionOrchestrator
    """
    return ThreeVersionOrchestrator(
        version_manager=VersionManager(),
        event_bus=event_bus,
        cycle_config=config,
    )


async def run_three_version_demo():
    """
    Demonstrate the three-version parallel system.
    
    This shows:
    1. Starting all three versions in parallel
    2. User code review (V2 only)
    3. Experimental review (V1)
    4. V1 error → V2 fix flow
    5. Technology promotion/degradation
    """
    logger.info("=" * 70)
    logger.info("THREE-VERSION PARALLEL SYSTEM DEMONSTRATION")
    logger.info("=" * 70)
    
    # Create orchestrator
    orchestrator = create_three_version_system()
    
    # Start all versions
    await orchestrator.start()
    
    # 1. User code review (goes to V2)
    logger.info("\n[DEMO] User Code Review (routed to V2):")
    user_result = await orchestrator.user_code_review(
        code="password = 'secret123'",
        language="python",
    )
    logger.info(f"  Result: {user_result}")
    
    # 2. Start an experiment in V1
    logger.info("\n[DEMO] Starting V1 Experiment:")
    exp_result = await orchestrator.run_experiment(
        tech_name="new_attention_mechanism",
        tech_config={"heads": 16, "dim": 512},
    )
    logger.info(f"  Experiment started: {exp_result}")
    
    # 3. Report V1 error for V2 to fix
    logger.info("\n[DEMO] V1 Error → V2 Fix:")
    error_result = await orchestrator.report_error(
        tech_id=exp_result["tech_id"],
        tech_name="new_attention_mechanism",
        error_type="compatibility",
        description="Incompatible with existing pipeline",
    )
    logger.info(f"  Error reported, V2 analyzing: {error_result}")
    
    # 4. Get system status
    logger.info("\n[DEMO] System Status:")
    status = orchestrator.get_parallel_status()
    logger.info(f"  All Running: {status.all_running}")
    logger.info(f"  V1 Technologies: {status.v1_status['technologies']}")
    logger.info(f"  V2 Technologies: {status.v2_status['technologies']}")
    logger.info(f"  V3 Technologies: {status.v3_status['technologies']}")
    logger.info(f"  Cycle Phase: {status.cycle_phase}")
    
    # Stop
    await orchestrator.stop()
    
    logger.info("\n" + "=" * 70)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 70)


# Export
__all__ = [
    "ThreeVersionOrchestrator",
    "ParallelVersionStatus",
    "create_three_version_system",
    "run_three_version_demo",
]
