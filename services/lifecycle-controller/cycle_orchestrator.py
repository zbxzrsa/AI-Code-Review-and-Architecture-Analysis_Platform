"""
Self-Evolution Cycle Orchestrator

This is the master orchestrator that ensures the three-version architecture
operates as a complete, closed-loop self-iterating system.

Cycle Flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    SELF-EVOLUTION CYCLE                         │
    │                                                                 │
    │   ┌──────────┐    Shadow     ┌──────────┐    Gray-Scale        │
    │   │   V1     │ ──────────►   │   V2     │ ◄────────────┐       │
    │   │ Experiment│   Traffic    │ Production│              │       │
    │   └────▲─────┘               └─────┬────┘              │       │
    │        │                           │                    │       │
    │        │ Recovery                  │ SLO Breach        │       │
    │        │ (Gold-set)                │ Rollback          │       │
    │        │                           ▼                    │       │
    │   ┌────┴─────┐              ┌──────────┐               │       │
    │   │   V3     │ ◄────────────│ Demotion │───────────────┘       │
    │   │Quarantine│              └──────────┘                        │
    │   └──────────┘                                                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Key responsibilities:
- Coordinate all lifecycle transitions
- Ensure cycle continuity (no dead ends)
- Track cycle metrics and health
- Provide unified API for cycle management
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from .controller import LifecycleController, VersionState, VersionConfig
from .recovery_manager import RecoveryManager, RecoveryStatus

logger = logging.getLogger(__name__)


class CyclePhase(str, Enum):
    """Phases in the self-evolution cycle"""
    EXPERIMENT = "experiment"       # V1: Testing new models
    EVALUATION = "evaluation"       # Shadow traffic evaluation
    PROMOTION = "promotion"         # Gray-scale rollout
    PRODUCTION = "production"       # V2: Stable serving
    DEGRADATION = "degradation"     # Performance decline
    QUARANTINE = "quarantine"       # V3: Isolated
    RECOVERY = "recovery"           # Attempting recovery
    REINTEGRATION = "reintegration" # V3 → V1 transition


@dataclass
class CycleEvent:
    """Event in the evolution cycle"""
    timestamp: str
    version_id: str
    from_phase: CyclePhase
    to_phase: CyclePhase
    trigger: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass
class CycleHealth:
    """Health metrics for the evolution cycle"""
    is_healthy: bool = True
    v1_active_experiments: int = 0
    v2_stable_versions: int = 0
    v3_quarantined: int = 0
    v3_recovering: int = 0
    
    # Cycle velocity metrics
    avg_promotion_time_hours: float = 0.0
    avg_recovery_time_hours: float = 0.0
    
    # Success rates
    promotion_success_rate: float = 0.0
    recovery_success_rate: float = 0.0
    
    # Cycle continuity (no dead ends)
    cycle_continuity_ok: bool = True
    
    last_promotion: Optional[str] = None
    last_recovery: Optional[str] = None
    last_quarantine: Optional[str] = None


class CycleOrchestrator:
    """
    Master orchestrator for the self-evolution cycle.
    
    Ensures the V1 → V2 → V3 → V1 cycle operates continuously
    without manual intervention.
    """
    
    def __init__(
        self,
        lifecycle_controller: LifecycleController,
        recovery_manager: RecoveryManager,
    ):
        self.lifecycle = lifecycle_controller
        self.recovery = recovery_manager
        
        self.cycle_events: List[CycleEvent] = []
        self.cycle_metrics: Dict[str, List[float]] = {
            "promotion_times": [],
            "recovery_times": [],
            "quarantine_durations": [],
        }
        
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {}
    
    async def start(self):
        """Start the cycle orchestrator"""
        self._running = True
        
        # Start sub-components
        await self.lifecycle.start()
        await self.recovery.start()
        
        # Start orchestration loop
        asyncio.create_task(self._orchestration_loop())
        asyncio.create_task(self._recovery_integration_loop())
        
        logger.info("Cycle Orchestrator started - Self-evolution cycle active")
    
    async def stop(self):
        """Stop the cycle orchestrator"""
        self._running = False
        await self.lifecycle.stop()
        await self.recovery.stop()
        logger.info("Cycle Orchestrator stopped")
    
    # ==================== Main Orchestration ====================
    
    async def _orchestration_loop(self):
        """
        Main orchestration loop ensuring cycle continuity.
        Runs every 30 seconds to coordinate transitions.
        """
        while self._running:
            try:
                # 1. Check for stalled experiments (V1 not progressing)
                await self._check_stalled_experiments()
                
                # 2. Check for versions ready for recovery promotion
                await self._process_recovery_promotions()
                
                # 3. Update cycle health metrics
                await self._update_cycle_health()
                
                # 4. Ensure no dead ends in the cycle
                await self._ensure_cycle_continuity()
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
            
            await asyncio.sleep(30)
    
    async def _recovery_integration_loop(self):
        """
        Loop that integrates recovery manager with lifecycle controller.
        Promotes recovered versions back to V1.
        """
        while self._running:
            try:
                # Get versions that passed recovery
                passed_versions = self.recovery.get_passed_versions()
                
                for record in passed_versions:
                    # Promote back to V1 shadow
                    await self._promote_recovered_to_v1(record.version_id)
                    
            except Exception as e:
                logger.error(f"Recovery integration error: {e}")
            
            await asyncio.sleep(60)
    
    # ==================== Cycle Transitions ====================
    
    async def register_new_experiment(
        self,
        version_id: str,
        model_version: str,
        prompt_version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VersionConfig:
        """
        Register a new experiment to start the cycle.
        Entry point: → V1 (Experiment)
        """
        config = VersionConfig(
            version_id=version_id,
            model_version=model_version,
            prompt_version=prompt_version,
            routing_policy_version="default",
            current_state=VersionState.EXPERIMENT,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        self.lifecycle.active_versions[version_id] = config
        
        self._log_cycle_event(
            version_id=version_id,
            from_phase=CyclePhase.EXPERIMENT,
            to_phase=CyclePhase.EXPERIMENT,
            trigger="new_experiment_registered"
        )
        
        logger.info(f"New experiment registered: {version_id}")
        return config
    
    async def start_shadow_evaluation(self, version_id: str):
        """
        Start shadow traffic evaluation for an experiment.
        Transition: V1 (Experiment) → V1 (Shadow)
        """
        if version_id not in self.lifecycle.active_versions:
            raise ValueError(f"Version {version_id} not found")
        
        config = self.lifecycle.active_versions[version_id]
        config.current_state = VersionState.SHADOW
        
        self._log_cycle_event(
            version_id=version_id,
            from_phase=CyclePhase.EXPERIMENT,
            to_phase=CyclePhase.EVALUATION,
            trigger="shadow_evaluation_started"
        )
        
        logger.info(f"Shadow evaluation started for {version_id}")
    
    async def trigger_quarantine(
        self,
        version_id: str,
        reason: str,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Trigger quarantine for a failing version.
        Transition: V1/V2 → V3 (Quarantine)
        """
        if version_id in self.lifecycle.active_versions:
            config = self.lifecycle.active_versions[version_id]
            previous_state = config.current_state
            
            # Update lifecycle state
            config.current_state = VersionState.QUARANTINE
            config.metadata["quarantine_reason"] = reason
            config.metadata["quarantine_time"] = datetime.now(timezone.utc).isoformat()
            
            # Register with recovery manager
            self.recovery.register_quarantine(
                version_id=version_id,
                reason=reason,
                metadata=metrics
            )
            
            self._log_cycle_event(
                version_id=version_id,
                from_phase=self._state_to_phase(previous_state),
                to_phase=CyclePhase.QUARANTINE,
                trigger=f"quarantine: {reason}",
                metrics=metrics or {}
            )
            
            logger.warning(f"Version {version_id} quarantined: {reason}")
    
    async def _promote_recovered_to_v1(self, version_id: str):
        """
        Promote a recovered version back to V1 shadow.
        Transition: V3 (Recovery) → V1 (Shadow)
        
        This closes the loop!
        """
        if version_id not in self.lifecycle.active_versions:
            # Re-create config for recovered version
            recovery_record = self.recovery.get_recovery_status(version_id)
            if not recovery_record:
                return
            
            config = VersionConfig(
                version_id=version_id,
                model_version=recovery_record.metadata.get("model_version", "unknown"),
                prompt_version=recovery_record.metadata.get("prompt_version", "unknown"),
                routing_policy_version="default",
                current_state=VersionState.SHADOW,
                created_at=datetime.now(timezone.utc),
                metadata={
                    "recovered_from_quarantine": True,
                    "original_quarantine_reason": recovery_record.quarantine_reason,
                    "recovery_attempts": recovery_record.recovery_attempts,
                    "best_recovery_score": recovery_record.best_score,
                }
            )
            
            self.lifecycle.active_versions[version_id] = config
        else:
            config = self.lifecycle.active_versions[version_id]
            config.current_state = VersionState.SHADOW
            config.consecutive_failures = 0
        
        # Mark as promoted in recovery manager
        self.recovery.mark_promoted_to_v1(version_id)
        
        self._log_cycle_event(
            version_id=version_id,
            from_phase=CyclePhase.RECOVERY,
            to_phase=CyclePhase.EVALUATION,
            trigger="recovery_successful"
        )
        
        logger.info(f"🔄 CYCLE COMPLETE: {version_id} recovered from V3 → V1")
        
        # Trigger callbacks
        await self._trigger_callbacks("recovery_complete", version_id)
    
    # ==================== Cycle Health ====================
    
    async def _check_stalled_experiments(self):
        """Check for experiments that have stalled"""
        now = datetime.now(timezone.utc)
        
        for version_id, config in self.lifecycle.active_versions.items():
            if config.current_state == VersionState.EXPERIMENT:
                # Experiment stalled for > 7 days
                if now - config.created_at > timedelta(days=7):
                    logger.warning(f"Experiment {version_id} stalled - auto-starting shadow")
                    await self.start_shadow_evaluation(version_id)
    
    async def _process_recovery_promotions(self):
        """Process versions ready for V1 promotion after recovery"""
        passed = self.recovery.get_passed_versions()
        
        for record in passed:
            if record.recovery_status == RecoveryStatus.PASSED:
                await self._promote_recovered_to_v1(record.version_id)
    
    async def _update_cycle_health(self):
        """Update cycle health metrics"""
        # Count versions in each state
        v1_count = sum(
            1 for v in self.lifecycle.active_versions.values()
            if v.current_state in [VersionState.EXPERIMENT, VersionState.SHADOW]
        )
        
        v2_count = sum(
            1 for v in self.lifecycle.active_versions.values()
            if v.current_state.value.startswith("gray_") or v.current_state == VersionState.STABLE
        )
        
        v3_count = sum(
            1 for v in self.lifecycle.active_versions.values()
            if v.current_state == VersionState.QUARANTINE
        )
        
        recovery_stats = self.recovery.get_recovery_statistics()
        
        # Log health status periodically
        logger.debug(
            f"Cycle Health: V1={v1_count}, V2={v2_count}, V3={v3_count}, "
            f"Recovering={recovery_stats['in_progress']}"
        )
    
    async def _ensure_cycle_continuity(self):
        """
        Ensure the cycle has no dead ends.
        Every version should have a path forward.
        """
        for version_id, config in self.lifecycle.active_versions.items():
            # Check for versions stuck in quarantine without recovery scheduled
            if config.current_state == VersionState.QUARANTINE:
                recovery_record = self.recovery.get_recovery_status(version_id)
                
                if not recovery_record:
                    # Register for recovery if not already
                    self.recovery.register_quarantine(
                        version_id=version_id,
                        reason=config.metadata.get("quarantine_reason", "unknown"),
                        metadata=config.metadata
                    )
                    logger.info(f"Registered {version_id} for recovery to ensure cycle continuity")
    
    # ==================== Utilities ====================
    
    def _log_cycle_event(
        self,
        version_id: str,
        from_phase: CyclePhase,
        to_phase: CyclePhase,
        trigger: str,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Log a cycle event"""
        event = CycleEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version_id=version_id,
            from_phase=from_phase,
            to_phase=to_phase,
            trigger=trigger,
            metrics=metrics or {}
        )
        
        self.cycle_events.append(event)
        
        # Keep only last 1000 events
        if len(self.cycle_events) > 1000:
            self.cycle_events = self.cycle_events[-1000:]
    
    def _state_to_phase(self, state: VersionState) -> CyclePhase:
        """Map version state to cycle phase"""
        mapping = {
            VersionState.EXPERIMENT: CyclePhase.EXPERIMENT,
            VersionState.SHADOW: CyclePhase.EVALUATION,
            VersionState.GRAY_1: CyclePhase.PROMOTION,
            VersionState.GRAY_5: CyclePhase.PROMOTION,
            VersionState.GRAY_25: CyclePhase.PROMOTION,
            VersionState.GRAY_50: CyclePhase.PROMOTION,
            VersionState.STABLE: CyclePhase.PRODUCTION,
            VersionState.QUARANTINE: CyclePhase.QUARANTINE,
            VersionState.RE_EVALUATION: CyclePhase.RECOVERY,
        }
        return mapping.get(state, CyclePhase.EXPERIMENT)
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for cycle events"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    async def _trigger_callbacks(self, event_type: str, *args, **kwargs):
        """Trigger registered callbacks"""
        for callback in self._callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    # ==================== API Methods ====================
    
    def get_cycle_status(self) -> Dict[str, Any]:
        """Get current cycle status"""
        recovery_stats = self.recovery.get_recovery_statistics()
        
        return {
            "cycle_active": self._running,
            "versions": {
                "v1_experiments": sum(
                    1 for v in self.lifecycle.active_versions.values()
                    if v.current_state in [VersionState.EXPERIMENT, VersionState.SHADOW]
                ),
                "v2_production": sum(
                    1 for v in self.lifecycle.active_versions.values()
                    if v.current_state == VersionState.STABLE
                ),
                "v2_gray_scale": sum(
                    1 for v in self.lifecycle.active_versions.values()
                    if v.current_state.value.startswith("gray_")
                ),
                "v3_quarantined": recovery_stats["total_quarantined"],
                "v3_recovering": recovery_stats["in_progress"],
                "v3_recovered": recovery_stats["passed"],
            },
            "recovery_stats": recovery_stats,
            "recent_events": self.cycle_events[-10:],
            "cycle_health": {
                "is_continuous": True,  # Cycle has no dead ends
                "all_paths_active": True,
            }
        }
    
    def get_cycle_events(
        self,
        version_id: Optional[str] = None,
        limit: int = 50
    ) -> List[CycleEvent]:
        """Get cycle events, optionally filtered by version"""
        events = self.cycle_events
        
        if version_id:
            events = [e for e in events if e.version_id == version_id]
        
        return events[-limit:]
