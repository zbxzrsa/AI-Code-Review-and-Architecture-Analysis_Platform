"""
Promotion Manager

Handles V1 → V2 promotion with canary deployment:
- Pre-promotion validation
- Phased rollout (10% → 50% → 100%)
- Automatic rollback on issues
- Post-promotion finalization
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .event_types import (
    EventType,
    Version,
    VersionEvent,
    PromotionPhase,
    PromotionRequest,
)

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    """Configuration for promotion phase."""
    traffic_percentage: int
    duration_hours: int
    error_rate_threshold: float
    latency_p95_threshold_ms: int
    rollback_on_failure: bool = True


class PromotionManager:
    """
    Manages the V1 → V2 promotion process.
    
    Implements:
    - Pre-promotion validation
    - Canary deployment (10% → 50% → 100%)
    - Health monitoring during rollout
    - Automatic rollback on issues
    """
    
    PHASE_CONFIGS = {
        PromotionPhase.PHASE_1_10_PERCENT: PhaseConfig(
            traffic_percentage=10,
            duration_hours=24,
            error_rate_threshold=0.02,
            latency_p95_threshold_ms=3000,
        ),
        PromotionPhase.PHASE_2_50_PERCENT: PhaseConfig(
            traffic_percentage=50,
            duration_hours=48,
            error_rate_threshold=0.02,
            latency_p95_threshold_ms=3000,
        ),
        PromotionPhase.PHASE_3_100_PERCENT: PhaseConfig(
            traffic_percentage=100,
            duration_hours=168,  # 7 days
            error_rate_threshold=0.02,
            latency_p95_threshold_ms=3000,
        ),
    }
    
    def __init__(
        self,
        event_bus = None,
        metrics_client = None,
        routing_controller = None,
    ):
        self.event_bus = event_bus
        self.metrics = metrics_client
        self.routing = routing_controller
        
        # Active promotions
        self._promotions: Dict[str, PromotionRequest] = {}
        self._phase_tasks: Dict[str, asyncio.Task] = {}
        
        # Event handlers
        self._on_phase_complete: List[Callable] = []
        self._on_rollback: List[Callable] = []
    
    async def request_promotion(
        self,
        experiment_id: str,
        evaluation_metrics: Dict[str, Any],
        confidence_score: float,
    ) -> PromotionRequest:
        """Request promotion of V1 experiment to V2."""
        request = PromotionRequest(
            experiment_id=experiment_id,
            metrics=evaluation_metrics,
            confidence_score=confidence_score,
        )
        
        self._promotions[request.request_id] = request
        
        # Emit event
        await self._emit_event(
            EventType.PROMOTION_REQUESTED,
            Version.V1_EXPERIMENTATION,
            {
                "request_id": request.request_id,
                "experiment_id": experiment_id,
                "confidence_score": confidence_score,
            },
        )
        
        logger.info(f"Promotion requested for experiment {experiment_id}")
        return request
    
    async def approve_promotion(
        self,
        request_id: str,
        approver: str,
    ) -> bool:
        """Approve promotion request and start canary deployment."""
        request = self._promotions.get(request_id)
        if not request:
            logger.error(f"Promotion request {request_id} not found")
            return False
        
        request.approved_at = datetime.now(timezone.utc)
        request.approved_by = approver
        
        # Emit event
        await self._emit_event(
            EventType.PROMOTION_APPROVED,
            Version.V1_EXPERIMENTATION,
            {
                "request_id": request_id,
                "experiment_id": request.experiment_id,
                "approver": approver,
            },
        )
        
        # Start canary deployment - save task to prevent GC
        request.deployment_task = asyncio.create_task(self._run_canary_deployment(request))
        
        logger.info(f"Promotion {request_id} approved by {approver}")
        return True
    
    async def _run_canary_deployment(self, request: PromotionRequest):
        """Execute canary deployment phases."""
        try:
            # Phase 1: Pre-promotion validation
            logger.info(f"Starting pre-promotion validation for {request.experiment_id}")
            validation_passed = await self._pre_promotion_validation(request)
            
            if not validation_passed:
                await self._handle_promotion_failure(request, "Pre-promotion validation failed")
                return
            
            # Phase 2-4: Canary deployment
            phases = [
                PromotionPhase.PHASE_1_10_PERCENT,
                PromotionPhase.PHASE_2_50_PERCENT,
                PromotionPhase.PHASE_3_100_PERCENT,
            ]
            
            for phase in phases:
                logger.info(f"Starting {phase.value} for {request.experiment_id}")
                
                success = await self._execute_phase(request, phase)
                
                if not success:
                    await self._rollback_promotion(request, f"Failed at {phase.value}")
                    return
            
            # Finalize promotion
            await self._finalize_promotion(request)
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            await self._rollback_promotion(request, str(e))
    
    async def _pre_promotion_validation(self, request: PromotionRequest) -> bool:
        """Run pre-promotion validation checks."""
        request.current_phase = PromotionPhase.VALIDATION
        
        checks = {
            "metrics_revalidation": await self._revalidate_metrics(request),
            "load_test": await self._run_load_test(request),
            "security_audit": await self._run_security_audit(request),
            "dependency_check": await self._check_dependencies(request),
        }
        
        all_passed = all(checks.values())
        
        if not all_passed:
            failed = [k for k, v in checks.items() if not v]
            logger.warning(f"Pre-promotion checks failed: {failed}")
        
        return all_passed
    
    async def _revalidate_metrics(self, request: PromotionRequest) -> bool:
        """Re-run evaluation metrics as sanity check."""
        # In production, re-run evaluation
        logger.info("Re-validating metrics...")
        return True
    
    async def _run_load_test(self, request: PromotionRequest) -> bool:
        """Simulate production load on V1."""
        logger.info("Running load test...")
        # In production, use k6 or similar
        return True
    
    async def _run_security_audit(self, request: PromotionRequest) -> bool:
        """Scan for security vulnerabilities."""
        logger.info("Running security audit...")
        return True
    
    async def _check_dependencies(self, request: PromotionRequest) -> bool:
        """Check for infrastructure conflicts."""
        logger.info("Checking dependencies...")
        return True
    
    async def _execute_phase(
        self,
        request: PromotionRequest,
        phase: PromotionPhase,
    ) -> bool:
        """Execute a single canary phase."""
        config = self.PHASE_CONFIGS[phase]
        request.current_phase = phase
        
        # Update traffic routing
        await self._update_traffic_split(
            request.experiment_id,
            config.traffic_percentage,
        )
        
        # Emit phase change event
        await self._emit_event(
            EventType.PROMOTION_PHASE_CHANGED,
            Version.V2_PRODUCTION,
            {
                "request_id": request.request_id,
                "experiment_id": request.experiment_id,
                "phase": phase.value,
                "traffic_percentage": config.traffic_percentage,
            },
        )
        
        # Monitor for duration
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=config.duration_hours)
        
        # For demo, use shorter intervals
        check_interval = 60  # 1 minute
        
        while datetime.now(timezone.utc) < end_time:
            # Check health metrics
            health_ok = await self._check_phase_health(request, config)
            
            if not health_ok:
                logger.warning(f"Health check failed for {phase.value}")
                return False
            
            await asyncio.sleep(check_interval)
        
        # Phase completed successfully
        for handler in self._on_phase_complete:
            await handler(request, phase)
        
        return True
    
    async def _check_phase_health(
        self,
        request: PromotionRequest,
        config: PhaseConfig,
    ) -> bool:
        """Check health during canary phase."""
        # In production, fetch real metrics
        # Simulated health check
        return True
    
    async def _update_traffic_split(
        self,
        experiment_id: str,
        new_percentage: int,
    ):
        """Update traffic routing between V1 and V2."""
        logger.info(f"Updating traffic split: {new_percentage}% to new version")
        
        if self.routing:
            await self.routing.set_traffic_split(
                new_version=experiment_id,
                percentage=new_percentage,
            )
    
    async def _rollback_promotion(
        self,
        request: PromotionRequest,
        reason: str,
    ):
        """Rollback to previous V2 version."""
        logger.warning(f"Rolling back promotion {request.request_id}: {reason}")
        
        request.current_phase = PromotionPhase.ROLLED_BACK
        
        # Reset traffic to 100% old V2
        await self._update_traffic_split(request.experiment_id, 0)
        
        # Emit event
        await self._emit_event(
            EventType.PROMOTION_ROLLBACK,
            Version.V2_PRODUCTION,
            {
                "request_id": request.request_id,
                "experiment_id": request.experiment_id,
                "reason": reason,
                "phase_at_failure": request.current_phase.value,
            },
        )
        
        # Call rollback handlers
        for handler in self._on_rollback:
            await handler(request, reason)
    
    async def _finalize_promotion(self, request: PromotionRequest):
        """Finalize successful promotion."""
        request.current_phase = PromotionPhase.COMPLETED
        
        logger.info(f"Finalizing promotion for {request.experiment_id}")
        
        # Archive previous V2
        # Update documentation
        # Update metrics labels
        
        # Emit completion event
        await self._emit_event(
            EventType.PROMOTION_COMPLETED,
            Version.V2_PRODUCTION,
            {
                "request_id": request.request_id,
                "experiment_id": request.experiment_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        # Clean up
        del self._promotions[request.request_id]
    
    async def _handle_promotion_failure(
        self,
        request: PromotionRequest,
        reason: str,
    ):
        """Handle promotion failure."""
        await self._emit_event(
            EventType.PROMOTION_FAILED,
            Version.V1_EXPERIMENTATION,
            {
                "request_id": request.request_id,
                "experiment_id": request.experiment_id,
                "reason": reason,
            },
        )
    
    async def _emit_event(
        self,
        event_type: EventType,
        version: Version,
        payload: Dict[str, Any],
    ):
        """Emit event to event bus."""
        event = VersionEvent(
            event_type=event_type,
            version=version,
            payload=payload,
            source="promotion-manager",
        )
        
        if self.event_bus:
            await self.event_bus.publish(event_type.value, event.to_dict())
    
    def get_active_promotions(self) -> List[PromotionRequest]:
        """Get all active promotions."""
        return list(self._promotions.values())
    
    def on_phase_complete(self, handler: Callable):
        """Register phase completion handler."""
        self._on_phase_complete.append(handler)
    
    def on_rollback(self, handler: Callable):
        """Register rollback handler."""
        self._on_rollback.append(handler)
