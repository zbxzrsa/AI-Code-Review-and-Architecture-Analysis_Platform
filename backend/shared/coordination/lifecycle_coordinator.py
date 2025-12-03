"""
Lifecycle Coordinator

Central orchestration for Three-Version Self-Evolving Cycle:
- Coordinates V1 experiments
- Manages V1 → V2 promotions
- Handles V1 → V3 quarantines
- Monitors V2 production health
- Enforces version isolation
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .event_types import (
    EventType,
    Version,
    VersionEvent,
    PromotionPhase,
    ExperimentProposal,
)
from .promotion_manager import PromotionManager
from .quarantine_manager import QuarantineManager
from .health_monitor import HealthMonitor, AlertSeverity
from .experiment_generator import ExperimentGenerator

logger = logging.getLogger(__name__)


class LifecycleCoordinator:
    """
    Central coordinator for Three-Version Self-Evolving Cycle.
    
    Responsibilities:
    - Orchestrate experiment lifecycle
    - Coordinate promotion/quarantine decisions
    - Enforce version isolation
    - Manage event flow between components
    """
    
    def __init__(
        self,
        event_bus = None,
        db_connection = None,
        metrics_client = None,
    ):
        self.event_bus = event_bus
        self.db = db_connection
        self.metrics = metrics_client
        
        # Initialize managers
        self.promotion = PromotionManager(
            event_bus=event_bus,
            metrics_client=metrics_client,
        )
        
        self.quarantine = QuarantineManager(
            event_bus=event_bus,
            db_connection=db_connection,
        )
        
        self.health = HealthMonitor(
            event_bus=event_bus,
            metrics_client=metrics_client,
        )
        
        self.experiments = ExperimentGenerator(
            event_bus=event_bus,
            quarantine_manager=self.quarantine,
            health_monitor=self.health,
        )
        
        # Active experiments
        self._experiments: Dict[str, Dict[str, Any]] = {}
        
        # Setup event handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup internal event handlers."""
        # Health monitor handlers
        self.health.register_alert_handler(
            AlertSeverity.CRITICAL,
            self._handle_critical_alert,
        )
        
        # Remediation handlers
        self.health.register_remediation_handler(
            "rollback",
            self._handle_rollback,
        )
        self.health.register_remediation_handler(
            "scale_up",
            self._handle_scale_up,
        )
        
        # Promotion handlers
        self.promotion.on_phase_complete(self._on_promotion_phase_complete)
        self.promotion.on_rollback(self._on_promotion_rollback)
    
    async def start(self):
        """Start the lifecycle coordinator."""
        logger.info("Starting Lifecycle Coordinator...")
        
        # Start health monitoring
        await self.health.start()
        
        # Subscribe to events if event bus available
        if self.event_bus:
            await self._subscribe_events()
        
        logger.info("Lifecycle Coordinator started")
    
    async def stop(self):
        """Stop the lifecycle coordinator."""
        logger.info("Stopping Lifecycle Coordinator...")
        
        await self.health.stop()
        
        logger.info("Lifecycle Coordinator stopped")
    
    async def _subscribe_events(self):
        """Subscribe to relevant events."""
        events_to_handle = [
            EventType.EXPERIMENT_EVALUATION_COMPLETED,
            EventType.MONITORING_ALERT,
            EventType.COST_THRESHOLD_EXCEEDED,
        ]
        
        for event_type in events_to_handle:
            await self.event_bus.subscribe(
                event_type.value,
                self._handle_event,
            )
    
    async def _handle_event(self, event_data: Dict[str, Any]):
        """Handle incoming events."""
        event = VersionEvent.from_dict(event_data)
        
        logger.debug(f"Received event: {event.event_type.value}")
        
        if event.event_type == EventType.EXPERIMENT_EVALUATION_COMPLETED:
            await self._process_evaluation_result(event)
    
    async def _process_evaluation_result(self, event: VersionEvent):
        """Process experiment evaluation result."""
        payload = event.payload
        decision = payload.get("decision", "").upper()
        experiment_id = payload.get("experiment_id")
        
        if decision == "PROMOTE":
            await self.promotion.request_promotion(
                experiment_id=experiment_id,
                evaluation_metrics=payload.get("metrics", {}),
                confidence_score=payload.get("confidence_score", 0),
            )
        
        elif decision == "QUARANTINE":
            await self.quarantine.quarantine_experiment(
                experiment_id=experiment_id,
                failure_type=payload.get("failure_type", "unknown"),
                failure_evidence=payload.get("evidence", {}),
                error_logs=payload.get("error_logs", ""),
                metrics_at_failure=payload.get("metrics", {}),
            )
    
    async def _handle_critical_alert(self, alert):
        """Handle critical health alerts."""
        logger.critical(f"Critical alert: {alert.message}")
        
        # Check if auto-rollback needed
        if alert.metric_name == "error_rate" and alert.current_value > 0.05:
            logger.warning("Error rate critical, triggering rollback")
            await self._trigger_emergency_rollback(alert)
    
    async def _handle_rollback(self, alert):
        """Handle rollback remediation."""
        logger.info(f"Executing rollback remediation for {alert.metric_name}")
        
        # Get active promotion if any
        active = self.promotion.get_active_promotions()
        if active:
            for promotion in active:
                await self.promotion._rollback_promotion(
                    promotion,
                    f"Auto-remediation due to {alert.metric_name} alert",
                )
    
    async def _handle_scale_up(self, alert):
        """Handle scale-up remediation."""
        logger.info(f"Executing scale-up remediation for {alert.metric_name}")
        
        # In production, trigger Kubernetes HPA or manual scaling
        # kubectl scale deployment v2-service --replicas=6
    
    async def _trigger_emergency_rollback(self, alert):
        """Trigger emergency rollback to last known good version."""
        logger.critical("EMERGENCY ROLLBACK TRIGGERED")
        
        await self._emit_event(
            EventType.ROLLBACK_TRIGGERED,
            Version.V2_PRODUCTION,
            {
                "trigger": "emergency",
                "alert_id": alert.alert_id,
                "metric": alert.metric_name,
                "value": alert.current_value,
            },
        )
    
    async def _on_promotion_phase_complete(self, request, phase):
        """Handle promotion phase completion."""
        logger.info(f"Promotion {request.request_id} completed phase {phase.value}")
    
    async def _on_promotion_rollback(self, request, reason):
        """Handle promotion rollback."""
        logger.warning(f"Promotion {request.request_id} rolled back: {reason}")
        
        # Consider quarantining the experiment
        await self.quarantine.quarantine_experiment(
            experiment_id=request.experiment_id,
            failure_type="promotion_failure",
            failure_evidence={"reason": reason, "phase": request.current_phase.value},
            error_logs="",
            metrics_at_failure=request.metrics,
        )
    
    # ==========================================================================
    # Public API
    # ==========================================================================
    
    async def create_experiment(
        self,
        title: str,
        hypothesis: str,
        success_criteria: Dict[str, str],
        evaluation_period_days: int = 14,
    ) -> ExperimentProposal:
        """Create new experiment proposal."""
        import uuid
        
        proposal = ExperimentProposal(
            experiment_id=str(uuid.uuid4()),
            title=title,
            hypothesis=hypothesis,
            success_criteria=success_criteria,
            evaluation_period_days=evaluation_period_days,
            source="manual",
            status="proposed",
        )
        
        self._experiments[proposal.experiment_id] = {
            "proposal": proposal,
            "status": "proposed",
            "created_at": datetime.utcnow(),
        }
        
        await self._emit_event(
            EventType.EXPERIMENT_CREATED,
            Version.V1_EXPERIMENTATION,
            {
                "experiment_id": proposal.experiment_id,
                "title": title,
                "hypothesis": hypothesis,
            },
        )
        
        return proposal
    
    async def start_experiment(
        self,
        experiment_id: str,
        approver: str,
    ) -> bool:
        """Start approved experiment."""
        if experiment_id not in self._experiments:
            return False
        
        # Check blacklist
        if self.quarantine.is_blacklisted(experiment_id):
            logger.warning(f"Experiment {experiment_id} is blacklisted")
            return False
        
        self._experiments[experiment_id]["status"] = "running"
        self._experiments[experiment_id]["started_at"] = datetime.utcnow()
        self._experiments[experiment_id]["approver"] = approver
        
        await self._emit_event(
            EventType.EXPERIMENT_STARTED,
            Version.V1_EXPERIMENTATION,
            {
                "experiment_id": experiment_id,
                "approver": approver,
            },
        )
        
        return True
    
    async def evaluate_experiment(
        self,
        experiment_id: str,
        metrics: Dict[str, Any],
    ) -> str:
        """Evaluate experiment and make decision."""
        await self._emit_event(
            EventType.EXPERIMENT_EVALUATION_STARTED,
            Version.V1_EXPERIMENTATION,
            {"experiment_id": experiment_id},
        )
        
        # Get success criteria
        exp = self._experiments.get(experiment_id)
        if not exp:
            return "HOLD"
        
        criteria = exp["proposal"].success_criteria
        
        # Evaluate against criteria
        decision, confidence = self._evaluate_against_criteria(metrics, criteria)
        
        await self._emit_event(
            EventType.EXPERIMENT_EVALUATION_COMPLETED,
            Version.V1_EXPERIMENTATION,
            {
                "experiment_id": experiment_id,
                "decision": decision,
                "confidence_score": confidence,
                "metrics": metrics,
            },
        )
        
        return decision
    
    def _evaluate_against_criteria(
        self,
        metrics: Dict[str, Any],
        criteria: Dict[str, str],
    ) -> tuple:
        """Evaluate metrics against success criteria."""
        passed = 0
        total = len(criteria)
        
        for criterion, threshold in criteria.items():
            value = metrics.get(criterion, 0)
            
            # Parse threshold (e.g., "> 85%", "< 3s")
            if threshold.startswith(">"):
                target = float(threshold[1:].strip().rstrip("%"))
                if "%" in threshold:
                    target /= 100
                if value > target:
                    passed += 1
            elif threshold.startswith("<"):
                target = float(threshold[1:].strip().rstrip("s"))
                if value < target:
                    passed += 1
        
        ratio = passed / total if total > 0 else 0
        
        if ratio >= 1.0:
            return "PROMOTE", ratio
        elif ratio >= 0.5:
            return "HOLD", ratio
        else:
            return "QUARANTINE", ratio
    
    async def approve_promotion(
        self,
        request_id: str,
        approver: str,
    ) -> bool:
        """Approve promotion request."""
        return await self.promotion.approve_promotion(request_id, approver)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        health = self.health.get_current_health()
        slo = self.health.calculate_slo_compliance()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "versions": {
                "v1_experiments": len([
                    e for e in self._experiments.values()
                    if e["status"] == "running"
                ]),
                "v2_health": "healthy" if slo["compliant"] else "degraded",
                "v3_quarantined": len(self.quarantine.get_quarantine_records()),
            },
            "active_promotions": len(self.promotion.get_active_promotions()),
            "active_alerts": len(self.health.get_active_alerts()),
            "slo_compliance": slo,
            "blacklist_size": len(self.quarantine.get_blacklist()),
        }
    
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
            source="lifecycle-coordinator",
        )
        
        if self.event_bus:
            await self.event_bus.publish(event_type.value, event.to_dict())
