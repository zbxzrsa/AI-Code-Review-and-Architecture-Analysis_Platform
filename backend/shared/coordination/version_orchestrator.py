"""
Version Orchestrator

Central coordinator for the Three-Version Self-Evolution Cycle.

Architecture:
- V1 (New/Experimentation): Tests new technologies, trial and error
- V2 (Stable/Production): User-facing, bug fixes, compatibility optimization
- V3 (Old/Quarantine): Deprecated technologies, bad reputation elimination

AI Models per Version:
- Version Control AI: Admin-only, manages version decisions
- Code Review AI: User-facing, performs code analysis

Access Control:
- Users: Can only access Code Review AI (CR-AI) on V2
- Admins: Can access both CR-AI and VC-AI on all versions
- System: Full access for self-evolution cycle
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class VersionState(str, Enum):
    """State of a version."""
    ACTIVE = "active"
    TESTING = "testing"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class AIModelType(str, Enum):
    """Types of AI models in each version."""
    VERSION_CONTROL = "version_control"  # Admin-only
    CODE_REVIEW = "code_review"          # User-facing


class UserRole(str, Enum):
    """User roles for access control."""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class VersionConfig:
    """Configuration for a version."""
    version: str  # v1, v2, v3
    state: VersionState
    cr_ai_endpoint: str
    vc_ai_endpoint: str
    cr_ai_accessible_by: List[UserRole]
    vc_ai_accessible_by: List[UserRole]
    description: str
    deployed_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EvolutionMetrics:
    """Metrics for version evolution decisions."""
    accuracy: float = 0.0
    error_rate: float = 0.0
    latency_p95_ms: float = 0.0
    user_satisfaction: float = 0.0
    cost_per_request: float = 0.0
    security_score: float = 0.0
    stability_score: float = 0.0


class VersionOrchestrator:
    """
    Orchestrates the Three-Version Self-Evolution Cycle.
    
    Responsibilities:
    1. Manage version states (V1, V2, V3)
    2. Control AI model access per user role
    3. Orchestrate the evolution cycle
    4. Handle version promotions and demotions
    """
    
    # Default version configurations
    DEFAULT_CONFIGS = {
        "v1": VersionConfig(
            version="v1",
            state=VersionState.TESTING,
            cr_ai_endpoint="/api/v1/cr-ai",
            vc_ai_endpoint="/api/v1/vc-ai",
            cr_ai_accessible_by=[UserRole.ADMIN, UserRole.SYSTEM],  # Testing only
            vc_ai_accessible_by=[UserRole.ADMIN, UserRole.SYSTEM],  # Admin only
            description="Experimentation zone - new technologies and trial/error",
        ),
        "v2": VersionConfig(
            version="v2",
            state=VersionState.ACTIVE,
            cr_ai_endpoint="/api/v2/cr-ai",
            vc_ai_endpoint="/api/v2/vc-ai",
            cr_ai_accessible_by=[UserRole.USER, UserRole.ADMIN, UserRole.SYSTEM],  # User-facing
            vc_ai_accessible_by=[UserRole.ADMIN, UserRole.SYSTEM],  # Admin only
            description="Production stable - user-facing, reliability focused",
        ),
        "v3": VersionConfig(
            version="v3",
            state=VersionState.DEPRECATED,
            cr_ai_endpoint="/api/v3/cr-ai",
            vc_ai_endpoint="/api/v3/vc-ai",
            cr_ai_accessible_by=[UserRole.ADMIN, UserRole.SYSTEM],  # Archive only
            vc_ai_accessible_by=[UserRole.ADMIN, UserRole.SYSTEM],  # Admin only
            description="Quarantine zone - deprecated technologies, elimination",
        ),
    }
    
    def __init__(
        self,
        event_bus = None,
        metrics_client = None,
        promotion_manager = None,
        quarantine_manager = None,
    ):
        self.event_bus = event_bus
        self.metrics = metrics_client
        self.promotion = promotion_manager
        self.quarantine = quarantine_manager
        
        # Version configurations
        self._versions: Dict[str, VersionConfig] = {
            k: VersionConfig(**v.__dict__) 
            for k, v in self.DEFAULT_CONFIGS.items()
        }
        
        # Evolution metrics per version
        self._metrics: Dict[str, EvolutionMetrics] = {
            "v1": EvolutionMetrics(),
            "v2": EvolutionMetrics(),
            "v3": EvolutionMetrics(),
        }
        
        # Running state
        self._running = False
        self._evolution_task: Optional[asyncio.Task] = None
    
    # =========================================================================
    # Access Control
    # =========================================================================
    
    def can_access_model(
        self,
        user_role: UserRole,
        version: str,
        model_type: AIModelType,
    ) -> bool:
        """
        Check if a user role can access a specific AI model.
        
        Access Rules:
        - Users can ONLY access Code Review AI on V2
        - Admins can access both AI models on all versions
        - System has full access everywhere
        """
        config = self._versions.get(version)
        if not config:
            return False
        
        if model_type == AIModelType.CODE_REVIEW:
            return user_role in config.cr_ai_accessible_by
        elif model_type == AIModelType.VERSION_CONTROL:
            return user_role in config.vc_ai_accessible_by
        
        return False
    
    def get_accessible_endpoints(
        self,
        user_role: UserRole,
    ) -> Dict[str, List[str]]:
        """Get all accessible endpoints for a user role."""
        endpoints = {}
        
        for version, config in self._versions.items():
            version_endpoints = []
            
            if user_role in config.cr_ai_accessible_by:
                version_endpoints.append(config.cr_ai_endpoint)
            
            if user_role in config.vc_ai_accessible_by:
                version_endpoints.append(config.vc_ai_endpoint)
            
            if version_endpoints:
                endpoints[version] = version_endpoints
        
        return endpoints
    
    def get_user_version(self) -> str:
        """Get the version available to regular users (always V2)."""
        return "v2"
    
    # =========================================================================
    # Evolution Cycle
    # =========================================================================
    
    async def start_evolution_cycle(self):
        """Start the self-evolution cycle."""
        if self._running:
            return
        
        self._running = True
        self._evolution_task = asyncio.create_task(self._evolution_loop())
        
        logger.info("Version Evolution Cycle started")
    
    async def stop_evolution_cycle(self):
        """Stop the self-evolution cycle."""
        self._running = False
        
        if self._evolution_task:
            self._evolution_task.cancel()
            try:
                await self._evolution_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Version Evolution Cycle stopped")
    
    async def _evolution_loop(self):
        """Main evolution cycle loop."""
        while self._running:
            try:
                # Step 1: Collect metrics from all versions
                await self._collect_version_metrics()
                
                # Step 2: Evaluate V1 experiments
                await self._evaluate_v1_experiments()
                
                # Step 3: Monitor V2 health
                await self._monitor_v2_health()
                
                # Step 4: Review V3 for potential revival
                await self._review_v3_quarantine()
                
                # Step 5: Execute pending evolutions
                await self._execute_evolutions()
                
                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Evolution cycle error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_version_metrics(self):
        """Collect metrics from all versions."""
        for version in ["v1", "v2", "v3"]:
            if self.metrics:
                metrics = await self.metrics.get_version_metrics(version)
                self._metrics[version] = EvolutionMetrics(
                    accuracy=metrics.get("accuracy", 0),
                    error_rate=metrics.get("error_rate", 0),
                    latency_p95_ms=metrics.get("latency_p95_ms", 0),
                    user_satisfaction=metrics.get("user_satisfaction", 0),
                    cost_per_request=metrics.get("cost_per_request", 0),
                    security_score=metrics.get("security_score", 0),
                    stability_score=metrics.get("stability_score", 0),
                )
    
    async def _evaluate_v1_experiments(self):
        """Evaluate V1 experiments for promotion to V2."""
        v1_metrics = self._metrics["v1"]
        v2_metrics = self._metrics["v2"]
        
        # Promotion criteria
        promotion_criteria = {
            "accuracy": v1_metrics.accuracy >= v2_metrics.accuracy * 0.98,  # No regression > 2%
            "error_rate": v1_metrics.error_rate <= 0.02,  # < 2% error rate
            "latency": v1_metrics.latency_p95_ms <= 3000,  # p95 < 3s
            "security": v1_metrics.security_score >= 0.95,  # High security
            "stability": v1_metrics.stability_score >= 0.90,  # Good stability
        }
        
        all_passed = all(promotion_criteria.values())
        
        if all_passed:
            logger.info("V1 meets promotion criteria, initiating V1 → V2 promotion")
            await self._initiate_promotion("v1", "v2")
        else:
            failed = [k for k, v in promotion_criteria.items() if not v]
            logger.debug(f"V1 promotion criteria not met: {failed}")
    
    async def _monitor_v2_health(self):
        """Monitor V2 production health."""
        v2_metrics = self._metrics["v2"]
        
        # Health thresholds
        if v2_metrics.error_rate > 0.05:  # > 5% error rate
            logger.critical("V2 error rate critical, considering rollback")
            await self._initiate_rollback("v2")
        
        elif v2_metrics.latency_p95_ms > 5000:  # > 5s latency
            logger.warning("V2 latency degraded, scaling up")
            await self._scale_version("v2", scale_factor=2)
    
    async def _review_v3_quarantine(self):
        """Review V3 quarantine for potential revival."""
        v3_config = self._versions["v3"]
        
        # Check if any quarantined technology can be re-evaluated
        if self.quarantine:
            pending_reviews = self.quarantine.get_quarantine_records(pending_review=True)
            
            for record in pending_reviews:
                # Check if context has changed
                if await self._context_allows_retry(record):
                    logger.info(f"Quarantined item {record.experiment_id} approved for retry in V1")
                    await self._move_to_v1(record)
    
    async def _execute_evolutions(self):
        """Execute pending version evolutions."""
        # Check for pending promotions
        if self.promotion:
            active = self.promotion.get_active_promotions()
            for promotion in active:
                logger.info(f"Promotion in progress: {promotion.request_id}")
    
    # =========================================================================
    # Version Transitions
    # =========================================================================
    
    async def _initiate_promotion(self, from_version: str, to_version: str):
        """Initiate version promotion (V1 → V2)."""
        logger.info(f"Initiating promotion: {from_version} → {to_version}")
        
        if self.promotion:
            await self.promotion.request_promotion(
                experiment_id=f"{from_version}_tech_{datetime.utcnow().strftime('%Y%m%d')}",
                evaluation_metrics=self._metrics[from_version].__dict__,
                confidence_score=0.95,
            )
        
        # Update version states
        # V2 becomes V3 (archive old stable)
        self._versions["v3"] = VersionConfig(
            **{**self._versions["v2"].__dict__, "state": VersionState.ARCHIVED}
        )
        
        # V1 becomes V2 (new stable)
        self._versions["v2"] = VersionConfig(
            **{**self._versions["v1"].__dict__, "state": VersionState.ACTIVE}
        )
        
        # Reset V1 for new experiments
        self._versions["v1"] = VersionConfig(
            version="v1",
            state=VersionState.TESTING,
            cr_ai_endpoint="/api/v1/cr-ai",
            vc_ai_endpoint="/api/v1/vc-ai",
            cr_ai_accessible_by=[UserRole.ADMIN, UserRole.SYSTEM],
            vc_ai_accessible_by=[UserRole.ADMIN, UserRole.SYSTEM],
            description="New experimentation zone",
        )
    
    async def _initiate_rollback(self, version: str):
        """Initiate rollback to previous stable version."""
        logger.warning(f"Initiating rollback for {version}")
        
        # In production, would restore from V3 archive
        # For now, reset V2 to safe defaults
    
    async def _scale_version(self, version: str, scale_factor: int):
        """Scale a version's resources."""
        logger.info(f"Scaling {version} by factor {scale_factor}")
        # In production, would trigger Kubernetes HPA
    
    async def _move_to_v1(self, record):
        """Move quarantined item back to V1 for retry."""
        logger.info(f"Moving {record.experiment_id} to V1 for retry")
        # Create new experiment in V1
    
    async def _context_allows_retry(self, record) -> bool:
        """Check if context allows retrying a quarantined item."""
        # Check if new models, fixes, or context changes allow retry
        days_since = (datetime.utcnow() - record.quarantined_at).days
        return days_since >= 90  # Quarterly review
    
    # =========================================================================
    # Status & Information
    # =========================================================================
    
    def get_version_status(self) -> Dict[str, Any]:
        """Get status of all versions."""
        return {
            version: {
                "state": config.state.value,
                "description": config.description,
                "cr_ai_endpoint": config.cr_ai_endpoint,
                "vc_ai_endpoint": config.vc_ai_endpoint,
                "last_updated": config.last_updated.isoformat(),
                "metrics": self._metrics[version].__dict__,
            }
            for version, config in self._versions.items()
        }
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get status of the evolution cycle."""
        return {
            "running": self._running,
            "versions": self.get_version_status(),
            "user_version": self.get_user_version(),
            "admin_versions": ["v1", "v2", "v3"],
        }


# =============================================================================
# Access Control Middleware
# =============================================================================

class AccessControlMiddleware:
    """
    Middleware to enforce AI model access control.
    
    Rules:
    - Regular users can ONLY access /api/v2/cr-ai/*
    - Admins can access all CR-AI and VC-AI endpoints
    - VC-AI endpoints are NEVER accessible to regular users
    """
    
    def __init__(self, orchestrator: VersionOrchestrator):
        self.orchestrator = orchestrator
    
    async def __call__(self, request, call_next):
        """Check access control before processing request."""
        from fastapi import HTTPException
        
        path = request.url.path
        user_role = self._get_user_role(request)
        
        # Determine which version and model type is being accessed
        version, model_type = self._parse_endpoint(path)
        
        if version and model_type:
            # Check access
            if not self.orchestrator.can_access_model(user_role, version, model_type):
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied: {user_role.value} cannot access {model_type.value} on {version}"
                )
        
        return await call_next(request)
    
    def _get_user_role(self, request) -> UserRole:
        """Extract user role from request."""
        # In production, extract from JWT token
        role_header = request.headers.get("X-User-Role", "user")
        
        try:
            return UserRole(role_header.lower())
        except ValueError:
            return UserRole.USER
    
    def _parse_endpoint(self, path: str) -> tuple:
        """Parse endpoint to determine version and model type."""
        if "/api/v1/cr-ai" in path:
            return "v1", AIModelType.CODE_REVIEW
        elif "/api/v1/vc-ai" in path:
            return "v1", AIModelType.VERSION_CONTROL
        elif "/api/v2/cr-ai" in path:
            return "v2", AIModelType.CODE_REVIEW
        elif "/api/v2/vc-ai" in path:
            return "v2", AIModelType.VERSION_CONTROL
        elif "/api/v3/cr-ai" in path:
            return "v3", AIModelType.CODE_REVIEW
        elif "/api/v3/vc-ai" in path:
            return "v3", AIModelType.VERSION_CONTROL
        
        return None, None


# =============================================================================
# Self-Evolution Engine
# =============================================================================

class SelfEvolutionEngine:
    """
    Engine that drives the self-evolution of the platform.
    
    The engine:
    1. Monitors all three versions continuously
    2. Evaluates new technologies in V1
    3. Maintains stability in V2
    4. Cleans up deprecated technologies in V3
    5. Automatically promotes successful experiments
    6. Automatically quarantines failed experiments
    """
    
    def __init__(
        self,
        orchestrator: VersionOrchestrator,
        promotion_manager = None,
        quarantine_manager = None,
        health_monitor = None,
    ):
        self.orchestrator = orchestrator
        self.promotion = promotion_manager
        self.quarantine = quarantine_manager
        self.health = health_monitor
        
        self._running = False
    
    async def start(self):
        """Start the self-evolution engine."""
        self._running = True
        
        # Start orchestrator
        await self.orchestrator.start_evolution_cycle()
        
        # Start health monitoring
        if self.health:
            await self.health.start()
        
        logger.info("Self-Evolution Engine started")
        logger.info("=" * 50)
        logger.info("Three-Version Self-Evolving Cycle Active:")
        logger.info("  V1 (New): Testing new technologies")
        logger.info("  V2 (Stable): User-facing production")
        logger.info("  V3 (Old): Quarantine for elimination")
        logger.info("=" * 50)
    
    async def stop(self):
        """Stop the self-evolution engine."""
        self._running = False
        
        await self.orchestrator.stop_evolution_cycle()
        
        if self.health:
            await self.health.stop()
        
        logger.info("Self-Evolution Engine stopped")
    
    def get_cycle_status(self) -> Dict[str, Any]:
        """Get current status of the evolution cycle."""
        return {
            "running": self._running,
            "orchestrator": self.orchestrator.get_evolution_status(),
            "timestamp": datetime.utcnow().isoformat(),
        }
