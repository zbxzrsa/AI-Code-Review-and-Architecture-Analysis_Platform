"""
Platform Startup Orchestrator

Initializes and coordinates the Three-Version Self-Evolution Platform.

This module:
1. Initializes all three versions (V1, V2, V3)
2. Sets up the self-evolution cycle
3. Configures access control
4. Starts health monitoring
5. Begins the continuous improvement loop
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .version_orchestrator import (
    VersionOrchestrator,
    SelfEvolutionEngine,
    AccessControlMiddleware,
    UserRole,
    AIModelType,
)
from .promotion_manager import PromotionManager
from .quarantine_manager import QuarantineManager
from .health_monitor import HealthMonitor
from .experiment_generator import ExperimentGenerator
from .lifecycle_coordinator import LifecycleCoordinator

logger = logging.getLogger(__name__)


class PlatformStartup:
    """
    Orchestrates the complete platform startup.
    
    Initializes:
    - Three-Version Architecture (V1, V2, V3)
    - AI Models (CR-AI user-facing, VC-AI admin-only)
    - Self-Evolution Cycle
    - Access Control
    - Health Monitoring
    """
    
    def __init__(
        self,
        event_bus = None,
        db_connection = None,
        redis_client = None,
        metrics_client = None,
    ):
        self.event_bus = event_bus
        self.db = db_connection
        self.redis = redis_client
        self.metrics = metrics_client
        
        # Core components
        self.version_orchestrator: Optional[VersionOrchestrator] = None
        self.lifecycle_coordinator: Optional[LifecycleCoordinator] = None
        self.evolution_engine: Optional[SelfEvolutionEngine] = None
        
        # Managers
        self.promotion_manager: Optional[PromotionManager] = None
        self.quarantine_manager: Optional[QuarantineManager] = None
        self.health_monitor: Optional[HealthMonitor] = None
        self.experiment_generator: Optional[ExperimentGenerator] = None
        
        # State
        self._initialized = False
        self._running = False
    
    async def initialize(self):
        """Initialize all platform components."""
        logger.info("=" * 60)
        logger.info("INITIALIZING THREE-VERSION SELF-EVOLUTION PLATFORM")
        logger.info("=" * 60)
        
        # Step 1: Initialize managers
        logger.info("Step 1: Initializing managers...")
        
        self.promotion_manager = PromotionManager(
            event_bus=self.event_bus,
            metrics_client=self.metrics,
        )
        
        self.quarantine_manager = QuarantineManager(
            event_bus=self.event_bus,
            db_connection=self.db,
        )
        
        self.health_monitor = HealthMonitor(
            event_bus=self.event_bus,
            metrics_client=self.metrics,
        )
        
        self.experiment_generator = ExperimentGenerator(
            event_bus=self.event_bus,
            quarantine_manager=self.quarantine_manager,
            health_monitor=self.health_monitor,
        )
        
        logger.info("  ✓ Promotion Manager initialized")
        logger.info("  ✓ Quarantine Manager initialized")
        logger.info("  ✓ Health Monitor initialized")
        logger.info("  ✓ Experiment Generator initialized")
        
        # Step 2: Initialize version orchestrator
        logger.info("Step 2: Initializing Version Orchestrator...")
        
        self.version_orchestrator = VersionOrchestrator(
            event_bus=self.event_bus,
            metrics_client=self.metrics,
            promotion_manager=self.promotion_manager,
            quarantine_manager=self.quarantine_manager,
        )
        
        logger.info("  ✓ Version Orchestrator initialized")
        logger.info("  → V1 (Experimentation): Testing new technologies")
        logger.info("  → V2 (Production): User-facing stable version")
        logger.info("  → V3 (Quarantine): Deprecated technology archive")
        
        # Step 3: Initialize lifecycle coordinator
        logger.info("Step 3: Initializing Lifecycle Coordinator...")
        
        self.lifecycle_coordinator = LifecycleCoordinator(
            event_bus=self.event_bus,
            db_connection=self.db,
            metrics_client=self.metrics,
        )
        
        logger.info("  ✓ Lifecycle Coordinator initialized")
        
        # Step 4: Initialize self-evolution engine
        logger.info("Step 4: Initializing Self-Evolution Engine...")
        
        self.evolution_engine = SelfEvolutionEngine(
            orchestrator=self.version_orchestrator,
            promotion_manager=self.promotion_manager,
            quarantine_manager=self.quarantine_manager,
            health_monitor=self.health_monitor,
        )
        
        logger.info("  ✓ Self-Evolution Engine initialized")
        
        self._initialized = True
        
        logger.info("=" * 60)
        logger.info("PLATFORM INITIALIZATION COMPLETE")
        logger.info("=" * 60)
    
    async def start(self):
        """Start the platform and begin self-evolution cycle."""
        if not self._initialized:
            await self.initialize()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("STARTING THREE-VERSION SELF-EVOLUTION CYCLE")
        logger.info("=" * 60)
        logger.info("")
        
        # Start health monitoring
        logger.info("Starting Health Monitor...")
        await self.health_monitor.start()
        
        # Start lifecycle coordinator
        logger.info("Starting Lifecycle Coordinator...")
        await self.lifecycle_coordinator.start()
        
        # Start self-evolution engine
        logger.info("Starting Self-Evolution Engine...")
        await self.evolution_engine.start()
        
        self._running = True
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("PLATFORM RUNNING")
        logger.info("=" * 60)
        logger.info("")
        logger.info("VERSION ARCHITECTURE:")
        logger.info("┌─────────────────────────────────────────────────────────┐")
        logger.info("│ V1 (Experimentation)                                    │")
        logger.info("│   ├─ CR-AI: Admin/System only (testing)                │")
        logger.info("│   └─ VC-AI: Admin/System only                          │")
        logger.info("├─────────────────────────────────────────────────────────┤")
        logger.info("│ V2 (Production) ★ USER-FACING                          │")
        logger.info("│   ├─ CR-AI: Users, Admins, System                      │")
        logger.info("│   └─ VC-AI: Admin/System only (NEVER users)            │")
        logger.info("├─────────────────────────────────────────────────────────┤")
        logger.info("│ V3 (Quarantine)                                        │")
        logger.info("│   ├─ CR-AI: Admin/System only (archived)               │")
        logger.info("│   └─ VC-AI: Admin/System only (archived)               │")
        logger.info("└─────────────────────────────────────────────────────────┘")
        logger.info("")
        logger.info("EVOLUTION CYCLE:")
        logger.info("  V1 tests new tech → Success? → Promote to V2")
        logger.info("                    → Failure? → Quarantine to V3")
        logger.info("  V2 serves users   → Degradation? → Rollback")
        logger.info("  V3 archives failures → Quarterly review → Retry in V1?")
        logger.info("")
        logger.info("ACCESS CONTROL:")
        logger.info("  • Users:  /api/v2/cr-ai/* ONLY")
        logger.info("  • Admins: All CR-AI and VC-AI endpoints")
        logger.info("  • System: Full access (self-evolution)")
        logger.info("")
    
    async def stop(self):
        """Stop the platform gracefully."""
        logger.info("Stopping platform...")
        
        if self.evolution_engine:
            await self.evolution_engine.stop()
        
        if self.lifecycle_coordinator:
            await self.lifecycle_coordinator.stop()
        
        if self.health_monitor:
            await self.health_monitor.stop()
        
        self._running = False
        
        logger.info("Platform stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current platform status."""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "timestamp": datetime.utcnow().isoformat(),
            "versions": self.version_orchestrator.get_version_status() if self.version_orchestrator else {},
            "evolution": self.evolution_engine.get_cycle_status() if self.evolution_engine else {},
            "access_control": {
                "user_version": "v2",
                "user_endpoints": ["/api/v2/cr-ai"],
                "admin_endpoints": [
                    "/api/v1/cr-ai", "/api/v1/vc-ai",
                    "/api/v2/cr-ai", "/api/v2/vc-ai",
                    "/api/v3/cr-ai", "/api/v3/vc-ai",
                ],
            },
        }


# =============================================================================
# Factory Function
# =============================================================================

async def create_platform(
    event_bus = None,
    db_connection = None,
    redis_client = None,
    metrics_client = None,
) -> PlatformStartup:
    """Create and initialize the platform."""
    platform = PlatformStartup(
        event_bus=event_bus,
        db_connection=db_connection,
        redis_client=redis_client,
        metrics_client=metrics_client,
    )
    
    await platform.initialize()
    
    return platform


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Main entry point for running the platform."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    platform = await create_platform()
    
    try:
        await platform.start()
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            
            status = platform.get_status()
            logger.info(f"Platform status: running={status['running']}")
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await platform.stop()


if __name__ == "__main__":
    asyncio.run(main())
