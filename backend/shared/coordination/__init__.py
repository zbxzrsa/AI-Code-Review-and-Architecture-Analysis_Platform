"""
System Coordination Protocol

Three-Version Self-Evolving Cycle Orchestration:
- V1 (Experimentation): New technologies, trial and error
- V2 (Production): Stable user-facing version
- V3 (Quarantine): Deprecated technologies archive

AI Models per Version:
- Version Control AI (VC-AI): Admin-only, manages version decisions
- Code Review AI (CR-AI): User-facing on V2, admin-only on V1/V3

Access Control:
- Users: Can ONLY access CR-AI on V2
- Admins: Can access both CR-AI and VC-AI on all versions
- System: Full access for self-evolution cycle
"""

from .lifecycle_coordinator import LifecycleCoordinator
from .promotion_manager import PromotionManager
from .quarantine_manager import QuarantineManager
from .health_monitor import HealthMonitor
from .experiment_generator import ExperimentGenerator
from .event_types import EventType, VersionEvent
from .version_orchestrator import (
    VersionOrchestrator,
    SelfEvolutionEngine,
    AccessControlMiddleware,
    UserRole,
    AIModelType,
    VersionState,
)
from .startup import PlatformStartup, create_platform

__all__ = [
    # Core Components
    "LifecycleCoordinator",
    "PromotionManager",
    "QuarantineManager",
    "HealthMonitor",
    "ExperimentGenerator",
    
    # Version Management
    "VersionOrchestrator",
    "SelfEvolutionEngine",
    "PlatformStartup",
    "create_platform",
    
    # Access Control
    "AccessControlMiddleware",
    "UserRole",
    "AIModelType",
    "VersionState",
    
    # Events
    "EventType",
    "VersionEvent",
]
