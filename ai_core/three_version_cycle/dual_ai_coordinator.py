"""
Dual-AI Coordinator

Manages the two types of AI per version:
1. Version Control AI (VC-AI): Admin-only, manages version decisions and self-evolution
2. Code Review AI (CR-AI): User-facing, performs code analysis

Each version (V1, V2, V3) has its own dedicated VC-AI and CR-AI instances.

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                        Dual-AI Coordinator                              │
├─────────────────────────────────────────────────────────────────────────┤
│  V1 (Experimental)        V2 (Stable)          V3 (Quarantine)         │
│  ┌─────────────────┐     ┌─────────────────┐   ┌─────────────────┐     │
│  │ V1-VCAI (Admin) │     │ V2-VCAI (Admin) │   │ V3-VCAI (Admin) │     │
│  │ - Experiments   │     │ - Fixes bugs    │   │ - Analyzes      │     │
│  │ - Trials        │     │ - Optimizes     │   │ - Compares      │     │
│  └─────────────────┘     └─────────────────┘   └─────────────────┘     │
│  ┌─────────────────┐     ┌─────────────────┐   ┌─────────────────┐     │
│  │ V1-CRAI (Test)  │     │ V2-CRAI (Users) │   │ V3-CRAI (Ref)   │     │
│  │ - Shadow tests  │     │ - Production    │   │ - Baseline      │     │
│  └─────────────────┘     └─────────────────┘   └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import uuid

logger = logging.getLogger(__name__)


class AIType(str, Enum):
    """Types of AI in each version."""
    VERSION_CONTROL = "vc_ai"  # Admin-only: manages versions and evolution
    CODE_REVIEW = "cr_ai"      # User-facing: performs code analysis


class AIStatus(str, Enum):
    """Status of an AI instance."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class AccessLevel(str, Enum):
    """Access level for AI endpoints."""
    PUBLIC = "public"       # All users (V2 CR-AI only)
    TESTING = "testing"     # Test users and admins (V1 CR-AI)
    ADMIN = "admin"         # Admins only (VC-AI, V3)
    SYSTEM = "system"       # System-only internal calls


@dataclass
class AIInstance:
    """Represents an AI instance (either VC-AI or CR-AI)."""
    instance_id: str
    version: str        # v1, v2, v3
    ai_type: AIType
    status: AIStatus
    access_level: AccessLevel
    
    # Configuration
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    capabilities: List[str] = field(default_factory=list)
    
    # Metrics
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def error_rate(self) -> float:
        return self.error_count / self.request_count if self.request_count > 0 else 0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.request_count if self.request_count > 0 else 0


@dataclass
class VersionAIPair:
    """A pair of VC-AI and CR-AI for a version."""
    version: str
    vc_ai: AIInstance
    cr_ai: AIInstance
    
    def get_status(self) -> Dict[str, str]:
        return {
            "version": self.version,
            "vc_ai_status": self.vc_ai.status.value,
            "cr_ai_status": self.cr_ai.status.value,
        }


class DualAICoordinator:
    """
    Coordinates the dual-AI architecture across all three versions.
    
    Responsibilities:
    1. Manage VC-AI and CR-AI instances per version
    2. Route requests to appropriate AI based on user role
    3. Coordinate cross-version AI communication
    4. Enforce access control policies
    """
    
    def __init__(
        self,
        event_bus=None,
        version_manager=None,
    ):
        self.event_bus = event_bus
        self.version_manager = version_manager
        
        # Initialize AI pairs for each version
        self._version_pairs: Dict[str, VersionAIPair] = {}
        self._initialize_ai_pairs()
        
        # Cross-version communication handlers
        self._cross_version_handlers: Dict[str, Callable] = {}
        
        # Lock for coordination
        self._lock = asyncio.Lock()
    
    def _initialize_ai_pairs(self):
        """Initialize AI pairs for all versions."""
        
        # V1 Experimental
        v1_vc = AIInstance(
            instance_id=str(uuid.uuid4()),
            version="v1",
            ai_type=AIType.VERSION_CONTROL,
            status=AIStatus.ACTIVE,
            access_level=AccessLevel.ADMIN,
            model_name="experimental-vcai",
            capabilities=[
                "experiment_management",
                "technology_trial",
                "error_detection",
                "metrics_collection",
            ],
        )
        
        v1_cr = AIInstance(
            instance_id=str(uuid.uuid4()),
            version="v1",
            ai_type=AIType.CODE_REVIEW,
            status=AIStatus.ACTIVE,
            access_level=AccessLevel.TESTING,
            model_name="experimental-crai",
            capabilities=[
                "code_review",
                "bug_detection",
                "security_scan",
                "shadow_analysis",
            ],
        )
        
        self._version_pairs["v1"] = VersionAIPair("v1", v1_vc, v1_cr)
        
        # V2 Production (Stable)
        v2_vc = AIInstance(
            instance_id=str(uuid.uuid4()),
            version="v2",
            ai_type=AIType.VERSION_CONTROL,
            status=AIStatus.ACTIVE,
            access_level=AccessLevel.ADMIN,
            model_name="stable-vcai",
            capabilities=[
                "error_fixing",
                "compatibility_optimization",
                "performance_tuning",
                "stability_monitoring",
                "v1_error_remediation",
            ],
        )
        
        v2_cr = AIInstance(
            instance_id=str(uuid.uuid4()),
            version="v2",
            ai_type=AIType.CODE_REVIEW,
            status=AIStatus.ACTIVE,
            access_level=AccessLevel.PUBLIC,
            model_name="stable-crai",
            capabilities=[
                "code_review",
                "bug_detection",
                "security_scan",
                "optimization_suggestions",
                "documentation_generation",
                "test_generation",
                "refactoring",
            ],
        )
        
        self._version_pairs["v2"] = VersionAIPair("v2", v2_vc, v2_cr)
        
        # V3 Quarantine
        v3_vc = AIInstance(
            instance_id=str(uuid.uuid4()),
            version="v3",
            ai_type=AIType.VERSION_CONTROL,
            status=AIStatus.ACTIVE,
            access_level=AccessLevel.ADMIN,
            model_name="quarantine-vcai",
            capabilities=[
                "failure_analysis",
                "comparison_baseline",
                "technology_exclusion",
                "pattern_learning",
            ],
        )
        
        v3_cr = AIInstance(
            instance_id=str(uuid.uuid4()),
            version="v3",
            ai_type=AIType.CODE_REVIEW,
            status=AIStatus.ACTIVE,
            access_level=AccessLevel.ADMIN,
            model_name="quarantine-crai",
            capabilities=[
                "baseline_analysis",
                "comparison_review",
            ],
        )
        
        self._version_pairs["v3"] = VersionAIPair("v3", v3_vc, v3_cr)
        
        logger.info("Initialized dual-AI pairs for all versions")
    
    # =========================================================================
    # AI Access
    # =========================================================================
    
    def get_ai(
        self,
        version: str,
        ai_type: AIType,
    ) -> Optional[AIInstance]:
        """Get an AI instance."""
        pair = self._version_pairs.get(version)
        if not pair:
            return None
        
        if ai_type == AIType.VERSION_CONTROL:
            return pair.vc_ai
        else:
            return pair.cr_ai
    
    def get_user_accessible_ai(self) -> AIInstance:
        """Get the AI accessible to regular users (V2 CR-AI)."""
        return self._version_pairs["v2"].cr_ai
    
    def get_admin_ais(self) -> Dict[str, List[AIInstance]]:
        """Get all AIs accessible to admins."""
        result = {}
        for version, pair in self._version_pairs.items():
            result[version] = [pair.vc_ai, pair.cr_ai]
        return result
    
    # =========================================================================
    # Access Control
    # =========================================================================
    
    def can_access(
        self,
        user_role: str,
        version: str,
        ai_type: AIType,
    ) -> bool:
        """Check if a user role can access an AI instance."""
        ai = self.get_ai(version, ai_type)
        if not ai:
            return False
        
        access_level = ai.access_level
        
        if user_role == "system":
            return True
        
        if user_role == "admin":
            return access_level in [
                AccessLevel.PUBLIC, 
                AccessLevel.TESTING, 
                AccessLevel.ADMIN
            ]
        
        if user_role == "tester":
            return access_level in [AccessLevel.PUBLIC, AccessLevel.TESTING]
        
        if user_role == "user":
            return access_level == AccessLevel.PUBLIC
        
        return False
    
    # =========================================================================
    # Request Routing
    # =========================================================================
    
    async def route_request(
        self,
        user_role: str,
        request_type: str,  # "code_review", "version_control"
        request_data: Dict[str, Any],
        preferred_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route a request to the appropriate AI."""
        
        # Determine AI type from request
        ai_type = AIType.VERSION_CONTROL if request_type == "version_control" else AIType.CODE_REVIEW
        
        # Determine version
        if ai_type == AIType.CODE_REVIEW and user_role == "user":
            # Users always go to V2
            version = "v2"
        elif preferred_version and self.can_access(user_role, preferred_version, ai_type):
            version = preferred_version
        else:
            version = "v2"  # Default to stable
        
        # Get AI instance
        ai = self.get_ai(version, ai_type)
        if not ai or ai.status != AIStatus.ACTIVE:
            return {"success": False, "error": "AI not available"}
        
        # Process request
        result = await self._process_request(ai, request_data)
        
        # Update metrics
        async with self._lock:
            ai.request_count += 1
            ai.total_latency_ms += result.get("latency_ms", 0)
            if not result.get("success", False):
                ai.error_count += 1
            ai.last_active = datetime.now(timezone.utc)
        
        return result
    
    async def _process_request(
        self,
        ai: AIInstance,
        request_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a request through an AI instance."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Simulated AI processing
            # In production, would call actual model
            result = {
                "success": True,
                "version": ai.version,
                "ai_type": ai.ai_type.value,
                "model": ai.model_name,
                "response": f"Processed by {ai.ai_type.value} on {ai.version}",
            }
            
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "version": ai.version,
                "ai_type": ai.ai_type.value,
            }
        
        result["latency_ms"] = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return result
    
    # =========================================================================
    # Cross-Version Communication
    # =========================================================================
    
    async def v2_fix_v1_error(
        self,
        error_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """V2 VC-AI analyzes and fixes V1 error."""
        v2_vc = self.get_ai("v2", AIType.VERSION_CONTROL)
        
        if not v2_vc or v2_vc.status != AIStatus.ACTIVE:
            return {"success": False, "error": "V2 VC-AI not available"}
        
        # V2 analyzes the error
        analysis = await self._v2_analyze_error(error_data)
        
        # Generate fix
        fix = await self._v2_generate_fix(analysis)
        
        logger.info(f"V2 VC-AI generated fix for V1 error: {error_data.get('error_id')}")
        
        return {
            "success": True,
            "analysis": analysis,
            "fix": fix,
            "source": "v2_vcai",
        }
    
    async def _v2_analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """V2 VC-AI analyzes V1 error."""
        return {
            "error_type": error_data.get("error_type"),
            "root_cause": "Identified compatibility issue",
            "affected_components": ["technology_adapter"],
            "severity": "medium",
            "fixable": True,
        }
    
    async def _v2_generate_fix(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """V2 VC-AI generates fix for error."""
        return {
            "fix_type": "compatibility_patch",
            "changes": [
                {"file": "adapter.py", "action": "add_wrapper"},
            ],
            "config_updates": {"timeout_ms": 5000},
            "estimated_success_rate": 0.95,
        }
    
    async def v3_compare_technology(
        self,
        tech_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """V3 VC-AI compares technology against quarantine baseline."""
        v3_vc = self.get_ai("v3", AIType.VERSION_CONTROL)
        
        if not v3_vc or v3_vc.status != AIStatus.ACTIVE:
            return {"success": False, "error": "V3 VC-AI not available"}
        
        comparison = {
            "technology": tech_data.get("name"),
            "baseline_comparison": {
                "accuracy_delta": 0.05,
                "error_rate_delta": -0.02,
            },
            "similar_failures": [],
            "recommendation": "proceed_with_caution",
        }
        
        return {
            "success": True,
            "comparison": comparison,
            "source": "v3_vcai",
        }
    
    async def coordinate_promotion(
        self,
        tech_id: str,
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """Coordinate AI handoff during promotion."""
        
        # Get both VC-AIs
        from_vc = self.get_ai(from_version, AIType.VERSION_CONTROL)
        to_vc = self.get_ai(to_version, AIType.VERSION_CONTROL)
        
        if not from_vc or not to_vc:
            return {"success": False, "error": "VC-AI not available"}
        
        # Transfer technology knowledge
        transfer_result = {
            "tech_id": tech_id,
            "from_version": from_version,
            "to_version": to_version,
            "knowledge_transferred": True,
            "configurations_synced": True,
        }
        
        logger.info(f"Coordinated promotion: {tech_id} from {from_version} to {to_version}")
        
        return {
            "success": True,
            "transfer": transfer_result,
        }
    
    # =========================================================================
    # Status & Metrics
    # =========================================================================
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all AI instances."""
        status = {}
        
        for version, pair in self._version_pairs.items():
            status[version] = {
                "vc_ai": {
                    "status": pair.vc_ai.status.value,
                    "access_level": pair.vc_ai.access_level.value,
                    "request_count": pair.vc_ai.request_count,
                    "error_rate": pair.vc_ai.error_rate,
                    "avg_latency_ms": pair.vc_ai.avg_latency_ms,
                    "capabilities": pair.vc_ai.capabilities,
                },
                "cr_ai": {
                    "status": pair.cr_ai.status.value,
                    "access_level": pair.cr_ai.access_level.value,
                    "request_count": pair.cr_ai.request_count,
                    "error_rate": pair.cr_ai.error_rate,
                    "avg_latency_ms": pair.cr_ai.avg_latency_ms,
                    "capabilities": pair.cr_ai.capabilities,
                },
            }
        
        return status
    
    def get_user_ai_status(self) -> Dict[str, Any]:
        """Get status of user-accessible AI (V2 CR-AI only)."""
        v2_cr = self._version_pairs["v2"].cr_ai
        
        return {
            "available": v2_cr.status == AIStatus.ACTIVE,
            "model": v2_cr.model_name,
            "capabilities": v2_cr.capabilities,
            "avg_latency_ms": v2_cr.avg_latency_ms,
        }
    
    def get_version_pair(self, version: str) -> Optional[VersionAIPair]:
        """Get the AI pair for a version."""
        return self._version_pairs.get(version)
    
    # =========================================================================
    # AI Lifecycle
    # =========================================================================
    
    async def set_ai_status(
        self,
        version: str,
        ai_type: AIType,
        status: AIStatus,
    ):
        """Set the status of an AI instance."""
        ai = self.get_ai(version, ai_type)
        if ai:
            old_status = ai.status
            ai.status = status
            
            logger.info(f"{version} {ai_type.value} status: {old_status.value} → {status.value}")
            
            if self.event_bus:
                await self.event_bus.publish("ai_status_changed", {
                    "version": version,
                    "ai_type": ai_type.value,
                    "old_status": old_status.value,
                    "new_status": status.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
    
    async def update_ai_capabilities(
        self,
        version: str,
        ai_type: AIType,
        capabilities: List[str],
    ):
        """Update capabilities of an AI instance."""
        ai = self.get_ai(version, ai_type)
        if ai:
            ai.capabilities = capabilities
            logger.info(f"Updated {version} {ai_type.value} capabilities")


# =============================================================================
# Dual-AI Request Handler
# =============================================================================

class DualAIRequestHandler:
    """
    Handles requests to the dual-AI system with proper routing.
    """
    
    def __init__(self, coordinator: DualAICoordinator):
        self.coordinator = coordinator
    
    async def handle_user_request(
        self,
        request_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a user request (routes to V2 CR-AI)."""
        return await self.coordinator.route_request(
            user_role="user",
            request_type="code_review",
            request_data=request_data,
        )
    
    async def handle_admin_request(
        self,
        request_type: str,
        version: str,
        ai_type: str,
        request_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle an admin request (can access any AI)."""
        return await self.coordinator.route_request(
            user_role="admin",
            request_type=request_type,
            request_data=request_data,
            preferred_version=version,
        )
    
    async def handle_system_request(
        self,
        request_type: str,
        version: str,
        request_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a system internal request."""
        return await self.coordinator.route_request(
            user_role="system",
            request_type=request_type,
            request_data=request_data,
            preferred_version=version,
        )
