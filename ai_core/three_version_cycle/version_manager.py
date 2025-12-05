"""
Version Manager

Manages the three concurrent versions of the AI system:
- V1 Experimentation: Testing new technologies
- V2 Production: Stable user-facing version
- V3 Quarantine: Archive for failed experiments

Handles promotion (V1→V2), degradation (V2→V3), and re-evaluation.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Version(str, Enum):
    """Version identifiers."""
    V1_EXPERIMENTAL = "v1"
    V2_PRODUCTION = "v2"
    V3_QUARANTINE = "v3"


class VersionState(str, Enum):
    """Version state."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    EVALUATING = "evaluating"
    PROMOTING = "promoting"
    DEGRADING = "degrading"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


@dataclass
class VersionConfig:
    """Configuration for a version."""
    version: Version
    name: str
    description: str
    max_concurrent_experiments: int = 10
    evaluation_interval_hours: int = 24
    min_samples_for_evaluation: int = 1000
    auto_promote: bool = False
    auto_degrade: bool = True
    
    # Thresholds
    accuracy_threshold: float = 0.85
    error_rate_threshold: float = 0.05
    latency_p95_threshold_ms: float = 3000
    cost_increase_threshold: float = 0.20


@dataclass
class VersionMetrics:
    """Metrics for a version."""
    version: Version
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    accuracy_sum: float = 0
    accuracy_count: int = 0
    cost_total: float = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.failed_requests / self.total_requests
    
    @property
    def avg_latency_ms(self) -> float:
        if self.successful_requests == 0:
            return 0
        return self.total_latency_ms / self.successful_requests
    
    @property
    def avg_accuracy(self) -> float:
        if self.accuracy_count == 0:
            return 0
        return self.accuracy_sum / self.accuracy_count
    
    @property
    def avg_cost_per_request(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.cost_total / self.total_requests


@dataclass
class Technology:
    """Represents a technology/technique being tested."""
    tech_id: str
    name: str
    category: str  # attention, architecture, training, optimization
    description: str
    config: Dict[str, Any]
    source: str  # e.g., "LLMs-from-scratch", "custom"
    version_introduced: Version
    status: str = "experimental"
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PromotionRecord:
    """Record of a promotion or degradation event."""
    record_id: str
    from_version: Version
    to_version: Version
    technologies: List[str]
    reason: str
    metrics_before: Dict[str, float]
    metrics_after: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    rollback_at: Optional[datetime] = None


# Default configurations for each version
DEFAULT_CONFIGS = {
    Version.V1_EXPERIMENTAL: VersionConfig(
        version=Version.V1_EXPERIMENTAL,
        name="V1 Experimentation",
        description="Testing ground for new AI technologies",
        max_concurrent_experiments=20,
        evaluation_interval_hours=6,
        min_samples_for_evaluation=500,
        auto_promote=False,
        auto_degrade=True,
        accuracy_threshold=0.80,  # Lower threshold for experiments
        error_rate_threshold=0.10,
        latency_p95_threshold_ms=5000,
        cost_increase_threshold=0.50,
    ),
    Version.V2_PRODUCTION: VersionConfig(
        version=Version.V2_PRODUCTION,
        name="V2 Production",
        description="Stable user-facing AI system",
        max_concurrent_experiments=0,  # No experiments in production
        evaluation_interval_hours=24,
        min_samples_for_evaluation=1000,
        auto_promote=False,
        auto_degrade=True,
        accuracy_threshold=0.85,
        error_rate_threshold=0.05,
        latency_p95_threshold_ms=3000,
        cost_increase_threshold=0.20,
    ),
    Version.V3_QUARANTINE: VersionConfig(
        version=Version.V3_QUARANTINE,
        name="V3 Quarantine",
        description="Archive for failed experiments",
        max_concurrent_experiments=0,
        evaluation_interval_hours=720,  # 30 days
        min_samples_for_evaluation=0,
        auto_promote=False,
        auto_degrade=False,
        accuracy_threshold=0.70,
        error_rate_threshold=0.20,
        latency_p95_threshold_ms=10000,
        cost_increase_threshold=1.0,
    ),
}


class VersionManager:
    """
    Manages the three-version self-evolution cycle.
    
    Responsibilities:
    - Track state and metrics for each version
    - Manage technology experiments
    - Handle promotions (V1→V2) and degradations (V2→V3)
    - Re-evaluate quarantined technologies
    """
    
    def __init__(
        self,
        configs: Optional[Dict[Version, VersionConfig]] = None,
        event_bus: Optional[Any] = None,
        db_connection: Optional[Any] = None,
    ):
        self.configs = configs or DEFAULT_CONFIGS.copy()
        self.event_bus = event_bus
        self.db = db_connection
        
        # State tracking
        self._states: Dict[Version, VersionState] = {
            Version.V1_EXPERIMENTAL: VersionState.ACTIVE,
            Version.V2_PRODUCTION: VersionState.ACTIVE,
            Version.V3_QUARANTINE: VersionState.ACTIVE,
        }
        
        # Metrics
        self._metrics: Dict[Version, VersionMetrics] = {
            v: VersionMetrics(version=v) for v in Version
        }
        
        # Technologies
        self._technologies: Dict[str, Technology] = {}
        self._version_technologies: Dict[Version, Set[str]] = {
            v: set() for v in Version
        }
        
        # Promotion history
        self._promotion_history: List[PromotionRecord] = []
        
        # Locks
        self._promotion_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def get_state(self, version: Version) -> VersionState:
        """Get current state of a version."""
        return self._states[version]
    
    async def set_state(self, version: Version, state: VersionState) -> bool:
        """Set state of a version."""
        async with self._state_lock:
            old_state = self._states[version]
            self._states[version] = state
            
            logger.info(f"Version {version.value} state: {old_state.value} → {state.value}")
            
            if self.event_bus:
                await self.event_bus.publish("version_state_changed", {
                    "version": version.value,
                    "old_state": old_state.value,
                    "new_state": state.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            
            return True
    
    # =========================================================================
    # Metrics Management
    # =========================================================================
    
    def record_request(
        self,
        version: Version,
        success: bool,
        latency_ms: float,
        accuracy: Optional[float] = None,
        cost: float = 0,
    ):
        """Record a request for metrics tracking."""
        metrics = self._metrics[version]
        metrics.total_requests += 1
        
        if success:
            metrics.successful_requests += 1
            metrics.total_latency_ms += latency_ms
        else:
            metrics.failed_requests += 1
        
        if accuracy is not None:
            metrics.accuracy_sum += accuracy
            metrics.accuracy_count += 1
        
        metrics.cost_total += cost
        metrics.last_updated = datetime.now(timezone.utc)
    
    def get_metrics(self, version: Version) -> VersionMetrics:
        """Get metrics for a version."""
        return self._metrics[version]
    
    def get_all_metrics(self) -> Dict[Version, VersionMetrics]:
        """Get metrics for all versions."""
        return self._metrics.copy()
    
    # =========================================================================
    # Technology Management
    # =========================================================================
    
    async def register_technology(
        self,
        name: str,
        category: str,
        description: str,
        config: Dict[str, Any],
        source: str = "custom",
        version: Version = Version.V1_EXPERIMENTAL,
    ) -> Technology:
        """Register a new technology for experimentation."""
        tech_id = str(uuid.uuid4())
        
        tech = Technology(
            tech_id=tech_id,
            name=name,
            category=category,
            description=description,
            config=config,
            source=source,
            version_introduced=version,
        )
        
        self._technologies[tech_id] = tech
        self._version_technologies[version].add(tech_id)
        
        logger.info(f"Registered technology: {name} ({tech_id}) in {version.value}")
        return tech
    
    async def get_technology(self, tech_id: str) -> Optional[Technology]:
        """Get a technology by ID."""
        return self._technologies.get(tech_id)
    
    async def get_version_technologies(self, version: Version) -> List[Technology]:
        """Get all technologies in a version."""
        tech_ids = self._version_technologies[version]
        return [self._technologies[tid] for tid in tech_ids if tid in self._technologies]
    
    async def update_technology_metrics(
        self,
        tech_id: str,
        metrics: Dict[str, float],
    ) -> bool:
        """Update metrics for a technology."""
        tech = self._technologies.get(tech_id)
        if not tech:
            return False
        
        tech.metrics.update(metrics)
        return True
    
    # =========================================================================
    # Promotion and Degradation
    # =========================================================================
    
    async def evaluate_for_promotion(self, tech_id: str) -> Dict[str, Any]:
        """Evaluate if a technology is ready for promotion."""
        tech = self._technologies.get(tech_id)
        if not tech:
            return {"eligible": False, "reason": "Technology not found"}
        
        if tech.version_introduced != Version.V1_EXPERIMENTAL:
            return {"eligible": False, "reason": "Technology not in V1"}
        
        config = self.configs[Version.V2_PRODUCTION]
        metrics = tech.metrics
        
        # Check thresholds
        checks = {
            "accuracy": metrics.get("accuracy", 0) >= config.accuracy_threshold,
            "error_rate": metrics.get("error_rate", 1) <= config.error_rate_threshold,
            "latency": metrics.get("latency_p95_ms", float("inf")) <= config.latency_p95_threshold_ms,
            "sample_size": metrics.get("sample_count", 0) >= config.min_samples_for_evaluation,
        }
        
        eligible = all(checks.values())
        
        return {
            "eligible": eligible,
            "tech_id": tech_id,
            "checks": checks,
            "metrics": metrics,
            "thresholds": {
                "accuracy": config.accuracy_threshold,
                "error_rate": config.error_rate_threshold,
                "latency_p95_ms": config.latency_p95_threshold_ms,
                "min_samples": config.min_samples_for_evaluation,
            },
        }
    
    async def promote_technology(
        self,
        tech_id: str,
        reason: str = "Passed all evaluation criteria",
    ) -> PromotionRecord:
        """Promote a technology from V1 to V2."""
        async with self._promotion_lock:
            tech = self._technologies.get(tech_id)
            if not tech:
                raise ValueError(f"Technology {tech_id} not found")
            
            if tech_id not in self._version_technologies[Version.V1_EXPERIMENTAL]:
                raise ValueError(f"Technology {tech_id} not in V1")
            
            # Record metrics before promotion
            metrics_before = tech.metrics.copy()
            
            # Move technology
            self._version_technologies[Version.V1_EXPERIMENTAL].discard(tech_id)
            self._version_technologies[Version.V2_PRODUCTION].add(tech_id)
            tech.status = "promoted"
            
            # Create promotion record
            record = PromotionRecord(
                record_id=str(uuid.uuid4()),
                from_version=Version.V1_EXPERIMENTAL,
                to_version=Version.V2_PRODUCTION,
                technologies=[tech_id],
                reason=reason,
                metrics_before=metrics_before,
            )
            
            self._promotion_history.append(record)
            
            logger.info(f"Promoted technology {tech.name} from V1 to V2")
            
            if self.event_bus:
                await self.event_bus.publish("technology_promoted", {
                    "tech_id": tech_id,
                    "name": tech.name,
                    "from_version": "v1",
                    "to_version": "v2",
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            
            return record
    
    async def degrade_technology(
        self,
        tech_id: str,
        reason: str,
        from_version: Version = Version.V2_PRODUCTION,
    ) -> PromotionRecord:
        """Degrade a technology to V3 quarantine."""
        async with self._promotion_lock:
            tech = self._technologies.get(tech_id)
            if not tech:
                raise ValueError(f"Technology {tech_id} not found")
            
            if tech_id not in self._version_technologies[from_version]:
                raise ValueError(f"Technology {tech_id} not in {from_version.value}")
            
            # Record metrics before degradation
            metrics_before = tech.metrics.copy()
            
            # Move technology
            self._version_technologies[from_version].discard(tech_id)
            self._version_technologies[Version.V3_QUARANTINE].add(tech_id)
            tech.status = "quarantined"
            
            # Create degradation record
            record = PromotionRecord(
                record_id=str(uuid.uuid4()),
                from_version=from_version,
                to_version=Version.V3_QUARANTINE,
                technologies=[tech_id],
                reason=reason,
                metrics_before=metrics_before,
            )
            
            self._promotion_history.append(record)
            
            logger.warning(f"Degraded technology {tech.name} to V3: {reason}")
            
            if self.event_bus:
                await self.event_bus.publish("technology_quarantined", {
                    "tech_id": tech_id,
                    "name": tech.name,
                    "from_version": from_version.value,
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            
            return record
    
    def re_evaluate_quarantined(
        self,
        tech_id: str,
    ) -> Dict[str, Any]:
        """Re-evaluate a quarantined technology for potential retry."""
        tech = self._technologies.get(tech_id)
        if not tech:
            return {"success": False, "reason": "Technology not found"}
        
        if tech_id not in self._version_technologies[Version.V3_QUARANTINE]:
            return {"success": False, "reason": "Technology not in V3"}
        
        # Check if enough time has passed (at least 30 days)
        config = self.configs[Version.V3_QUARANTINE]
        time_since_quarantine = datetime.now(timezone.utc) - tech.created_at
        
        if time_since_quarantine.days < 30:
            return {
                "success": False,
                "reason": f"Minimum quarantine period not met ({time_since_quarantine.days}/30 days)",
            }
        
        # Move back to V1 for re-experimentation
        self._version_technologies[Version.V3_QUARANTINE].discard(tech_id)
        self._version_technologies[Version.V1_EXPERIMENTAL].add(tech_id)
        tech.status = "re-evaluating"
        tech.metrics = {}  # Reset metrics
        
        logger.info(f"Re-evaluating quarantined technology: {tech.name}")
        
        return {
            "success": True,
            "tech_id": tech_id,
            "name": tech.name,
            "new_version": "v1",
        }
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        return {
            "versions": {
                v.value: {
                    "state": self._states[v].value,
                    "config": {
                        "name": self.configs[v].name,
                        "description": self.configs[v].description,
                    },
                    "metrics": {
                        "total_requests": self._metrics[v].total_requests,
                        "success_rate": self._metrics[v].success_rate,
                        "error_rate": self._metrics[v].error_rate,
                        "avg_latency_ms": self._metrics[v].avg_latency_ms,
                        "avg_accuracy": self._metrics[v].avg_accuracy,
                        "avg_cost": self._metrics[v].avg_cost_per_request,
                    },
                    "technology_count": len(self._version_technologies[v]),
                }
                for v in Version
            },
            "technologies": {
                tech_id: {
                    "name": tech.name,
                    "category": tech.category,
                    "status": tech.status,
                    "version": next(
                        (v.value for v in Version if tech_id in self._version_technologies[v]),
                        "unknown"
                    ),
                }
                for tech_id, tech in self._technologies.items()
            },
            "promotion_history": [
                {
                    "record_id": r.record_id,
                    "from": r.from_version.value,
                    "to": r.to_version.value,
                    "technologies": r.technologies,
                    "reason": r.reason,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in self._promotion_history[-10:]  # Last 10
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
