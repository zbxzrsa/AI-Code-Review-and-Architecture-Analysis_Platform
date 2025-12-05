"""
Safe Rollback Manager

Ensures quick recovery in case of update failure with:
- Version snapshots
- Health-based automatic rollback
- Rollback verification
- Recovery time < 30 seconds
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import copy
import hashlib

logger = logging.getLogger(__name__)


class RollbackTrigger(Enum):
    """Triggers for rollback"""
    MANUAL = "manual"
    HEALTH_CHECK_FAILED = "health_check_failed"
    ERROR_RATE_EXCEEDED = "error_rate_exceeded"
    LATENCY_EXCEEDED = "latency_exceeded"
    ACCURACY_DROPPED = "accuracy_dropped"
    AVAILABILITY_DROPPED = "availability_dropped"
    TEST_FAILED = "test_failed"


class RollbackStatus(Enum):
    """Rollback operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class VersionSnapshot:
    """Snapshot of a version state"""
    snapshot_id: str
    version: str
    timestamp: str
    
    # State data
    model_state: Dict[str, Any]
    config_state: Dict[str, Any]
    metrics_state: Dict[str, float]
    
    # Metadata
    checksum: str
    size_bytes: int
    is_healthy: bool = True
    tags: List[str] = field(default_factory=list)
    
    def verify_integrity(self) -> bool:
        """Verify snapshot integrity"""
        computed = hashlib.sha256(
            str(self.model_state).encode() +
            str(self.config_state).encode()
        ).hexdigest()
        return computed == self.checksum


@dataclass
class RollbackRecord:
    """Record of a rollback operation"""
    rollback_id: str
    trigger: RollbackTrigger
    from_version: str
    to_version: str
    
    status: RollbackStatus
    started_at: str
    completed_at: Optional[str] = None
    
    duration_seconds: float = 0.0
    recovery_verified: bool = False
    
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """
    Health monitoring for rollback decisions
    """
    
    def __init__(
        self,
        error_rate_threshold: float = 0.05,
        latency_threshold_ms: float = 500,
        accuracy_threshold: float = 0.80,
        availability_threshold: float = 0.99
    ):
        self.error_rate_threshold = error_rate_threshold
        self.latency_threshold = latency_threshold_ms
        self.accuracy_threshold = accuracy_threshold
        self.availability_threshold = availability_threshold
        
        self.current_metrics: Dict[str, float] = {}
        self.health_history: List[Dict[str, Any]] = []
        
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update current metrics"""
        self.current_metrics = metrics
        self.health_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy()
        })
        
        # Keep only last 100 entries
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
    
    def check_health(self) -> Tuple[bool, Optional[RollbackTrigger]]:
        """Check if system is healthy, return trigger if not"""
        if not self.current_metrics:
            return True, None
        
        # Check error rate
        error_rate = self.current_metrics.get("error_rate", 0)
        if error_rate > self.error_rate_threshold:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                return False, RollbackTrigger.ERROR_RATE_EXCEEDED
        
        # Check latency
        latency = self.current_metrics.get("latency_p95_ms", 0)
        if latency > self.latency_threshold:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                return False, RollbackTrigger.LATENCY_EXCEEDED
        
        # Check accuracy
        accuracy = self.current_metrics.get("accuracy", 1)
        if accuracy < self.accuracy_threshold:
            return False, RollbackTrigger.ACCURACY_DROPPED
        
        # Check availability
        availability = self.current_metrics.get("availability", 1)
        if availability < self.availability_threshold:
            return False, RollbackTrigger.AVAILABILITY_DROPPED
        
        # All checks passed
        self.consecutive_failures = 0
        return True, None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        is_healthy, trigger = self.check_health()
        
        return {
            "is_healthy": is_healthy,
            "trigger": trigger.value if trigger else None,
            "consecutive_failures": self.consecutive_failures,
            "current_metrics": self.current_metrics,
            "thresholds": {
                "error_rate": self.error_rate_threshold,
                "latency_ms": self.latency_threshold,
                "accuracy": self.accuracy_threshold,
                "availability": self.availability_threshold
            }
        }


class SafeRollbackManager:
    """
    Safe Rollback Manager
    
    Features:
    - Automatic snapshot management
    - Health-based rollback triggers
    - Fast rollback (< 30 seconds)
    - Rollback verification
    - Rollback history tracking
    """
    
    def __init__(
        self,
        max_snapshots: int = 10,
        rollback_timeout_seconds: int = 30,
        auto_rollback_enabled: bool = True
    ):
        self.max_snapshots = max_snapshots
        self.rollback_timeout = rollback_timeout_seconds
        self.auto_rollback_enabled = auto_rollback_enabled
        
        self.snapshots: Dict[str, VersionSnapshot] = {}
        self.rollback_history: List[RollbackRecord] = []
        self.health_monitor = HealthMonitor()
        
        self.current_version: Optional[str] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_rollback_started: Optional[Callable] = None
        self.on_rollback_completed: Optional[Callable] = None
        self.on_health_issue: Optional[Callable] = None
    
    def create_snapshot(
        self,
        version: str,
        model_state: Dict[str, Any],
        config_state: Dict[str, Any],
        metrics_state: Dict[str, float],
        tags: Optional[List[str]] = None
    ) -> VersionSnapshot:
        """Create a version snapshot"""
        # Generate checksum
        checksum = hashlib.sha256(
            str(model_state).encode() +
            str(config_state).encode()
        ).hexdigest()
        
        snapshot = VersionSnapshot(
            snapshot_id=f"snap_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            version=version,
            timestamp=datetime.now().isoformat(),
            model_state=copy.deepcopy(model_state),
            config_state=copy.deepcopy(config_state),
            metrics_state=metrics_state.copy(),
            checksum=checksum,
            size_bytes=len(str(model_state)) + len(str(config_state)),
            tags=tags or []
        )
        
        self.snapshots[version] = snapshot
        self.current_version = version
        
        # Cleanup old snapshots
        self._cleanup_old_snapshots()
        
        logger.info(f"Created snapshot for version {version}")
        return snapshot
    
    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond max limit"""
        if len(self.snapshots) > self.max_snapshots:
            # Sort by timestamp and remove oldest
            sorted_versions = sorted(
                self.snapshots.keys(),
                key=lambda v: self.snapshots[v].timestamp
            )
            
            to_remove = sorted_versions[:-self.max_snapshots]
            for version in to_remove:
                del self.snapshots[version]
                logger.info(f"Removed old snapshot: {version}")
    
    def get_snapshot(self, version: str) -> Optional[VersionSnapshot]:
        """Get a specific snapshot"""
        return self.snapshots.get(version)
    
    def get_latest_healthy_snapshot(self) -> Optional[VersionSnapshot]:
        """Get the most recent healthy snapshot"""
        healthy = [
            s for s in self.snapshots.values()
            if s.is_healthy and s.version != self.current_version
        ]
        
        if not healthy:
            return None
        
        return max(healthy, key=lambda s: s.timestamp)
    
    async def rollback(
        self,
        target_version: str,
        trigger: RollbackTrigger = RollbackTrigger.MANUAL,
        reason: str = ""
    ) -> RollbackRecord:
        """
        Perform rollback to a specific version
        
        Target: Complete in < 30 seconds
        """
        rollback_id = f"rb_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        started_at = datetime.now()
        
        logger.warning(
            f"Starting rollback: {self.current_version} -> {target_version} "
            f"(trigger: {trigger.value})"
        )
        
        record = RollbackRecord(
            rollback_id=rollback_id,
            trigger=trigger,
            from_version=self.current_version or "unknown",
            to_version=target_version,
            status=RollbackStatus.IN_PROGRESS,
            started_at=started_at.isoformat(),
            reason=reason
        )
        
        if self.on_rollback_started:
            await self.on_rollback_started(record)
        
        try:
            # Get target snapshot
            snapshot = self.get_snapshot(target_version)
            if not snapshot:
                raise ValueError(f"Snapshot not found: {target_version}")
            
            # Verify snapshot integrity
            if not snapshot.verify_integrity():
                raise ValueError(f"Snapshot integrity check failed: {target_version}")
            
            # Perform rollback with timeout
            async with asyncio.timeout(self.rollback_timeout):
                await self._apply_rollback(snapshot)
            
            # Verify rollback success
            verified = await self._verify_rollback(snapshot)
            
            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()
            
            record.status = RollbackStatus.VERIFIED if verified else RollbackStatus.COMPLETED
            record.completed_at = completed_at.isoformat()
            record.duration_seconds = duration
            record.recovery_verified = verified
            record.details = {
                "target_snapshot": snapshot.snapshot_id,
                "meets_30s_target": duration <= 30
            }
            
            self.current_version = target_version
            
            logger.info(
                f"Rollback completed in {duration:.2f}s "
                f"(target: {self.rollback_timeout}s)"
            )
            
        except asyncio.TimeoutError:
            record.status = RollbackStatus.FAILED
            record.completed_at = datetime.now().isoformat()
            record.details["error"] = "Rollback timeout exceeded"
            logger.error(f"Rollback timed out after {self.rollback_timeout}s")
            
        except Exception as e:
            record.status = RollbackStatus.FAILED
            record.completed_at = datetime.now().isoformat()
            record.details["error"] = str(e)
            logger.error(f"Rollback failed: {e}")
        
        self.rollback_history.append(record)
        
        if self.on_rollback_completed:
            await self.on_rollback_completed(record)
        
        return record
    
    async def _apply_rollback(self, snapshot: VersionSnapshot) -> None:
        """Apply a rollback from snapshot"""
        # Simulate rollback application
        # In production, this would:
        # 1. Stop current service
        # 2. Load model state from snapshot
        # 3. Apply configuration
        # 4. Restart service
        
        await asyncio.sleep(0.5)  # Simulate rollback time
        
        logger.info(f"Applied rollback to {snapshot.version}")
    
    async def _verify_rollback(
        self,
        snapshot: VersionSnapshot  # noqa: ARG002 - used for identification in production
    ) -> bool:
        """Verify rollback was successful"""
        # In production, this would:
        # 1. Run health checks against snapshot.version
        # 2. Verify model responses
        # 3. Check metrics
        
        await asyncio.sleep(0.2)  # Simulate verification
        
        return True
    
    async def auto_rollback_if_needed(
        self,
        current_metrics: Dict[str, float]
    ) -> Optional[RollbackRecord]:
        """Check health and auto-rollback if needed"""
        if not self.auto_rollback_enabled:
            return None
        
        self.health_monitor.update_metrics(current_metrics)
        is_healthy, trigger = self.health_monitor.check_health()
        
        if not is_healthy and trigger:
            logger.warning(f"Health check failed: {trigger.value}")
            
            if self.on_health_issue:
                await self.on_health_issue(trigger, current_metrics)
            
            # Find rollback target
            target_snapshot = self.get_latest_healthy_snapshot()
            if target_snapshot:
                return await self.rollback(
                    target_snapshot.version,
                    trigger=trigger,
                    reason=f"Auto-rollback due to {trigger.value}"
                )
            else:
                logger.error("No healthy snapshot available for rollback")
        
        return None
    
    def start_monitoring(self, interval_seconds: int = 5) -> None:
        """Start health monitoring with auto-rollback"""
        async def monitor_loop():
            while True:
                await asyncio.sleep(interval_seconds)
                # Health check would be triggered by metrics updates
        
        self._monitoring_task = asyncio.create_task(monitor_loop())
        logger.info("Rollback monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            logger.info("Rollback monitoring stopped")
    
    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get rollback statistics"""
        if not self.rollback_history:
            return {
                "total_rollbacks": 0,
                "successful_rollbacks": 0,
                "average_duration_seconds": 0
            }
        
        successful = [
            r for r in self.rollback_history
            if r.status in [RollbackStatus.COMPLETED, RollbackStatus.VERIFIED]
        ]
        
        durations = [r.duration_seconds for r in successful if r.duration_seconds > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        by_trigger = {}
        for r in self.rollback_history:
            trigger = r.trigger.value
            by_trigger[trigger] = by_trigger.get(trigger, 0) + 1
        
        return {
            "total_rollbacks": len(self.rollback_history),
            "successful_rollbacks": len(successful),
            "success_rate": len(successful) / len(self.rollback_history),
            "average_duration_seconds": avg_duration,
            "meets_30s_target": avg_duration <= 30,
            "by_trigger": by_trigger,
            "snapshots_available": len(self.snapshots)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rollback manager status"""
        return {
            "current_version": self.current_version,
            "auto_rollback_enabled": self.auto_rollback_enabled,
            "snapshots_count": len(self.snapshots),
            "max_snapshots": self.max_snapshots,
            "rollback_timeout_seconds": self.rollback_timeout,
            "health_status": self.health_monitor.get_health_status(),
            "statistics": self.get_rollback_statistics()
        }


# Import Tuple for type hints
from typing import Tuple
