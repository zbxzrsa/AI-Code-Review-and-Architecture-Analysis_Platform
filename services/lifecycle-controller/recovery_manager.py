"""
V3 Recovery Manager

Manages the recovery cycle for quarantined versions (V3 → V1).
This completes the closed-loop self-iteration cycle:

    V1 (Experiment) → V2 (Production) → V3 (Quarantine) → V1 (Re-experiment)
                ↑_________________________________________________|

Key responsibilities:
- Monitor quarantined versions for recovery eligibility
- Schedule periodic re-evaluation attempts
- Track improvement over time
- Manage recovery criteria and cooldown periods
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class RecoveryStatus(str, Enum):
    """Status of a recovery attempt"""
    PENDING = "pending"           # Waiting for cooldown
    ELIGIBLE = "eligible"         # Ready for re-evaluation
    IN_PROGRESS = "in_progress"   # Currently being evaluated
    PASSED = "passed"             # Ready for V1 promotion
    FAILED = "failed"             # Needs more improvement
    ABANDONED = "abandoned"       # Too many failures, archived


@dataclass
class RecoveryConfig:
    """Configuration for recovery process"""
    # Cooldown before first recovery attempt
    initial_cooldown_hours: int = 24
    
    # Cooldown between retry attempts (exponential backoff)
    retry_cooldown_hours: int = 12
    max_cooldown_hours: int = 168  # 1 week
    
    # Maximum recovery attempts before abandonment
    max_recovery_attempts: int = 5
    
    # Gold-set pass thresholds (must be stricter for recovery)
    gold_set_accuracy_threshold: float = 0.90
    gold_set_security_threshold: float = 0.99
    gold_set_false_positive_threshold: float = 0.02
    
    # Improvement requirements
    min_improvement_per_attempt: float = 0.02  # 2% improvement required


@dataclass
class RecoveryRecord:
    """Record of a version's recovery history"""
    version_id: str
    quarantine_time: datetime
    quarantine_reason: str
    recovery_status: RecoveryStatus = RecoveryStatus.PENDING
    recovery_attempts: int = 0
    last_attempt_time: Optional[datetime] = None
    last_attempt_score: float = 0.0
    best_score: float = 0.0
    improvement_history: List[float] = field(default_factory=list)
    next_eligible_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecoveryManager:
    """
    Manages the V3 → V1 recovery cycle.
    
    This is the critical component that closes the self-evolution loop,
    allowing failed experiments to improve and re-enter the pipeline.
    """
    
    def __init__(
        self,
        evaluation_url: str = "http://evaluation-pipeline.platform-control-plane.svc:8080",
        config: Optional[RecoveryConfig] = None
    ):
        self.evaluation_url = evaluation_url
        self.config = config or RecoveryConfig()
        
        self.recovery_records: Dict[str, RecoveryRecord] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._running = False
        self._recovery_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the recovery manager"""
        self._http_client = httpx.AsyncClient(timeout=60.0)
        self._running = True
        
        logger.info("Recovery Manager started - V3→V1 cycle active")
        
        # Start recovery check loop and store task reference to prevent GC
        self._recovery_task = asyncio.create_task(self._recovery_loop())
    
    async def stop(self):
        """Stop the recovery manager"""
        self._running = False
        
        # Cancel recovery task
        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                # Expected when we cancel - swallow since we initiated the cancellation
                pass
            finally:
                self._recovery_task = None
        
        if self._http_client:
            await self._http_client.aclose()
        logger.info("Recovery Manager stopped")
    
    # ==================== Main Recovery Loop ====================
    
    async def _recovery_loop(self):
        """
        Continuous loop checking for recovery-eligible versions.
        Runs every 5 minutes to balance responsiveness and resource usage.
        """
        while self._running:
            try:
                await self._process_recovery_candidates()
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _process_recovery_candidates(self):
        """Process all versions eligible for recovery"""
        now = datetime.now(timezone.utc)
        
        for version_id, record in self.recovery_records.items():
            # Skip if not eligible yet
            if record.recovery_status == RecoveryStatus.PENDING:
                if record.next_eligible_time and now >= record.next_eligible_time:
                    record.recovery_status = RecoveryStatus.ELIGIBLE
                    logger.info(f"Version {version_id} now eligible for recovery")
            
            # Process eligible versions
            if record.recovery_status == RecoveryStatus.ELIGIBLE:
                await self._attempt_recovery(record)
    
    # ==================== Recovery Attempt ====================
    
    async def _attempt_recovery(self, record: RecoveryRecord):
        """Attempt recovery for a quarantined version"""
        version_id = record.version_id
        
        logger.info(f"Starting recovery attempt {record.recovery_attempts + 1} for {version_id}")
        record.recovery_status = RecoveryStatus.IN_PROGRESS
        record.last_attempt_time = datetime.now(timezone.utc)
        
        # Run gold-set evaluation
        result = await self._run_gold_set_evaluation(version_id)
        
        if result["success"]:
            score = result["score"]
            record.improvement_history.append(score)
            record.last_attempt_score = score
            record.best_score = max(record.best_score, score)
            record.recovery_attempts += 1
            
            # Check if passed all thresholds
            if self._check_recovery_passed(result):
                record.recovery_status = RecoveryStatus.PASSED
                logger.info(f"Version {version_id} PASSED recovery - ready for V1!")
                return True
            
            # Check for improvement
            if self._check_sufficient_improvement(record):
                logger.info(f"Version {version_id} showing improvement: {score:.2%}")
            else:
                logger.warning(f"Version {version_id} not improving: {score:.2%}")
            
            # Check for max attempts
            if record.recovery_attempts >= self.config.max_recovery_attempts:
                record.recovery_status = RecoveryStatus.ABANDONED
                logger.warning(f"Version {version_id} ABANDONED after {record.recovery_attempts} attempts")
                return False
            
            # Schedule next attempt with exponential backoff
            record.recovery_status = RecoveryStatus.PENDING
            cooldown = self._calculate_cooldown(record.recovery_attempts)
            record.next_eligible_time = datetime.now(timezone.utc) + timedelta(hours=cooldown)
            logger.info(f"Version {version_id} scheduled for retry in {cooldown}h")
            
        else:
            record.recovery_status = RecoveryStatus.FAILED
            logger.error(f"Version {version_id} recovery evaluation failed: {result.get('error')}")
        
        return False
    
    async def _run_gold_set_evaluation(self, version_id: str) -> Dict[str, Any]:
        """Run comprehensive gold-set evaluation"""
        try:
            response = await self._http_client.post(
                f"{self.evaluation_url}/evaluate/gold-set",
                json={
                    "version_id": version_id,
                    "evaluation_type": "recovery",
                    "include_categories": [
                        "security",
                        "quality", 
                        "performance",
                        "false_positive"
                    ]
                },
                timeout=120.0  # Gold-set can take time
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Gold-set evaluation failed for {version_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _check_recovery_passed(self, result: Dict[str, Any]) -> bool:
        """Check if recovery evaluation passed all thresholds"""
        metrics = result.get("metrics", {})
        
        accuracy = metrics.get("accuracy", 0)
        security = metrics.get("security_pass_rate", 0)
        false_positive = metrics.get("false_positive_rate", 1)
        
        return (
            accuracy >= self.config.gold_set_accuracy_threshold and
            security >= self.config.gold_set_security_threshold and
            false_positive <= self.config.gold_set_false_positive_threshold
        )
    
    def _check_sufficient_improvement(self, record: RecoveryRecord) -> bool:
        """Check if version is showing sufficient improvement"""
        if len(record.improvement_history) < 2:
            return True  # First attempt, no comparison
        
        current = record.improvement_history[-1]
        previous = record.improvement_history[-2]
        
        return (current - previous) >= self.config.min_improvement_per_attempt
    
    def _calculate_cooldown(self, attempts: int) -> float:
        """Calculate cooldown with exponential backoff"""
        cooldown = self.config.retry_cooldown_hours * (2 ** (attempts - 1))
        return min(cooldown, self.config.max_cooldown_hours)
    
    # ==================== External Interface ====================
    
    def register_quarantine(
        self,
        version_id: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RecoveryRecord:
        """Register a version entering quarantine (V3)"""
        now = datetime.now(timezone.utc)
        
        record = RecoveryRecord(
            version_id=version_id,
            quarantine_time=now,
            quarantine_reason=reason,
            recovery_status=RecoveryStatus.PENDING,
            next_eligible_time=now + timedelta(hours=self.config.initial_cooldown_hours),
            metadata=metadata or {}
        )
        
        self.recovery_records[version_id] = record
        
        logger.info(
            f"Registered {version_id} for recovery - eligible in {self.config.initial_cooldown_hours}h"
        )
        
        return record
    
    def get_recovery_status(self, version_id: str) -> Optional[RecoveryRecord]:
        """Get recovery status for a version"""
        return self.recovery_records.get(version_id)
    
    def get_passed_versions(self) -> List[RecoveryRecord]:
        """Get all versions that passed recovery and are ready for V1"""
        return [
            r for r in self.recovery_records.values()
            if r.recovery_status == RecoveryStatus.PASSED
        ]
    
    def mark_promoted_to_v1(self, version_id: str):
        """Mark a version as promoted back to V1"""
        if version_id in self.recovery_records:
            record = self.recovery_records[version_id]
            record.metadata["promoted_to_v1_at"] = datetime.now(timezone.utc).isoformat()
            record.metadata["total_quarantine_time"] = str(
                datetime.now(timezone.utc) - record.quarantine_time
            )
            
            # Archive but don't delete - keep for analytics
            logger.info(f"Version {version_id} promoted back to V1 after recovery")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery process"""
        records = list(self.recovery_records.values())
        
        return {
            "total_quarantined": len(records),
            "pending": sum(1 for r in records if r.recovery_status == RecoveryStatus.PENDING),
            "eligible": sum(1 for r in records if r.recovery_status == RecoveryStatus.ELIGIBLE),
            "in_progress": sum(1 for r in records if r.recovery_status == RecoveryStatus.IN_PROGRESS),
            "passed": sum(1 for r in records if r.recovery_status == RecoveryStatus.PASSED),
            "failed": sum(1 for r in records if r.recovery_status == RecoveryStatus.FAILED),
            "abandoned": sum(1 for r in records if r.recovery_status == RecoveryStatus.ABANDONED),
            "average_attempts": (
                sum(r.recovery_attempts for r in records) / len(records)
                if records else 0
            ),
            "average_best_score": (
                sum(r.best_score for r in records) / len(records)
                if records else 0
            ),
            "recovery_success_rate": (
                sum(1 for r in records if r.recovery_status == RecoveryStatus.PASSED) / len(records)
                if records else 0
            ),
        }
