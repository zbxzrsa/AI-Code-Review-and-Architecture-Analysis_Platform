"""
SelfHealing_V2 - Enhanced Recovery Manager

Production recovery with automated runbooks.
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    RESTART = "restart"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    FAILOVER = "failover"
    ROLLBACK = "rollback"
    CLEAR_CACHE = "clear_cache"
    RECONNECT = "reconnect"


class RecoveryStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RecoveryStep:
    """Single recovery step"""
    action: RecoveryAction
    handler: Callable[[], Awaitable[bool]]
    timeout_seconds: int = 60
    retry_count: int = 1
    description: str = ""


@dataclass
class Runbook:
    """Recovery runbook with ordered steps"""
    name: str
    description: str
    steps: List[RecoveryStep] = field(default_factory=list)
    cooldown_seconds: int = 300
    enabled: bool = True


@dataclass
class RecoveryExecution:
    """Record of recovery execution"""
    runbook_name: str
    service: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: RecoveryStatus = RecoveryStatus.PENDING
    steps_completed: int = 0
    total_steps: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runbook_name": self.runbook_name,
            "service": self.service,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "error": self.error,
        }


class RecoveryManager:
    """
    Production recovery manager.

    V2 Features:
    - Runbook-based recovery
    - Step sequencing
    - Cooldown periods
    - Recovery history
    """

    def __init__(self):
        self._runbooks: Dict[str, Runbook] = {}
        self._service_runbooks: Dict[str, str] = {}  # service -> runbook
        self._last_recovery: Dict[str, datetime] = {}
        self._executions: List[RecoveryExecution] = []
        self._max_executions = 100

    def register_runbook(self, runbook: Runbook):
        """Register a recovery runbook"""
        self._runbooks[runbook.name] = runbook
        logger.info(f"Registered runbook: {runbook.name}")

    def assign_runbook(self, service: str, runbook_name: str):
        """Assign runbook to service"""
        if runbook_name not in self._runbooks:
            raise ValueError(f"Unknown runbook: {runbook_name}")

        self._service_runbooks[service] = runbook_name

    async def execute_recovery(
        self,
        service: str,
        runbook_name: Optional[str] = None,
    ) -> RecoveryExecution:
        """Execute recovery for a service"""
        # Get runbook
        rb_name = runbook_name or self._service_runbooks.get(service)
        if not rb_name or rb_name not in self._runbooks:
            return self._create_skipped_execution(service, "No runbook")

        runbook = self._runbooks[rb_name]

        if not runbook.enabled:
            return self._create_skipped_execution(service, "Runbook disabled")

        # Check cooldown
        if service in self._last_recovery:
            elapsed = (datetime.now(timezone.utc) - self._last_recovery[service]).total_seconds()
            if elapsed < runbook.cooldown_seconds:
                return self._create_skipped_execution(service, "Cooldown active")

        # Create execution record
        execution = RecoveryExecution(
            runbook_name=rb_name,
            service=service,
            started_at=datetime.now(timezone.utc),
            status=RecoveryStatus.IN_PROGRESS,
            total_steps=len(runbook.steps),
        )

        self._executions.append(execution)

        logger.info(f"Starting recovery for {service} using runbook {rb_name}")

        # Execute steps
        try:
            for i, step in enumerate(runbook.steps):
                success = await self._execute_step(step)

                if success:
                    execution.steps_completed = i + 1
                else:
                    execution.status = RecoveryStatus.FAILED
                    execution.error = f"Step {i+1} ({step.action.value}) failed"
                    break

            if execution.status == RecoveryStatus.IN_PROGRESS:
                execution.status = RecoveryStatus.SUCCESS

        except Exception as e:
            execution.status = RecoveryStatus.FAILED
            execution.error = str(e)
            logger.error(f"Recovery failed for {service}: {e}")

        execution.completed_at = datetime.now(timezone.utc)
        self._last_recovery[service] = datetime.now(timezone.utc)

        # Limit history
        if len(self._executions) > self._max_executions:
            self._executions = self._executions[-self._max_executions:]

        logger.info(f"Recovery {execution.status.value} for {service}")
        return execution

    async def _execute_step(self, step: RecoveryStep) -> bool:
        """Execute a single recovery step"""
        for attempt in range(step.retry_count):
            try:
                result = await asyncio.wait_for(
                    step.handler(),
                    timeout=step.timeout_seconds,
                )

                if result:
                    logger.debug(f"Step {step.action.value} succeeded")
                    return True

            except asyncio.TimeoutError:
                logger.warning(f"Step {step.action.value} timed out (attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"Step {step.action.value} failed: {e} (attempt {attempt+1})")

            if attempt < step.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return False

    def _create_skipped_execution(self, service: str, reason: str) -> RecoveryExecution:
        """Create skipped execution record"""
        execution = RecoveryExecution(
            runbook_name="",
            service=service,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            status=RecoveryStatus.SKIPPED,
            error=reason,
        )
        self._executions.append(execution)
        return execution

    def get_history(self, service: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Get recovery history"""
        executions = self._executions

        if service:
            executions = [e for e in executions if e.service == service]

        return [e.to_dict() for e in executions[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        success = sum(1 for e in self._executions if e.status == RecoveryStatus.SUCCESS)
        failed = sum(1 for e in self._executions if e.status == RecoveryStatus.FAILED)

        return {
            "total_executions": len(self._executions),
            "successful": success,
            "failed": failed,
            "skipped": len(self._executions) - success - failed,
            "success_rate": success / max(1, success + failed),
            "runbooks": len(self._runbooks),
        }
