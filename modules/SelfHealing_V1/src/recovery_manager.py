"""
SelfHealing_V1 - Recovery Manager

Automatic recovery actions for unhealthy services.
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .health_monitor import HealthStatus, ServiceHealth, SystemHealth

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    RESTART = "restart"
    RECONNECT = "reconnect"
    FAILOVER = "failover"
    SCALE_UP = "scale_up"
    ALERT = "alert"


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    service_name: str
    action: RecoveryAction
    timestamp: datetime
    success: bool
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "details": self.details,
        }


class RecoveryManager:
    """
    Automatic recovery manager.

    Features:
    - Recovery action execution
    - Retry logic
    - Recovery history
    """

    def __init__(
        self,
        max_retry_attempts: int = 3,
        retry_delay: int = 10,
        cooldown_period: int = 300,
    ):
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        self.cooldown_period = cooldown_period

        self._recovery_handlers: Dict[str, Dict[RecoveryAction, Callable]] = {}
        self._recovery_history: List[RecoveryAttempt] = []
        self._last_recovery: Dict[str, datetime] = {}

    def register_recovery_handler(
        self,
        service_name: str,
        action: RecoveryAction,
        handler: Callable,
    ):
        """Register recovery handler for service"""
        if service_name not in self._recovery_handlers:
            self._recovery_handlers[service_name] = {}

        self._recovery_handlers[service_name][action] = handler
        logger.info(f"Registered {action.value} handler for {service_name}")

    async def attempt_recovery(
        self,
        service: ServiceHealth,
        action: Optional[RecoveryAction] = None,
    ) -> bool:
        """Attempt to recover unhealthy service"""
        service_name = service.service_name

        # Check cooldown
        if service_name in self._last_recovery:
            elapsed = (datetime.now(timezone.utc) - self._last_recovery[service_name]).total_seconds()
            if elapsed < self.cooldown_period:
                logger.warning(f"Recovery cooldown active for {service_name}")
                return False

        # Determine action
        if action is None:
            action = self._select_action(service)

        # Get handler
        handlers = self._recovery_handlers.get(service_name, {})
        handler = handlers.get(action)

        if handler is None:
            logger.warning(f"No handler for {action.value} on {service_name}")
            return False

        # Execute recovery
        success = False
        for attempt in range(self.max_retry_attempts):
            try:
                logger.info(f"Recovery attempt {attempt + 1}/{self.max_retry_attempts} for {service_name}")

                result = await handler()
                success = bool(result)

                if success:
                    logger.info(f"Recovery successful for {service_name}")
                    break

                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Recovery error: {e}")
                await asyncio.sleep(self.retry_delay)

        # Record attempt
        attempt_record = RecoveryAttempt(
            service_name=service_name,
            action=action,
            timestamp=datetime.now(timezone.utc),
            success=success,
            details=f"Completed after {attempt + 1} attempts",
        )

        self._recovery_history.append(attempt_record)
        self._last_recovery[service_name] = datetime.now(timezone.utc)

        return success

    def _select_action(self, service: ServiceHealth) -> RecoveryAction:
        """Select appropriate recovery action"""
        if service.consecutive_failures >= 5:
            return RecoveryAction.FAILOVER
        elif service.consecutive_failures >= 3:
            return RecoveryAction.RESTART
        else:
            return RecoveryAction.RECONNECT

    async def recover_system(self, health: SystemHealth) -> Dict[str, bool]:
        """Attempt recovery for all unhealthy services"""
        results = {}

        for service_name, service in health.services.items():
            if service.status == HealthStatus.UNHEALTHY:
                results[service_name] = await self.attempt_recovery(service)

        return results

    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recovery history"""
        return [r.to_dict() for r in self._recovery_history[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        total = len(self._recovery_history)
        successful = sum(1 for r in self._recovery_history if r.success)

        return {
            "total_attempts": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / max(1, total),
        }
