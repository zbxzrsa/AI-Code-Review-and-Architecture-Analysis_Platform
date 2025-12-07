"""
Self-Healing Orchestrator - Coordinates all self-healing components

Integrates:
- Health monitoring
- Auto-repair
- Alert management
- Metrics collection

Provides unified interface for self-healing operations.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .health_monitor import HealthMonitor, HealthStatus, HealthMetrics
from .auto_repair import AutoRepair, RepairAction, RepairResult
from .alert_manager import AlertManager, Alert, AlertSeverity
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class SelfHealingOrchestrator:
    """
    Orchestrates self-healing operations.

    Workflow:
    1. Monitor health continuously
    2. Detect issues
    3. Generate alerts
    4. Trigger repairs
    5. Verify recovery
    6. Log results
    """

    def __init__(
        self,
        enable_auto_repair: bool = True,
        dry_run: bool = False
    ):
        self.enable_auto_repair = enable_auto_repair
        self.dry_run = dry_run

        # Components
        self.health_monitor = HealthMonitor()
        self.auto_repair = AutoRepair(dry_run=dry_run)
        self.alert_manager = AlertManager()
        self.metrics_collector = MetricsCollector()

        # State
        self.is_running = False
        self.start_time: Optional[datetime] = None

        # Setup callbacks
        self._setup_callbacks()

        # Statistics
        self.stats = {
            "issues_detected": 0,
            "repairs_attempted": 0,
            "repairs_successful": 0,
            "alerts_sent": 0
        }

    def _setup_callbacks(self) -> None:
        """Setup callbacks between components."""
        # Health monitor callbacks
        self.health_monitor.on_threshold_exceeded = self._on_threshold_exceeded
        self.health_monitor.on_anomaly_detected = self._on_anomaly_detected
        self.health_monitor.on_status_change = self._on_status_change

        # Auto-repair callbacks
        self.auto_repair.on_repair_started = self._on_repair_started
        self.auto_repair.on_repair_completed = self._on_repair_completed

    async def start(self) -> None:
        """Start self-healing system."""
        logger.info("Starting self-healing orchestrator...")

        self.is_running = True
        self.start_time = datetime.now()

        # Start health monitor
        await self.health_monitor.start()

        logger.info(
            f"Self-healing orchestrator started "
            f"(auto_repair={'enabled' if self.enable_auto_repair else 'disabled'}, "
            f"dry_run={self.dry_run})"
        )

    async def stop(self) -> None:
        """Stop self-healing system."""
        logger.info("Stopping self-healing orchestrator...")

        self.is_running = False

        # Stop health monitor
        await self.health_monitor.stop()

        logger.info("Self-healing orchestrator stopped")

    async def _on_threshold_exceeded(
        self,
        check: Any,
        result: Dict[str, Any]
    ) -> None:
        """Handle threshold exceeded event."""
        self.stats["issues_detected"] += 1

        # Generate alert
        severity = (
            AlertSeverity.CRITICAL if result["status"] == HealthStatus.CRITICAL
            else AlertSeverity.WARNING
        )

        await self.alert_manager.generate_alert(
            severity=severity,
            title=f"{check.name} threshold exceeded",
            message=result["message"],
            source="health_monitor",
            labels={"check_id": check.check_id},
            annotations={"value": str(result["value"])}
        )

        self.stats["alerts_sent"] += 1

        # Trigger repair if enabled and critical
        if self.enable_auto_repair and result["status"] == HealthStatus.CRITICAL:
            if check.repair_action:
                await self._execute_repair(
                    RepairAction[check.repair_action.upper()],
                    {
                        "triggered_by": check.check_id,
                        "issue": result["message"],
                        "metrics": {"value": result["value"]}
                    }
                )

    async def _on_anomaly_detected(self, anomalies: List[Dict[str, Any]]) -> None:
        """Handle anomaly detection."""
        for anomaly in anomalies:
            await self.alert_manager.generate_alert(
                severity=AlertSeverity.WARNING,
                title=f"Anomaly detected: {anomaly['type']}",
                message=f"Current: {anomaly['current']}, Baseline: {anomaly['baseline']}",
                source="anomaly_detector",
                labels={"type": anomaly["type"]},
                annotations={"deviation": str(anomaly["deviation"])}
            )

    async def _on_status_change(
        self,
        old_status: HealthStatus,
        new_status: HealthStatus
    ) -> None:
        """Handle system status change."""
        severity = {
            HealthStatus.HEALTHY: AlertSeverity.INFO,
            HealthStatus.DEGRADED: AlertSeverity.WARNING,
            HealthStatus.UNHEALTHY: AlertSeverity.ERROR,
            HealthStatus.CRITICAL: AlertSeverity.CRITICAL
        }.get(new_status, AlertSeverity.INFO)

        await self.alert_manager.generate_alert(
            severity=severity,
            title=f"System status changed: {old_status.value} → {new_status.value}",
            message=f"System health transitioned from {old_status.value} to {new_status.value}",
            source="health_monitor",
            labels={"old_status": old_status.value, "new_status": new_status.value}
        )

    async def _on_repair_started(self, record: Any) -> None:
        """Handle repair started event."""
        logger.info(f"Repair started: {record.action.value}")

    async def _on_repair_completed(self, record: Any) -> None:
        """Handle repair completed event."""
        logger.info(
            f"Repair completed: {record.action.value} - "
            f"Result: {record.result.value}"
        )

        # Generate alert for repair result
        if record.result == RepairResult.SUCCESS:
            severity = AlertSeverity.INFO
            title = f"Repair successful: {record.action.value}"
        elif record.result == RepairResult.FAILED:
            severity = AlertSeverity.ERROR
            title = f"Repair failed: {record.action.value}"
        else:
            severity = AlertSeverity.WARNING
            title = f"Repair {record.result.value}: {record.action.value}"

        await self.alert_manager.generate_alert(
            severity=severity,
            title=title,
            message=f"Duration: {record.duration_seconds:.2f}s",
            source="auto_repair",
            labels={"action": record.action.value, "result": record.result.value}
        )

    async def _execute_repair(
        self,
        action: RepairAction,
        context: Dict[str, Any]
    ) -> None:
        """Execute repair action."""
        self.stats["repairs_attempted"] += 1

        record = await self.auto_repair.execute_repair(action, context)

        if record.success:
            self.stats["repairs_successful"] += 1

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_report = self.health_monitor.get_health_report()
        repair_stats = self.auto_repair.get_statistics()
        alert_stats = self.alert_manager.get_statistics()

        uptime = (
            (datetime.now() - self.start_time).total_seconds()
            if self.start_time else 0
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "is_running": self.is_running,
            "health": health_report,
            "repairs": repair_stats,
            "alerts": alert_stats,
            "self_healing_stats": self.stats,
            "auto_repair_enabled": self.enable_auto_repair,
            "dry_run": self.dry_run
        }
