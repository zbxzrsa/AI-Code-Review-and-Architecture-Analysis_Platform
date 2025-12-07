"""
System Monitor

Real-time monitoring of networked learning system.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..config import MonitoringConfig
from .alerts import Alert, AlertManager
from .metrics import MetricsCollector, SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """System health status."""
    healthy: bool
    status: str  # "healthy", "degraded", "unhealthy"
    components: Dict[str, bool]
    last_check: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.healthy,
            "status": self.status,
            "components": self.components,
            "last_check": self.last_check.isoformat(),
        }


class SystemMonitor:
    """
    Real-time system monitoring.
    
    Features:
    - Periodic health checks
    - Metrics collection and aggregation
    - Alert evaluation
    - Dashboard data export
    """
    
    def __init__(
        self,
        config: MonitoringConfig,
        metrics_collector: Optional[MetricsCollector] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        """
        Initialize system monitor.
        
        Args:
            config: Monitoring configuration
            metrics_collector: Optional existing metrics collector
            alert_manager: Optional existing alert manager
        """
        self.config = config
        self.metrics = metrics_collector or MetricsCollector(config.metrics_interval_seconds)
        self.alerts = alert_manager or AlertManager()
        
        # Component health checkers
        self._health_checkers: Dict[str, Callable[[], bool]] = {}
        self._component_health: Dict[str, bool] = {}
        
        # Monitoring state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_health_check: Optional[datetime] = None
    
    def register_health_checker(self, name: str, checker: Callable[[], bool]):
        """
        Register a component health checker.
        
        Args:
            name: Component name
            checker: Function that returns True if component is healthy
        """
        self._health_checkers[name] = checker
    
    async def start(self):
        """Start monitoring."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitor started")
    
    async def stop(self):
        """Stop monitoring."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Run health checks
                await self._run_health_checks()
                
                # Get current metrics
                current_metrics = self.metrics.get_metrics()
                
                # Evaluate alert rules
                metrics_dict = {
                    "avg_latency_ms": current_metrics.avg_processing_latency_ms,
                    "availability": current_metrics.availability_percent,
                    "memory_percent": current_metrics.memory_usage_percent,
                    "error_rate": current_metrics.collection_errors / max(current_metrics.items_collected, 1),
                    "quality_pass_rate": current_metrics.quality_pass_rate,
                }
                
                triggered_alerts = self.alerts.evaluate_rules(metrics_dict)
                
                if triggered_alerts:
                    for alert in triggered_alerts:
                        logger.warning(f"Alert: {alert.message}")
                
                # Log metrics periodically
                if self.config.enable_real_time_monitoring:
                    self._log_metrics(current_metrics)
                
                await asyncio.sleep(self.config.metrics_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.config.metrics_interval_seconds)
    
    async def _run_health_checks(self):
        """Run all registered health checks."""
        for name, checker in self._health_checkers.items():
            try:
                healthy = checker()
                self._component_health[name] = healthy
                self.metrics.record_health_check(healthy)
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                self._component_health[name] = False
                self.metrics.record_health_check(False)
        
        self._last_health_check = datetime.now(timezone.utc)
    
    def _log_metrics(self, metrics: SystemMetrics):
        """Log metrics summary."""
        logger.info(
            f"Metrics: collected={metrics.items_collected}, "
            f"processed={metrics.items_processed}, "
            f"latency={metrics.avg_processing_latency_ms:.1f}ms, "
            f"availability={metrics.availability_percent:.2f}%"
        )
    
    def get_health(self) -> HealthStatus:
        """Get current health status."""
        # Determine overall status
        all_healthy = all(self._component_health.values()) if self._component_health else True
        any_unhealthy = any(not h for h in self._component_health.values())
        
        if all_healthy:
            status = "healthy"
        elif any_unhealthy:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return HealthStatus(
            healthy=all_healthy,
            status=status,
            components=dict(self._component_health),
            last_check=self._last_health_check or datetime.now(timezone.utc),
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        metrics = self.metrics.get_metrics()
        health = self.get_health()
        active_alerts = self.alerts.get_active_alerts()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": health.to_dict(),
            "metrics": metrics.to_dict(),
            "alerts": {
                "active_count": len(active_alerts),
                "alerts": [a.to_dict() for a in active_alerts[:10]],  # Latest 10
            },
            "sla_status": {
                "latency_ok": metrics.avg_processing_latency_ms < 500,
                "availability_ok": metrics.availability_percent >= 99.9,
                "throughput_ok": metrics.throughput_items_per_sec > 0,
            },
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        return self.metrics.get_metrics().to_prometheus()
