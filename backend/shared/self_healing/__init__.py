"""
Self-Healing System for AI Code Review Platform

This module provides automated detection, prevention, and repair
mechanisms for common system issues.

Components:
- HealthMonitor: Real-time system health monitoring
- AutoRepair: Automated repair mechanisms
- AlertManager: Alert generation and routing
- MetricsCollector: System metrics collection
"""

from .health_monitor import HealthMonitor, HealthStatus, HealthMetrics
from .auto_repair import AutoRepair, RepairAction, RepairResult
from .alert_manager import AlertManager, Alert, AlertSeverity
from .metrics_collector import (
    MetricsCollector,
    Metric,
    MetricSource,
    MetricType,
    MetricSeries,
    AggregationType,
)
from .orchestrator import SelfHealingOrchestrator

__all__ = [
    # Health Monitoring
    "HealthMonitor",
    "HealthStatus",
    "HealthMetrics",
    # Auto Repair
    "AutoRepair",
    "RepairAction",
    "RepairResult",
    # Alert Management
    "AlertManager",
    "Alert",
    "AlertSeverity",
    # Metrics Collection
    "MetricsCollector",
    "Metric",
    "MetricSource",
    "MetricType",
    "MetricSeries",
    "AggregationType",
    # Orchestration
    "SelfHealingOrchestrator",
]

__version__ = "1.0.0"
