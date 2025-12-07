"""
Self-Healing System for AI Code Review Platform

This module provides automated detection, prevention, and repair
mechanisms for common system issues.

Components:
- HealthMonitor: Real-time system health monitoring
- AutoRepair: Automated repair mechanisms
- AlertManager: Alert generation and routing
- MetricsCollector: System metrics collection
- ErrorKnowledgeBase: Error pattern storage and auto-repair
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
from .error_knowledge_base import (
    ErrorKnowledgeBase,
    ErrorRecord,
    ErrorPattern,
    CodeFix,
    VerificationTest,
    RepairExecutionLog,
    ErrorSeverity,
    ErrorCategory,
    RepairStatus,
    get_knowledge_base,
)

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
    # Error Knowledge Base
    "ErrorKnowledgeBase",
    "ErrorRecord",
    "ErrorPattern",
    "CodeFix",
    "VerificationTest",
    "RepairExecutionLog",
    "ErrorSeverity",
    "ErrorCategory",
    "RepairStatus",
    "get_knowledge_base",
]

__version__ = "1.1.0"
