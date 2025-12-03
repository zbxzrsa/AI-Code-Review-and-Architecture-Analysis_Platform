"""
V2 VC-AI SLO Configuration

Strict Service Level Objective definitions for production operations.
"""

from typing import Dict, Any, List
from enum import Enum


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# =============================================================================
# SLO Definitions
# =============================================================================

SLO_DEFINITIONS: Dict[str, Any] = {
    "availability": {
        "target": 0.9999,  # 99.99%
        "measurement": "successful_responses / total_requests",
        "window": "30-day rolling",
        "budget_minutes_per_year": 52,  # 52 minutes downtime/year
        "alert_thresholds": {
            "warning": 0.9995,  # >= 99.95%
            "critical": 0.9995,  # < 99.95%
        },
    },
    
    "latency": {
        "p50": {
            "target_ms": 100,
            "alert_threshold_ms": 150,
        },
        "p99": {
            "target_ms": 500,
            "alert_threshold_ms": 600,
        },
        "p99_9": {
            "target_ms": 1000,
            "alert_threshold_ms": 1200,
        },
    },
    
    "accuracy": {
        "target": 0.98,  # >= 98%
        "measurement": "correct_analyses / total_analyses",
        "alert_threshold": 0.97,  # < 97%
    },
    
    "error_rate": {
        "target": 0.001,  # < 0.1%
        "measurement": "error_responses / total_requests",
        "alert_threshold": 0.0015,  # >= 0.15%
    },
    
    "cost_efficiency": {
        "target": "within_monthly_budget",
        "measurement": "actual_cost / projected_cost",
        "alert_threshold": 1.10,  # >= 110% of budget
    },
}


# =============================================================================
# Monitoring Setup
# =============================================================================

MONITORING_SETUP: Dict[str, Any] = {
    "metrics_collection": {
        "collection_interval_seconds": 10,
        "retention_years": 3,
        "cardinality_limit": 1_000_000,
    },
    
    "key_metrics": [
        "request_latency_seconds",       # histogram
        "request_total",                  # counter
        "errors_total",                   # counter
        "accuracy_score",                 # gauge
        "model_load_time_seconds",        # histogram
        "cache_hit_rate",                 # gauge
        "cost_per_request_dollars",       # gauge
        "slo_compliance_percentage",      # gauge
        "error_budget_remaining",         # gauge
        "failover_events_total",          # counter
        "circuit_breaker_state",          # gauge (0=closed, 1=open, 2=half-open)
    ],
    
    "dashboards": {
        "operational": {
            "displays": [
                "availability",
                "latency_p50_p99_p999",
                "error_rate",
                "current_cost_rate",
                "circuit_breaker_status",
            ],
            "refresh_seconds": 5,
            "audience": "ops_team",
        },
        "business": {
            "displays": [
                "requests_per_day",
                "avg_latency",
                "user_satisfaction",
                "cost_trend",
            ],
            "refresh_seconds": 3600,
            "audience": "management",
        },
        "slo_compliance": {
            "displays": [
                "current_slo_compliance_percent",
                "error_budget_remaining",
                "projection_for_month",
                "historical_slo_trends",
            ],
            "refresh_seconds": 5,
            "audience": "all_teams",
        },
    },
}


# =============================================================================
# Alerting Configuration
# =============================================================================

ALERTING_CONFIG: Dict[str, Any] = {
    "critical_alerts": [
        {
            "name": "availability_critical",
            "condition": "availability < 0.9995",
            "severity": AlertSeverity.CRITICAL,
            "message": "Availability below critical threshold",
        },
        {
            "name": "p99_latency_critical",
            "condition": "p99_latency > 800ms",
            "severity": AlertSeverity.CRITICAL,
            "message": "P99 latency exceeds critical threshold",
        },
        {
            "name": "error_rate_critical",
            "condition": "error_rate > 0.005",
            "severity": AlertSeverity.CRITICAL,
            "message": "Error rate exceeds critical threshold",
        },
        {
            "name": "accuracy_critical",
            "condition": "accuracy < 0.96",
            "severity": AlertSeverity.CRITICAL,
            "message": "Accuracy below critical threshold",
        },
        {
            "name": "cost_overrun",
            "condition": "cost > budget * 1.2",
            "severity": AlertSeverity.CRITICAL,
            "message": "Cost exceeds 120% of budget",
        },
    ],
    
    "warning_alerts": [
        {
            "name": "p99_latency_warning",
            "condition": "p99_latency > 600ms",
            "severity": AlertSeverity.WARNING,
            "message": "P99 latency approaching threshold",
        },
        {
            "name": "error_rate_warning",
            "condition": "error_rate > 0.001",
            "severity": AlertSeverity.WARNING,
            "message": "Error rate elevated",
        },
        {
            "name": "error_budget_low",
            "condition": "error_budget_remaining < 25%",
            "severity": AlertSeverity.WARNING,
            "message": "Error budget running low",
        },
    ],
    
    "alert_routing": {
        "immediate": {
            "channel": "pagerduty",
            "latency_seconds": 30,
        },
        "escalation_1": {
            "channel": "on_call_engineer",
            "latency_minutes": 10,
        },
        "escalation_2": {
            "channel": "platform_team",
            "latency_minutes": 30,
        },
        "escalation_3": {
            "channel": "engineering_director",
            "latency_minutes": 60,
        },
    },
}


# =============================================================================
# Error Budget Tracking
# =============================================================================

ERROR_BUDGET_CONFIG: Dict[str, Any] = {
    "calculation": {
        "window_days": 30,
        "slo_target": 0.9999,
        "formula": "1 - SLO_target = error_budget_ratio",
        # 1 - 0.9999 = 0.0001 = 0.01% error budget
    },
    
    "budget_policy": {
        "green_zone": {  # > 50% budget remaining
            "action": "normal_operations",
            "deployment_policy": "standard_canary",
        },
        "yellow_zone": {  # 25-50% budget remaining
            "action": "increased_caution",
            "deployment_policy": "extended_canary",
            "change_freeze_recommendation": False,
        },
        "red_zone": {  # < 25% budget remaining
            "action": "emergency_mode",
            "deployment_policy": "critical_only",
            "change_freeze_recommendation": True,
        },
    },
    
    "reporting": {
        "daily_report": True,
        "weekly_summary": True,
        "monthly_review": True,
        "stakeholder_notification": "on_zone_transition",
    },
}


# =============================================================================
# Rollback Configuration
# =============================================================================

ROLLBACK_CONFIG: Dict[str, Any] = {
    "triggers": [
        {
            "metric": "error_rate",
            "threshold": 0.003,  # > 0.3%
            "action": "automatic_rollback",
        },
        {
            "metric": "p99_latency_ms",
            "threshold": 800,
            "action": "automatic_rollback",
        },
        {
            "metric": "cost_per_request",
            "threshold": "budget_plus_20_percent",
            "action": "alert_and_manual_review",
        },
        {
            "metric": "user_satisfaction_drop",
            "threshold": 0.5,  # points
            "action": "automatic_rollback",
        },
        {
            "metric": "security_issue",
            "threshold": "any",
            "action": "immediate_rollback",
        },
        {
            "metric": "data_corruption",
            "threshold": "any",
            "action": "immediate_rollback",
        },
        {
            "metric": "compliance_violation",
            "threshold": "any",
            "action": "immediate_rollback",
        },
    ],
    
    "procedure": {
        "max_duration_minutes": 5,
        "verification": "all_metrics_return_to_baseline",
        "post_mortem": "auto_triggered",
        "notification": ["ops_team", "development_team", "stakeholders"],
    },
}
