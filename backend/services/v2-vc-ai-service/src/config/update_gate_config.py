"""
V2 VC-AI Update Gate Configuration

Multi-stage validation pipeline for accepting innovations from V1.
"""

from typing import Dict, Any, List
from enum import Enum


class GateDecision(str, Enum):
    """Gate decision outcomes"""
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    REJECT = "REJECT"
    APPROVED = "APPROVED"
    NEEDS_FIX = "NEEDS_FIX"
    PROCEED = "PROCEED"
    ROLLBACK = "ROLLBACK"
    CONTINUE = "CONTINUE"
    PAUSE = "PAUSE"


class ValidationStage(str, Enum):
    """Validation pipeline stages"""
    V1_QUALIFICATION = "stage_1_v1_qualification"
    STAGING_DEPLOYMENT = "stage_2_staging_deployment"
    CANARY_DEPLOYMENT = "stage_3_canary_deployment"
    PROGRESSIVE_ROLLOUT = "stage_4_progressive_rollout"
    FULL_PRODUCTION = "stage_5_full_production"


# =============================================================================
# Update Validation Pipeline
# =============================================================================

UPDATE_VALIDATION_PIPELINE: Dict[str, Any] = {
    "stage_1_v1_qualification": {
        "entry_requirements": [
            "V1 experiment must have >= 1 week of data",
            "All metrics must be >= V2 baseline + 5%",
            "Zero regressions on validation test set",
            "Security audit completed and passed",
            "Cost per request must be <= V2 or < 5% increase",
        ],
        
        "validation_tasks": {
            "regression_test": {
                "dataset_size": 500_000,
                "dataset_description": "commits from production over 6 months",
                "success_criteria": "accuracy >= V2 baseline on all commits",
                "failure_action": "REJECT_AND_SEND_TO_V3",
            },
            
            "security_audit": {
                "checklist": [
                    "Input validation coverage",
                    "SQL injection prevention",
                    "XSS prevention",
                    "Privilege escalation checks",
                    "Data leakage prevention",
                    "Rate limiting correctness",
                    "Authentication verification",
                    "Authorization verification",
                ],
                "required_score": 0.95,  # >= 95%
                "third_party_review_required": True,
            },
            
            "performance_validation": {
                "latency_p99": "must be <= V2_p99_latency",
                "throughput": "must be >= V2_throughput OR cost_reduction > 20%",
                "memory_usage": "must be <= V2 + 10%",
                "cpu_usage": "must be <= V2 + 15%",
            },
            
            "compliance_check": {
                "gdpr_compliance": "verified",
                "hipaa_compliance": "verified_if_applicable",
                "soc2_alignment": "verified",
                "iso27001_alignment": "verified",
                "audit_logging": "complete",
                "data_retention": "compliant",
            },
        },
        
        "decision_matrix": {
            "all_passed": GateDecision.PASS,
            "minor_issues": GateDecision.CONDITIONAL,
            "any_critical_failure": GateDecision.REJECT,
        },
        
        "duration_hours": 24,
    },
    
    "stage_2_staging_deployment": {
        "duration_hours_min": 24,
        "duration_hours_max": 48,
        
        "environment": {
            "type": "staging_cluster",
            "specification": "identical_to_production",
            "data": "anonymized_production_data",
        },
        
        "tests": {
            "full_regression_suite": {
                "test_count": 2000,
                "pass_requirement": "all_must_pass",
            },
            "load_testing": {
                "load_multiplier": 5,  # 5x production peak
                "duration_minutes": 60,
                "success_criteria": "no_degradation",
            },
            "stress_testing": {
                "load_multiplier": 10,  # 10x production peak
                "duration_minutes": 30,
                "success_criteria": "graceful_degradation",
            },
            "chaos_engineering": {
                "scenarios": [
                    "network_partition",
                    "service_restart",
                    "database_failover",
                    "memory_pressure",
                    "cpu_spike",
                ],
                "success_criteria": "resilience_to_all_failures",
            },
            "disaster_recovery": {
                "scenarios": [
                    "full_database_restore",
                    "service_recovery",
                    "config_rollback",
                ],
                "rto_minutes": 30,  # Recovery Time Objective
                "rpo_minutes": 5,   # Recovery Point Objective
            },
        },
        
        "decision_matrix": {
            "all_tests_passed": GateDecision.APPROVED,
            "minor_failures": GateDecision.NEEDS_FIX,
            "critical_failures": GateDecision.REJECT,
        },
    },
    
    "stage_3_canary_deployment": {
        "duration_hours_min": 4,
        "duration_hours_max": 8,
        "traffic_percentage": 5,
        
        "monitoring": {
            "metrics": {
                "error_rate": {
                    "threshold": 0.002,  # must stay < 0.2%
                    "breach_action": "immediate_rollback",
                },
                "latency_p99_ms": {
                    "threshold": 600,  # must stay < 600ms
                    "breach_action": "immediate_rollback",
                },
                "cost_per_request": {
                    "threshold": "within_budget",
                    "breach_action": "alert_and_review",
                },
                "user_satisfaction": {
                    "threshold": -0.2,  # no decline > 0.2 points
                    "breach_action": "alert_and_review",
                },
            },
            "frequency": "real_time_second_granularity",
            "alerting": "instant_on_threshold_breach",
        },
        
        "comparison": {
            "shadow_mode": {
                "enabled": True,
                "description": "Run V1 and V2 in parallel on same requests",
            },
            "blind_comparison": {
                "enabled": True,
                "description": "Compare outputs without knowing which is which",
            },
            "agreement_rate_threshold": 0.98,  # must be >= 98% on critical issues
        },
        
        "decision_matrix": {
            "all_metrics_nominal": GateDecision.PROCEED,
            "any_threshold_breach": GateDecision.ROLLBACK,
        },
    },
    
    "stage_4_progressive_rollout": {
        "schedule": [
            {"hour": 1, "traffic_percentage": 5},
            {"hour": 4, "traffic_percentage": 25},
            {"hour": 12, "traffic_percentage": 50},
            {"hour": 24, "traffic_percentage": 75},
            {"hour": 48, "traffic_percentage": 100},
        ],
        
        "at_each_step": {
            "wait_duration_hours": 4,
            "monitoring_check": "all_metrics_nominal",
            "user_feedback_sampling": True,
            "decision_options": [
                GateDecision.CONTINUE,
                GateDecision.PAUSE,
                GateDecision.ROLLBACK,
            ],
        },
        
        "rollback_triggers": [
            {"metric": "error_rate", "threshold": 0.003},        # > 0.3%
            {"metric": "latency_p99_ms", "threshold": 800},
            {"metric": "cost_increase_percent", "threshold": 20},
            {"metric": "user_satisfaction_drop", "threshold": 0.5},
            {"metric": "security_issue", "threshold": "any"},
            {"metric": "data_corruption", "threshold": "any"},
            {"metric": "compliance_violation", "threshold": "any"},
        ],
        
        "rollback_procedure": {
            "duration_minutes_max": 5,
            "verification": "all_metrics_return_to_baseline",
            "post_mortem": "auto_triggered",
            "notification_channels": [
                "ops_team",
                "development_team",
                "stakeholders",
            ],
        },
    },
    
    "stage_5_full_production": {
        "duration": "ongoing",
        
        "slo_monitoring": {
            "availability": "continuous",
            "latency": "per_minute_tracking",
            "error_rate": "real_time_dashboards",
            "cost": "daily_reconciliation",
        },
        
        "regression_detection": {
            "daily_test": "compare_performance_vs_historical_baseline",
            "threshold_drop_percent": 2,  # alert if accuracy drops > 2%
            "response": "immediate_investigation_potential_rollback",
        },
        
        "user_feedback": {
            "sampling_rate": 0.001,  # 0.1% of all requests
            "feedback_channels": ["NPS", "feature_requests", "bug_reports"],
            "analysis": "weekly_sentiment_analysis",
        },
        
        "continuous_improvement": {
            "a_b_testing": True,
            "feature_flags": True,
            "gradual_rollout": True,
        },
    },
}


# =============================================================================
# Validation Metrics Thresholds
# =============================================================================

VALIDATION_THRESHOLDS: Dict[str, Any] = {
    "accuracy": {
        "minimum_for_promotion": 0.95,
        "improvement_over_baseline": 0.05,  # 5% improvement required
    },
    
    "latency_p99_ms": {
        "maximum_allowed": 500,
        "degradation_tolerance_percent": 0,  # No degradation allowed
    },
    
    "error_rate": {
        "maximum_allowed": 0.001,  # 0.1%
        "degradation_tolerance_percent": 0,
    },
    
    "throughput_rps": {
        "minimum_required": "match_v2_baseline",
        "acceptable_decrease_percent": 0,
    },
    
    "cost_per_request": {
        "maximum_increase_percent": 5,
        "preferred": "decrease_or_neutral",
    },
    
    "security_score": {
        "minimum_required": 0.95,
        "third_party_audit": "required",
    },
}


# =============================================================================
# Automated Gate Checks
# =============================================================================

AUTOMATED_GATE_CHECKS: List[Dict[str, Any]] = [
    {
        "name": "code_quality_check",
        "tool": "sonarqube",
        "criteria": {
            "coverage": ">= 80%",
            "duplications": "< 3%",
            "code_smells": "< 10 per KLOC",
            "security_hotspots": "0 unreviewed",
        },
    },
    {
        "name": "dependency_vulnerability_scan",
        "tool": "snyk",
        "criteria": {
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
            "medium_vulnerabilities": "< 5",
        },
    },
    {
        "name": "container_scan",
        "tool": "trivy",
        "criteria": {
            "critical_vulnerabilities": 0,
            "high_vulnerabilities": 0,
        },
    },
    {
        "name": "api_compatibility_check",
        "tool": "openapi_diff",
        "criteria": {
            "breaking_changes": 0,
            "deprecation_notice_required": True,
        },
    },
    {
        "name": "performance_benchmark",
        "tool": "k6",
        "criteria": {
            "p99_latency": "<= baseline",
            "throughput": ">= baseline",
            "error_rate": "< 0.1%",
        },
    },
]
