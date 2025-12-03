"""
V2 VC-AI Model Configuration

Production model stack with consistency guarantees and failover strategy.
"""

from typing import Dict, Any


# =============================================================================
# Primary Production Stack
# =============================================================================

MODEL_CONFIG: Dict[str, Any] = {
    "primary": {
        "model": "gpt-4-turbo-2024-04-09",
        "provider": "openai",
        "rationale": "Highest reliability & maturity",
        "capabilities": [
            "commit_analysis",
            "change_type_detection",
            "impact_assessment",
            "risk_evaluation",
            "semantic_versioning"
        ],
    },
    "backup": {
        "model": "claude-3-opus-20240229",
        "provider": "anthropic",
        "rationale": "Automatic failover with excellent reliability",
        "capabilities": [
            "commit_analysis",
            "change_type_detection",
            "impact_assessment",
            "risk_evaluation"
        ],
    },
    "deployment_pattern": "active_passive",
    "health_check_interval_seconds": 30,
    "circuit_breaker_enabled": True,
}


# =============================================================================
# Consistency Guarantees
# =============================================================================

CONSISTENCY_GUARANTEES: Dict[str, Any] = {
    "deterministic_output": {
        "lock": {
            "temperature": 0.3,  # Fixed for reproducibility
            "top_p": 0.9,        # Locked
            "top_k": 40,         # Locked
            "seed": "hash(commit_id)",  # Deterministic seeding
        },
        "implication": "Same commit always gets same analysis",
        "verification": "Run 3x on same input, compare outputs, must be identical",
    },
    "version_pinning": {
        "model_version": "gpt-4-turbo-2024-04-09",
        "api_version": "2024-01-15",
        "commitment": "No breaking changes without 30-day notice",
    },
    "fallback_strategy": {
        "primary_timeout": "5 seconds",
        "failover_trigger": "timeout OR error_rate > 1%",
        "fallback_model": "claude-3-opus",
        "user_notification": "transparent_about_which_model_handled_request",
    },
}


# =============================================================================
# Failover Strategy
# =============================================================================

FAILOVER_STRATEGY: Dict[str, Any] = {
    "primary_model_failure": {
        "timeout": "5 seconds",
        "fallback": "activate_secondary_model",
        "user_notification": "transparent (mention which model handled request)",
    },
    "both_models_unavailable": {
        "response": "return_cached_result_if_available",
        "user_notification": "inform_user_of_degradation",
        "sla_exception": "temporarily relax latency SLO",
    },
    "circuit_breaker": {
        "failure_threshold": 5,
        "success_threshold": 3,
        "timeout_seconds": 60,
        "half_open_requests": 3,
    },
    "health_checks": {
        "interval_seconds": 30,
        "timeout_seconds": 5,
        "unhealthy_threshold": 3,
        "healthy_threshold": 2,
    },
}


# =============================================================================
# Model Prompts (Production-Grade)
# =============================================================================

SYSTEM_PROMPTS: Dict[str, str] = {
    "commit_analysis": """You are an enterprise-grade version control AI assistant.
Your role is to analyze commits with highest accuracy and consistency.

Requirements:
1. Always provide deterministic, reproducible analysis
2. Identify change type accurately (feat, fix, docs, style, refactor, test, chore)
3. Assess impact level precisely (LOW, MEDIUM, HIGH, CRITICAL)
4. List all affected components and services
5. Identify breaking changes and migration requirements
6. Generate rollback plans when applicable
7. Provide risk assessment (SAFE, CAUTION, RISKY)

Output must be JSON-formatted and comply with production SLOs.""",

    "version_management": """You are an enterprise version management AI.
Your responsibilities:
1. Semantic versioning recommendations
2. Release note generation
3. Changelog compilation
4. Version history analysis
5. Dependency impact assessment

All outputs must be consistent, auditable, and compliant.""",

    "impact_analysis": """You are a change impact analysis AI.
Analyze code changes to determine:
1. Affected modules and services
2. Dependency graph impacts
3. API compatibility (breaking vs non-breaking)
4. Performance implications
5. Security considerations
6. Required testing scope

Precision target: >= 98%. False positives are costly.""",
}
