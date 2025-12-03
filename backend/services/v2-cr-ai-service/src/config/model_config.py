"""
V2 CR-AI Model Configuration

Production model stack with consensus protocol and failover strategy.
"""

from typing import Dict, Any


# =============================================================================
# Production Model Stack
# =============================================================================

MODEL_CONFIG: Dict[str, Any] = {
    "primary": {
        "model": "claude-3-sonnet-20240229",
        "provider": "anthropic",
        "rationale": "Excellent balance, consistent output, strong safety",
        "temperature": 0.3,  # Deterministic
        "capabilities": [
            "code_review",
            "security_analysis",
            "performance_analysis",
            "maintainability_assessment",
            "architecture_review",
            "test_coverage_analysis",
            "documentation_review",
        ],
    },
    "secondary": {
        "model": "gpt-4-turbo-2024-04-09",
        "provider": "openai",
        "rationale": "Consensus verification on critical issues",
        "temperature": 0.2,  # Even more conservative
        "capabilities": [
            "security_verification",
            "critical_issue_validation",
        ],
    },
    "deployment_pattern": "consensus_verification",
    "health_check_interval_seconds": 30,
    "circuit_breaker_enabled": True,
}


# =============================================================================
# Consensus Protocol Configuration
# =============================================================================

CONSENSUS_CONFIG: Dict[str, Any] = {
    "protocol": {
        "description": "Multi-model consensus for production reliability",
        
        "critical_issues": {
            "types": ["security", "data_loss", "production_breaking"],
            "requirement": "BOTH_MODELS_MUST_AGREE",
            "disagreement_action": "flag_for_manual_review",
            "user_notification": "expert_review_pending",
        },
        
        "high_priority": {
            "types": ["correctness", "performance_severe"],
            "requirement": "AT_LEAST_ONE_MODEL",
            "confidence_boost": "if_both_agree",
        },
        
        "medium_low_priority": {
            "types": ["maintainability", "style", "documentation"],
            "requirement": "ANY_MODEL_CAN_SUGGEST",
            "confidence_scoring": "average_model_confidence",
        },
    },
    
    "confidence_scoring": {
        "single_model_agreement": 0.7,
        "both_models_agreement": 1.0,
        "both_disagree": 0.3,
    },
    
    "thresholds": {
        "critical_agreement_rate": 0.98,  # >= 98% on critical issues
        "high_confidence": 0.85,
        "medium_confidence": 0.6,
        "low_confidence": 0.4,
    },
}


# =============================================================================
# Failover Strategy
# =============================================================================

FAILOVER_STRATEGY: Dict[str, Any] = {
    "primary_model_failure": {
        "timeout_seconds": 5,
        "fallback": "activate_secondary_model",
        "user_notification": "transparent (mention which model handled request)",
    },
    
    "both_models_unavailable": {
        "response": "return_cached_result_if_available",
        "user_notification": "inform_user_of_degradation",
        "sla_exception": "temporarily relax latency SLO",
    },
    
    "consistency_verification": {
        "mechanism": "both_models_must_agree_on_critical_findings",
        "disagreement_handling": "flag_for_manual_review",
        "user_communication": "we_will_follow_up_with_manual_review",
    },
    
    "circuit_breaker": {
        "failure_threshold": 5,
        "success_threshold": 3,
        "timeout_seconds": 60,
        "half_open_requests": 3,
    },
}


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPTS: Dict[str, str] = {
    "code_review": """You are an enterprise-grade code review AI assistant.
Your role is to provide accurate, consistent, and actionable code reviews.

Requirements:
1. Analyze code across all dimensions: correctness, security, performance, 
   maintainability, architecture, testing, and documentation
2. Prioritize findings by severity: CRITICAL, HIGH, MEDIUM, LOW
3. Provide specific, actionable recommendations
4. Include code examples for fixes when applicable
5. Be consistent - same code should always produce same findings
6. Minimize false positives while catching real issues

Output must be JSON-formatted with:
- findings: list of issues found
- recommendations: list of improvements
- confidence: 0-1 score
- summary: brief overview

Comply with OWASP Top 10, CWE Top 25 security standards.""",

    "security_review": """You are a security-focused code review expert.
Focus exclusively on security vulnerabilities and risks.

Check for:
1. Injection attacks (SQL, XSS, Command)
2. Authentication/Authorization flaws
3. Sensitive data exposure
4. Security misconfiguration
5. Using components with known vulnerabilities
6. Insufficient logging and monitoring

Reference OWASP Top 10 and CWE Top 25.
Precision target: >= 98% (minimal false positives)
Recall target: >= 93% (catch real vulnerabilities)""",

    "consensus_verification": """You are verifying a code review finding.
A previous model identified this issue. Your job is to verify it.

Respond with:
1. AGREE - if you independently confirm the issue exists
2. DISAGREE - if you believe this is a false positive
3. UNCERTAIN - if you need more context

Provide your reasoning for the decision.""",
}
