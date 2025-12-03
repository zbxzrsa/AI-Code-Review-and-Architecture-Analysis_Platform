"""
V2 CR-AI Review Configuration

Comprehensive review dimensions and accuracy targets.
"""

from typing import Dict, Any, List
from enum import Enum


class ReviewDimension(str, Enum):
    """Review dimensions"""
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


class IssueSeverity(str, Enum):
    """Issue severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


# =============================================================================
# Review Dimensions Configuration
# =============================================================================

REVIEW_DIMENSIONS: Dict[str, Any] = {
    "correctness": {
        "enabled": True,
        "critical": True,
        "precision_target": 0.96,  # >= 96%
        "recall_target": 0.94,    # >= 94%
        "checks": [
            "logic_errors",
            "boundary_conditions",
            "null_safety",
            "type_correctness",
            "error_handling",
            "resource_management",
            "concurrency_issues",
            "data_validation",
        ],
        "severity_mapping": {
            "logic_errors": IssueSeverity.CRITICAL,
            "null_safety": IssueSeverity.HIGH,
            "boundary_conditions": IssueSeverity.HIGH,
            "error_handling": IssueSeverity.MEDIUM,
        },
    },
    
    "security": {
        "enabled": True,
        "critical": True,
        "precision_target": 0.98,  # >= 98% (minimal false positives)
        "recall_target": 0.93,    # >= 93% (catch vulnerabilities)
        "standards": ["OWASP_Top_10", "CWE_Top_25"],
        "checks": [
            "sql_injection",
            "xss",
            "command_injection",
            "path_traversal",
            "authentication_bypass",
            "authorization_bypass",
            "sensitive_data_exposure",
            "insecure_deserialization",
            "insufficient_logging",
            "cryptographic_failures",
            "ssrf",
            "xxe",
        ],
        "severity_mapping": {
            "sql_injection": IssueSeverity.CRITICAL,
            "command_injection": IssueSeverity.CRITICAL,
            "authentication_bypass": IssueSeverity.CRITICAL,
            "authorization_bypass": IssueSeverity.CRITICAL,
            "xss": IssueSeverity.HIGH,
            "sensitive_data_exposure": IssueSeverity.HIGH,
        },
    },
    
    "performance": {
        "enabled": True,
        "critical": False,
        "precision_target": 0.92,  # >= 92%
        "recall_target": 0.85,    # >= 85%
        "checks": [
            "algorithmic_complexity",
            "memory_efficiency",
            "database_queries",
            "network_calls",
            "caching_opportunities",
            "unnecessary_computation",
            "resource_leaks",
            "blocking_operations",
        ],
        "severity_mapping": {
            "algorithmic_complexity": IssueSeverity.HIGH,
            "resource_leaks": IssueSeverity.HIGH,
            "database_queries": IssueSeverity.MEDIUM,
        },
    },
    
    "maintainability": {
        "enabled": True,
        "critical": False,
        "precision_target": 0.90,  # >= 90%
        "recall_target": 0.88,    # >= 88%
        "checks": [
            "code_complexity",
            "naming_conventions",
            "code_duplication",
            "function_length",
            "class_cohesion",
            "code_smells",
            "magic_numbers",
            "dead_code",
        ],
        "severity_mapping": {
            "code_complexity": IssueSeverity.MEDIUM,
            "code_duplication": IssueSeverity.MEDIUM,
            "dead_code": IssueSeverity.LOW,
        },
    },
    
    "architecture": {
        "enabled": True,
        "critical": False,
        "precision_target": 0.89,  # >= 89%
        "recall_target": 0.87,    # >= 87%
        "checks": [
            "solid_principles",
            "design_patterns",
            "dependency_injection",
            "layering_violations",
            "circular_dependencies",
            "coupling",
            "cohesion",
            "api_design",
        ],
        "severity_mapping": {
            "circular_dependencies": IssueSeverity.HIGH,
            "layering_violations": IssueSeverity.MEDIUM,
            "coupling": IssueSeverity.MEDIUM,
        },
    },
    
    "testing": {
        "enabled": True,
        "critical": False,
        "precision_target": 0.88,  # >= 88%
        "recall_target": 0.85,    # >= 85%
        "checks": [
            "test_coverage",
            "test_quality",
            "edge_case_coverage",
            "mock_usage",
            "test_isolation",
            "test_naming",
            "assertion_quality",
            "flaky_tests",
        ],
        "severity_mapping": {
            "test_coverage": IssueSeverity.MEDIUM,
            "flaky_tests": IssueSeverity.HIGH,
            "test_quality": IssueSeverity.LOW,
        },
    },
    
    "documentation": {
        "enabled": True,
        "critical": False,
        "precision_target": 0.85,  # >= 85%
        "recall_target": 0.80,    # >= 80%
        "checks": [
            "api_documentation",
            "inline_comments",
            "readme_quality",
            "changelog",
            "type_hints",
            "docstrings",
            "example_code",
        ],
        "severity_mapping": {
            "api_documentation": IssueSeverity.MEDIUM,
            "type_hints": IssueSeverity.LOW,
            "inline_comments": IssueSeverity.INFO,
        },
    },
}


# =============================================================================
# Review Configuration
# =============================================================================

REVIEW_CONFIG: Dict[str, Any] = {
    "output_structure": {
        "high_confidence_findings": {
            "position": "first",
            "description": "Issues with high confidence score",
        },
        "medium_confidence": {
            "position": "second",
            "description": "Issues with medium confidence, shown with caveats",
        },
        "low_confidence": {
            "position": "third",
            "description": "Optional insights for consideration",
        },
        "manual_review_needed": {
            "position": "flagged_separately",
            "description": "Issues requiring human verification",
        },
    },
    
    "response_format": {
        "include_line_numbers": True,
        "include_code_snippets": True,
        "include_fix_suggestions": True,
        "include_confidence_scores": True,
        "include_references": True,
        "max_findings_per_file": 50,
    },
    
    "language_support": [
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "rust",
        "c",
        "cpp",
        "csharp",
        "ruby",
        "php",
        "swift",
        "kotlin",
    ],
    
    "file_size_limits": {
        "max_file_size_kb": 500,
        "max_total_size_kb": 5000,
        "max_files_per_review": 100,
    },
}


# =============================================================================
# Production Guarantees
# =============================================================================

PRODUCTION_GUARANTEES: Dict[str, Any] = {
    "accuracy_guarantee": {
        "claim": "Each finding is verified by at least one production-grade LLM",
        "verification": "Consensus protocol ensures quality",
        "false_positive_rate": 0.02,  # <= 2%
        "false_negative_rate": 0.05,  # <= 5%
    },
    
    "consistency_guarantee": {
        "claim": "Same code gets same feedback every time",
        "mechanism": "Locked model, locked temperature, deterministic seeding",
        "verification": "Review same code 3x, outputs must be identical",
    },
    
    "compliance_guarantee": {
        "claim": "All reviews meet regulatory standards",
        "standards": ["SOC2", "GDPR", "HIPAA", "ISO27001"],
        "audit_trail": "complete_logging_and_traceability",
    },
    
    "performance_guarantee": {
        "claim": "Reviews complete within SLA",
        "sla_latency_p99_ms": 500,
        "availability": 0.9999,
        "uptime_guarantee": "backed_by_SLA_credits",
    },
    
    "security_guarantee": {
        "claim": "Code is never stored or used for training",
        "mechanism": "ephemeral_processing_only",
        "verification": "no_code_persisted_to_disk",
    },
}


# =============================================================================
# CI/CD Integration
# =============================================================================

CICD_INTEGRATION: Dict[str, Any] = {
    "github": {
        "setup": "GitHub App + OAuth",
        "triggers": ["pull_request_opened", "pull_request_updated"],
        "features": {
            "auto_review": "Automatically review each PR",
            "inline_comments": "Leave comments on specific lines",
            "review_blocking": "Make review status required check",
            "auto_dismiss": "Dismiss review on new commits",
        },
        "output_format": "GitHub Review API format",
    },
    
    "gitlab": {
        "setup": "GitLab OAuth app",
        "triggers": ["merge_request_created", "merge_request_updated"],
        "features": {
            "auto_review": "Automatic review",
            "inline_comments": "Line-level comments",
            "merge_blocking": "Block merge on critical issues",
        },
        "output_format": "GitLab MR Discussions API",
    },
    
    "bitbucket": {
        "setup": "Bitbucket OAuth 2.0",
        "triggers": ["pullrequest:created", "pullrequest:updated"],
        "features": {
            "auto_review": "Automatic review",
            "inline_comments": "File-level comments",
        },
        "output_format": "Bitbucket PR Comments API",
    },
    
    "azure_devops": {
        "setup": "Azure DevOps Service Connection",
        "triggers": ["pull_request.created", "pull_request.updated"],
        "features": {
            "auto_review": "Pipeline-triggered review",
            "inline_comments": "Thread comments",
            "policy_enforcement": "Branch policy integration",
        },
        "output_format": "Azure DevOps Pull Request Threads API",
    },
}
