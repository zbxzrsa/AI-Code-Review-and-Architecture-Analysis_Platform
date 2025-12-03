"""
Self-Evolution Module

Automated bug detection, fixing, and continuous improvement system.

Components:
- BugFixerEngine: Core vulnerability detection and fix generation
- AutoFixCycle: Enhanced automated fix cycle with learning
- FixVerifier: Fix verification pipeline
"""

from .bug_fixer import (
    BugFixerEngine,
    Severity,
    BugCategory,
    FixStatus,
    DetectedVulnerability,
    CodeFix,
    FixResult,
    VulnerabilityPattern,
    VULNERABILITY_PATTERNS,
    create_bug_fixer,
)

from .fix_verifier import (
    FixVerifier,
    VerificationResult,
    VerificationPipeline,
    TestResult,
    create_test_runner,
)

from .auto_fix_cycle import (
    AutoFixCycle,
    FixCycleConfig,
    FixCyclePhase,
    FixStrategy,
    create_auto_fix_cycle,
)

__all__ = [
    # Bug Fixer
    "BugFixerEngine",
    "Severity",
    "BugCategory",
    "FixStatus",
    "DetectedVulnerability",
    "CodeFix",
    "FixResult",
    "VulnerabilityPattern",
    "VULNERABILITY_PATTERNS",
    "create_bug_fixer",
    # Fix Verifier
    "FixVerifier",
    "VerificationResult",
    "VerificationPipeline",
    "TestResult",
    "create_test_runner",
    # Auto Fix Cycle
    "AutoFixCycle",
    "FixCycleConfig",
    "FixCyclePhase",
    "FixStrategy",
    "create_auto_fix_cycle",
]
