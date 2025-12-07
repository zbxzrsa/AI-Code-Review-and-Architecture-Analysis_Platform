"""
自演化模块 (Self-Evolution Module)

模块功能描述:
    自动化错误检测、修复和持续改进系统。

主要功能:
    - 自动漏洞检测
    - 代码修复生成
    - 修复验证
    - 持续学习改进

主要组件:
    - BugFixerEngine: 核心漏洞检测和修复生成引擎
    - AutoFixCycle: 增强型自动修复循环（带学习功能）
    - FixVerifier: 修复验证管道

导出类:
    - BugFixerEngine, Severity, BugCategory, FixStatus
    - DetectedVulnerability, CodeFix, FixResult
    - FixVerifier, VerificationResult, VerificationPipeline
    - AutoFixCycle, FixCycleConfig, FixCyclePhase, FixStrategy

最后修改日期: 2024-12-07
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
