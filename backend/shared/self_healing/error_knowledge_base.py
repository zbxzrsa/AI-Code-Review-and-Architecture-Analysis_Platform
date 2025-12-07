"""
Error Knowledge Base - Self-Healing Error Pattern Storage and Auto-Repair

This module stores comprehensive error patterns, solutions, and repair procedures
for automated error detection and resolution.

Features:
- Structured error pattern storage with identification patterns
- Complete repair solutions with code modifications
- Verification test cases for each repair
- Auto-matching and auto-repair capabilities
- Repair execution logging and verification
"""

import re
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels matching SonarQube."""
    BLOCKER = "blocker"
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories."""
    CODE_SMELL = "code_smell"
    BUG = "bug"
    VULNERABILITY = "vulnerability"
    SECURITY_HOTSPOT = "security_hotspot"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"


class RepairStatus(Enum):
    """Status of repair attempt."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    VERIFIED = "verified"
    SKIPPED = "skipped"


@dataclass
class ErrorPattern:
    """Pattern for identifying specific errors."""
    pattern_id: str
    regex_pattern: str
    file_pattern: str  # Glob pattern for matching files
    line_pattern: Optional[str] = None  # Pattern for specific line content
    context_patterns: List[str] = field(default_factory=list)  # Additional context

    def matches(self, error_message: str, file_path: str, line_content: str = "") -> bool:
        """Check if this pattern matches the given error."""
        # Check message pattern
        if not re.search(self.regex_pattern, error_message, re.IGNORECASE):
            return False

        # Check file pattern
        if self.file_pattern and not Path(file_path).match(self.file_pattern):
            return False

        # Check line pattern
        if self.line_pattern and line_content:
            if not re.search(self.line_pattern, line_content):
                return False

        return True


@dataclass
class CodeFix:
    """Code modification to fix an error."""
    file_path: str
    old_content: str
    new_content: str
    description: str
    line_number: Optional[int] = None
    is_regex: bool = False  # If True, old_content is a regex pattern


@dataclass
class VerificationTest:
    """Test case to verify a repair."""
    test_name: str
    test_type: str  # "unit", "integration", "lint", "static_analysis"
    test_command: str
    expected_result: str
    timeout_seconds: int = 60


@dataclass
class ErrorRecord:
    """Complete record of an error with solution."""
    # Identification
    error_id: str
    error_type: str
    category: ErrorCategory
    severity: ErrorSeverity

    # Description
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None

    # Environment
    trigger_conditions: List[str] = field(default_factory=list)
    reproduction_steps: List[str] = field(default_factory=list)
    system_state: Dict[str, Any] = field(default_factory=dict)
    environment_info: Dict[str, str] = field(default_factory=dict)

    # Pattern matching
    patterns: List[ErrorPattern] = field(default_factory=list)

    # Solution
    fixes: List[CodeFix] = field(default_factory=list)
    verification_tests: List[VerificationTest] = field(default_factory=list)

    # Priority and automation
    priority: int = 5  # 1-10, 1 being highest
    auto_repair_enabled: bool = True
    requires_manual_review: bool = False

    # Timestamps
    first_detected: str = ""
    last_detected: str = ""
    repair_count: int = 0

    # Repair history
    repair_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "trigger_conditions": self.trigger_conditions,
            "reproduction_steps": self.reproduction_steps,
            "system_state": self.system_state,
            "environment_info": self.environment_info,
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "regex_pattern": p.regex_pattern,
                    "file_pattern": p.file_pattern,
                    "line_pattern": p.line_pattern,
                    "context_patterns": p.context_patterns
                }
                for p in self.patterns
            ],
            "fixes": [
                {
                    "file_path": f.file_path,
                    "old_content": f.old_content,
                    "new_content": f.new_content,
                    "description": f.description,
                    "line_number": f.line_number,
                    "is_regex": f.is_regex
                }
                for f in self.fixes
            ],
            "verification_tests": [
                {
                    "test_name": t.test_name,
                    "test_type": t.test_type,
                    "test_command": t.test_command,
                    "expected_result": t.expected_result,
                    "timeout_seconds": t.timeout_seconds
                }
                for t in self.verification_tests
            ],
            "priority": self.priority,
            "auto_repair_enabled": self.auto_repair_enabled,
            "requires_manual_review": self.requires_manual_review,
            "first_detected": self.first_detected,
            "last_detected": self.last_detected,
            "repair_count": self.repair_count,
            "repair_history": self.repair_history
        }


@dataclass
class RepairExecutionLog:
    """Log of repair execution."""
    execution_id: str
    error_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: RepairStatus = RepairStatus.PENDING
    files_modified: List[str] = field(default_factory=list)
    verification_results: Dict[str, bool] = field(default_factory=dict)
    error_message: Optional[str] = None
    rollback_available: bool = False


class ErrorKnowledgeBase:
    """
    Central knowledge base for error patterns and solutions.

    Provides:
    - Error pattern storage and matching
    - Automated repair execution
    - Verification and rollback
    - Learning from new errors
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("data/error_knowledge")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Error records by ID
        self.errors: Dict[str, ErrorRecord] = {}

        # Pattern index for fast matching
        self.pattern_index: Dict[str, List[str]] = {}  # pattern_id -> error_ids

        # Repair execution logs
        self.execution_logs: List[RepairExecutionLog] = []

        # Statistics
        self.stats = {
            "total_errors_stored": 0,
            "total_repairs_executed": 0,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "patterns_matched": 0
        }

        # Load existing knowledge base
        self._load_knowledge_base()

        # Initialize with known error patterns
        self._initialize_known_patterns()

    def _load_knowledge_base(self):
        """Load existing error records from storage."""
        kb_file = self.storage_path / "knowledge_base.json"
        if kb_file.exists():
            try:
                with open(kb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for error_data in data.get("errors", []):
                        error = self._dict_to_error_record(error_data)
                        self.errors[error.error_id] = error
                    self.stats = data.get("stats", self.stats)
                logger.info(f"Loaded {len(self.errors)} error records from knowledge base")
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")

    def _save_knowledge_base(self):
        """Save error records to storage."""
        kb_file = self.storage_path / "knowledge_base.json"
        try:
            data = {
                "errors": [e.to_dict() for e in self.errors.values()],
                "stats": self.stats,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug("Knowledge base saved")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def _dict_to_error_record(self, data: Dict) -> ErrorRecord:
        """Convert dictionary to ErrorRecord."""
        return ErrorRecord(
            error_id=data["error_id"],
            error_type=data["error_type"],
            category=ErrorCategory(data["category"]),
            severity=ErrorSeverity(data["severity"]),
            title=data["title"],
            description=data["description"],
            file_path=data["file_path"],
            line_number=data.get("line_number"),
            trigger_conditions=data.get("trigger_conditions", []),
            reproduction_steps=data.get("reproduction_steps", []),
            system_state=data.get("system_state", {}),
            environment_info=data.get("environment_info", {}),
            patterns=[
                ErrorPattern(
                    pattern_id=p["pattern_id"],
                    regex_pattern=p["regex_pattern"],
                    file_pattern=p["file_pattern"],
                    line_pattern=p.get("line_pattern"),
                    context_patterns=p.get("context_patterns", [])
                )
                for p in data.get("patterns", [])
            ],
            fixes=[
                CodeFix(
                    file_path=f["file_path"],
                    old_content=f["old_content"],
                    new_content=f["new_content"],
                    description=f["description"],
                    line_number=f.get("line_number"),
                    is_regex=f.get("is_regex", False)
                )
                for f in data.get("fixes", [])
            ],
            verification_tests=[
                VerificationTest(
                    test_name=t["test_name"],
                    test_type=t["test_type"],
                    test_command=t["test_command"],
                    expected_result=t["expected_result"],
                    timeout_seconds=t.get("timeout_seconds", 60)
                )
                for t in data.get("verification_tests", [])
            ],
            priority=data.get("priority", 5),
            auto_repair_enabled=data.get("auto_repair_enabled", True),
            requires_manual_review=data.get("requires_manual_review", False),
            first_detected=data.get("first_detected", ""),
            last_detected=data.get("last_detected", ""),
            repair_count=data.get("repair_count", 0),
            repair_history=data.get("repair_history", [])
        )

    def _initialize_known_patterns(self):
        """Initialize with known error patterns from SonarQube analysis."""
        known_errors = self._get_known_errors()
        for error in known_errors:
            if error.error_id not in self.errors:
                self.add_error(error)
        logger.info(f"Initialized {len(known_errors)} known error patterns")

    def _get_known_errors(self) -> List[ErrorRecord]:
        """Define known error patterns and their solutions."""
        errors = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # ================================================================
        # ERROR TYPE 1: Method always returns same value (Code Smell)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_METHOD_ALWAYS_SAME_VALUE_001",
            error_type="code_smell",
            category=ErrorCategory.CODE_SMELL,
            severity=ErrorSeverity.BLOCKER,
            title="Method always returns the same value",
            description="Method returns a constant value regardless of input. This is confusing and indicates either dead code or incomplete implementation.",
            file_path="**/deployment/system.py",
            line_number=251,
            trigger_conditions=[
                "Method contains return statement with constant value",
                "No conditional logic affecting return value",
                "Static analysis detects constant return"
            ],
            reproduction_steps=[
                "Run SonarQube analysis on the codebase",
                "Check for 'Refactor this method to not always return the same value' warning",
                "Identify methods with constant returns"
            ],
            system_state={"analyzer": "sonarqube", "rule": "python:S3516"},
            environment_info={"python_version": "3.10+", "analyzer_version": "9.x"},
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_CONST_RETURN",
                    regex_pattern=r"Refactor this method to not always return the same value",
                    file_pattern="**/*.py",
                    line_pattern=r"return\s+(True|False|None|\d+|\"[^\"]*\"|\'[^\']*\')"
                )
            ],
            fixes=[
                CodeFix(
                    file_path="ai_core/foundation_model/deployment/system.py",
                    old_content='def is_healthy(self) -> bool:\n        """Check system health."""\n        return True',
                    new_content='def is_healthy(self) -> bool:\n        """Check system health based on actual metrics."""\n        try:\n            # Check actual health indicators\n            cpu_ok = self.metrics.get("cpu_usage", 0) < 90\n            memory_ok = self.metrics.get("memory_usage", 0) < 90\n            errors_ok = self.metrics.get("error_rate", 0) < 0.05\n            return cpu_ok and memory_ok and errors_ok\n        except Exception:\n            return False',
                    description="Replace constant return with actual health check logic"
                ),
                CodeFix(
                    file_path="ai_core/foundation_model/practical_deployment.py",
                    old_content='def validate_deployment(self) -> bool:\n        """Validate deployment configuration."""\n        return True',
                    new_content='def validate_deployment(self) -> bool:\n        """Validate deployment configuration."""\n        try:\n            # Validate required configurations\n            has_model = bool(self.config.get("model_path"))\n            has_endpoint = bool(self.config.get("endpoint"))\n            valid_replicas = self.config.get("replicas", 0) > 0\n            return has_model and has_endpoint and valid_replicas\n        except Exception:\n            return False',
                    description="Replace constant return with actual validation logic"
                ),
                CodeFix(
                    file_path="modules/Caching_V2/src/redis_client.py",
                    old_content='def is_connected(self) -> bool:\n        """Check if connected to Redis."""\n        return True',
                    new_content='def is_connected(self) -> bool:\n        """Check if connected to Redis."""\n        try:\n            if self._client is None:\n                return False\n            self._client.ping()\n            return True\n        except Exception:\n            return False',
                    description="Replace constant return with actual connection check"
                )
            ],
            verification_tests=[
                VerificationTest(
                    test_name="test_health_check_returns_dynamic_value",
                    test_type="unit",
                    test_command="pytest tests/ -k 'health' -v",
                    expected_result="All health check tests pass"
                ),
                VerificationTest(
                    test_name="sonarqube_code_smell_check",
                    test_type="static_analysis",
                    test_command="sonar-scanner -Dsonar.projectKey=test",
                    expected_result="No S3516 violations"
                )
            ],
            priority=2,
            auto_repair_enabled=True,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 2: Undefined exports in __all__ (Bug)
        # ================================================================
        module_exports_errors = [
            ("modules/AIOrchestration_V1/__init__.py", ["Orchestrator", "ProviderRouter", "FallbackChain"], 9),
            ("modules/AIOrchestration_V2/__init__.py", ["Orchestrator", "ProviderRouter", "FallbackChain", "LoadBalancer", "CircuitBreaker"], 9),
            ("modules/Authentication_V1/__init__.py", ["AuthManager", "SessionManager", "TokenService"], 11),
            ("modules/Authentication_V2/__init__.py", ["MFAService", "AuthManager", "SessionManager", "TokenService", "OAuthProvider"], 11),
            ("modules/Authentication_V3/__init__.py", ["AuthManager", "SessionManager"], 12),
            ("modules/Caching_V1/__init__.py", ["CacheManager", "RedisClient", "SemanticCache"], 9),
            ("modules/Caching_V2/__init__.py", ["CacheManager", "RedisClient", "SemanticCache", "CacheWarmer"], 9),
            ("modules/Monitoring_V1/__init__.py", ["MetricsCollector", "AlertManager", "DashboardService"], 9),
            ("modules/Monitoring_V2/__init__.py", ["AlertManager", "SLOTracker", "TracingService", "MetricsCollector"], 9),
            ("modules/SelfHealing_V1/__init__.py", ["HealthMonitor", "RecoveryManager", "IncidentDetector"], 11),
            ("modules/SelfHealing_V2/__init__.py", ["HealthMonitor", "RecoveryManager", "IncidentDetector", "PredictiveHealer"], 11),
            ("modules/SelfHealing_V3/__init__.py", ["HealthMonitor", "RecoveryManager"], 12),
        ]

        for file_path, undefined_exports, line_num in module_exports_errors:
            error_id = f"ERR_UNDEFINED_EXPORT_{hashlib.md5(file_path.encode()).hexdigest()[:8]}"

            errors.append(ErrorRecord(
                error_id=error_id,
                error_type="bug",
                category=ErrorCategory.BUG,
                severity=ErrorSeverity.BLOCKER,
                title=f"Undefined exports in __all__: {', '.join(undefined_exports)}",
                description=f"The __all__ list in {file_path} references names that are not defined in the module. This will cause ImportError when someone tries to import these names.",
                file_path=file_path,
                line_number=line_num,
                trigger_conditions=[
                    "__all__ list contains string names",
                    "Names in __all__ are not defined/imported in module",
                    "Module is a placeholder or stub"
                ],
                reproduction_steps=[
                    f"Try to import: from {file_path.replace('/', '.').replace('.py', '')} import {undefined_exports[0]}",
                    "Observe ImportError or AttributeError",
                    "Run static analysis to detect undefined names"
                ],
                patterns=[
                    ErrorPattern(
                        pattern_id=f"PAT_UNDEFINED_EXPORT_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                        regex_pattern=r"is not defined",
                        file_pattern=file_path,
                        line_pattern=r"__all__\s*="
                    )
                ],
                fixes=[
                    CodeFix(
                        file_path=file_path,
                        old_content=f'__all__ = ["{"\", \"".join(undefined_exports)}"]',
                        new_content=self._generate_module_fix(file_path, undefined_exports),
                        description=f"Add placeholder implementations for {', '.join(undefined_exports)}",
                        line_number=line_num
                    )
                ],
                verification_tests=[
                    VerificationTest(
                        test_name=f"test_import_{file_path.replace('/', '_').replace('.py', '')}",
                        test_type="unit",
                        test_command=f"python -c \"from {file_path.replace('/', '.').replace('.py', '').replace('modules.', '')} import *\"",
                        expected_result="No ImportError"
                    )
                ],
                priority=1,
                auto_repair_enabled=True,
                first_detected=timestamp,
                last_detected=timestamp
            ))

        # ================================================================
        # ERROR TYPE 3: Unexpected named arguments in tests (Bug)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_UNEXPECTED_KWARGS_COST_CONTROL",
            error_type="bug",
            category=ErrorCategory.BUG,
            severity=ErrorSeverity.BLOCKER,
            title="Unexpected named arguments 'tokens' and 'cost' in test",
            description="Test file uses named arguments 'tokens' and 'cost' that are not accepted by the function being called.",
            file_path="tests/foundation_model/deployment/test_cost_control.py",
            line_number=186,
            trigger_conditions=[
                "Function signature does not include 'tokens' or 'cost' parameters",
                "Test calls function with unexpected keyword arguments",
                "API signature changed but tests not updated"
            ],
            reproduction_steps=[
                "Run: pytest tests/foundation_model/deployment/test_cost_control.py",
                "Observe TypeError: unexpected keyword argument",
                "Check function signature vs test call"
            ],
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_UNEXPECTED_KWARGS",
                    regex_pattern=r"Remove this unexpected named argument",
                    file_pattern="**/test_*.py",
                    line_pattern=r"(tokens|cost)\s*="
                )
            ],
            fixes=[
                CodeFix(
                    file_path="tests/foundation_model/deployment/test_cost_control.py",
                    old_content="track_usage(tokens=100, cost=0.01)",
                    new_content="track_usage(token_count=100, cost_amount=0.01)",
                    description="Update parameter names to match function signature",
                    is_regex=True
                )
            ],
            verification_tests=[
                VerificationTest(
                    test_name="test_cost_control_runs",
                    test_type="unit",
                    test_command="pytest tests/foundation_model/deployment/test_cost_control.py -v",
                    expected_result="All tests pass"
                )
            ],
            priority=1,
            auto_repair_enabled=True,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 4: asyncio.CancelledError not re-raised (Bug)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_CANCELLED_ERROR_NOT_RERAISED",
            error_type="bug",
            category=ErrorCategory.BUG,
            severity=ErrorSeverity.MAJOR,
            title="asyncio.CancelledError not re-raised after cleanup",
            description="When catching asyncio.CancelledError, it should be re-raised after cleanup to properly propagate cancellation.",
            file_path="**/*.py",
            trigger_conditions=[
                "Code catches asyncio.CancelledError",
                "Exception is swallowed (pass or break) instead of re-raised",
                "Async function with try/except CancelledError"
            ],
            reproduction_steps=[
                "Run SonarQube analysis",
                "Check for 'Ensure that the asyncio.CancelledError exception is re-raised' warning"
            ],
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_CANCELLED_NOT_RERAISED",
                    regex_pattern=r"Ensure that the asyncio\.CancelledError exception is re-raised",
                    file_pattern="**/*.py",
                    line_pattern=r"except\s+asyncio\.CancelledError"
                )
            ],
            fixes=[
                CodeFix(
                    file_path="",
                    old_content="except asyncio.CancelledError:\n                pass",
                    new_content="except asyncio.CancelledError:\n                # Re-raise after cleanup\n                raise",
                    description="Re-raise CancelledError after cleanup",
                    is_regex=False
                )
            ],
            verification_tests=[
                VerificationTest(
                    test_name="test_cancelled_error_propagation",
                    test_type="static_analysis",
                    test_command="sonar-scanner",
                    expected_result="No CancelledError warnings"
                )
            ],
            priority=2,
            auto_repair_enabled=False,  # Requires manual review
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 5: Cognitive Complexity > 15 (Code Smell)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_COGNITIVE_COMPLEXITY",
            error_type="code_smell",
            category=ErrorCategory.CODE_SMELL,
            severity=ErrorSeverity.CRITICAL,
            title="Function cognitive complexity exceeds limit of 15",
            description="Function has high cognitive complexity making it difficult to understand and maintain. Refactor by extracting helper functions.",
            file_path="**/*.py",
            trigger_conditions=[
                "Function has deeply nested control flow",
                "Multiple conditional branches",
                "Complex boolean expressions",
                "Many loops and break/continue statements"
            ],
            reproduction_steps=[
                "Run SonarQube analysis",
                "Check for 'Refactor this function to reduce its Cognitive Complexity' warning"
            ],
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_COGNITIVE_COMPLEXITY",
                    regex_pattern=r"Refactor this function to reduce its Cognitive Complexity from \d+ to the 15 allowed",
                    file_pattern="**/*.py"
                )
            ],
            fixes=[],  # Manual refactoring required
            verification_tests=[
                VerificationTest(
                    test_name="test_cognitive_complexity",
                    test_type="static_analysis",
                    test_command="sonar-scanner",
                    expected_result="No cognitive complexity warnings"
                )
            ],
            priority=3,
            auto_repair_enabled=False,  # Requires manual refactoring
            requires_manual_review=True,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 6: Duplicate string literals (Code Smell)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_DUPLICATE_LITERALS",
            error_type="code_smell",
            category=ErrorCategory.CODE_SMELL,
            severity=ErrorSeverity.CRITICAL,
            title="Duplicate string literal should be a constant",
            description="String literal is duplicated multiple times. Extract to a constant for maintainability.",
            file_path="**/*.py",
            trigger_conditions=[
                "Same string literal appears 3+ times",
                "String is used in comparisons or assignments",
                "Not a simple punctuation or single character"
            ],
            reproduction_steps=[
                "Run SonarQube analysis",
                "Check for 'Define a constant instead of duplicating this literal' warning"
            ],
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_DUPLICATE_LITERAL",
                    regex_pattern=r"Define a constant instead of duplicating this literal",
                    file_pattern="**/*.py"
                )
            ],
            fixes=[],  # File-specific, needs analysis
            verification_tests=[
                VerificationTest(
                    test_name="test_no_duplicate_literals",
                    test_type="static_analysis",
                    test_command="sonar-scanner",
                    expected_result="No duplicate literal warnings"
                )
            ],
            priority=4,
            auto_repair_enabled=False,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 7: Empty methods without explanation (Code Smell)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_EMPTY_METHOD",
            error_type="code_smell",
            category=ErrorCategory.CODE_SMELL,
            severity=ErrorSeverity.CRITICAL,
            title="Empty method needs explanation comment",
            description="Empty method should have a comment explaining why it's empty or be implemented.",
            file_path="**/*.py",
            trigger_conditions=[
                "Method body contains only 'pass'",
                "No comment explaining the empty implementation",
                "Not a placeholder for abstract method"
            ],
            reproduction_steps=[
                "Run SonarQube analysis",
                "Check for 'Add a nested comment explaining why this method is empty' warning"
            ],
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_EMPTY_METHOD",
                    regex_pattern=r"Add a nested comment explaining why this method is empty",
                    file_pattern="**/*.py",
                    line_pattern=r"def \w+\([^)]*\):\s*pass"
                )
            ],
            fixes=[
                CodeFix(
                    file_path="",
                    old_content="pass",
                    new_content="# Intentionally empty - no-op implementation\n            pass",
                    description="Add explanation comment",
                    is_regex=False
                )
            ],
            verification_tests=[],
            priority=4,
            auto_repair_enabled=False,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 8: Synchronous file API in async function (Bug)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_SYNC_FILE_IN_ASYNC",
            error_type="bug",
            category=ErrorCategory.BUG,
            severity=ErrorSeverity.MAJOR,
            title="Synchronous open() used in async function",
            description="Using synchronous file operations in async functions blocks the event loop. Use aiofiles or run_in_executor.",
            file_path="**/*.py",
            trigger_conditions=[
                "Async function contains open() call",
                "File operations not wrapped in run_in_executor",
                "aiofiles not being used"
            ],
            reproduction_steps=[
                "Run SonarQube analysis",
                "Check for 'Use an asynchronous file API' warning"
            ],
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_SYNC_FILE_ASYNC",
                    regex_pattern=r"Use an asynchronous file API instead of synchronous open\(\)",
                    file_pattern="**/*.py",
                    line_pattern=r"open\("
                )
            ],
            fixes=[],  # Requires refactoring to use aiofiles
            verification_tests=[],
            priority=3,
            auto_repair_enabled=False,
            requires_manual_review=True,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 9: TypeScript Mock Export Not Found (Bug)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_TS_MOCK_EXPORT_NOT_FOUND",
            error_type="bug",
            category=ErrorCategory.BUG,
            severity=ErrorSeverity.CRITICAL,
            title="Module has no exported member from vi.mock",
            description="TypeScript cannot find exports defined within vi.mock(). Use namespace import with type cast to access mock helpers.",
            file_path="**/__tests__/*.ts",
            line_number=None,
            trigger_conditions=[
                "vi.mock() defines custom exports like __resetMockState",
                "Direct import of mock-only exports fails",
                "TypeScript strict mode enabled"
            ],
            reproduction_steps=[
                "Define vi.mock() with custom helper exports",
                "Import the custom exports directly",
                "TypeScript error: Module has no exported member"
            ],
            system_state={"typescript_version": "5.x", "vitest_version": "1.x"},
            environment_info={"framework": "Vitest", "language": "TypeScript"},
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_TS_MOCK_EXPORT",
                    regex_pattern=r"Module .* has no exported member",
                    file_pattern="**/__tests__/*.ts",
                    line_pattern=r"import\s*\{.*__\w+.*\}\s*from"
                ),
                ErrorPattern(
                    pattern_id="PAT_TS_MOCK_HELPER",
                    regex_pattern=r"Did you mean to use 'import __\w+ from",
                    file_pattern="**/__tests__/*.ts"
                )
            ],
            fixes=[
                CodeFix(
                    file_path="",
                    old_content='import { __resetMockState, __setMockState } from "../../store/authStore";',
                    new_content='''import * as authStoreModule from "../../store/authStore";

// Get mock helpers - these are added by vi.mock above
const { __resetMockState, __setMockState } = authStoreModule as unknown as {
  __resetMockState: () => void;
  __setMockState: (state: Record<string, unknown>) => void;
};''',
                    description="Use namespace import with type cast for mock helpers",
                    is_regex=False
                )
            ],
            verification_tests=[
                VerificationTest(
                    test_name="test_typescript_compile",
                    test_type="lint",
                    test_command="npx tsc --noEmit",
                    expected_result="No compilation errors"
                ),
                VerificationTest(
                    test_name="test_vitest_run",
                    test_type="unit",
                    test_command="npx vitest run --reporter=verbose",
                    expected_result="All tests pass"
                )
            ],
            priority=2,
            auto_repair_enabled=True,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 10: TypeScript Non-Null Assertion Forbidden (Code Smell)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_TS_NON_NULL_ASSERTION",
            error_type="code_smell",
            category=ErrorCategory.CODE_SMELL,
            severity=ErrorSeverity.MINOR,
            title="Forbidden non-null assertion",
            description="Non-null assertion (!) bypasses TypeScript's null checking. Initialize variables with default values instead.",
            file_path="**/*.ts",
            line_number=None,
            trigger_conditions=[
                "Variable declared with let and used with ! assertion",
                "Variable assigned inside async callback",
                "ESLint/TypeScript strict mode enabled"
            ],
            reproduction_steps=[
                "Declare let variable: boolean;",
                "Assign inside await act(async () => { ... })",
                "Use expect(variable!).toBe(...) with assertion"
            ],
            system_state={"eslint_rule": "@typescript-eslint/no-non-null-assertion"},
            environment_info={"framework": "Vitest", "language": "TypeScript"},
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_TS_NON_NULL",
                    regex_pattern=r"Forbidden non-null assertion",
                    file_pattern="**/*.ts",
                    line_pattern=r"\w+!"
                )
            ],
            fixes=[
                CodeFix(
                    file_path="",
                    old_content="let success: boolean;\n",
                    new_content="let success = false;\n",
                    description="Initialize with default value instead of using ! assertion",
                    is_regex=False
                ),
                CodeFix(
                    file_path="",
                    old_content="expect(success!)",
                    new_content="expect(success)",
                    description="Remove non-null assertion after proper initialization",
                    is_regex=False
                )
            ],
            verification_tests=[
                VerificationTest(
                    test_name="test_eslint_no_assertions",
                    test_type="lint",
                    test_command="npx eslint --ext .ts --rule '@typescript-eslint/no-non-null-assertion: error'",
                    expected_result="No non-null assertion warnings"
                )
            ],
            priority=3,
            auto_repair_enabled=True,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 11: TypeScript Role Type Incompatibility (Bug)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_TS_ROLE_TYPE_INCOMPATIBLE",
            error_type="bug",
            category=ErrorCategory.BUG,
            severity=ErrorSeverity.MAJOR,
            title="Role type incompatibility in mock objects",
            description="Spread operator preserves original type's literal role. Define explicit type with union roles to allow reassignment.",
            file_path="**/__tests__/*.ts",
            line_number=None,
            trigger_conditions=[
                "Mock user object has role: 'user' as const",
                "Spread into new object with different role",
                "TypeScript strict literal types enabled"
            ],
            reproduction_steps=[
                "Define mockUser with role: 'user' as const",
                "Create mockAdminUser = {...mockUser, role: 'admin' as const}",
                "Assign to state with original type - Type error occurs"
            ],
            system_state={"typescript_strict": True},
            environment_info={"framework": "Vitest", "language": "TypeScript"},
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_TS_ROLE_TYPE",
                    regex_pattern=r"Types of property 'role' are incompatible",
                    file_pattern="**/__tests__/*.ts",
                    line_pattern=r"role.*as const"
                ),
                ErrorPattern(
                    pattern_id="PAT_TS_ROLE_LITERAL",
                    regex_pattern=r"Type '\"(admin|viewer|guest)\"' is not assignable to type '\"user\"'",
                    file_pattern="**/__tests__/*.ts"
                )
            ],
            fixes=[
                CodeFix(
                    file_path="",
                    old_content='''const mockUser = {
  id: "1",
  email: "test@example.com",
  name: "Test User",
  role: "user" as const,
  avatar: null,
};

const mockAdminUser = {
  ...mockUser,
  role: "admin" as const,
  name: "Admin User",
};''',
                    new_content='''// Mock user type
type MockUser = {
  id: string;
  email: string;
  name: string;
  role: "admin" | "user" | "viewer" | "guest";
  avatar: string | null;
};

// Mock the auth store
const mockUser: MockUser = {
  id: "1",
  email: "test@example.com",
  name: "Test User",
  role: "user",
  avatar: null,
};

const mockAdminUser: MockUser = {
  id: "1",
  email: "admin@example.com",
  name: "Admin User",
  role: "admin",
  avatar: null,
};''',
                    description="Define explicit MockUser type with union role types",
                    is_regex=False
                )
            ],
            verification_tests=[
                VerificationTest(
                    test_name="test_typescript_types",
                    test_type="lint",
                    test_command="npx tsc --noEmit",
                    expected_result="No type errors"
                )
            ],
            priority=2,
            auto_repair_enabled=True,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        # ================================================================
        # ERROR TYPE 12: TypeScript Mock State Type Too Narrow (Bug)
        # ================================================================
        errors.append(ErrorRecord(
            error_id="ERR_TS_MOCK_STATE_TYPE_NARROW",
            error_type="bug",
            category=ErrorCategory.BUG,
            severity=ErrorSeverity.MAJOR,
            title="Mock state type inferred too narrowly",
            description="Variable type inferred from initial value prevents reassignment. Add explicit type annotation with union types.",
            file_path="**/__tests__/*.ts",
            line_number=None,
            trigger_conditions=[
                "let mockState = { user: mockUser, ... }",
                "mockState.user = mockAdminUser in tests",
                "Type inference locks to initial mockUser type"
            ],
            reproduction_steps=[
                "Define let mockAuthState = { user: mockUser, isAuthenticated: true }",
                "In beforeEach: mockAuthState.user = mockAdminUser",
                "Type error: mockAdminUser not assignable to type of mockUser"
            ],
            system_state={"typescript_strict": True},
            environment_info={"framework": "Vitest", "language": "TypeScript"},
            patterns=[
                ErrorPattern(
                    pattern_id="PAT_TS_STATE_NARROW",
                    regex_pattern=r"is not assignable to type '\{ id: string;.*role: \"user\"",
                    file_pattern="**/__tests__/*.ts"
                )
            ],
            fixes=[
                CodeFix(
                    file_path="",
                    old_content="let mockAuthState = {\n  user: mockUser,\n  isAuthenticated: true,\n};",
                    new_content="let mockAuthState: { user: MockUser | null; isAuthenticated: boolean } = {\n  user: mockUser,\n  isAuthenticated: true,\n};",
                    description="Add explicit type annotation allowing any MockUser",
                    is_regex=False
                )
            ],
            verification_tests=[
                VerificationTest(
                    test_name="test_mock_state_types",
                    test_type="lint",
                    test_command="npx tsc --noEmit",
                    expected_result="No type errors"
                )
            ],
            priority=2,
            auto_repair_enabled=True,
            first_detected=timestamp,
            last_detected=timestamp
        ))

        return errors

    def _generate_module_fix(self, file_path: str, exports: List[str]) -> str:
        """Generate placeholder implementations for undefined exports."""
        module_name = Path(file_path).parent.name

        implementations = []
        for name in exports:
            implementations.append(f'''
class {name}:
    """Placeholder for {name} - implement actual functionality."""

    def __init__(self):
        self._initialized = True

    def __repr__(self):
        return f"<{name} placeholder>"
''')

        return f'''"""
{module_name} Module

This module provides {', '.join(exports)}.
"""

{"".join(implementations)}

__all__ = {exports}
'''

    def add_error(self, error: ErrorRecord) -> str:
        """Add an error record to the knowledge base."""
        error.first_detected = error.first_detected or datetime.now(timezone.utc).isoformat()
        error.last_detected = datetime.now(timezone.utc).isoformat()

        self.errors[error.error_id] = error
        self.stats["total_errors_stored"] = len(self.errors)

        # Index patterns
        for pattern in error.patterns:
            if pattern.pattern_id not in self.pattern_index:
                self.pattern_index[pattern.pattern_id] = []
            if error.error_id not in self.pattern_index[pattern.pattern_id]:
                self.pattern_index[pattern.pattern_id].append(error.error_id)

        self._save_knowledge_base()
        logger.info(f"Added error record: {error.error_id}")

        return error.error_id

    def find_matching_errors(
        self,
        error_message: str,
        file_path: str,
        line_content: str = ""
    ) -> List[ErrorRecord]:
        """Find error records matching the given error."""
        matches = []

        for error in self.errors.values():
            for pattern in error.patterns:
                if pattern.matches(error_message, file_path, line_content):
                    matches.append(error)
                    self.stats["patterns_matched"] += 1
                    break

        # Sort by priority
        matches.sort(key=lambda e: e.priority)
        return matches

    async def auto_repair(
        self,
        error_id: str,
        dry_run: bool = False
    ) -> RepairExecutionLog:
        """Execute auto-repair for a specific error."""
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d%H%M%S')}_{error_id[:8]}"

        log = RepairExecutionLog(
            execution_id=execution_id,
            error_id=error_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            status=RepairStatus.IN_PROGRESS
        )

        try:
            error = self.errors.get(error_id)
            if not error:
                log.status = RepairStatus.FAILED
                log.error_message = f"Error {error_id} not found in knowledge base"
                return log

            if not error.auto_repair_enabled:
                log.status = RepairStatus.SKIPPED
                log.error_message = "Auto-repair disabled for this error"
                return log

            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Starting auto-repair for {error_id}")

            # Apply fixes
            for fix in error.fixes:
                if dry_run:
                    logger.info(f"[DRY RUN] Would apply fix to {fix.file_path}")
                else:
                    success = await self._apply_fix(fix)
                    if success:
                        log.files_modified.append(fix.file_path)
                    else:
                        log.status = RepairStatus.FAILED
                        log.error_message = f"Failed to apply fix to {fix.file_path}"
                        return log

            # Run verification tests
            if not dry_run:
                for test in error.verification_tests:
                    result = await self._run_verification(test)
                    log.verification_results[test.test_name] = result

            # Update status
            if all(log.verification_results.values()) or dry_run:
                log.status = RepairStatus.VERIFIED if not dry_run else RepairStatus.SUCCESS
                self.stats["successful_repairs"] += 1
                error.repair_count += 1
                error.repair_history.append({
                    "execution_id": execution_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": log.status.value
                })
            else:
                log.status = RepairStatus.FAILED
                log.error_message = "Verification failed"
                self.stats["failed_repairs"] += 1

            self.stats["total_repairs_executed"] += 1

        except Exception as e:
            log.status = RepairStatus.FAILED
            log.error_message = str(e)
            logger.error(f"Auto-repair failed: {e}")

        finally:
            log.completed_at = datetime.now(timezone.utc).isoformat()
            self.execution_logs.append(log)
            self._save_knowledge_base()

        return log

    async def _apply_fix(self, fix: CodeFix) -> bool:
        """Apply a code fix to a file."""
        try:
            file_path = Path(fix.file_path)

            if not file_path.exists():
                # Create file with new content
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(fix.new_content, encoding='utf-8')
                logger.info(f"Created new file: {fix.file_path}")
                return True

            content = file_path.read_text(encoding='utf-8')

            if fix.is_regex:
                new_content = re.sub(fix.old_content, fix.new_content, content)
            else:
                new_content = content.replace(fix.old_content, fix.new_content)

            if new_content != content:
                # Backup original
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                backup_path.write_text(content, encoding='utf-8')

                # Write new content
                file_path.write_text(new_content, encoding='utf-8')
                logger.info(f"Applied fix to {fix.file_path}")
                return True
            else:
                logger.warning(f"Fix pattern not found in {fix.file_path}")
                return True  # Not an error, pattern might already be fixed

        except Exception as e:
            logger.error(f"Failed to apply fix to {fix.file_path}: {e}")
            return False

    async def _run_verification(self, test: VerificationTest) -> bool:
        """Run a verification test."""
        try:
            logger.info(f"Running verification: {test.test_name}")

            process = await asyncio.create_subprocess_shell(
                test.test_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=test.timeout_seconds
                )

                return process.returncode == 0

            except asyncio.TimeoutError:
                process.kill()
                logger.warning(f"Verification {test.test_name} timed out")
                return False

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            **self.stats,
            "errors_by_severity": {
                severity.value: len([e for e in self.errors.values() if e.severity == severity])
                for severity in ErrorSeverity
            },
            "errors_by_category": {
                category.value: len([e for e in self.errors.values() if e.category == category])
                for category in ErrorCategory
            },
            "auto_repair_enabled_count": len([e for e in self.errors.values() if e.auto_repair_enabled]),
            "recent_executions": [
                {
                    "execution_id": log.execution_id,
                    "error_id": log.error_id,
                    "status": log.status.value,
                    "started_at": log.started_at
                }
                for log in self.execution_logs[-10:]
            ]
        }

    def get_all_errors(self) -> List[Dict[str, Any]]:
        """Get all error records as dictionaries."""
        return [e.to_dict() for e in self.errors.values()]

    def get_error(self, error_id: str) -> Optional[ErrorRecord]:
        """Get a specific error record."""
        return self.errors.get(error_id)


# Global knowledge base instance
_knowledge_base: Optional[ErrorKnowledgeBase] = None


def get_knowledge_base() -> ErrorKnowledgeBase:
    """Get or create the global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = ErrorKnowledgeBase()
    return _knowledge_base


__all__ = [
    "ErrorSeverity",
    "ErrorCategory",
    "RepairStatus",
    "ErrorPattern",
    "CodeFix",
    "VerificationTest",
    "ErrorRecord",
    "RepairExecutionLog",
    "ErrorKnowledgeBase",
    "get_knowledge_base",
]
