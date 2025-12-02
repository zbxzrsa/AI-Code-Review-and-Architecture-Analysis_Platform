"""
Review Dimension Configuration for V1 Code Review AI

Defines the multi-dimensional review framework:
- Correctness, Security, Performance
- Maintainability, Architecture, Testing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class ReviewDimension(str, Enum):
    """Review dimensions for multi-dimensional analysis"""
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    ARCHITECTURE = "architecture"
    TESTING = "testing"


class SeverityLevel(str, Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DimensionCheck:
    """A specific check within a review dimension"""
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    severity_default: SeverityLevel = SeverityLevel.MEDIUM


@dataclass
class CorrectnessConfig:
    """Configuration for correctness review dimension"""
    definition: str = "Does the code do what it's supposed to do?"
    false_positive_tolerance: float = 0.02  # < 2%
    
    checks: List[DimensionCheck] = field(default_factory=lambda: [
        DimensionCheck(
            name="logic_errors",
            description="Logic errors in conditionals and loops",
            examples=["if x > 0 and x < 0", "while True without break"],
            severity_default=SeverityLevel.HIGH,
        ),
        DimensionCheck(
            name="boundary_conditions",
            description="Boundary conditions handled correctly",
            examples=["array[len(array)]", "range(n) vs range(n+1)"],
            severity_default=SeverityLevel.HIGH,
        ),
        DimensionCheck(
            name="null_safety",
            description="Null/undefined reference safety",
            examples=["obj.method() without null check", "optional?.value"],
            severity_default=SeverityLevel.HIGH,
        ),
        DimensionCheck(
            name="type_safety",
            description="Type safety and contract compliance",
            examples=["int + string", "wrong return type"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="off_by_one",
            description="Off-by-one and fencepost errors",
            examples=["for i in range(len(arr)-1): arr[i+1]"],
            severity_default=SeverityLevel.HIGH,
        ),
    ])
    
    error_examples: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "code": "for i in range(len(arr)-1): arr[i+1]",
            "issue": "off-by-one: can access arr[len(arr)]",
            "fix": "for i in range(len(arr)): arr[i]",
        },
        {
            "code": "if user and user.name:",
            "issue": "potential AttributeError if user has no name attribute",
            "fix": "if user and hasattr(user, 'name') and user.name:",
        },
    ])


@dataclass
class SecurityConfig:
    """Configuration for security review dimension"""
    definition: str = "Are there security vulnerabilities?"
    false_negative_tolerance: float = 0.01  # < 1% - must catch real vulnerabilities
    
    checks: List[DimensionCheck] = field(default_factory=lambda: [
        DimensionCheck(
            name="sql_injection",
            description="SQL injection vulnerabilities",
            examples=["f'SELECT * FROM users WHERE id={user_id}'"],
            severity_default=SeverityLevel.CRITICAL,
        ),
        DimensionCheck(
            name="xss",
            description="Cross-site scripting (XSS) risks",
            examples=["innerHTML = userInput", "dangerouslySetInnerHTML"],
            severity_default=SeverityLevel.CRITICAL,
        ),
        DimensionCheck(
            name="auth_flaws",
            description="Authentication/authorization flaws",
            examples=["hardcoded passwords", "missing permission checks"],
            severity_default=SeverityLevel.CRITICAL,
        ),
        DimensionCheck(
            name="data_exposure",
            description="Sensitive data exposure",
            examples=["logging passwords", "exposing API keys"],
            severity_default=SeverityLevel.CRITICAL,
        ),
        DimensionCheck(
            name="crypto_misuse",
            description="Cryptography misuse",
            examples=["MD5 for passwords", "hardcoded encryption keys"],
            severity_default=SeverityLevel.HIGH,
        ),
        DimensionCheck(
            name="dependency_vuln",
            description="Dependency vulnerabilities",
            examples=["outdated packages with CVEs"],
            severity_default=SeverityLevel.HIGH,
        ),
        DimensionCheck(
            name="command_injection",
            description="Command injection vulnerabilities",
            examples=["os.system(user_input)", "subprocess with shell=True"],
            severity_default=SeverityLevel.CRITICAL,
        ),
        DimensionCheck(
            name="deserialization",
            description="Insecure deserialization",
            examples=["pickle.loads(untrusted_data)", "eval(user_input)"],
            severity_default=SeverityLevel.CRITICAL,
        ),
    ])
    
    cwe_coverage: List[str] = field(default_factory=lambda: [
        "CWE-79",   # XSS
        "CWE-89",   # SQL Injection
        "CWE-78",   # OS Command Injection
        "CWE-22",   # Path Traversal
        "CWE-352",  # CSRF
        "CWE-434",  # Unrestricted Upload
        "CWE-611",  # XXE
        "CWE-918",  # SSRF
        "CWE-502",  # Deserialization
        "CWE-798",  # Hardcoded Credentials
    ])


@dataclass
class PerformanceConfig:
    """Configuration for performance review dimension"""
    definition: str = "Is the code efficient?"
    
    checks: List[DimensionCheck] = field(default_factory=lambda: [
        DimensionCheck(
            name="algorithmic_complexity",
            description="Algorithmic complexity analysis",
            examples=["O(n²) when O(n) possible", "nested loops over large data"],
            severity_default=SeverityLevel.HIGH,
        ),
        DimensionCheck(
            name="memory_allocation",
            description="Memory allocation patterns",
            examples=["creating objects in loops", "memory leaks"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="cache_efficiency",
            description="Cache efficiency",
            examples=["missing memoization", "repeated computations"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="io_optimization",
            description="IO operations optimization",
            examples=["sync IO in async context", "unbuffered writes"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="unnecessary_copies",
            description="Unnecessary allocations or copies",
            examples=["list(list(x))", "string concatenation in loops"],
            severity_default=SeverityLevel.LOW,
        ),
        DimensionCheck(
            name="data_structures",
            description="Inefficient data structures",
            examples=["list for membership tests vs set", "dict vs dataclass"],
            severity_default=SeverityLevel.MEDIUM,
        ),
    ])
    
    complexity_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "max_nested_loops": 2,
        "max_function_calls_per_line": 3,
        "max_recursion_depth": 10,
    })


@dataclass
class MaintainabilityConfig:
    """Configuration for maintainability review dimension"""
    definition: str = "Is code easy to understand and modify?"
    
    checks: List[DimensionCheck] = field(default_factory=lambda: [
        DimensionCheck(
            name="naming_clarity",
            description="Variable/function naming clarity",
            examples=["x, y, z as variable names", "do_stuff()"],
            severity_default=SeverityLevel.LOW,
        ),
        DimensionCheck(
            name="code_complexity",
            description="Code complexity (cyclomatic, cognitive)",
            examples=["deeply nested conditions", "long switch statements"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="function_length",
            description="Function length and cohesion",
            examples=["functions > 50 lines", "doing multiple unrelated things"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="documentation",
            description="Comments and documentation quality",
            examples=["missing docstrings", "outdated comments"],
            severity_default=SeverityLevel.LOW,
        ),
        DimensionCheck(
            name="dry_violations",
            description="DRY principle violations",
            examples=["copy-pasted code blocks", "repeated logic"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="magic_numbers",
            description="Magic numbers vs constants",
            examples=["if status == 200", "sleep(86400)"],
            severity_default=SeverityLevel.LOW,
        ),
    ])
    
    metrics_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "cyclomatic_complexity": 10,
        "cognitive_complexity": 15,
        "function_length": 50,
        "class_length": 300,
        "parameter_count": 5,
    })


@dataclass
class ArchitectureConfig:
    """Configuration for architecture review dimension"""
    definition: str = "Does code follow good design patterns?"
    
    checks: List[DimensionCheck] = field(default_factory=lambda: [
        DimensionCheck(
            name="design_patterns",
            description="Design pattern application correctness",
            examples=["incorrect singleton", "over-engineered factory"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="coupling",
            description="Module coupling assessment",
            examples=["importing internal modules", "god classes"],
            severity_default=SeverityLevel.HIGH,
        ),
        DimensionCheck(
            name="cohesion",
            description="Cohesion analysis",
            examples=["unrelated methods in same class"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="solid_principles",
            description="SOLID principles compliance",
            examples=["classes with multiple responsibilities"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="dependency_injection",
            description="Dependency injection usage",
            examples=["hardcoded dependencies", "global state"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="separation_of_concerns",
            description="Separation of concerns",
            examples=["business logic in controllers", "UI in data layer"],
            severity_default=SeverityLevel.HIGH,
        ),
    ])
    
    patterns_to_detect: List[str] = field(default_factory=lambda: [
        "singleton", "factory", "observer", "strategy",
        "decorator", "adapter", "facade", "repository",
    ])


@dataclass
class TestingConfig:
    """Configuration for testing review dimension"""
    definition: str = "Is code sufficiently tested?"
    
    checks: List[DimensionCheck] = field(default_factory=lambda: [
        DimensionCheck(
            name="coverage",
            description="Code coverage percentage",
            examples=["untested branches", "missing unit tests"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="test_isolation",
            description="Test isolation and independence",
            examples=["tests depending on order", "shared state"],
            severity_default=SeverityLevel.HIGH,
        ),
        DimensionCheck(
            name="edge_cases",
            description="Edge case coverage",
            examples=["empty inputs", "boundary values"],
            severity_default=SeverityLevel.MEDIUM,
        ),
        DimensionCheck(
            name="mock_usage",
            description="Mock/stub appropriateness",
            examples=["over-mocking", "mocking implementation details"],
            severity_default=SeverityLevel.LOW,
        ),
        DimensionCheck(
            name="test_quality",
            description="Test quality and readability",
            examples=["unclear assertions", "no test descriptions"],
            severity_default=SeverityLevel.LOW,
        ),
    ])
    
    coverage_targets: Dict[str, float] = field(default_factory=lambda: {
        "unit_tests": 0.80,          # >= 80%
        "integration_tests": 0.60,   # >= 60%
        "critical_paths": 1.00,      # 100%
    })


@dataclass
class ReviewDimensionConfig:
    """Complete review dimension configuration"""
    correctness: CorrectnessConfig = field(default_factory=CorrectnessConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    maintainability: MaintainabilityConfig = field(default_factory=MaintainabilityConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    
    # Default dimensions to review
    default_dimensions: List[ReviewDimension] = field(default_factory=lambda: [
        ReviewDimension.CORRECTNESS,
        ReviewDimension.SECURITY,
        ReviewDimension.PERFORMANCE,
    ])
    
    # Dimension weights for overall scoring
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        ReviewDimension.CORRECTNESS.value: 0.25,
        ReviewDimension.SECURITY.value: 0.25,
        ReviewDimension.PERFORMANCE.value: 0.15,
        ReviewDimension.MAINTAINABILITY.value: 0.15,
        ReviewDimension.ARCHITECTURE.value: 0.10,
        ReviewDimension.TESTING.value: 0.10,
    })
