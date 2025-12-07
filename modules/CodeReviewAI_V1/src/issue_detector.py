"""
CodeReviewAI_V1 - Issue Detector

Specialized issue detection for different dimensions:
- Security vulnerabilities
- Performance bottlenecks
- Code quality issues
- Maintainability concerns
"""

import re
import logging
from typing import List, Dict, Optional, Any, Pattern
from dataclasses import dataclass

from .models import Finding, Dimension, Severity

logger = logging.getLogger(__name__)


@dataclass
class DetectionRule:
    """Rule for detecting specific issues"""
    rule_id: str
    dimension: Dimension
    severity: Severity
    pattern: Pattern
    message: str
    suggestion: str
    explanation: str
    cwe_id: Optional[str] = None
    enabled: bool = True


class IssueDetector:
    """
    Pattern-based and semantic issue detection.

    Detects security vulnerabilities, performance issues,
    code quality problems, and maintainability concerns.
    """

    def __init__(self):
        """Initialize detector with default rules"""
        self.rules: Dict[str, DetectionRule] = {}
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default detection rules"""

        # Security rules
        self.add_rule(DetectionRule(
            rule_id="SEC001",
            dimension=Dimension.SECURITY,
            severity=Severity.CRITICAL,
            pattern=re.compile(r'\beval\s*\(', re.IGNORECASE),
            message="Use of eval() detected",
            suggestion="Avoid using eval(). Use ast.literal_eval() for safe parsing or find alternative approaches.",
            explanation="eval() executes arbitrary code and can lead to code injection attacks.",
            cwe_id="CWE-95",
        ))

        self.add_rule(DetectionRule(
            rule_id="SEC002",
            dimension=Dimension.SECURITY,
            severity=Severity.CRITICAL,
            pattern=re.compile(r'execute\s*\(\s*[f"\'].*\{.*\}', re.IGNORECASE),
            message="Potential SQL injection vulnerability",
            suggestion="Use parameterized queries with placeholders (?, %s) instead of string formatting.",
            explanation="String interpolation in SQL queries allows attackers to inject malicious SQL.",
            cwe_id="CWE-89",
        ))

        self.add_rule(DetectionRule(
            rule_id="SEC003",
            dimension=Dimension.SECURITY,
            severity=Severity.HIGH,
            pattern=re.compile(r'(password|secret|api_key|apikey|token)\s*=\s*["\'][^"\']+["\']', re.IGNORECASE),
            message="Hardcoded credential detected",
            suggestion="Use environment variables or a secrets manager like HashiCorp Vault.",
            explanation="Hardcoded credentials can be exposed in version control or compiled binaries.",
            cwe_id="CWE-798",
        ))

        self.add_rule(DetectionRule(
            rule_id="SEC004",
            dimension=Dimension.SECURITY,
            severity=Severity.HIGH,
            pattern=re.compile(r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True', re.IGNORECASE),
            message="Shell injection risk with shell=True",
            suggestion="Use shell=False and pass command as a list of arguments.",
            explanation="shell=True allows shell injection through user-controlled input.",
            cwe_id="CWE-78",
        ))

        self.add_rule(DetectionRule(
            rule_id="SEC005",
            dimension=Dimension.SECURITY,
            severity=Severity.MEDIUM,
            pattern=re.compile(r'pickle\.(load|loads)\s*\(', re.IGNORECASE),
            message="Unsafe deserialization with pickle",
            suggestion="Use safer serialization formats like JSON, or validate pickle source.",
            explanation="Pickle can execute arbitrary code during deserialization.",
            cwe_id="CWE-502",
        ))

        # Performance rules
        self.add_rule(DetectionRule(
            rule_id="PERF001",
            dimension=Dimension.PERFORMANCE,
            severity=Severity.MEDIUM,
            pattern=re.compile(r'for\s+\w+\s+in\s+.*:\s*\n\s+for\s+\w+\s+in', re.MULTILINE),
            message="Nested loop detected",
            suggestion="Consider using list comprehensions, vectorization, or algorithmic optimization.",
            explanation="Nested loops can lead to O(n²) or worse time complexity.",
        ))

        self.add_rule(DetectionRule(
            rule_id="PERF002",
            dimension=Dimension.PERFORMANCE,
            severity=Severity.LOW,
            pattern=re.compile(r'\+\s*=\s*["\'][^"\']*["\']', re.IGNORECASE),
            message="String concatenation in loop may be inefficient",
            suggestion="Use join() or StringIO for building strings.",
            explanation="Repeated string concatenation creates new string objects.",
        ))

        self.add_rule(DetectionRule(
            rule_id="PERF003",
            dimension=Dimension.PERFORMANCE,
            severity=Severity.MEDIUM,
            pattern=re.compile(r'time\.sleep\s*\(\s*\d+\s*\)', re.IGNORECASE),
            message="Blocking sleep call detected",
            suggestion="Use asyncio.sleep() in async code or consider event-driven approaches.",
            explanation="Blocking sleep wastes resources and blocks the thread.",
        ))

        # Maintainability rules
        self.add_rule(DetectionRule(
            rule_id="MAINT001",
            dimension=Dimension.MAINTAINABILITY,
            severity=Severity.LOW,
            pattern=re.compile(r'^def\s+\w+\s*\([^)]*\)\s*:\s*\n(?!\s*["\'])'),
            message="Function missing docstring",
            suggestion="Add a docstring explaining function purpose, parameters, and return value.",
            explanation="Docstrings improve code readability and enable auto-documentation.",
        ))

        self.add_rule(DetectionRule(
            rule_id="MAINT002",
            dimension=Dimension.MAINTAINABILITY,
            severity=Severity.MEDIUM,
            pattern=re.compile(r'#\s*TODO|#\s*FIXME|#\s*HACK|#\s*XXX', re.IGNORECASE),
            message="TODO/FIXME comment found",
            suggestion="Address the TODO item or create a tracked issue.",
            explanation="TODO comments indicate incomplete or problematic code.",
        ))

        self.add_rule(DetectionRule(
            rule_id="MAINT003",
            dimension=Dimension.MAINTAINABILITY,
            severity=Severity.HIGH,
            pattern=re.compile(r'except\s*:\s*\n\s*(pass|\.\.\.)', re.MULTILINE),
            message="Bare except with pass suppresses all errors",
            suggestion="Catch specific exceptions and handle them appropriately.",
            explanation="Silently ignoring all exceptions hides bugs and makes debugging difficult.",
        ))

        # Correctness rules
        self.add_rule(DetectionRule(
            rule_id="CORR001",
            dimension=Dimension.CORRECTNESS,
            severity=Severity.HIGH,
            pattern=re.compile(r'==\s*(None|True|False)\b'),
            message="Use 'is' for None/bool comparison",
            suggestion="Use 'is None', 'is True', 'is False' for identity comparison.",
            explanation="Use == for value comparison and 'is' for identity comparison.",
        ))

        self.add_rule(DetectionRule(
            rule_id="CORR002",
            dimension=Dimension.CORRECTNESS,
            severity=Severity.MEDIUM,
            pattern=re.compile(r'def\s+\w+\s*\([^)]*=\s*\[\s*\]|\[mutable\]'),
            message="Mutable default argument",
            suggestion="Use None as default and initialize inside function.",
            explanation="Mutable default arguments are shared across calls.",
        ))

    def add_rule(self, rule: DetectionRule):
        """Add a detection rule"""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str):
        """Remove a detection rule"""
        self.rules.pop(rule_id, None)

    def enable_rule(self, rule_id: str):
        """Enable a detection rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str):
        """Disable a detection rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False

    async def detect(
        self,
        code: str,
        language: str = "python",
        dimensions: Optional[List[Dimension]] = None,
    ) -> List[Finding]:
        """
        Detect issues in code.

        Args:
            code: Source code to analyze
            language: Programming language
            dimensions: Dimensions to check (all if None)

        Returns:
            List of detected findings
        """
        findings = []
        lines = code.split('\n')

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            if dimensions and rule.dimension not in dimensions:
                continue

            # Find all matches
            for match in rule.pattern.finditer(code):
                # Calculate line number
                line_num = code[:match.start()].count('\n') + 1

                # Get code snippet
                snippet_start = max(0, line_num - 2)
                snippet_end = min(len(lines), line_num + 2)
                code_snippet = '\n'.join(lines[snippet_start:snippet_end])

                finding = Finding(
                    dimension=rule.dimension.value,
                    issue=rule.message,
                    line_numbers=[line_num],
                    severity=rule.severity.value,
                    confidence=0.9,
                    suggestion=rule.suggestion,
                    explanation=rule.explanation,
                    cwe_id=rule.cwe_id,
                    rule_id=rule.rule_id,
                    code_snippet=code_snippet,
                )

                findings.append(finding)

        logger.info(f"Detected {len(findings)} issues using {len(self.rules)} rules")
        return findings

    def get_rules_summary(self) -> Dict[str, Any]:
        """Get summary of loaded rules"""
        summary = {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "by_dimension": {},
            "by_severity": {},
        }

        for rule in self.rules.values():
            dim = rule.dimension.value
            sev = rule.severity.value

            summary["by_dimension"][dim] = summary["by_dimension"].get(dim, 0) + 1
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1

        return summary
