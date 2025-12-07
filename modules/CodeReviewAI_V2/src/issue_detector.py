"""
CodeReviewAI_V2 - Enhanced Issue Detector

Production-grade issue detection with:
- Extended rule set (OWASP Top 10)
- Multi-language support
- Configurable rule severity
"""

import re
import logging
from typing import List, Dict, Optional, Pattern
from dataclasses import dataclass

from .models import Finding, Dimension, Severity

logger = logging.getLogger(__name__)


@dataclass
class DetectionRule:
    """Enhanced detection rule with language support"""
    rule_id: str
    dimension: Dimension
    severity: Severity
    pattern: Pattern
    message: str
    suggestion: str
    explanation: str
    cwe_id: Optional[str] = None
    enabled: bool = True
    languages: List[str] = None  # V2: Language-specific rules

    def __post_init__(self):
        if self.languages is None:
            self.languages = ["python", "javascript", "typescript", "java", "go"]


class IssueDetector:
    """
    Enhanced issue detector with extended rule coverage.

    V2 Improvements:
    - OWASP Top 10 coverage
    - Language-specific rules
    - Performance optimizations
    """

    def __init__(self):
        self.rules: Dict[str, DetectionRule] = {}
        self._load_default_rules()
        self._load_owasp_rules()

    def _load_default_rules(self):
        """Load standard detection rules"""
        # Import patterns from V1 and enhance
        standard_rules = [
            # Security
            ("SEC001", Dimension.SECURITY, Severity.CRITICAL,
             r'\beval\s*\(', "eval() usage", "CWE-95"),
            ("SEC002", Dimension.SECURITY, Severity.CRITICAL,
             r'execute\s*\(\s*[f"\']', "SQL injection risk", "CWE-89"),
            ("SEC003", Dimension.SECURITY, Severity.HIGH,
             r'(password|secret|api_key)\s*=\s*["\'][^"\']+["\']', "Hardcoded credential", "CWE-798"),
            ("SEC004", Dimension.SECURITY, Severity.HIGH,
             r'subprocess.*shell\s*=\s*True', "Shell injection", "CWE-78"),
            ("SEC005", Dimension.SECURITY, Severity.MEDIUM,
             r'pickle\.(load|loads)\s*\(', "Unsafe deserialization", "CWE-502"),

            # Performance
            ("PERF001", Dimension.PERFORMANCE, Severity.MEDIUM,
             r'for\s+\w+\s+in\s+.*:\s*\n\s+for\s+\w+\s+in', "Nested loop", None),
            ("PERF002", Dimension.PERFORMANCE, Severity.LOW,
             r'\+\s*=\s*["\'][^"\']*["\']', "String concatenation", None),
            ("PERF003", Dimension.PERFORMANCE, Severity.MEDIUM,
             r'time\.sleep\s*\(', "Blocking sleep", None),

            # Maintainability
            ("MAINT001", Dimension.MAINTAINABILITY, Severity.LOW,
             r'^def\s+\w+\s*\([^)]*\)\s*:', "Function definition", None),
            ("MAINT002", Dimension.MAINTAINABILITY, Severity.MEDIUM,
             r'#\s*(TODO|FIXME|HACK|XXX)', "TODO comment", None),
            ("MAINT003", Dimension.MAINTAINABILITY, Severity.HIGH,
             r'except\s*:\s*\n\s*(pass|\.\.\.)', "Bare except", None),
        ]

        for rule_id, dim, sev, pattern, msg, cwe in standard_rules:
            self.add_rule(DetectionRule(
                rule_id=rule_id,
                dimension=dim,
                severity=sev,
                pattern=re.compile(pattern, re.MULTILINE | re.IGNORECASE),
                message=msg,
                suggestion=f"Address: {msg}",
                explanation=f"Detection rule {rule_id}",
                cwe_id=cwe,
            ))

    def _load_owasp_rules(self):
        """V2: Load OWASP Top 10 aligned rules"""
        owasp_rules = [
            # A01: Broken Access Control
            ("OWASP-A01-01", Dimension.SECURITY, Severity.HIGH,
             r'@app\.route.*methods.*DELETE', "DELETE without auth check", "CWE-862"),

            # A02: Cryptographic Failures
            ("OWASP-A02-01", Dimension.SECURITY, Severity.HIGH,
             r'hashlib\.md5\s*\(', "Weak hash (MD5)", "CWE-328"),
            ("OWASP-A02-02", Dimension.SECURITY, Severity.HIGH,
             r'hashlib\.sha1\s*\(', "Weak hash (SHA1)", "CWE-328"),

            # A03: Injection
            ("OWASP-A03-01", Dimension.SECURITY, Severity.CRITICAL,
             r'os\.system\s*\(', "OS command injection risk", "CWE-78"),
            ("OWASP-A03-02", Dimension.SECURITY, Severity.CRITICAL,
             r'subprocess\.call\s*\([^)]*,\s*shell\s*=\s*True', "Command injection", "CWE-78"),

            # A04: Insecure Design
            ("OWASP-A04-01", Dimension.SECURITY, Severity.MEDIUM,
             r'random\.(random|randint|choice)\s*\(', "Insecure random", "CWE-330"),

            # A05: Security Misconfiguration
            ("OWASP-A05-01", Dimension.SECURITY, Severity.HIGH,
             r'debug\s*=\s*True', "Debug mode enabled", "CWE-489"),
            ("OWASP-A05-02", Dimension.SECURITY, Severity.MEDIUM,
             r'verify\s*=\s*False', "SSL verification disabled", "CWE-295"),

            # A06: Vulnerable Components (would need dependency scanning)

            # A07: Authentication Failures
            ("OWASP-A07-01", Dimension.SECURITY, Severity.HIGH,
             r'jwt\.decode\s*\([^)]*verify\s*=\s*False', "JWT verification disabled", "CWE-347"),

            # A08: Software and Data Integrity
            ("OWASP-A08-01", Dimension.SECURITY, Severity.HIGH,
             r'yaml\.load\s*\([^)]*\)', "Unsafe YAML load", "CWE-502"),

            # A09: Logging Failures
            ("OWASP-A09-01", Dimension.SECURITY, Severity.MEDIUM,
             r'print\s*\([^)]*password', "Password in log", "CWE-532"),

            # A10: SSRF
            ("OWASP-A10-01", Dimension.SECURITY, Severity.HIGH,
             r'requests\.(get|post)\s*\([^)]*\+', "Potential SSRF", "CWE-918"),
        ]

        for rule_id, dim, sev, pattern, msg, cwe in owasp_rules:
            self.add_rule(DetectionRule(
                rule_id=rule_id,
                dimension=dim,
                severity=sev,
                pattern=re.compile(pattern, re.MULTILINE | re.IGNORECASE),
                message=msg,
                suggestion=f"OWASP: {msg}",
                explanation=f"OWASP Top 10 rule: {cwe}",
                cwe_id=cwe,
            ))

    def add_rule(self, rule: DetectionRule):
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str):
        self.rules.pop(rule_id, None)

    def enable_rule(self, rule_id: str):
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str):
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False

    async def detect(
        self,
        code: str,
        language: str = "python",
        dimensions: Optional[List[Dimension]] = None,
    ) -> List[Finding]:
        """Detect issues with language awareness"""
        findings = []
        lines = code.split('\n')

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            if dimensions and rule.dimension not in dimensions:
                continue

            if language not in rule.languages:
                continue

            for match in rule.pattern.finditer(code):
                line_num = code[:match.start()].count('\n') + 1

                snippet_start = max(0, line_num - 2)
                snippet_end = min(len(lines), line_num + 2)
                code_snippet = '\n'.join(lines[snippet_start:snippet_end])

                findings.append(Finding(
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
                ))

        logger.info(f"Detected {len(findings)} issues for {language}")
        return findings

    def get_rules_summary(self) -> Dict:
        """Get summary with OWASP coverage"""
        summary = {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "owasp_rules": sum(1 for r in self.rules if r.startswith("OWASP")),
            "by_dimension": {},
            "by_severity": {},
        }

        for rule in self.rules.values():
            dim = rule.dimension.value
            sev = rule.severity.value
            summary["by_dimension"][dim] = summary["by_dimension"].get(dim, 0) + 1
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1

        return summary
