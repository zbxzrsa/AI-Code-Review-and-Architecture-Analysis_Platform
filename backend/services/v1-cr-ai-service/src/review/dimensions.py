"""
Dimension Analyzers for V1 Code Review AI

Specialized analyzers for each review dimension:
- Correctness
- Security
- Performance
- Maintainability
- Architecture
- Testing
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


class DimensionAnalyzer(ABC):
    """Abstract base class for dimension analyzers"""
    
    @property
    @abstractmethod
    def dimension_name(self) -> str:
        """Name of the dimension"""
        pass
    
    @abstractmethod
    def analyze(
        self,
        code: str,
        lines: List[str],
        language: str,
    ) -> List[Dict[str, Any]]:
        """Analyze code for this dimension"""
        pass


class CorrectnessAnalyzer(DimensionAnalyzer):
    """Analyzer for correctness issues"""
    
    @property
    def dimension_name(self) -> str:
        return "correctness"
    
    # Patterns for correctness issues
    PATTERNS = {
        "off_by_one": [
            (r'range\(len\(\w+\)\s*-\s*1\)', "Off-by-one in range()"),
            (r'for\s+\w+\s+in\s+range\(.*-1\)', "Potential off-by-one"),
            (r'\[\w+\s*\+\s*1\]', "Index+1 may exceed bounds"),
        ],
        "null_check": [
            (r'(\w+)\.(\w+)\s*\(.*\)\s*if\s+\1', "Check before access pattern"),
        ],
        "comparison": [
            (r'==\s*True\b', "Use 'if x:' instead of 'if x == True'"),
            (r'==\s*False\b', "Use 'if not x:' instead of 'if x == False'"),
            (r'==\s*None\b', "Use 'is None' instead of '== None'"),
        ],
    }
    
    def analyze(
        self,
        code: str,
        lines: List[str],
        language: str,
    ) -> List[Dict[str, Any]]:
        findings = []
        
        for category, patterns in self.PATTERNS.items():
            for i, line in enumerate(lines, 1):
                for pattern, description in patterns:
                    if re.search(pattern, line):
                        findings.append({
                            "dimension": self.dimension_name,
                            "category": category,
                            "issue": description,
                            "line_number": i,
                            "code_snippet": line.strip(),
                            "severity": "high" if category == "off_by_one" else "medium",
                        })
        
        return findings


class SecurityAnalyzer(DimensionAnalyzer):
    """Analyzer for security vulnerabilities"""
    
    @property
    def dimension_name(self) -> str:
        return "security"
    
    # Security vulnerability patterns
    PATTERNS = {
        "sql_injection": [
            (r'execute\s*\(\s*f["\']', "CWE-89", "SQL Injection via f-string"),
            (r'execute\s*\(\s*["\'].*%s', "CWE-89", "SQL Injection via %s"),
            (r'execute\s*\(\s*["\'].*\+', "CWE-89", "SQL Injection via concatenation"),
        ],
        "command_injection": [
            (r'os\.system\s*\(', "CWE-78", "Command Injection via os.system"),
            (r'subprocess.*shell\s*=\s*True', "CWE-78", "Command Injection with shell=True"),
            (r'eval\s*\(', "CWE-94", "Code Injection via eval()"),
            (r'exec\s*\(', "CWE-94", "Code Injection via exec()"),
        ],
        "hardcoded_secrets": [
            (r'password\s*=\s*["\'][^"\']+["\']', "CWE-798", "Hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "CWE-798", "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "CWE-798", "Hardcoded secret"),
        ],
        "xss": [
            (r'innerHTML\s*=', "CWE-79", "XSS via innerHTML"),
            (r'dangerouslySetInnerHTML', "CWE-79", "XSS via dangerouslySetInnerHTML"),
        ],
        "insecure_deserialization": [
            (r'pickle\.loads?\s*\(', "CWE-502", "Insecure deserialization with pickle"),
            (r'yaml\.load\s*\([^)]*\)', "CWE-502", "Insecure YAML loading"),
        ],
    }
    
    def analyze(
        self,
        code: str,
        lines: List[str],
        language: str,
    ) -> List[Dict[str, Any]]:
        findings = []
        
        for category, patterns in self.PATTERNS.items():
            for i, line in enumerate(lines, 1):
                for pattern, cwe, description in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        findings.append({
                            "dimension": self.dimension_name,
                            "category": category,
                            "issue": description,
                            "cwe_id": cwe,
                            "line_number": i,
                            "code_snippet": line.strip(),
                            "severity": "critical",
                        })
        
        return findings


class PerformanceAnalyzer(DimensionAnalyzer):
    """Analyzer for performance issues"""
    
    @property
    def dimension_name(self) -> str:
        return "performance"
    
    PATTERNS = {
        "complexity": [
            (r'for.*:\s*\n\s*for.*:\s*\n\s*for', "Triple nested loop (O(n³))"),
        ],
        "string_concat": [
            (r'\+=\s*["\']', "String concatenation in loop (use join())"),
        ],
        "inefficient_list": [
            (r'if\s+\w+\s+in\s+list\(', "Convert to list for membership test"),
            (r'sorted\(.*\)\[0\]', "Use min() instead of sorted()[0]"),
            (r'sorted\(.*\)\[-1\]', "Use max() instead of sorted()[-1]"),
        ],
        "repeated_computation": [
            (r'for.*:\s*\n.*len\(\w+\)', "len() called in loop"),
        ],
    }
    
    def analyze(
        self,
        code: str,
        lines: List[str],
        language: str,
    ) -> List[Dict[str, Any]]:
        findings = []
        
        # Check patterns
        for category, patterns in self.PATTERNS.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, code, re.MULTILINE)
                for match in matches:
                    line_num = code[:match.start()].count('\n') + 1
                    findings.append({
                        "dimension": self.dimension_name,
                        "category": category,
                        "issue": description,
                        "line_number": line_num,
                        "severity": "medium",
                    })
        
        # Check nested loop depth
        indent_stack = []
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith('for ') or stripped.startswith('while '):
                indent = len(line) - len(stripped)
                while indent_stack and indent_stack[-1] >= indent:
                    indent_stack.pop()
                indent_stack.append(indent)
                
                if len(indent_stack) > 2:
                    findings.append({
                        "dimension": self.dimension_name,
                        "category": "complexity",
                        "issue": f"Nested loop depth: {len(indent_stack)}",
                        "line_number": i,
                        "severity": "high" if len(indent_stack) > 3 else "medium",
                    })
        
        return findings


# Factory for dimension analyzers
ANALYZERS = {
    "correctness": CorrectnessAnalyzer,
    "security": SecurityAnalyzer,
    "performance": PerformanceAnalyzer,
}


def get_analyzer(dimension: str) -> Optional[DimensionAnalyzer]:
    """Get analyzer for a dimension"""
    analyzer_class = ANALYZERS.get(dimension)
    if analyzer_class:
        return analyzer_class()
    return None
