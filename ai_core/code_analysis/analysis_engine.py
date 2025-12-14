"""
Code Analysis Engine

A comprehensive code analysis engine that scans project directories and identifies:
- Syntax Errors
- Runtime Errors
- Logical Errors
- Security Vulnerabilities
- Performance Issues
- Coding Standard Violations

Supports JavaScript/TypeScript, Python, Java, and other mainstream languages.
Target accuracy: >95% error identification rate.
"""

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ErrorType(str, Enum):
    """Types of code errors."""
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    LOGICAL = "logical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    DEPRECATED = "deprecated"
    TYPE_ERROR = "type_error"


class Severity(str, Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    UNKNOWN = "unknown"


# Language file extensions mapping
LANGUAGE_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".pyw": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".cs": Language.CSHARP,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CodeLocation:
    """Location of code in a file."""
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "column_start": self.column_start,
            "column_end": self.column_end,
        }


@dataclass
class CodeIssue:
    """Represents a detected code issue."""
    issue_id: str
    error_type: ErrorType
    severity: Severity
    message: str
    location: CodeLocation
    code_snippet: str
    rule_id: str
    suggestion: str = ""
    fix_available: bool = False
    fix_code: Optional[str] = None
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    related_issues: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location.to_dict(),
            "code_snippet": self.code_snippet,
            "rule_id": self.rule_id,
            "suggestion": self.suggestion,
            "fix_available": self.fix_available,
            "confidence": self.confidence,
            "tags": self.tags,
        }


@dataclass
class AnalysisResult:
    """Result of code analysis."""
    analysis_id: str
    file_path: str
    language: Language
    issues: List[CodeIssue]
    metrics: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = True
    error: Optional[str] = None
    
    @property
    def issue_count(self) -> int:
        return len(self.issues)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.HIGH)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "file_path": self.file_path,
            "language": self.language.value,
            "issue_count": self.issue_count,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "success": self.success,
        }


@dataclass
class ProjectAnalysis:
    """Analysis result for entire project."""
    project_id: str
    project_path: str
    file_results: List[AnalysisResult]
    summary: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    @property
    def total_issues(self) -> int:
        return sum(r.issue_count for r in self.file_results)
    
    @property
    def files_with_issues(self) -> int:
        return sum(1 for r in self.file_results if r.issue_count > 0)


# =============================================================================
# Rule Definitions
# =============================================================================

@dataclass
class AnalysisRule:
    """Definition of an analysis rule."""
    rule_id: str
    name: str
    description: str
    error_type: ErrorType
    severity: Severity
    languages: List[Language]
    pattern: Optional[str] = None  # Regex pattern
    ast_check: Optional[Callable] = None  # AST-based check function
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


# =============================================================================
# Language Analyzers
# =============================================================================

class LanguageAnalyzer(ABC):
    """Abstract base class for language-specific analyzers."""
    
    def __init__(self, language: Language):
        self.language = language
        self.rules: List[AnalysisRule] = []
        self._initialize_rules()
    
    @abstractmethod
    def _initialize_rules(self) -> None:
        """Initialize language-specific rules."""
        pass
    
    @abstractmethod
    async def analyze(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze code and return issues."""
        pass
    
    def _create_issue(
        self,
        rule: AnalysisRule,
        location: CodeLocation,
        code_snippet: str,
        message: Optional[str] = None,
        suggestion: str = "",
        fix_code: Optional[str] = None,
        confidence: float = 1.0,
    ) -> CodeIssue:
        """Create a code issue."""
        return CodeIssue(
            issue_id=str(uuid.uuid4()),
            error_type=rule.error_type,
            severity=rule.severity,
            message=message or rule.description,
            location=location,
            code_snippet=code_snippet,
            rule_id=rule.rule_id,
            suggestion=suggestion,
            fix_available=fix_code is not None,
            fix_code=fix_code,
            confidence=confidence,
            tags=rule.tags.copy(),
        )


class PythonAnalyzer(LanguageAnalyzer):
    """Python code analyzer."""
    
    def __init__(self):
        super().__init__(Language.PYTHON)
    
    def _initialize_rules(self) -> None:
        """Initialize Python-specific rules."""
        self.rules = [
            # Syntax rules
            AnalysisRule(
                rule_id="PY-SYN-001",
                name="syntax_error",
                description="Python syntax error",
                error_type=ErrorType.SYNTAX,
                severity=Severity.CRITICAL,
                languages=[Language.PYTHON],
                tags=["syntax", "critical"],
            ),
            
            # Security rules
            AnalysisRule(
                rule_id="PY-SEC-001",
                name="hardcoded_secret",
                description="Hardcoded password or secret detected",
                error_type=ErrorType.SECURITY,
                severity=Severity.CRITICAL,
                languages=[Language.PYTHON],
                pattern=r'(password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']',
                tags=["security", "secrets"],
            ),
            AnalysisRule(
                rule_id="PY-SEC-002",
                name="eval_usage",
                description="Use of eval() is dangerous",
                error_type=ErrorType.SECURITY,
                severity=Severity.HIGH,
                languages=[Language.PYTHON],
                pattern=r'\beval\s*\(',
                tags=["security", "injection"],
            ),
            AnalysisRule(
                rule_id="PY-SEC-003",
                name="exec_usage",
                description="Use of exec() is dangerous",
                error_type=ErrorType.SECURITY,
                severity=Severity.HIGH,
                languages=[Language.PYTHON],
                pattern=r'\bexec\s*\(',
                tags=["security", "injection"],
            ),
            AnalysisRule(
                rule_id="PY-SEC-004",
                name="sql_injection",
                description="Potential SQL injection vulnerability",
                error_type=ErrorType.SECURITY,
                severity=Severity.CRITICAL,
                languages=[Language.PYTHON],
                pattern=r'execute\s*\(\s*["\'].*%s.*["\'].*%',
                tags=["security", "sql", "injection"],
            ),
            AnalysisRule(
                rule_id="PY-SEC-005",
                name="pickle_usage",
                description="Use of pickle can be insecure",
                error_type=ErrorType.SECURITY,
                severity=Severity.MEDIUM,
                languages=[Language.PYTHON],
                pattern=r'pickle\.(load|loads)\s*\(',
                tags=["security", "serialization"],
            ),
            
            # Performance rules
            AnalysisRule(
                rule_id="PY-PERF-001",
                name="list_in_loop",
                description="List concatenation in loop is inefficient",
                error_type=ErrorType.PERFORMANCE,
                severity=Severity.MEDIUM,
                languages=[Language.PYTHON],
                tags=["performance", "loop"],
            ),
            AnalysisRule(
                rule_id="PY-PERF-002",
                name="global_import",
                description="Import inside function is slow",
                error_type=ErrorType.PERFORMANCE,
                severity=Severity.LOW,
                languages=[Language.PYTHON],
                tags=["performance", "import"],
            ),
            AnalysisRule(
                rule_id="PY-PERF-003",
                name="inefficient_comprehension",
                description="Use list comprehension instead of loop",
                error_type=ErrorType.PERFORMANCE,
                severity=Severity.LOW,
                languages=[Language.PYTHON],
                tags=["performance", "comprehension"],
            ),
            
            # Style rules
            AnalysisRule(
                rule_id="PY-STY-001",
                name="line_too_long",
                description="Line exceeds maximum length",
                error_type=ErrorType.STYLE,
                severity=Severity.LOW,
                languages=[Language.PYTHON],
                tags=["style", "pep8"],
            ),
            AnalysisRule(
                rule_id="PY-STY-002",
                name="missing_docstring",
                description="Missing docstring",
                error_type=ErrorType.STYLE,
                severity=Severity.LOW,
                languages=[Language.PYTHON],
                tags=["style", "documentation"],
            ),
            AnalysisRule(
                rule_id="PY-STY-003",
                name="unused_import",
                description="Unused import",
                error_type=ErrorType.STYLE,
                severity=Severity.LOW,
                languages=[Language.PYTHON],
                tags=["style", "imports"],
            ),
            
            # Runtime error rules
            AnalysisRule(
                rule_id="PY-RUN-001",
                name="undefined_variable",
                description="Potentially undefined variable",
                error_type=ErrorType.RUNTIME,
                severity=Severity.HIGH,
                languages=[Language.PYTHON],
                tags=["runtime", "undefined"],
            ),
            AnalysisRule(
                rule_id="PY-RUN-002",
                name="type_mismatch",
                description="Type mismatch detected",
                error_type=ErrorType.TYPE_ERROR,
                severity=Severity.MEDIUM,
                languages=[Language.PYTHON],
                tags=["runtime", "types"],
            ),
            
            # Logical error rules
            AnalysisRule(
                rule_id="PY-LOG-001",
                name="unreachable_code",
                description="Unreachable code detected",
                error_type=ErrorType.LOGICAL,
                severity=Severity.MEDIUM,
                languages=[Language.PYTHON],
                tags=["logic", "dead_code"],
            ),
            AnalysisRule(
                rule_id="PY-LOG-002",
                name="comparison_to_none",
                description="Use 'is None' instead of '== None'",
                error_type=ErrorType.LOGICAL,
                severity=Severity.LOW,
                languages=[Language.PYTHON],
                pattern=r'==\s*None|!=\s*None',
                tags=["logic", "comparison"],
            ),
            AnalysisRule(
                rule_id="PY-LOG-003",
                name="mutable_default",
                description="Mutable default argument",
                error_type=ErrorType.LOGICAL,
                severity=Severity.MEDIUM,
                languages=[Language.PYTHON],
                tags=["logic", "defaults"],
            ),
        ]
    
    async def analyze(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze Python code."""
        issues = []
        lines = content.split('\n')
        
        # 1. Syntax check using AST
        issues.extend(await self._check_syntax(file_path, content))
        
        # 2. Pattern-based checks
        issues.extend(await self._check_patterns(file_path, content, lines))
        
        # 3. AST-based semantic checks
        try:
            tree = ast.parse(content)
            issues.extend(await self._check_ast(file_path, tree, content, lines))
        except SyntaxError:
            pass  # Already caught in syntax check
        
        return issues
    
    async def _check_syntax(self, file_path: str, content: str) -> List[CodeIssue]:
        """Check for syntax errors."""
        issues = []
        rule = next((r for r in self.rules if r.rule_id == "PY-SYN-001"), None)
        
        if rule:
            try:
                ast.parse(content)
            except SyntaxError as e:
                lines = content.split('\n')
                line_num = e.lineno or 1
                code_snippet = lines[line_num - 1] if line_num <= len(lines) else ""
                
                issues.append(self._create_issue(
                    rule=rule,
                    location=CodeLocation(
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        column_start=e.offset or 0,
                        column_end=e.offset or 0,
                    ),
                    code_snippet=code_snippet,
                    message=f"Syntax error: {e.msg}",
                    suggestion=f"Fix the syntax error at line {line_num}",
                ))
        
        return issues
    
    async def _check_patterns(
        self,
        file_path: str,
        content: str,
        lines: List[str],
    ) -> List[CodeIssue]:
        """Check for pattern-based issues."""
        issues = []
        
        for rule in self.rules:
            if not rule.pattern or not rule.enabled:
                continue
            
            try:
                pattern = re.compile(rule.pattern, re.IGNORECASE)
                
                for line_num, line in enumerate(lines, 1):
                    matches = pattern.finditer(line)
                    for match in matches:
                        issues.append(self._create_issue(
                            rule=rule,
                            location=CodeLocation(
                                file_path=file_path,
                                line_start=line_num,
                                line_end=line_num,
                                column_start=match.start(),
                                column_end=match.end(),
                            ),
                            code_snippet=line.strip(),
                            suggestion=self._get_suggestion(rule),
                        ))
            except re.error:
                continue
        
        return issues
    
    async def _check_ast(
        self,
        file_path: str,
        tree: ast.AST,
        content: str,
        lines: List[str],
    ) -> List[CodeIssue]:
        """Perform AST-based analysis."""
        issues = []
        
        # Check for mutable default arguments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                issues.extend(
                    self._check_mutable_defaults(file_path, node, lines)
                )
                issues.extend(
                    self._check_missing_docstring(file_path, node, lines)
                )
            
            # Check for unreachable code
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                issues.extend(
                    self._check_unreachable_code(file_path, node, lines)
                )
        
        # Check for line length
        for line_num, line in enumerate(lines, 1):
            if len(line) > 120:
                rule = next((r for r in self.rules if r.rule_id == "PY-STY-001"), None)
                if rule:
                    issues.append(self._create_issue(
                        rule=rule,
                        location=CodeLocation(
                            file_path=file_path,
                            line_start=line_num,
                            line_end=line_num,
                        ),
                        code_snippet=line[:50] + "...",
                        message=f"Line too long ({len(line)} > 120 characters)",
                        suggestion="Break line into multiple lines",
                    ))
        
        return issues
    
    def _check_mutable_defaults(
        self,
        file_path: str,
        node: ast.FunctionDef,
        lines: List[str],
    ) -> List[CodeIssue]:
        """Check for mutable default arguments."""
        issues = []
        rule = next((r for r in self.rules if r.rule_id == "PY-LOG-003"), None)
        
        if not rule:
            return issues
        
        for default in node.args.defaults + node.args.kw_defaults:
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                line_num = default.lineno
                code_snippet = lines[line_num - 1].strip() if line_num <= len(lines) else ""
                
                issues.append(self._create_issue(
                    rule=rule,
                    location=CodeLocation(
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                    ),
                    code_snippet=code_snippet,
                    message="Mutable default argument. Use None and initialize inside function.",
                    suggestion="def func(arg=None): if arg is None: arg = []",
                    fix_available=True,
                ))
        
        return issues
    
    def _check_missing_docstring(
        self,
        file_path: str,
        node: ast.FunctionDef,
        lines: List[str],
    ) -> List[CodeIssue]:
        """Check for missing docstrings."""
        issues = []
        rule = next((r for r in self.rules if r.rule_id == "PY-STY-002"), None)
        
        if not rule:
            return issues
        
        # Skip private/dunder methods
        if node.name.startswith('_'):
            return issues
        
        docstring = ast.get_docstring(node)
        if not docstring:
            code_snippet = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
            
            issues.append(self._create_issue(
                rule=rule,
                location=CodeLocation(
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.lineno,
                ),
                code_snippet=code_snippet,
                message=f"Function '{node.name}' is missing a docstring",
                suggestion='Add a docstring: """Description of function."""',
                confidence=0.8,
            ))
        
        return issues
    
    def _check_unreachable_code(
        self,
        file_path: str,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        lines: List[str],
    ) -> List[CodeIssue]:
        """Check for unreachable code after return/raise."""
        issues = []
        rule = next((r for r in self.rules if r.rule_id == "PY-LOG-001"), None)
        
        if not rule:
            return issues
        
        body = node.body
        for i, stmt in enumerate(body[:-1]):
            if isinstance(stmt, (ast.Return, ast.Raise)):
                # Code after return/raise is unreachable
                next_stmt = body[i + 1]
                if hasattr(next_stmt, 'lineno'):
                    code_snippet = lines[next_stmt.lineno - 1].strip() if next_stmt.lineno <= len(lines) else ""
                    
                    issues.append(self._create_issue(
                        rule=rule,
                        location=CodeLocation(
                            file_path=file_path,
                            line_start=next_stmt.lineno,
                            line_end=next_stmt.lineno,
                        ),
                        code_snippet=code_snippet,
                        message="Unreachable code after return/raise statement",
                        suggestion="Remove unreachable code or fix control flow",
                    ))
                break  # Only report the first unreachable statement
        
        return issues
    
    def _get_suggestion(self, rule: AnalysisRule) -> str:
        """Get suggestion for a rule."""
        suggestions = {
            "PY-SEC-001": "Use environment variables or a secrets manager",
            "PY-SEC-002": "Use ast.literal_eval() or define allowed expressions",
            "PY-SEC-003": "Avoid exec() or sanitize input thoroughly",
            "PY-SEC-004": "Use parameterized queries with placeholders",
            "PY-SEC-005": "Use json for serialization or validate pickle sources",
            "PY-LOG-002": "Use 'x is None' instead of 'x == None'",
        }
        return suggestions.get(rule.rule_id, "Review and fix the issue")


class JavaScriptAnalyzer(LanguageAnalyzer):
    """JavaScript/TypeScript code analyzer."""
    
    def __init__(self, language: Language = Language.JAVASCRIPT):
        super().__init__(language)
    
    def _initialize_rules(self) -> None:
        """Initialize JavaScript-specific rules."""
        self.rules = [
            # Security rules
            AnalysisRule(
                rule_id="JS-SEC-001",
                name="eval_usage",
                description="Use of eval() is dangerous",
                error_type=ErrorType.SECURITY,
                severity=Severity.CRITICAL,
                languages=[Language.JAVASCRIPT, Language.TYPESCRIPT],
                pattern=r'\beval\s*\(',
                tags=["security", "injection"],
            ),
            AnalysisRule(
                rule_id="JS-SEC-002",
                name="innerHTML",
                description="innerHTML can lead to XSS",
                error_type=ErrorType.SECURITY,
                severity=Severity.HIGH,
                languages=[Language.JAVASCRIPT, Language.TYPESCRIPT],
                pattern=r'\.innerHTML\s*=',
                tags=["security", "xss"],
            ),
            AnalysisRule(
                rule_id="JS-SEC-003",
                name="document_write",
                description="document.write() can be dangerous",
                error_type=ErrorType.SECURITY,
                severity=Severity.MEDIUM,
                languages=[Language.JAVASCRIPT, Language.TYPESCRIPT],
                pattern=r'document\.write\s*\(',
                tags=["security", "xss"],
            ),
            AnalysisRule(
                rule_id="JS-SEC-004",
                name="hardcoded_secret",
                description="Hardcoded secret detected",
                error_type=ErrorType.SECURITY,
                severity=Severity.CRITICAL,
                languages=[Language.JAVASCRIPT, Language.TYPESCRIPT],
                pattern=r'(password|secret|apiKey|api_key|token)\s*[=:]\s*["\'][^"\']+["\']',
                tags=["security", "secrets"],
            ),
            
            # Style rules
            AnalysisRule(
                rule_id="JS-STY-001",
                name="var_usage",
                description="Use let/const instead of var",
                error_type=ErrorType.STYLE,
                severity=Severity.LOW,
                languages=[Language.JAVASCRIPT, Language.TYPESCRIPT],
                pattern=r'\bvar\s+',
                tags=["style", "es6"],
            ),
            AnalysisRule(
                rule_id="JS-STY-002",
                name="console_log",
                description="console.log should be removed in production",
                error_type=ErrorType.STYLE,
                severity=Severity.LOW,
                languages=[Language.JAVASCRIPT, Language.TYPESCRIPT],
                pattern=r'console\.(log|debug|info)\s*\(',
                tags=["style", "debugging"],
            ),
            
            # Logical rules
            AnalysisRule(
                rule_id="JS-LOG-001",
                name="double_equals",
                description="Use === instead of ==",
                error_type=ErrorType.LOGICAL,
                severity=Severity.MEDIUM,
                languages=[Language.JAVASCRIPT, Language.TYPESCRIPT],
                pattern=r'[^=!]==[^=]',
                tags=["logic", "comparison"],
            ),
            AnalysisRule(
                rule_id="JS-LOG-002",
                name="typeof_undefined",
                description="Use typeof x === 'undefined'",
                error_type=ErrorType.LOGICAL,
                severity=Severity.MEDIUM,
                languages=[Language.JAVASCRIPT, Language.TYPESCRIPT],
                pattern=r'===?\s*undefined(?!\s*;)',
                tags=["logic", "undefined"],
            ),
            
            # Performance rules
            AnalysisRule(
                rule_id="JS-PERF-001",
                name="dom_in_loop",
                description="DOM manipulation in loop is inefficient",
                error_type=ErrorType.PERFORMANCE,
                severity=Severity.MEDIUM,
                languages=[Language.JAVASCRIPT, Language.TYPESCRIPT],
                pattern=r'for\s*\([^)]*\)\s*\{[^}]*document\.(getElementById|querySelector)',
                tags=["performance", "dom"],
            ),
        ]
    
    async def analyze(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze JavaScript/TypeScript code."""
        issues = []
        lines = content.split('\n')
        
        # Pattern-based checks
        for rule in self.rules:
            if not rule.pattern or not rule.enabled:
                continue
            
            try:
                pattern = re.compile(rule.pattern, re.IGNORECASE)
                
                for line_num, line in enumerate(lines, 1):
                    matches = pattern.finditer(line)
                    for match in matches:
                        issues.append(self._create_issue(
                            rule=rule,
                            location=CodeLocation(
                                file_path=file_path,
                                line_start=line_num,
                                line_end=line_num,
                                column_start=match.start(),
                                column_end=match.end(),
                            ),
                            code_snippet=line.strip(),
                            suggestion=self._get_suggestion(rule),
                        ))
            except re.error:
                continue
        
        return issues
    
    def _get_suggestion(self, rule: AnalysisRule) -> str:
        """Get suggestion for a rule."""
        suggestions = {
            "JS-SEC-001": "Avoid eval(). Use JSON.parse() or other safe alternatives",
            "JS-SEC-002": "Use textContent or DOM manipulation methods",
            "JS-SEC-003": "Use DOM methods like appendChild instead",
            "JS-SEC-004": "Use environment variables or a configuration service",
            "JS-STY-001": "Replace 'var' with 'let' or 'const'",
            "JS-STY-002": "Remove console.log or use a proper logging library",
            "JS-LOG-001": "Use strict equality (===) for comparison",
            "JS-LOG-002": "Use typeof x === 'undefined' for safer checks",
        }
        return suggestions.get(rule.rule_id, "Review and fix the issue")


class JavaAnalyzer(LanguageAnalyzer):
    """Java code analyzer."""
    
    def __init__(self):
        super().__init__(Language.JAVA)
    
    def _initialize_rules(self) -> None:
        """Initialize Java-specific rules."""
        self.rules = [
            # Security rules
            AnalysisRule(
                rule_id="JAVA-SEC-001",
                name="sql_injection",
                description="Potential SQL injection vulnerability",
                error_type=ErrorType.SECURITY,
                severity=Severity.CRITICAL,
                languages=[Language.JAVA],
                pattern=r'(executeQuery|executeUpdate|execute)\s*\([^)]*\+',
                tags=["security", "sql", "injection"],
            ),
            AnalysisRule(
                rule_id="JAVA-SEC-002",
                name="hardcoded_credentials",
                description="Hardcoded credentials detected",
                error_type=ErrorType.SECURITY,
                severity=Severity.CRITICAL,
                languages=[Language.JAVA],
                pattern=r'(password|secret|apiKey)\s*=\s*"[^"]+"',
                tags=["security", "secrets"],
            ),
            
            # Performance rules
            AnalysisRule(
                rule_id="JAVA-PERF-001",
                name="string_concat_loop",
                description="String concatenation in loop. Use StringBuilder",
                error_type=ErrorType.PERFORMANCE,
                severity=Severity.MEDIUM,
                languages=[Language.JAVA],
                pattern=r'for\s*\([^)]*\)\s*\{[^}]*\+\s*=\s*"',
                tags=["performance", "string"],
            ),
            
            # Style rules
            AnalysisRule(
                rule_id="JAVA-STY-001",
                name="system_out",
                description="Use a logging framework instead of System.out",
                error_type=ErrorType.STYLE,
                severity=Severity.LOW,
                languages=[Language.JAVA],
                pattern=r'System\.(out|err)\.(print|println)',
                tags=["style", "logging"],
            ),
        ]
    
    async def analyze(self, file_path: str, content: str) -> List[CodeIssue]:
        """Analyze Java code."""
        issues = []
        lines = content.split('\n')
        
        for rule in self.rules:
            if not rule.pattern or not rule.enabled:
                continue
            
            try:
                pattern = re.compile(rule.pattern)
                
                for line_num, line in enumerate(lines, 1):
                    matches = pattern.finditer(line)
                    for match in matches:
                        issues.append(self._create_issue(
                            rule=rule,
                            location=CodeLocation(
                                file_path=file_path,
                                line_start=line_num,
                                line_end=line_num,
                            ),
                            code_snippet=line.strip(),
                        ))
            except re.error:
                continue
        
        return issues


# =============================================================================
# Code Analysis Engine
# =============================================================================

class CodeAnalysisEngine:
    """
    Main code analysis engine.
    
    Analyzes code for:
    - Syntax Errors
    - Runtime Errors
    - Logical Errors
    - Security Vulnerabilities
    - Performance Issues
    - Coding Standard Violations
    """
    
    def __init__(self):
        self.analyzers: Dict[Language, LanguageAnalyzer] = {
            Language.PYTHON: PythonAnalyzer(),
            Language.JAVASCRIPT: JavaScriptAnalyzer(Language.JAVASCRIPT),
            Language.TYPESCRIPT: JavaScriptAnalyzer(Language.TYPESCRIPT),
            Language.JAVA: JavaAnalyzer(),
        }
        
        # Metrics tracking
        self._analysis_count = 0
        self._issue_count = 0
        self._files_analyzed = 0
        
        # Ignore patterns
        self.ignore_patterns = {
            "node_modules",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "dist",
            "build",
            ".mypy_cache",
            ".pytest_cache",
            "*.min.js",
            "*.min.css",
        }
        
        logger.info("Code Analysis Engine initialized")
    
    def detect_language(self, file_path: str) -> Language:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return LANGUAGE_EXTENSIONS.get(ext, Language.UNKNOWN)
    
    def should_ignore(self, path: str) -> bool:
        """Check if path should be ignored."""
        path_parts = Path(path).parts
        for pattern in self.ignore_patterns:
            if pattern in path_parts:
                return True
            if pattern.startswith("*") and path.endswith(pattern[1:]):
                return True
        return False
    
    async def analyze_file(self, file_path: str) -> AnalysisResult:
        """
        Analyze a single file.
        
        Args:
            file_path: Path to file to analyze
            
        Returns:
            Analysis result with detected issues
        """
        started_at = datetime.now(timezone.utc)
        language = self.detect_language(file_path)
        
        result = AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            file_path=file_path,
            language=language,
            issues=[],
            metrics={},
            started_at=started_at,
        )
        
        if language == Language.UNKNOWN:
            result.completed_at = datetime.now(timezone.utc)
            return result
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Get appropriate analyzer
            analyzer = self.analyzers.get(language)
            if analyzer:
                issues = await analyzer.analyze(file_path, content)
                result.issues = issues
            
            # Calculate metrics
            lines = content.split('\n')
            result.metrics = {
                "lines_of_code": len(lines),
                "blank_lines": sum(1 for l in lines if not l.strip()),
                "comment_lines": self._count_comments(content, language),
                "file_size_bytes": len(content.encode('utf-8')),
            }
            
            self._files_analyzed += 1
            self._issue_count += len(result.issues)
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Error analyzing {file_path}: {e}")
        
        result.completed_at = datetime.now(timezone.utc)
        return result
    
    async def analyze_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_limit: int = 1000,
    ) -> ProjectAnalysis:
        """
        Analyze all files in a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to scan subdirectories
            file_limit: Maximum files to analyze
            
        Returns:
            Project analysis with all file results
        """
        started_at = datetime.now(timezone.utc)
        project_id = str(uuid.uuid4())
        
        logger.info(f"Starting project analysis: {directory}")
        
        # Collect files
        files_to_analyze = []
        dir_path = Path(directory)
        
        if recursive:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and not self.should_ignore(str(file_path)):
                    language = self.detect_language(str(file_path))
                    if language != Language.UNKNOWN:
                        files_to_analyze.append(str(file_path))
        else:
            for file_path in dir_path.iterdir():
                if file_path.is_file() and not self.should_ignore(str(file_path)):
                    language = self.detect_language(str(file_path))
                    if language != Language.UNKNOWN:
                        files_to_analyze.append(str(file_path))
        
        # Limit files
        files_to_analyze = files_to_analyze[:file_limit]
        
        logger.info(f"Found {len(files_to_analyze)} files to analyze")
        
        # Analyze files concurrently
        tasks = [self.analyze_file(f) for f in files_to_analyze]
        file_results = await asyncio.gather(*tasks)
        
        # Calculate summary
        total_issues = sum(r.issue_count for r in file_results)
        issues_by_type = {}
        issues_by_severity = {}
        
        for result in file_results:
            for issue in result.issues:
                # By type
                type_key = issue.error_type.value
                issues_by_type[type_key] = issues_by_type.get(type_key, 0) + 1
                
                # By severity
                sev_key = issue.severity.value
                issues_by_severity[sev_key] = issues_by_severity.get(sev_key, 0) + 1
        
        summary = {
            "total_files": len(file_results),
            "files_with_issues": sum(1 for r in file_results if r.issue_count > 0),
            "total_issues": total_issues,
            "issues_by_type": issues_by_type,
            "issues_by_severity": issues_by_severity,
            "total_lines": sum(r.metrics.get("lines_of_code", 0) for r in file_results),
        }
        
        self._analysis_count += 1
        
        logger.info(f"Analysis complete: {total_issues} issues in {len(file_results)} files")
        
        return ProjectAnalysis(
            project_id=project_id,
            project_path=directory,
            file_results=list(file_results),
            summary=summary,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
    
    def _count_comments(self, content: str, language: Language) -> int:
        """Count comment lines based on language."""
        count = 0
        lines = content.split('\n')
        
        if language == Language.PYTHON:
            in_docstring = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = not in_docstring
                    count += 1
                elif in_docstring or stripped.startswith('#'):
                    count += 1
        elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT, Language.JAVA]:
            in_block = False
            for line in lines:
                stripped = line.strip()
                if '/*' in stripped:
                    in_block = True
                    count += 1
                elif '*/' in stripped:
                    in_block = False
                    count += 1
                elif in_block or stripped.startswith('//'):
                    count += 1
        
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "analyses_performed": self._analysis_count,
            "files_analyzed": self._files_analyzed,
            "issues_found": self._issue_count,
            "supported_languages": list(self.analyzers.keys()),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ErrorType",
    "Severity",
    "Language",
    # Data classes
    "CodeLocation",
    "CodeIssue",
    "AnalysisResult",
    "ProjectAnalysis",
    "AnalysisRule",
    # Analyzers
    "LanguageAnalyzer",
    "PythonAnalyzer",
    "JavaScriptAnalyzer",
    "JavaAnalyzer",
    # Main engine
    "CodeAnalysisEngine",
]
