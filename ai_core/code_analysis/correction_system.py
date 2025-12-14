"""
Intelligent Correction System

Provides multi-level code correction capabilities:
- Basic Mode: Detailed error explanations and step-by-step correction instructions
- Advanced Mode: Automatic correction with user authorization
- Teaching Mode: Interactive examples demonstrating error correction

Integrates with:
- Machine learning models for pattern recognition
- Version control for traceable, reversible corrections
- User feedback mechanism for continuous improvement
"""

import asyncio
import difflib
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

from .analysis_engine import (
    CodeIssue,
    CodeLocation,
    ErrorType,
    Severity,
    Language,
    AnalysisResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class CorrectionMode(str, Enum):
    """Correction mode types."""
    BASIC = "basic"       # Explanations and instructions
    ADVANCED = "advanced" # Automatic corrections with authorization
    TEACHING = "teaching" # Interactive learning mode


class CorrectionStatus(str, Enum):
    """Status of a correction."""
    PENDING = "pending"
    SUGGESTED = "suggested"
    APPROVED = "approved"
    APPLIED = "applied"
    VERIFIED = "verified"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class FeedbackType(str, Enum):
    """Types of user feedback."""
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CorrectionStep:
    """A single step in a correction process."""
    step_number: int
    action: str
    description: str
    before_code: str
    after_code: str
    explanation: str
    completed: bool = False


@dataclass
class CorrectionSuggestion:
    """A suggested correction for an issue."""
    suggestion_id: str
    issue_id: str
    issue: CodeIssue
    mode: CorrectionMode
    
    # Correction details
    original_code: str
    corrected_code: str
    diff: str
    
    # Explanation
    explanation: str
    steps: List[CorrectionStep]
    
    # Metadata
    confidence: float
    risk_level: str  # low, medium, high
    reversible: bool = True
    requires_review: bool = False
    
    # Status tracking
    status: CorrectionStatus = CorrectionStatus.SUGGESTED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied_at: Optional[datetime] = None
    
    # Version control
    backup_path: Optional[str] = None
    commit_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "issue_id": self.issue_id,
            "mode": self.mode.value,
            "original_code": self.original_code,
            "corrected_code": self.corrected_code,
            "diff": self.diff,
            "explanation": self.explanation,
            "steps": [
                {
                    "step": s.step_number,
                    "action": s.action,
                    "description": s.description,
                }
                for s in self.steps
            ],
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "status": self.status.value,
            "reversible": self.reversible,
        }


@dataclass
class CorrectionResult:
    """Result of applying a correction."""
    result_id: str
    suggestion_id: str
    success: bool
    message: str
    
    # Changes made
    files_modified: List[str]
    lines_changed: int
    
    # Verification
    verified: bool = False
    tests_passed: Optional[bool] = None
    
    # Rollback info
    rollback_available: bool = True
    backup_created: bool = False
    
    applied_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserFeedback:
    """User feedback on a correction."""
    feedback_id: str
    suggestion_id: str
    feedback_type: FeedbackType
    rating: int  # 1-5
    comment: str
    user_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TeachingExample:
    """An interactive teaching example."""
    example_id: str
    title: str
    description: str
    error_type: ErrorType
    language: Language
    
    # The problematic code
    buggy_code: str
    
    # The correct code
    fixed_code: str
    
    # Step-by-step explanation
    steps: List[CorrectionStep]
    
    # Interactive elements
    hints: List[str]
    quiz_questions: List[Dict[str, Any]]
    
    difficulty: str  # beginner, intermediate, advanced
    estimated_time_minutes: int


# =============================================================================
# Correction Strategies
# =============================================================================

class CorrectionStrategy(ABC):
    """Abstract base class for correction strategies."""
    
    @abstractmethod
    def can_handle(self, issue: CodeIssue) -> bool:
        """Check if this strategy can handle the issue."""
        pass
    
    @abstractmethod
    def generate_correction(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> Optional[CorrectionSuggestion]:
        """Generate a correction for the issue."""
        pass


class SecurityCorrectionStrategy(CorrectionStrategy):
    """Correction strategy for security issues."""
    
    def can_handle(self, issue: CodeIssue) -> bool:
        return issue.error_type == ErrorType.SECURITY
    
    def generate_correction(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> Optional[CorrectionSuggestion]:
        """Generate security-focused correction."""
        
        # Determine correction based on rule ID
        correction_map = {
            "PY-SEC-001": self._fix_hardcoded_secret_python,
            "PY-SEC-002": self._fix_eval_usage,
            "PY-SEC-003": self._fix_exec_usage,
            "PY-SEC-004": self._fix_sql_injection,
            "JS-SEC-001": self._fix_eval_usage_js,
            "JS-SEC-002": self._fix_innerhtml,
            "JS-SEC-004": self._fix_hardcoded_secret_js,
        }
        
        fixer = correction_map.get(issue.rule_id)
        if fixer:
            return fixer(issue, code, mode)
        
        return self._generate_generic_security_fix(issue, code, mode)
    
    def _fix_hardcoded_secret_python(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix hardcoded secrets in Python."""
        lines = code.split('\n')
        line_idx = issue.location.line_start - 1
        original_line = lines[line_idx] if line_idx < len(lines) else ""
        
        # Extract variable name
        match = re.match(r'\s*(\w+)\s*=\s*["\'].*["\']', original_line)
        var_name = match.group(1) if match else "secret"
        
        # Generate fix
        indent = len(original_line) - len(original_line.lstrip())
        fixed_line = f'{" " * indent}{var_name} = os.environ.get("{var_name.upper()}")'
        
        # Create steps
        steps = [
            CorrectionStep(
                step_number=1,
                action="identify",
                description="Identify the hardcoded secret",
                before_code=original_line.strip(),
                after_code="",
                explanation="The secret value is directly in the source code, making it visible in version control.",
            ),
            CorrectionStep(
                step_number=2,
                action="replace",
                description="Replace with environment variable",
                before_code=original_line.strip(),
                after_code=fixed_line.strip(),
                explanation="Environment variables keep secrets out of code and allow different values per environment.",
            ),
            CorrectionStep(
                step_number=3,
                action="configure",
                description="Set environment variable",
                before_code="",
                after_code=f'export {var_name.upper()}="your_secret_value"',
                explanation="Set the environment variable in your shell or .env file.",
            ),
        ]
        
        # Generate diff
        lines_fixed = lines.copy()
        lines_fixed[line_idx] = fixed_line
        diff = self._generate_diff(lines, lines_fixed, issue.location.file_path)
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=original_line,
            corrected_code=fixed_line,
            diff=diff,
            explanation=(
                "Hardcoded secrets are a critical security vulnerability. "
                "Anyone with access to the code can see the secret value. "
                "Using environment variables keeps secrets separate from code."
            ),
            steps=steps,
            confidence=0.95,
            risk_level="low",
            reversible=True,
            requires_review=True,
        )
    
    def _fix_eval_usage(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix eval() usage in Python."""
        lines = code.split('\n')
        line_idx = issue.location.line_start - 1
        original_line = lines[line_idx] if line_idx < len(lines) else ""
        
        steps = [
            CorrectionStep(
                step_number=1,
                action="understand",
                description="Understand why eval() is dangerous",
                before_code=original_line.strip(),
                after_code="",
                explanation="eval() executes arbitrary Python code, allowing code injection attacks.",
            ),
            CorrectionStep(
                step_number=2,
                action="alternative",
                description="Use safer alternatives",
                before_code="result = eval(user_input)",
                after_code="result = ast.literal_eval(user_input)  # For literals only",
                explanation="ast.literal_eval() only evaluates Python literals, not arbitrary code.",
            ),
            CorrectionStep(
                step_number=3,
                action="validate",
                description="Add input validation",
                before_code="",
                after_code="if not validate_input(user_input): raise ValueError('Invalid input')",
                explanation="Always validate and sanitize input before processing.",
            ),
        ]
        
        # Suggest replacement
        fixed_line = original_line.replace("eval(", "ast.literal_eval(")
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=original_line,
            corrected_code=fixed_line,
            diff=f"- {original_line}\n+ {fixed_line}",
            explanation=(
                "eval() is dangerous because it executes any Python code passed to it. "
                "If user input reaches eval(), attackers can execute malicious code. "
                "Use ast.literal_eval() for safe evaluation of Python literals, or "
                "implement a whitelist of allowed operations."
            ),
            steps=steps,
            confidence=0.85,
            risk_level="medium",
            reversible=True,
            requires_review=True,
        )
    
    def _fix_exec_usage(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix exec() usage."""
        return self._fix_eval_usage(issue, code, mode)  # Similar approach
    
    def _fix_sql_injection(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix SQL injection vulnerability."""
        lines = code.split('\n')
        line_idx = issue.location.line_start - 1
        original_line = lines[line_idx] if line_idx < len(lines) else ""
        
        steps = [
            CorrectionStep(
                step_number=1,
                action="understand",
                description="Understand SQL injection",
                before_code='cursor.execute("SELECT * FROM users WHERE id=" + user_id)',
                after_code="",
                explanation="String concatenation in SQL allows attackers to inject malicious SQL.",
            ),
            CorrectionStep(
                step_number=2,
                action="fix",
                description="Use parameterized queries",
                before_code='cursor.execute("SELECT * FROM users WHERE id=" + user_id)',
                after_code='cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
                explanation="Parameterized queries separate SQL from data, preventing injection.",
            ),
            CorrectionStep(
                step_number=3,
                action="validate",
                description="Add input validation",
                before_code="",
                after_code="user_id = int(user_id)  # Validate type",
                explanation="Always validate input type and format before use.",
            ),
        ]
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=original_line,
            corrected_code="# Use parameterized query: cursor.execute(sql, params)",
            diff="",
            explanation=(
                "SQL injection is one of the most dangerous vulnerabilities. "
                "Always use parameterized queries or prepared statements. "
                "Never concatenate user input into SQL strings."
            ),
            steps=steps,
            confidence=0.90,
            risk_level="high",
            reversible=True,
            requires_review=True,
        )
    
    def _fix_eval_usage_js(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix eval() usage in JavaScript."""
        lines = code.split('\n')
        line_idx = issue.location.line_start - 1
        original_line = lines[line_idx] if line_idx < len(lines) else ""
        
        steps = [
            CorrectionStep(
                step_number=1,
                action="understand",
                description="Understand eval() risks",
                before_code=original_line.strip(),
                after_code="",
                explanation="eval() executes any JavaScript, enabling XSS and code injection.",
            ),
            CorrectionStep(
                step_number=2,
                action="alternative",
                description="Use JSON.parse for data",
                before_code="const data = eval(jsonString)",
                after_code="const data = JSON.parse(jsonString)",
                explanation="JSON.parse safely parses JSON without code execution.",
            ),
        ]
        
        fixed_line = original_line.replace("eval(", "JSON.parse(")
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=original_line,
            corrected_code=fixed_line,
            diff=f"- {original_line}\n+ {fixed_line}",
            explanation="Use JSON.parse() for JSON data. For other cases, reconsider the design.",
            steps=steps,
            confidence=0.80,
            risk_level="high",
            reversible=True,
            requires_review=True,
        )
    
    def _fix_innerhtml(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix innerHTML usage."""
        lines = code.split('\n')
        line_idx = issue.location.line_start - 1
        original_line = lines[line_idx] if line_idx < len(lines) else ""
        
        fixed_line = original_line.replace(".innerHTML", ".textContent")
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=original_line,
            corrected_code=fixed_line,
            diff=f"- {original_line}\n+ {fixed_line}",
            explanation="Use textContent for text, or sanitize HTML input before using innerHTML.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="replace",
                    description="Use textContent instead",
                    before_code=".innerHTML = userText",
                    after_code=".textContent = userText",
                    explanation="textContent doesn't parse HTML, preventing XSS.",
                ),
            ],
            confidence=0.90,
            risk_level="low",
            reversible=True,
        )
    
    def _fix_hardcoded_secret_js(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix hardcoded secrets in JavaScript."""
        lines = code.split('\n')
        line_idx = issue.location.line_start - 1
        original_line = lines[line_idx] if line_idx < len(lines) else ""
        
        match = re.match(r'\s*(const|let|var)?\s*(\w+)\s*[=:]\s*["\'].*["\']', original_line)
        var_name = match.group(2) if match else "secret"
        
        fixed_line = f'const {var_name} = process.env.{var_name.upper()};'
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=original_line,
            corrected_code=fixed_line,
            diff=f"- {original_line}\n+ {fixed_line}",
            explanation="Use environment variables for secrets. Never commit secrets to code.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="replace",
                    description="Use environment variable",
                    before_code=original_line.strip(),
                    after_code=fixed_line,
                    explanation="Access secrets via process.env in Node.js.",
                ),
            ],
            confidence=0.95,
            risk_level="low",
            reversible=True,
        )
    
    def _generate_generic_security_fix(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Generate generic security fix suggestion."""
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=issue.code_snippet,
            corrected_code="# Review and fix security issue manually",
            diff="",
            explanation=f"Security issue detected: {issue.message}. Manual review recommended.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="review",
                    description="Review the security issue",
                    before_code=issue.code_snippet,
                    after_code="",
                    explanation=issue.suggestion or "Consult security best practices.",
                ),
            ],
            confidence=0.5,
            risk_level="high",
            reversible=True,
            requires_review=True,
        )
    
    def _generate_diff(
        self,
        original_lines: List[str],
        fixed_lines: List[str],
        filename: str,
    ) -> str:
        """Generate unified diff."""
        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
        return "\n".join(diff)


class StyleCorrectionStrategy(CorrectionStrategy):
    """Correction strategy for style issues."""
    
    def can_handle(self, issue: CodeIssue) -> bool:
        return issue.error_type == ErrorType.STYLE
    
    def generate_correction(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> Optional[CorrectionSuggestion]:
        """Generate style correction."""
        
        if issue.rule_id == "PY-STY-001":  # Line too long
            return self._fix_line_too_long(issue, code, mode)
        elif issue.rule_id == "PY-STY-002":  # Missing docstring
            return self._fix_missing_docstring(issue, code, mode)
        elif issue.rule_id == "JS-STY-001":  # var usage
            return self._fix_var_usage(issue, code, mode)
        elif issue.rule_id == "JS-STY-002":  # console.log
            return self._fix_console_log(issue, code, mode)
        
        return None
    
    def _fix_line_too_long(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix line too long."""
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=issue.code_snippet,
            corrected_code="# Break into multiple lines",
            diff="",
            explanation="Long lines reduce readability. Break into logical segments.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="break",
                    description="Break line at logical points",
                    before_code=issue.code_snippet[:50] + "...",
                    after_code="line1 = (\n    part1 +\n    part2\n)",
                    explanation="Use parentheses or backslash for line continuation.",
                ),
            ],
            confidence=0.90,
            risk_level="low",
            reversible=True,
        )
    
    def _fix_missing_docstring(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix missing docstring."""
        # Extract function name
        match = re.match(r'def\s+(\w+)', issue.code_snippet)
        func_name = match.group(1) if match else "function"
        
        docstring = f'    """{func_name.replace("_", " ").title()}."""'
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=issue.code_snippet,
            corrected_code=f"{issue.code_snippet}\n{docstring}",
            diff=f"+ {docstring}",
            explanation="Docstrings document the purpose of functions for maintainability.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="add",
                    description="Add docstring after function definition",
                    before_code=issue.code_snippet,
                    after_code=f'{issue.code_snippet}\n    """Description."""',
                    explanation="Use triple quotes for docstrings. Describe what, not how.",
                ),
            ],
            confidence=0.95,
            risk_level="low",
            reversible=True,
        )
    
    def _fix_var_usage(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix var usage in JavaScript."""
        fixed = issue.code_snippet.replace("var ", "const ")
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=issue.code_snippet,
            corrected_code=fixed,
            diff=f"- {issue.code_snippet}\n+ {fixed}",
            explanation="Use 'const' for values that don't change, 'let' for those that do.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="replace",
                    description="Replace var with const or let",
                    before_code="var x = 1;",
                    after_code="const x = 1;  // or let if reassigned",
                    explanation="const and let have block scope and prevent accidental globals.",
                ),
            ],
            confidence=0.95,
            risk_level="low",
            reversible=True,
        )
    
    def _fix_console_log(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix console.log usage."""
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=issue.code_snippet,
            corrected_code="// Remove or use proper logging: logger.debug(...)",
            diff=f"- {issue.code_snippet}",
            explanation="console.log statements should be removed or replaced with a logging library.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="remove_or_replace",
                    description="Remove console.log or use logger",
                    before_code="console.log('debug', data);",
                    after_code="logger.debug('debug', data);",
                    explanation="Logging libraries allow control over log levels and output.",
                ),
            ],
            confidence=0.90,
            risk_level="low",
            reversible=True,
        )


class LogicalCorrectionStrategy(CorrectionStrategy):
    """Correction strategy for logical errors."""
    
    def can_handle(self, issue: CodeIssue) -> bool:
        return issue.error_type == ErrorType.LOGICAL
    
    def generate_correction(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> Optional[CorrectionSuggestion]:
        """Generate logical error correction."""
        
        if issue.rule_id == "PY-LOG-002":  # == None
            return self._fix_none_comparison(issue, code, mode)
        elif issue.rule_id == "PY-LOG-003":  # Mutable default
            return self._fix_mutable_default(issue, code, mode)
        elif issue.rule_id == "JS-LOG-001":  # == vs ===
            return self._fix_loose_equality(issue, code, mode)
        
        return None
    
    def _fix_none_comparison(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix == None comparison."""
        fixed = issue.code_snippet.replace("== None", "is None").replace("!= None", "is not None")
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=issue.code_snippet,
            corrected_code=fixed,
            diff=f"- {issue.code_snippet}\n+ {fixed}",
            explanation="Use 'is None' for identity comparison with None, not equality.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="replace",
                    description="Use identity comparison",
                    before_code="if x == None:",
                    after_code="if x is None:",
                    explanation="'is' checks identity (same object), '==' checks equality.",
                ),
            ],
            confidence=0.98,
            risk_level="low",
            reversible=True,
        )
    
    def _fix_mutable_default(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix mutable default argument."""
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=issue.code_snippet,
            corrected_code="def func(arg=None):\n    if arg is None:\n        arg = []",
            diff="",
            explanation="Mutable defaults are shared between calls. Use None and initialize inside.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="change_default",
                    description="Use None as default",
                    before_code="def func(items=[]):",
                    after_code="def func(items=None):",
                    explanation="None is immutable and signals 'no value provided'.",
                ),
                CorrectionStep(
                    step_number=2,
                    action="initialize",
                    description="Initialize inside function",
                    before_code="",
                    after_code="    if items is None:\n        items = []",
                    explanation="Create new list for each call when no value provided.",
                ),
            ],
            confidence=0.95,
            risk_level="low",
            reversible=True,
        )
    
    def _fix_loose_equality(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode,
    ) -> CorrectionSuggestion:
        """Fix == vs === in JavaScript."""
        fixed = issue.code_snippet.replace(" == ", " === ").replace(" != ", " !== ")
        
        return CorrectionSuggestion(
            suggestion_id=str(uuid.uuid4()),
            issue_id=issue.issue_id,
            issue=issue,
            mode=mode,
            original_code=issue.code_snippet,
            corrected_code=fixed,
            diff=f"- {issue.code_snippet}\n+ {fixed}",
            explanation="Strict equality (===) avoids type coercion bugs.",
            steps=[
                CorrectionStep(
                    step_number=1,
                    action="replace",
                    description="Use strict equality",
                    before_code="if (x == 0)",
                    after_code="if (x === 0)",
                    explanation="=== compares value and type, == may coerce unexpectedly.",
                ),
            ],
            confidence=0.95,
            risk_level="low",
            reversible=True,
        )


# =============================================================================
# Intelligent Correction System
# =============================================================================

class IntelligentCorrectionSystem:
    """
    Main intelligent correction system.
    
    Provides:
    - Basic Mode: Explanations and step-by-step instructions
    - Advanced Mode: Automatic corrections with authorization
    - Teaching Mode: Interactive learning examples
    
    Features:
    - Machine learning pattern recognition integration
    - Version control integration for traceability
    - User feedback mechanism for continuous improvement
    """
    
    def __init__(
        self,
        backup_dir: str = ".code_corrections/backups",
        enable_ml: bool = True,
    ):
        self.backup_dir = Path(backup_dir)
        self.enable_ml = enable_ml
        
        # Correction strategies
        self.strategies: List[CorrectionStrategy] = [
            SecurityCorrectionStrategy(),
            StyleCorrectionStrategy(),
            LogicalCorrectionStrategy(),
        ]
        
        # Storage
        self._suggestions: Dict[str, CorrectionSuggestion] = {}
        self._results: Dict[str, CorrectionResult] = {}
        self._feedback: Dict[str, List[UserFeedback]] = {}
        self._teaching_examples: Dict[str, TeachingExample] = {}
        
        # Statistics
        self._corrections_suggested = 0
        self._corrections_applied = 0
        self._corrections_successful = 0
        self._user_satisfaction_sum = 0
        self._feedback_count = 0
        
        # Initialize teaching examples
        self._initialize_teaching_examples()
        
        logger.info("Intelligent Correction System initialized")
    
    def _initialize_teaching_examples(self) -> None:
        """Initialize built-in teaching examples."""
        examples = [
            TeachingExample(
                example_id="teach-sec-001",
                title="SQL Injection Prevention",
                description="Learn how to prevent SQL injection attacks",
                error_type=ErrorType.SECURITY,
                language=Language.PYTHON,
                buggy_code='''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    return cursor.fetchone()
''',
                fixed_code='''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()
''',
                steps=[
                    CorrectionStep(
                        step_number=1,
                        action="identify",
                        description="Identify string concatenation in SQL",
                        before_code='"SELECT * FROM users WHERE id = " + user_id',
                        after_code="",
                        explanation="String concatenation allows malicious input to modify the query.",
                    ),
                    CorrectionStep(
                        step_number=2,
                        action="fix",
                        description="Use parameter placeholder",
                        before_code="",
                        after_code='"SELECT * FROM users WHERE id = ?"',
                        explanation="? is a placeholder that the database driver handles safely.",
                    ),
                    CorrectionStep(
                        step_number=3,
                        action="pass_params",
                        description="Pass parameters separately",
                        before_code="cursor.execute(query)",
                        after_code="cursor.execute(query, (user_id,))",
                        explanation="Parameters are escaped by the driver, preventing injection.",
                    ),
                ],
                hints=[
                    "Look for string concatenation with user input",
                    "Database drivers have built-in protection - use it!",
                    "Parameterized queries separate code from data",
                ],
                quiz_questions=[
                    {
                        "question": "Why is string concatenation dangerous in SQL?",
                        "options": [
                            "It's slower",
                            "Attackers can inject SQL code",
                            "It increases database size",
                            "It causes syntax errors",
                        ],
                        "correct": 1,
                    },
                ],
                difficulty="intermediate",
                estimated_time_minutes=10,
            ),
            TeachingExample(
                example_id="teach-log-001",
                title="Mutable Default Arguments",
                description="Understand the Python mutable default trap",
                error_type=ErrorType.LOGICAL,
                language=Language.PYTHON,
                buggy_code='''
def add_item(item, items=[]):
    items.append(item)
    return items
''',
                fixed_code='''
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
''',
                steps=[
                    CorrectionStep(
                        step_number=1,
                        action="understand",
                        description="Understand the problem",
                        before_code="def add_item(item, items=[]):",
                        after_code="",
                        explanation="Default arguments are evaluated once at function definition, not each call.",
                    ),
                    CorrectionStep(
                        step_number=2,
                        action="fix",
                        description="Use None as default",
                        before_code="items=[]",
                        after_code="items=None",
                        explanation="None is immutable and signals 'no value provided'.",
                    ),
                    CorrectionStep(
                        step_number=3,
                        action="initialize",
                        description="Initialize in function body",
                        before_code="",
                        after_code="if items is None:\n    items = []",
                        explanation="Create a new list each time when no argument is provided.",
                    ),
                ],
                hints=[
                    "Run the function multiple times - what happens?",
                    "Default values are shared between all calls",
                    "None is a safe sentinel value",
                ],
                quiz_questions=[
                    {
                        "question": "When are default arguments evaluated?",
                        "options": [
                            "Each time the function is called",
                            "Once when the module is imported",
                            "Once when the function is defined",
                            "Never - they're just templates",
                        ],
                        "correct": 2,
                    },
                ],
                difficulty="intermediate",
                estimated_time_minutes=8,
            ),
        ]
        
        for example in examples:
            self._teaching_examples[example.example_id] = example
    
    # =========================================================================
    # Core Methods
    # =========================================================================
    
    async def suggest_correction(
        self,
        issue: CodeIssue,
        code: str,
        mode: CorrectionMode = CorrectionMode.BASIC,
    ) -> Optional[CorrectionSuggestion]:
        """
        Generate a correction suggestion for an issue.
        
        Args:
            issue: The code issue to fix
            code: The full source code
            mode: Correction mode (basic, advanced, teaching)
            
        Returns:
            Correction suggestion or None if no fix available
        """
        for strategy in self.strategies:
            if strategy.can_handle(issue):
                suggestion = strategy.generate_correction(issue, code, mode)
                if suggestion:
                    self._suggestions[suggestion.suggestion_id] = suggestion
                    self._corrections_suggested += 1
                    
                    logger.info(
                        f"Generated {mode.value} correction for {issue.rule_id}: "
                        f"confidence={suggestion.confidence:.2f}"
                    )
                    
                    return suggestion
        
        return None
    
    async def suggest_corrections_for_file(
        self,
        analysis_result: AnalysisResult,
        code: str,
        mode: CorrectionMode = CorrectionMode.BASIC,
    ) -> List[CorrectionSuggestion]:
        """
        Generate correction suggestions for all issues in a file.
        
        Args:
            analysis_result: Analysis result with issues
            code: The source code
            mode: Correction mode
            
        Returns:
            List of correction suggestions
        """
        suggestions = []
        
        for issue in analysis_result.issues:
            suggestion = await self.suggest_correction(issue, code, mode)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    async def apply_correction(
        self,
        suggestion_id: str,
        file_path: str,
        authorized: bool = False,
    ) -> CorrectionResult:
        """
        Apply a correction to a file.
        
        Args:
            suggestion_id: ID of the suggestion to apply
            file_path: Path to the file
            authorized: User authorization for the change
            
        Returns:
            Result of applying the correction
        """
        suggestion = self._suggestions.get(suggestion_id)
        if not suggestion:
            return CorrectionResult(
                result_id=str(uuid.uuid4()),
                suggestion_id=suggestion_id,
                success=False,
                message="Suggestion not found",
                files_modified=[],
                lines_changed=0,
            )
        
        # Check authorization for advanced mode
        if suggestion.mode == CorrectionMode.ADVANCED and not authorized:
            return CorrectionResult(
                result_id=str(uuid.uuid4()),
                suggestion_id=suggestion_id,
                success=False,
                message="User authorization required for advanced corrections",
                files_modified=[],
                lines_changed=0,
            )
        
        # Check if high-risk correction requires review
        if suggestion.requires_review and not authorized:
            return CorrectionResult(
                result_id=str(uuid.uuid4()),
                suggestion_id=suggestion_id,
                success=False,
                message="This correction requires manual review before applying",
                files_modified=[],
                lines_changed=0,
            )
        
        try:
            # Create backup
            backup_path = await self._create_backup(file_path)
            suggestion.backup_path = backup_path
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply correction
            lines = content.split('\n')
            line_idx = suggestion.issue.location.line_start - 1
            
            if 0 <= line_idx < len(lines):
                # Replace the problematic line
                original_line = lines[line_idx]
                
                # If we have a direct fix, apply it
                if suggestion.corrected_code and suggestion.corrected_code != suggestion.original_code:
                    lines[line_idx] = suggestion.corrected_code
                    
                    # Write back
                    new_content = '\n'.join(lines)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    suggestion.status = CorrectionStatus.APPLIED
                    suggestion.applied_at = datetime.now(timezone.utc)
                    
                    self._corrections_applied += 1
                    
                    result = CorrectionResult(
                        result_id=str(uuid.uuid4()),
                        suggestion_id=suggestion_id,
                        success=True,
                        message="Correction applied successfully",
                        files_modified=[file_path],
                        lines_changed=1,
                        backup_created=True,
                    )
                    
                    self._results[result.result_id] = result
                    self._corrections_successful += 1
                    
                    logger.info(f"Applied correction to {file_path}")
                    
                    return result
            
            return CorrectionResult(
                result_id=str(uuid.uuid4()),
                suggestion_id=suggestion_id,
                success=False,
                message="Unable to apply correction - line not found",
                files_modified=[],
                lines_changed=0,
            )
            
        except Exception as e:
            logger.error(f"Error applying correction: {e}")
            return CorrectionResult(
                result_id=str(uuid.uuid4()),
                suggestion_id=suggestion_id,
                success=False,
                message=f"Error: {str(e)}",
                files_modified=[],
                lines_changed=0,
            )
    
    async def rollback_correction(self, result_id: str) -> bool:
        """
        Rollback a previously applied correction.
        
        Args:
            result_id: ID of the correction result
            
        Returns:
            True if rollback successful
        """
        result = self._results.get(result_id)
        if not result or not result.rollback_available:
            return False
        
        suggestion = self._suggestions.get(result.suggestion_id)
        if not suggestion or not suggestion.backup_path:
            return False
        
        try:
            # Restore from backup
            backup_path = Path(suggestion.backup_path)
            if backup_path.exists():
                file_path = result.files_modified[0] if result.files_modified else None
                if file_path:
                    with open(backup_path, 'r', encoding='utf-8') as f:
                        backup_content = f.read()
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(backup_content)
                    
                    suggestion.status = CorrectionStatus.ROLLED_BACK
                    result.rollback_available = False
                    
                    logger.info(f"Rolled back correction on {file_path}")
                    return True
                    
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
        
        return False
    
    async def _create_backup(self, file_path: str) -> str:
        """Create backup of file before modification."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(file_path).name
        backup_name = f"{filename}.{timestamp}.bak"
        backup_path = self.backup_dir / backup_name
        
        with open(file_path, 'r', encoding='utf-8') as src:
            content = src.read()
        with open(backup_path, 'w', encoding='utf-8') as dst:
            dst.write(content)
        
        return str(backup_path)
    
    # =========================================================================
    # Teaching Mode
    # =========================================================================
    
    def get_teaching_example(self, example_id: str) -> Optional[TeachingExample]:
        """Get a teaching example by ID."""
        return self._teaching_examples.get(example_id)
    
    def list_teaching_examples(
        self,
        error_type: Optional[ErrorType] = None,
        language: Optional[Language] = None,
        difficulty: Optional[str] = None,
    ) -> List[TeachingExample]:
        """List teaching examples with optional filters."""
        examples = list(self._teaching_examples.values())
        
        if error_type:
            examples = [e for e in examples if e.error_type == error_type]
        if language:
            examples = [e for e in examples if e.language == language]
        if difficulty:
            examples = [e for e in examples if e.difficulty == difficulty]
        
        return examples
    
    def generate_teaching_content(
        self,
        issue: CodeIssue,
    ) -> Dict[str, Any]:
        """Generate teaching content for an issue."""
        # Find matching teaching example
        matching_examples = [
            e for e in self._teaching_examples.values()
            if e.error_type == issue.error_type
        ]
        
        content = {
            "issue": issue.to_dict(),
            "explanation": issue.suggestion or f"Issue: {issue.message}",
            "learning_objectives": [],
            "examples": [],
            "next_steps": [],
        }
        
        if matching_examples:
            example = matching_examples[0]
            content["examples"].append({
                "id": example.example_id,
                "title": example.title,
                "buggy_code": example.buggy_code,
                "fixed_code": example.fixed_code,
                "hints": example.hints,
            })
            content["learning_objectives"] = [
                f"Understand {issue.error_type.value} errors",
                f"Learn to identify {issue.rule_id} pattern",
                "Apply the fix correctly",
            ]
        
        content["next_steps"] = [
            "Review the issue explanation",
            "Study the example code",
            "Apply the fix to your code",
            "Run tests to verify",
        ]
        
        return content
    
    # =========================================================================
    # Feedback System
    # =========================================================================
    
    async def submit_feedback(
        self,
        suggestion_id: str,
        feedback_type: FeedbackType,
        rating: int,
        comment: str,
        user_id: str,
    ) -> UserFeedback:
        """
        Submit feedback on a correction suggestion.
        
        This helps improve the system's accuracy over time.
        """
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            suggestion_id=suggestion_id,
            feedback_type=feedback_type,
            rating=max(1, min(5, rating)),  # Clamp 1-5
            comment=comment,
            user_id=user_id,
        )
        
        if suggestion_id not in self._feedback:
            self._feedback[suggestion_id] = []
        self._feedback[suggestion_id].append(feedback)
        
        # Update statistics
        self._user_satisfaction_sum += rating
        self._feedback_count += 1
        
        # Learn from feedback (for ML integration)
        await self._process_feedback(feedback)
        
        logger.info(f"Received feedback for {suggestion_id}: {feedback_type.value}, rating={rating}")
        
        return feedback
    
    async def _process_feedback(self, feedback: UserFeedback) -> None:
        """Process feedback for learning (ML integration point)."""
        # This would integrate with ML models to improve accuracy
        if feedback.feedback_type == FeedbackType.INCORRECT:
            # Flag for review and model retraining
            logger.warning(f"Incorrect suggestion flagged: {feedback.suggestion_id}")
            # TODO: Add to training data for model improvement
        elif feedback.feedback_type == FeedbackType.HELPFUL:
            # Reinforce positive examples
            pass
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        avg_satisfaction = (
            self._user_satisfaction_sum / self._feedback_count
            if self._feedback_count > 0 else 0
        )
        
        success_rate = (
            self._corrections_successful / self._corrections_applied
            if self._corrections_applied > 0 else 0
        )
        
        return {
            "corrections_suggested": self._corrections_suggested,
            "corrections_applied": self._corrections_applied,
            "corrections_successful": self._corrections_successful,
            "success_rate": success_rate,
            "feedback_count": self._feedback_count,
            "average_satisfaction": avg_satisfaction,
            "teaching_examples_available": len(self._teaching_examples),
            "strategies_active": len(self.strategies),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "CorrectionMode",
    "CorrectionStatus",
    "FeedbackType",
    # Data classes
    "CorrectionStep",
    "CorrectionSuggestion",
    "CorrectionResult",
    "UserFeedback",
    "TeachingExample",
    # Strategies
    "CorrectionStrategy",
    "SecurityCorrectionStrategy",
    "StyleCorrectionStrategy",
    "LogicalCorrectionStrategy",
    # Main system
    "IntelligentCorrectionSystem",
]
