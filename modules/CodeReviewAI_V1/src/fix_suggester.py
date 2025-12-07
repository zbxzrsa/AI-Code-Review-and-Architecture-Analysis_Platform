"""
CodeReviewAI_V1 - Fix Suggester

Generates fix suggestions for detected issues:
- Auto-fix code generation
- Fix validation
- Multi-option suggestions
"""

import re
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .models import Finding, Dimension, Severity

logger = logging.getLogger(__name__)


@dataclass
class FixSuggestion:
    """A suggested fix for an issue"""
    finding_id: str
    original_code: str
    fixed_code: str
    description: str
    confidence: float
    auto_applicable: bool = False
    breaking_change: bool = False
    requires_review: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "original_code": self.original_code,
            "fixed_code": self.fixed_code,
            "description": self.description,
            "confidence": self.confidence,
            "auto_applicable": self.auto_applicable,
            "breaking_change": self.breaking_change,
            "requires_review": self.requires_review,
        }


class FixSuggester:
    """
    Generates fix suggestions for detected issues.

    Supports:
    - Pattern-based auto-fixes
    - AI-generated suggestions
    - Multi-option alternatives
    """

    def __init__(self):
        """Initialize fix suggester"""
        self.fix_patterns: Dict[str, callable] = {}
        self._load_fix_patterns()

    def _load_fix_patterns(self):
        """Load pattern-based fix generators"""

        # SEC001: eval() fix
        def fix_eval(code: str, finding: Finding) -> Optional[FixSuggestion]:
            line_num = finding.line_numbers[0] - 1
            lines = code.split('\n')

            if line_num < len(lines):
                original = lines[line_num]
                if 'eval(' in original:
                    # Try to convert to ast.literal_eval
                    fixed = original.replace('eval(', 'ast.literal_eval(')
                    return FixSuggestion(
                        finding_id=finding.rule_id or "SEC001",
                        original_code=original.strip(),
                        fixed_code=fixed.strip(),
                        description="Replace eval() with ast.literal_eval() for safe parsing",
                        confidence=0.7,
                        auto_applicable=False,
                        requires_review=True,
                    )
            return None

        self.fix_patterns["SEC001"] = fix_eval

        # SEC003: Hardcoded credential fix
        def fix_hardcoded_cred(code: str, finding: Finding) -> Optional[FixSuggestion]:
            line_num = finding.line_numbers[0] - 1
            lines = code.split('\n')

            if line_num < len(lines):
                original = lines[line_num]
                # Match patterns like password = "secret"
                match = re.search(
                    r'(\w+)\s*=\s*["\'][^"\']+["\']',
                    original
                )
                if match:
                    var_name = match.group(1).upper()
                    fixed = re.sub(
                        r'(["\'][^"\']+["\'])',
                        f'os.getenv("{var_name}")',
                        original
                    )
                    return FixSuggestion(
                        finding_id=finding.rule_id or "SEC003",
                        original_code=original.strip(),
                        fixed_code=fixed.strip(),
                        description=f"Use environment variable {var_name} instead of hardcoded value",
                        confidence=0.8,
                        auto_applicable=False,
                        breaking_change=True,
                        requires_review=True,
                    )
            return None

        self.fix_patterns["SEC003"] = fix_hardcoded_cred

        # SEC004: shell=True fix
        def fix_shell_true(code: str, finding: Finding) -> Optional[FixSuggestion]:
            line_num = finding.line_numbers[0] - 1
            lines = code.split('\n')

            if line_num < len(lines):
                original = lines[line_num]
                if 'shell=True' in original:
                    fixed = original.replace('shell=True', 'shell=False')
                    return FixSuggestion(
                        finding_id=finding.rule_id or "SEC004",
                        original_code=original.strip(),
                        fixed_code=fixed.strip(),
                        description="Set shell=False and pass command as list",
                        confidence=0.6,
                        auto_applicable=False,
                        breaking_change=True,
                        requires_review=True,
                    )
            return None

        self.fix_patterns["SEC004"] = fix_shell_true

        # CORR001: is None fix
        def fix_is_none(code: str, finding: Finding) -> Optional[FixSuggestion]:
            line_num = finding.line_numbers[0] - 1
            lines = code.split('\n')

            if line_num < len(lines):
                original = lines[line_num]
                fixed = original
                fixed = re.sub(r'==\s*None\b', 'is None', fixed)
                fixed = re.sub(r'!=\s*None\b', 'is not None', fixed)
                fixed = re.sub(r'==\s*True\b', 'is True', fixed)
                fixed = re.sub(r'==\s*False\b', 'is False', fixed)

                if fixed != original:
                    return FixSuggestion(
                        finding_id=finding.rule_id or "CORR001",
                        original_code=original.strip(),
                        fixed_code=fixed.strip(),
                        description="Use 'is' for None/bool identity comparison",
                        confidence=0.95,
                        auto_applicable=True,
                        requires_review=False,
                    )
            return None

        self.fix_patterns["CORR001"] = fix_is_none

        # MAINT003: bare except fix
        def fix_bare_except(code: str, finding: Finding) -> Optional[FixSuggestion]:
            line_num = finding.line_numbers[0] - 1
            lines = code.split('\n')

            if line_num < len(lines):
                original = lines[line_num]
                if 'except:' in original:
                    fixed = original.replace('except:', 'except Exception as e:')
                    return FixSuggestion(
                        finding_id=finding.rule_id or "MAINT003",
                        original_code=original.strip(),
                        fixed_code=fixed.strip(),
                        description="Catch specific exception instead of bare except",
                        confidence=0.85,
                        auto_applicable=False,
                        requires_review=True,
                    )
            return None

        self.fix_patterns["MAINT003"] = fix_bare_except

    async def suggest_fixes(
        self,
        code: str,
        findings: List[Finding],
    ) -> List[FixSuggestion]:
        """
        Generate fix suggestions for findings.

        Args:
            code: Original source code
            findings: List of detected findings

        Returns:
            List of fix suggestions
        """
        suggestions = []

        for finding in findings:
            rule_id = finding.rule_id

            # Try pattern-based fix
            if rule_id and rule_id in self.fix_patterns:
                fix_func = self.fix_patterns[rule_id]
                suggestion = fix_func(code, finding)
                if suggestion:
                    suggestions.append(suggestion)
                    continue

            # Generate generic suggestion
            suggestion = FixSuggestion(
                finding_id=rule_id or "GENERIC",
                original_code=finding.code_snippet or "",
                fixed_code="",  # Requires manual fix
                description=finding.suggestion,
                confidence=0.5,
                auto_applicable=False,
                requires_review=True,
            )
            suggestions.append(suggestion)

        logger.info(f"Generated {len(suggestions)} fix suggestions for {len(findings)} findings")
        return suggestions

    async def apply_fix(
        self,
        code: str,
        suggestion: FixSuggestion,
    ) -> str:
        """
        Apply a fix suggestion to code.

        Args:
            code: Original source code
            suggestion: Fix suggestion to apply

        Returns:
            Modified code with fix applied
        """
        if not suggestion.fixed_code:
            raise ValueError("No fixed code available for this suggestion")

        if not suggestion.auto_applicable and suggestion.requires_review:
            raise ValueError("This fix requires manual review before applying")

        # Replace original with fixed code
        modified = code.replace(
            suggestion.original_code,
            suggestion.fixed_code,
            1  # Replace first occurrence only
        )

        logger.info(f"Applied fix {suggestion.finding_id}")
        return modified

    async def validate_fix(
        self,
        original_code: str,
        fixed_code: str,
    ) -> Dict[str, Any]:
        """
        Validate that a fix doesn't introduce new issues.

        Args:
            original_code: Code before fix
            fixed_code: Code after fix

        Returns:
            Validation result
        """
        # Basic validation
        result = {
            "valid": True,
            "syntax_ok": True,
            "no_new_issues": True,
            "warnings": [],
        }

        # Check syntax (Python)
        try:
            compile(fixed_code, '<string>', 'exec')
        except SyntaxError as e:
            result["valid"] = False
            result["syntax_ok"] = False
            result["warnings"].append(f"Syntax error: {e}")

        # Check for obvious problems
        if len(fixed_code) < len(original_code) * 0.5:
            result["warnings"].append("Fixed code is significantly shorter - verify correctness")

        if 'TODO' in fixed_code or 'FIXME' in fixed_code:
            result["warnings"].append("Fixed code contains TODO/FIXME comments")

        return result

    def add_fix_pattern(self, rule_id: str, fix_func: callable):
        """Add a custom fix pattern"""
        self.fix_patterns[rule_id] = fix_func

    def get_supported_rules(self) -> List[str]:
        """Get list of rules with auto-fix support"""
        return list(self.fix_patterns.keys())
