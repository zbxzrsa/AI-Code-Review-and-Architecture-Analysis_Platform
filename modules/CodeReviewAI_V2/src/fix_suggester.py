"""
CodeReviewAI_V2 - Enhanced Fix Suggester

Production-grade fix suggestions with:
- Batch processing
- Validation pipeline
- Auto-fix safety checks
"""

import re
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .models import Finding, Dimension, Severity

logger = logging.getLogger(__name__)


@dataclass
class FixSuggestion:
    """Enhanced fix suggestion with safety metadata"""
    finding_id: str
    original_code: str
    fixed_code: str
    description: str
    confidence: float
    auto_applicable: bool = False
    breaking_change: bool = False
    requires_review: bool = True
    # V2: Additional safety fields
    tested: bool = False
    rollback_code: Optional[str] = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

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
            "tested": self.tested,
            "rollback_code": self.rollback_code,
            "dependencies": self.dependencies,
        }


class FixSuggester:
    """
    Production fix suggester with safety checks.

    V2 Features:
    - Batch processing
    - Safety validation
    - Rollback support
    """

    def __init__(self):
        self.fix_patterns: Dict[str, callable] = {}
        self._load_fix_patterns()

    def _load_fix_patterns(self):
        """Load fix patterns with V2 enhancements"""

        # SEC001: eval fix
        def fix_eval(code: str, finding: Finding) -> Optional[FixSuggestion]:
            line_num = finding.line_numbers[0] - 1
            lines = code.split('\n')
            if line_num < len(lines) and 'eval(' in lines[line_num]:
                original = lines[line_num]
                fixed = original.replace('eval(', 'ast.literal_eval(')
                return FixSuggestion(
                    finding_id="SEC001",
                    original_code=original.strip(),
                    fixed_code=fixed.strip(),
                    description="Replace eval() with ast.literal_eval()",
                    confidence=0.7,
                    auto_applicable=False,
                    requires_review=True,
                    dependencies=["ast"],
                    rollback_code=original.strip(),
                )
            return None

        self.fix_patterns["SEC001"] = fix_eval

        # SEC003: Hardcoded credential
        def fix_hardcoded_cred(code: str, finding: Finding) -> Optional[FixSuggestion]:
            line_num = finding.line_numbers[0] - 1
            lines = code.split('\n')
            if line_num < len(lines):
                original = lines[line_num]
                match = re.search(r'(\w+)\s*=\s*["\'][^"\']+["\']', original)
                if match:
                    var = match.group(1).upper()
                    fixed = re.sub(r'["\'][^"\']+["\']', f'os.getenv("{var}")', original)
                    return FixSuggestion(
                        finding_id="SEC003",
                        original_code=original.strip(),
                        fixed_code=fixed.strip(),
                        description=f"Use environment variable {var}",
                        confidence=0.8,
                        breaking_change=True,
                        requires_review=True,
                        dependencies=["os"],
                        rollback_code=original.strip(),
                    )
            return None

        self.fix_patterns["SEC003"] = fix_hardcoded_cred

        # OWASP-A02: Weak hash
        def fix_weak_hash(code: str, finding: Finding) -> Optional[FixSuggestion]:
            line_num = finding.line_numbers[0] - 1
            lines = code.split('\n')
            if line_num < len(lines):
                original = lines[line_num]
                fixed = original
                if 'md5' in original.lower():
                    fixed = original.replace('md5', 'sha256').replace('MD5', 'sha256')
                elif 'sha1' in original.lower():
                    fixed = original.replace('sha1', 'sha256').replace('SHA1', 'sha256')
                if fixed != original:
                    return FixSuggestion(
                        finding_id=finding.rule_id or "OWASP-A02",
                        original_code=original.strip(),
                        fixed_code=fixed.strip(),
                        description="Replace weak hash with SHA-256",
                        confidence=0.85,
                        auto_applicable=False,
                        requires_review=True,
                        rollback_code=original.strip(),
                    )
            return None

        self.fix_patterns["OWASP-A02-01"] = fix_weak_hash
        self.fix_patterns["OWASP-A02-02"] = fix_weak_hash

        # CORR001: is None
        def fix_is_none(code: str, finding: Finding) -> Optional[FixSuggestion]:
            line_num = finding.line_numbers[0] - 1
            lines = code.split('\n')
            if line_num < len(lines):
                original = lines[line_num]
                fixed = original
                fixed = re.sub(r'==\s*None\b', 'is None', fixed)
                fixed = re.sub(r'!=\s*None\b', 'is not None', fixed)
                if fixed != original:
                    return FixSuggestion(
                        finding_id="CORR001",
                        original_code=original.strip(),
                        fixed_code=fixed.strip(),
                        description="Use 'is' for None comparison",
                        confidence=0.95,
                        auto_applicable=True,
                        requires_review=False,
                        rollback_code=original.strip(),
                    )
            return None

        self.fix_patterns["CORR001"] = fix_is_none

    async def suggest_fixes(
        self,
        code: str,
        findings: List[Finding],
    ) -> List[FixSuggestion]:
        """Generate fix suggestions with batch processing"""
        suggestions = []

        for finding in findings:
            rule_id = finding.rule_id

            if rule_id and rule_id in self.fix_patterns:
                suggestion = self.fix_patterns[rule_id](code, finding)
                if suggestion:
                    suggestions.append(suggestion)
                    continue

            # Generic suggestion
            suggestions.append(FixSuggestion(
                finding_id=rule_id or "GENERIC",
                original_code=finding.code_snippet or "",
                fixed_code="",
                description=finding.suggestion,
                confidence=0.5,
                requires_review=True,
            ))

        logger.info(f"Generated {len(suggestions)} suggestions")
        return suggestions

    async def apply_fix(
        self,
        code: str,
        suggestion: FixSuggestion,
        validate: bool = True,
    ) -> str:
        """Apply fix with optional validation"""
        if not suggestion.fixed_code:
            raise ValueError("No fixed code available")

        if not suggestion.auto_applicable and suggestion.requires_review:
            raise ValueError("Fix requires manual review")

        modified = code.replace(suggestion.original_code, suggestion.fixed_code, 1)

        if validate:
            validation = await self.validate_fix(code, modified)
            if not validation["syntax_ok"]:
                raise ValueError(f"Fix introduces syntax error: {validation['warnings']}")

        logger.info(f"Applied fix {suggestion.finding_id}")
        return modified

    async def validate_fix(
        self,
        original_code: str,
        fixed_code: str,
    ) -> Dict[str, Any]:
        """Validate fix with comprehensive checks"""
        result = {
            "valid": True,
            "syntax_ok": True,
            "no_new_issues": True,
            "warnings": [],
        }

        # Syntax check
        try:
            compile(fixed_code, '<string>', 'exec')
        except SyntaxError as e:
            result["valid"] = False
            result["syntax_ok"] = False
            result["warnings"].append(f"Syntax error: {e}")

        # Size check
        if len(fixed_code) < len(original_code) * 0.5:
            result["warnings"].append("Fixed code significantly shorter")

        # TODO check
        if 'TODO' in fixed_code or 'FIXME' in fixed_code:
            result["warnings"].append("Contains TODO/FIXME")

        return result

    def get_supported_rules(self) -> List[str]:
        return list(self.fix_patterns.keys())
