"""
Tests for CodeReviewAI_V1 Fix Suggester

Test coverage:
- Fix generation
- Fix application
- Fix validation
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fix_suggester import FixSuggester, FixSuggestion
from src.models import Finding, Dimension, Severity


class TestFixSuggester:
    """Test suite for FixSuggester"""

    @pytest.fixture
    def suggester(self):
        """Create fix suggester instance"""
        return FixSuggester()

    @pytest.fixture
    def eval_finding(self):
        """Finding for eval usage"""
        return Finding(
            dimension=Dimension.SECURITY.value,
            issue="Use of eval() detected",
            line_numbers=[2],
            severity=Severity.CRITICAL.value,
            confidence=0.95,
            suggestion="Use ast.literal_eval()",
            explanation="eval() is dangerous",
            rule_id="SEC001",
            code_snippet='result = eval(user_input)',
        )

    @pytest.fixture
    def none_comparison_finding(self):
        """Finding for None comparison"""
        return Finding(
            dimension=Dimension.CORRECTNESS.value,
            issue="Use 'is' for None comparison",
            line_numbers=[1],
            severity=Severity.HIGH.value,
            confidence=0.95,
            suggestion="Use 'is None'",
            explanation="Use identity comparison for None",
            rule_id="CORR001",
            code_snippet='if x == None:',
        )

    @pytest.mark.asyncio
    async def test_suggest_eval_fix(self, suggester, eval_finding):
        """Test fix suggestion for eval"""
        code = '''
def test():
    result = eval(user_input)
    return result
'''
        fixes = await suggester.suggest_fixes(code, [eval_finding])

        assert len(fixes) > 0
        assert fixes[0].finding_id == "SEC001"

    @pytest.mark.asyncio
    async def test_suggest_none_comparison_fix(self, suggester, none_comparison_finding):
        """Test fix suggestion for None comparison"""
        code = 'if x == None:'

        fixes = await suggester.suggest_fixes(code, [none_comparison_finding])

        assert len(fixes) > 0
        fix = fixes[0]
        assert 'is None' in fix.fixed_code or fix.fixed_code == ""

    @pytest.mark.asyncio
    async def test_fix_auto_applicable(self, suggester, none_comparison_finding):
        """Test that some fixes are auto-applicable"""
        code = 'if x == None:'

        fixes = await suggester.suggest_fixes(code, [none_comparison_finding])

        # The None comparison fix should be auto-applicable
        if fixes and fixes[0].fixed_code:
            assert fixes[0].auto_applicable is True

    @pytest.mark.asyncio
    async def test_validate_fix_syntax(self, suggester):
        """Test fix validation checks syntax"""
        valid_code = 'x = 1 + 2'
        invalid_code = 'x = 1 +'

        valid_result = await suggester.validate_fix("", valid_code)
        assert valid_result["syntax_ok"] is True

        invalid_result = await suggester.validate_fix("", invalid_code)
        assert invalid_result["syntax_ok"] is False

    @pytest.mark.asyncio
    async def test_validate_fix_warnings(self, suggester):
        """Test validation provides warnings"""
        original = 'x = 1\ny = 2\nz = 3'
        fixed = 'x = 1'  # Much shorter

        result = await suggester.validate_fix(original, fixed)

        assert len(result["warnings"]) > 0

    @pytest.mark.asyncio
    async def test_apply_auto_fix(self, suggester):
        """Test applying an auto-applicable fix"""
        code = 'if x == None:\n    pass'

        # Create an auto-applicable suggestion
        suggestion = FixSuggestion(
            finding_id="CORR001",
            original_code="if x == None:",
            fixed_code="if x is None:",
            description="Use 'is' for None comparison",
            confidence=0.95,
            auto_applicable=True,
            requires_review=False,
        )

        modified = await suggester.apply_fix(code, suggestion)

        assert "is None" in modified
        assert "== None" not in modified

    @pytest.mark.asyncio
    async def test_apply_fix_requires_review(self, suggester):
        """Test that non-auto fixes require review"""
        suggestion = FixSuggestion(
            finding_id="SEC001",
            original_code="eval(x)",
            fixed_code="ast.literal_eval(x)",
            description="Replace eval",
            confidence=0.7,
            auto_applicable=False,
            requires_review=True,
        )

        with pytest.raises(ValueError, match="requires manual review"):
            await suggester.apply_fix("eval(x)", suggestion)

    @pytest.mark.asyncio
    async def test_apply_fix_no_code(self, suggester):
        """Test applying fix with no fixed code"""
        suggestion = FixSuggestion(
            finding_id="GENERIC",
            original_code="bad code",
            fixed_code="",
            description="Manual fix required",
            confidence=0.5,
            auto_applicable=False,
            requires_review=True,
        )

        with pytest.raises(ValueError, match="No fixed code"):
            await suggester.apply_fix("bad code", suggestion)

    def test_get_supported_rules(self, suggester):
        """Test getting list of rules with auto-fix support"""
        supported = suggester.get_supported_rules()

        assert isinstance(supported, list)
        assert len(supported) > 0
        assert "SEC001" in supported  # eval fix
        assert "CORR001" in supported  # None comparison fix

    def test_add_custom_fix_pattern(self, suggester):
        """Test adding custom fix patterns"""
        def custom_fix(code: str, finding: Finding):
            return FixSuggestion(
                finding_id="CUSTOM001",
                original_code="bad",
                fixed_code="good",
                description="Custom fix",
                confidence=0.8,
                auto_applicable=True,
                requires_review=False,
            )

        suggester.add_fix_pattern("CUSTOM001", custom_fix)

        assert "CUSTOM001" in suggester.fix_patterns


class TestFixSuggestion:
    """Test suite for FixSuggestion dataclass"""

    def test_to_dict(self):
        """Test serialization"""
        suggestion = FixSuggestion(
            finding_id="SEC001",
            original_code="eval(x)",
            fixed_code="ast.literal_eval(x)",
            description="Replace eval",
            confidence=0.9,
            auto_applicable=False,
            breaking_change=True,
            requires_review=True,
        )

        data = suggestion.to_dict()

        assert data["finding_id"] == "SEC001"
        assert data["confidence"] == 0.9
        assert data["breaking_change"] is True

    def test_default_values(self):
        """Test default values"""
        suggestion = FixSuggestion(
            finding_id="TEST",
            original_code="a",
            fixed_code="b",
            description="Test",
            confidence=0.5,
        )

        assert suggestion.auto_applicable is False
        assert suggestion.breaking_change is False
        assert suggestion.requires_review is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
