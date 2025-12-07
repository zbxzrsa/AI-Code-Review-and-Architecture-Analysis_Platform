"""
Tests for CodeReviewAI_V1 Issue Detector

Test coverage:
- Rule loading
- Issue detection
- Custom rules
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.issue_detector import IssueDetector, DetectionRule
from src.models import Dimension, Severity


class TestIssueDetector:
    """Test suite for IssueDetector"""

    @pytest.fixture
    def detector(self):
        """Create issue detector instance"""
        return IssueDetector()

    @pytest.mark.asyncio
    async def test_detect_eval_usage(self, detector):
        """Test detection of eval() usage"""
        code = '''
result = eval(user_input)
'''
        findings = await detector.detect(code)

        eval_findings = [f for f in findings if 'eval' in f.issue.lower()]
        assert len(eval_findings) > 0
        assert eval_findings[0].severity == Severity.CRITICAL.value

    @pytest.mark.asyncio
    async def test_detect_hardcoded_password(self, detector):
        """Test detection of hardcoded passwords"""
        code = '''
password = "my_secret_password"
api_key = "sk-1234567890"
'''
        findings = await detector.detect(code)

        cred_findings = [f for f in findings if 'credential' in f.issue.lower() or 'hardcoded' in f.issue.lower()]
        assert len(cred_findings) >= 1

    @pytest.mark.asyncio
    async def test_detect_shell_injection(self, detector):
        """Test detection of shell injection risk"""
        code = '''
import subprocess
subprocess.run(cmd, shell=True)
'''
        findings = await detector.detect(code)

        shell_findings = [f for f in findings if 'shell' in f.issue.lower()]
        assert len(shell_findings) > 0

    @pytest.mark.asyncio
    async def test_detect_todo_comments(self, detector):
        """Test detection of TODO comments"""
        code = '''
# TODO: Fix this later
# FIXME: This is broken
'''
        findings = await detector.detect(code)

        todo_findings = [f for f in findings if 'todo' in f.issue.lower() or 'fixme' in f.issue.lower()]
        assert len(todo_findings) > 0

    @pytest.mark.asyncio
    async def test_detect_bare_except(self, detector):
        """Test detection of bare except clauses"""
        code = '''
try:
    risky_operation()
except:
    pass
'''
        findings = await detector.detect(code)

        except_findings = [f for f in findings if 'except' in f.issue.lower()]
        assert len(except_findings) > 0

    @pytest.mark.asyncio
    async def test_filter_by_dimension(self, detector):
        """Test filtering findings by dimension"""
        code = '''
password = "secret"
# TODO: fix this
'''

        # Only security
        security_findings = await detector.detect(
            code,
            dimensions=[Dimension.SECURITY]
        )

        for finding in security_findings:
            assert finding.dimension == Dimension.SECURITY.value

    @pytest.mark.asyncio
    async def test_finding_has_line_numbers(self, detector):
        """Test that findings include line numbers"""
        code = '''
line1
password = "test"
line3
'''
        findings = await detector.detect(code)

        for finding in findings:
            assert len(finding.line_numbers) > 0
            assert finding.line_numbers[0] > 0

    @pytest.mark.asyncio
    async def test_finding_has_code_snippet(self, detector):
        """Test that findings include code snippets"""
        code = '''
line1
result = eval("test")
line3
'''
        findings = await detector.detect(code)

        for finding in findings:
            if finding.code_snippet:
                assert len(finding.code_snippet) > 0

    def test_add_custom_rule(self, detector):
        """Test adding custom detection rules"""
        import re

        custom_rule = DetectionRule(
            rule_id="CUSTOM001",
            dimension=Dimension.MAINTAINABILITY,
            severity=Severity.LOW,
            pattern=re.compile(r'print\('),
            message="Debug print statement found",
            suggestion="Remove debug print statements before production",
            explanation="Print statements should not be in production code",
        )

        detector.add_rule(custom_rule)

        assert "CUSTOM001" in detector.rules

    def test_remove_rule(self, detector):
        """Test removing detection rules"""
        initial_count = len(detector.rules)

        detector.remove_rule("SEC001")

        assert len(detector.rules) == initial_count - 1
        assert "SEC001" not in detector.rules

    def test_enable_disable_rule(self, detector):
        """Test enabling/disabling rules"""
        detector.disable_rule("SEC001")
        assert detector.rules["SEC001"].enabled is False

        detector.enable_rule("SEC001")
        assert detector.rules["SEC001"].enabled is True

    @pytest.mark.asyncio
    async def test_disabled_rule_not_applied(self, detector):
        """Test that disabled rules are not applied"""
        code = 'result = eval("test")'

        # Disable eval detection
        detector.disable_rule("SEC001")

        findings = await detector.detect(code)

        eval_findings = [f for f in findings if f.rule_id == "SEC001"]
        assert len(eval_findings) == 0

    def test_get_rules_summary(self, detector):
        """Test getting rules summary"""
        summary = detector.get_rules_summary()

        assert "total_rules" in summary
        assert "enabled_rules" in summary
        assert "by_dimension" in summary
        assert "by_severity" in summary

        assert summary["total_rules"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
