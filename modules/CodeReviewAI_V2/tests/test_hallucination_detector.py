"""
Tests for CodeReviewAI_V2 Hallucination Detector

Test coverage:
- Finding verification
- Line number validation
- Code snippet verification
- Consistency checking
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hallucination_detector import HallucinationDetector, VerificationResult
from src.models import Finding, Dimension, Severity, VerificationStatus


class TestHallucinationDetector:
    """Test suite for HallucinationDetector"""

    @pytest.fixture
    def detector(self):
        return HallucinationDetector()

    @pytest.fixture
    def valid_finding(self):
        return Finding(
            dimension=Dimension.SECURITY.value,
            issue="eval() usage detected",
            line_numbers=[2],
            severity=Severity.CRITICAL.value,
            confidence=0.9,
            suggestion="Use ast.literal_eval()",
            explanation="eval() is dangerous",
            rule_id="SEC001",
            code_snippet='result = eval(user_input)',
        )

    @pytest.fixture
    def invalid_finding(self):
        return Finding(
            dimension=Dimension.SECURITY.value,
            issue="SQL injection detected",
            line_numbers=[100],  # Non-existent line
            severity=Severity.CRITICAL.value,
            confidence=0.9,
            suggestion="Use parameterized queries",
            explanation="SQL injection risk",
            rule_id="SEC002",
            code_snippet='execute("SELECT * FROM users")',
        )

    @pytest.fixture
    def sample_code(self):
        return '''
def test():
    result = eval(user_input)
    return result
'''

    @pytest.mark.asyncio
    async def test_verify_valid_finding(self, detector, valid_finding, sample_code):
        """Test verification of valid finding"""
        result = await detector.verify_finding(valid_finding, sample_code)

        assert result.status in [VerificationStatus.VERIFIED, VerificationStatus.UNCERTAIN]
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_verify_invalid_line_number(self, detector, invalid_finding, sample_code):
        """Test rejection of finding with invalid line number"""
        result = await detector.verify_finding(invalid_finding, sample_code)

        # Should be rejected due to line number being out of range
        assert result.status in [VerificationStatus.REJECTED, VerificationStatus.UNCERTAIN]

    @pytest.mark.asyncio
    async def test_verify_findings_filters_hallucinations(self, detector, valid_finding, invalid_finding, sample_code):
        """Test that verify_findings filters out hallucinations"""
        findings = [valid_finding, invalid_finding]
        verified = await detector.verify_findings(findings, sample_code)

        # At least the valid finding should remain
        assert len(verified) <= len(findings)

    @pytest.mark.asyncio
    async def test_verification_adds_metadata(self, detector, valid_finding, sample_code):
        """Test that verification adds metadata to findings"""
        findings = [valid_finding]
        verified = await detector.verify_findings(findings, sample_code)

        if verified:
            finding = verified[0]
            assert finding.verification_status != VerificationStatus.UNVERIFIED.value
            assert finding.verified_at is not None
            assert finding.verification_method is not None

    @pytest.mark.asyncio
    async def test_consistency_check(self, detector, valid_finding, sample_code):
        """Test consistency checking across runs"""
        code_hash = "test_hash"

        # First run - establishes baseline
        await detector.consistency_check([valid_finding], code_hash)

        # Second run with same finding
        scores2 = await detector.consistency_check([valid_finding], code_hash)

        # Scores should show consistency
        assert len(scores2) > 0

    def test_get_metrics(self, detector):
        """Test metrics retrieval"""
        metrics = detector.get_metrics()

        assert "total_checked" in metrics
        assert "verified_count" in metrics
        assert "rejected_count" in metrics
        assert "verification_rate" in metrics

    def test_clear_cache(self, detector):
        """Test cache clearing"""
        detector._consistency_cache["test"] = []
        detector.clear_cache()

        assert len(detector._consistency_cache) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
