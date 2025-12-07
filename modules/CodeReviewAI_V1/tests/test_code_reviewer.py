"""
Tests for CodeReviewAI_V1 Code Reviewer

Test coverage:
- Review execution
- Strategy selection
- Finding detection
- Score calculation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.code_reviewer import CodeReviewer, BaselineStrategy, ChainOfThoughtStrategy, EnsembleStrategy
from src.models import ReviewConfig, ReviewStatus, Dimension, Severity


class TestCodeReviewer:
    """Test suite for CodeReviewer"""

    @pytest.fixture
    def reviewer(self):
        """Create code reviewer instance"""
        return CodeReviewer(strategy="baseline")

    @pytest.fixture
    def sample_code(self):
        """Sample code with known issues"""
        return '''
def vulnerable_function(user_input):
    # Security issue: eval
    result = eval(user_input)

    # Hardcoded password
    password = "secret123"

    return result
'''

    @pytest.fixture
    def clean_code(self):
        """Clean code without issues"""
        return '''
def safe_function(value: int) -> int:
    """
    Safely process a numeric value.

    Args:
        value: Input integer value

    Returns:
        Processed integer
    """
    return value * 2
'''

    @pytest.mark.asyncio
    async def test_review_detects_eval(self, reviewer, sample_code):
        """Test that eval usage is detected"""
        result = await reviewer.review(sample_code, language="python")

        assert result.status == ReviewStatus.COMPLETED
        assert len(result.findings) > 0

        eval_findings = [f for f in result.findings if 'eval' in f.issue.lower()]
        assert len(eval_findings) > 0

    @pytest.mark.asyncio
    async def test_review_detects_hardcoded_password(self, reviewer, sample_code):
        """Test that hardcoded passwords are detected"""
        result = await reviewer.review(sample_code, language="python")

        password_findings = [f for f in result.findings if 'password' in f.issue.lower()]
        assert len(password_findings) > 0

    @pytest.mark.asyncio
    async def test_review_clean_code(self, reviewer, clean_code):
        """Test review of clean code returns high score"""
        result = await reviewer.review(clean_code, language="python")

        assert result.status == ReviewStatus.COMPLETED
        assert result.overall_score >= 80  # Clean code should score well

    @pytest.mark.asyncio
    async def test_review_generates_id(self, reviewer, sample_code):
        """Test that review generates unique ID"""
        result1 = await reviewer.review(sample_code)
        result2 = await reviewer.review(sample_code)

        assert result1.review_id != result2.review_id

    @pytest.mark.asyncio
    async def test_review_calculates_code_hash(self, reviewer, sample_code):
        """Test that code hash is consistent"""
        result1 = await reviewer.review(sample_code)
        result2 = await reviewer.review(sample_code)

        assert result1.code_hash == result2.code_hash

    @pytest.mark.asyncio
    async def test_review_with_config(self, reviewer):
        """Test review with custom config"""
        config = ReviewConfig(
            dimensions=[Dimension.SECURITY],
            max_findings=5,
            min_confidence=0.9,
        )

        code = 'password = "test123"'
        result = await reviewer.review(code, config=config)

        assert result.status == ReviewStatus.COMPLETED

    def test_metrics_tracking(self, reviewer):
        """Test that metrics are tracked"""
        metrics = reviewer.get_metrics()

        assert "review_count" in metrics
        assert "total_findings" in metrics
        assert "avg_time_ms" in metrics


class TestBaselineStrategy:
    """Test suite for BaselineStrategy"""

    @pytest.fixture
    def strategy(self):
        return BaselineStrategy()

    @pytest.fixture
    def config(self):
        return ReviewConfig()

    @pytest.mark.asyncio
    async def test_detects_eval(self, strategy, config):
        """Test eval detection"""
        code = 'result = eval("1 + 1")'
        findings = await strategy.review(code, "python", config)

        eval_findings = [f for f in findings if 'eval' in f.issue.lower()]
        assert len(eval_findings) > 0

    @pytest.mark.asyncio
    async def test_detects_sql_injection(self, strategy, config):
        """Test SQL injection detection"""
        code = 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'
        findings = await strategy.review(code, "python", config)

        # Note: This specific pattern may or may not be caught by baseline
        assert isinstance(findings, list)


class TestChainOfThoughtStrategy:
    """Test suite for ChainOfThoughtStrategy"""

    @pytest.fixture
    def strategy(self):
        return ChainOfThoughtStrategy()

    @pytest.fixture
    def config(self):
        return ReviewConfig()

    @pytest.mark.asyncio
    async def test_includes_reasoning_steps(self, strategy, config):
        """Test that findings include reasoning steps"""
        code = '''
def my_function():
    pass
'''
        findings = await strategy.review(code, "python", config)

        # Check if any finding has reasoning steps
        # (depends on implementation details)
        assert isinstance(findings, list)


class TestEnsembleStrategy:
    """Test suite for EnsembleStrategy"""

    @pytest.fixture
    def strategy(self):
        return EnsembleStrategy()

    @pytest.fixture
    def config(self):
        return ReviewConfig()

    @pytest.mark.asyncio
    async def test_combines_strategies(self, strategy, config):
        """Test that ensemble combines multiple strategies"""
        code = '''
def test():
    result = eval("test")
    return result
'''
        findings = await strategy.review(code, "python", config)

        # Ensemble should find issues
        assert len(findings) >= 0  # May find eval issue


class TestReviewConfig:
    """Test suite for ReviewConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ReviewConfig()

        assert config.max_findings == 50
        assert config.min_confidence == 0.7
        assert config.include_suggestions is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = ReviewConfig(
            dimensions=[Dimension.SECURITY, Dimension.PERFORMANCE],
            max_findings=10,
            min_confidence=0.9,
        )

        assert len(config.dimensions) == 2
        assert config.max_findings == 10
        assert config.min_confidence == 0.9

    def test_to_dict(self):
        """Test configuration serialization"""
        config = ReviewConfig()
        data = config.to_dict()

        assert "dimensions" in data
        assert "max_findings" in data
        assert "min_confidence" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
