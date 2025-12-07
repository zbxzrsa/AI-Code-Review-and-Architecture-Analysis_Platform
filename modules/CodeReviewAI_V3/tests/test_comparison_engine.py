"""
Tests for CodeReviewAI_V3 Comparison Engine
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.comparison_engine import ComparisonEngine
from src.models import ReviewResult, ReviewStatus, Finding


class TestComparisonEngine:
    """Test suite for ComparisonEngine"""

    @pytest.fixture
    def engine(self):
        return ComparisonEngine()

    @pytest.fixture
    def baseline_result(self):
        return ReviewResult(
            review_id="baseline-001",
            code_hash="abc123",
            status=ReviewStatus.QUARANTINED,
            findings=[],
            overall_score=70.0,
            dimension_scores={"security": 70.0},
            model_version="3.0.0",
            processing_time_ms=100,
        )

    @pytest.fixture
    def improved_result(self):
        return ReviewResult(
            review_id="compare-001",
            code_hash="abc123",
            status=ReviewStatus.COMPLETED,
            findings=[],
            overall_score=85.0,
            dimension_scores={"security": 85.0},
            model_version="2.0.0",
            processing_time_ms=150,
        )

    @pytest.fixture
    def worse_result(self):
        return ReviewResult(
            review_id="compare-002",
            code_hash="abc123",
            status=ReviewStatus.COMPLETED,
            findings=[],
            overall_score=55.0,
            dimension_scores={"security": 55.0},
            model_version="1.0.0",
            processing_time_ms=200,
        )

    @pytest.mark.asyncio
    async def test_compare_improved(self, engine, baseline_result, improved_result):
        """Test comparison with improved result"""
        result = await engine.compare(baseline_result, improved_result)

        assert result.score_delta == 15.0
        assert result.recommendation == "promote"

    @pytest.mark.asyncio
    async def test_compare_worse(self, engine, baseline_result, worse_result):
        """Test comparison with worse result"""
        result = await engine.compare(baseline_result, worse_result)

        assert result.score_delta == -15.0
        assert result.recommendation == "quarantine"

    @pytest.mark.asyncio
    async def test_compare_similar(self, engine, baseline_result):
        """Test comparison with similar result"""
        similar = ReviewResult(
            review_id="compare-003",
            code_hash="abc123",
            status=ReviewStatus.COMPLETED,
            findings=[],
            overall_score=72.0,  # Only 2 points better
            dimension_scores={"security": 72.0},
            model_version="2.0.0",
            processing_time_ms=100,
        )

        result = await engine.compare(baseline_result, similar)

        assert result.recommendation == "keep"

    @pytest.mark.asyncio
    async def test_batch_compare(self, engine, baseline_result, improved_result):
        """Test batch comparison"""
        baselines = [baseline_result, baseline_result]
        compares = [improved_result, improved_result]

        result = await engine.batch_compare(baselines, compares)

        assert result["total_comparisons"] == 2
        assert result["avg_score_delta"] == 15.0

    def test_get_history(self, engine):
        """Test history retrieval"""
        history = engine.get_history()
        assert isinstance(history, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
