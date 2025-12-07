"""
Unit tests for cost control module.
"""

import pytest
import threading
import time
from datetime import datetime

from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
from ai_core.foundation_model.deployment.cost_control import CostController


class TestCostController:
    """Tests for CostController class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PracticalDeploymentConfig(
            max_daily_tokens=1_000_000,  # 1M tokens/day
            max_monthly_cost_usd=1000.0,  # $1000/month
        )

    @pytest.fixture
    def controller(self, config):
        """Create cost controller for testing."""
        return CostController(config)

    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.daily_tokens == 0
        assert controller.monthly_cost == 0.0
        assert len(controller.token_history) == 0
        assert len(controller.cost_history) == 0

    def test_record_usage_tokens(self, controller):
        """Test recording token usage."""
        controller.record_usage(tokens=1000)
        
        assert controller.daily_tokens == 1000
        assert len(controller.token_history) == 1

    def test_record_usage_with_cost(self, controller):
        """Test recording usage with cost."""
        controller.record_usage(tokens=1000, cost=0.002)
        
        assert controller.daily_tokens == 1000
        assert controller.monthly_cost == 0.002

    def test_record_usage_cumulative(self, controller):
        """Test cumulative usage recording."""
        controller.record_usage(tokens=500)
        controller.record_usage(tokens=500)
        controller.record_usage(tokens=500)
        
        assert controller.daily_tokens == 1500
        assert len(controller.token_history) == 3

    def test_check_limits_within(self, controller):
        """Test check limits when within limits."""
        controller.record_usage(tokens=1000)
        
        within_limits, message = controller.check_limits()
        
        assert within_limits is True

    def test_check_limits_token_exceeded(self, controller, config):
        """Test check limits when tokens exceeded."""
        # Exceed daily token limit
        controller.record_usage(tokens=config.max_daily_tokens + 1)
        
        within_limits, message = controller.check_limits()
        
        assert within_limits is False
        assert "token limit exceeded" in message.lower()

    def test_check_limits_cost_exceeded(self, controller, config):
        """Test check limits when cost exceeded."""
        # Exceed monthly cost limit
        controller.record_usage(tokens=0, cost=config.max_monthly_cost_usd + 1)
        
        within_limits, message = controller.check_limits()
        
        assert within_limits is False
        assert "cost limit exceeded" in message.lower()

    def test_reset_daily(self, controller):
        """Test resetting daily counters."""
        controller.record_usage(tokens=10000)
        assert controller.daily_tokens == 10000
        
        controller.reset_daily()
        
        assert controller.daily_tokens == 0
        # Monthly cost should remain
        
    def test_get_usage_summary(self, controller, config):
        """Test getting usage summary."""
        controller.record_usage(tokens=500000, cost=500.0)
        
        summary = controller.get_usage_summary()
        
        assert summary["daily_tokens"] == 500000
        assert summary["daily_limit"] == config.max_daily_tokens
        assert summary["daily_usage_percent"] == 50.0
        assert summary["monthly_cost"] == 500.0
        assert summary["monthly_limit"] == config.max_monthly_cost_usd
        assert summary["monthly_usage_percent"] == 50.0

    def test_usage_history_timestamp(self, controller):
        """Test usage history includes timestamp."""
        controller.record_usage(tokens=100, operation="test_op")
        
        history_entry = controller.token_history[0]
        
        assert "timestamp" in history_entry
        assert "tokens" in history_entry
        assert "operation" in history_entry
        assert history_entry["operation"] == "test_op"

    def test_thread_safety(self, controller):
        """Test thread-safe usage recording."""
        num_threads = 10
        tokens_per_thread = 1000
        
        def record_usage():
            for _ in range(10):
                controller.record_usage(tokens=100)
                time.sleep(0.001)
        
        threads = [threading.Thread(target=record_usage) for _ in range(num_threads)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All recordings should be captured
        assert controller.daily_tokens == num_threads * tokens_per_thread


class TestCostControllerWarnings:
    """Tests for cost controller warning thresholds."""

    @pytest.fixture
    def controller(self):
        """Create controller with low limits for testing warnings."""
        config = PracticalDeploymentConfig(
            max_daily_tokens=100,
            max_monthly_cost_usd=100.0,
        )
        return CostController(config)

    def test_approaching_daily_limit(self, controller):
        """Test warning when approaching daily limit."""
        # Use 80% of daily tokens
        controller.record_usage(tokens=80)
        
        within_limits, _ = controller.check_limits()
        summary = controller.get_usage_summary()
        
        assert within_limits is True
        assert summary["daily_usage_percent"] == 80.0

    def test_approaching_monthly_cost(self, controller):
        """Test warning when approaching monthly cost."""
        # Use 90% of monthly cost
        controller.record_usage(tokens=0, cost=90.0)
        
        within_limits, _ = controller.check_limits()
        summary = controller.get_usage_summary()
        
        assert within_limits is True
        assert summary["monthly_usage_percent"] == 90.0


class TestCostControllerEdgeCases:
    """Edge case tests for CostController."""

    def test_zero_usage(self):
        """Test with zero usage."""
        config = PracticalDeploymentConfig()
        controller = CostController(config)
        
        controller.record_usage(tokens=0, cost=0.0)
        
        assert controller.daily_tokens == 0
        assert controller.monthly_cost == 0.0

    def test_large_single_usage(self):
        """Test recording very large single usage."""
        config = PracticalDeploymentConfig(
            max_daily_tokens=1_000_000_000,
            max_monthly_cost_usd=1_000_000.0,
        )
        controller = CostController(config)
        
        # Record 100M tokens
        controller.record_usage(tokens=100_000_000, cost=10000.0)
        
        assert controller.daily_tokens == 100_000_000
        assert controller.monthly_cost == 10000.0

    def test_multiple_operations(self):
        """Test tracking different operations."""
        config = PracticalDeploymentConfig()
        controller = CostController(config)
        
        controller.record_usage(tokens=100, operation="inference")
        controller.record_usage(tokens=200, operation="training")
        controller.record_usage(tokens=50, operation="embedding")
        
        operations = [h["operation"] for h in controller.token_history]
        
        assert "inference" in operations
        assert "training" in operations
        assert "embedding" in operations


class TestCostControllerIntegration:
    """Integration tests for cost control."""

    def test_realistic_usage_pattern(self):
        """Test realistic daily usage pattern."""
        config = PracticalDeploymentConfig(
            max_daily_tokens=10_000_000,  # 10M tokens/day
            max_monthly_cost_usd=5000.0,
        )
        controller = CostController(config)
        
        # Simulate hourly usage for a day
        for hour in range(24):
            tokens = 300000 + (hour * 10000)  # Increasing usage throughout day
            cost = tokens * 0.00001  # $0.01 per 1K tokens
            controller.record_usage(tokens=tokens, cost=cost, operation=f"hour_{hour}")
        
        summary = controller.get_usage_summary()
        
        assert summary["daily_tokens"] < config.max_daily_tokens
        assert summary["monthly_cost"] < config.max_monthly_cost_usd
        assert len(controller.token_history) == 24
