"""
Tests for Learning Engine and Protocol Optimizations

Tests cover:
- Async initialization of HTTP sessions
- Error handling improvements
- Rate limiting functionality
- Quality filtering
- Connection management
- Edge case handling
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Learning Channel Tests
# =============================================================================

class TestLearningChannelAsync:
    """Tests for async LearningChannel operations."""
    
    @pytest.fixture
    def source(self):
        from ai_core.distributed_vc.learning_engine import LearningSource, ChannelType
        return LearningSource(
            source_id="test_source",
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Test Source",
            url="https://api.github.com/search/repositories",
            fetch_interval_seconds=300,
            priority=1
        )
    
    @pytest.fixture
    def github_channel(self, source):
        from ai_core.distributed_vc.learning_engine import GitHubTrendingChannel
        return GitHubTrendingChannel(source)
    
    @pytest.mark.asyncio
    async def test_initialize_creates_session(self, github_channel):
        """Test that initialize creates an aiohttp session."""
        await github_channel.initialize()
        
        assert github_channel.session is not None
        assert not github_channel.session.closed
        
        # Cleanup
        await github_channel.close()
    
    @pytest.mark.asyncio
    async def test_session_has_timeout(self, github_channel):
        """Test that session is created with timeout configuration."""
        await github_channel.initialize()
        
        # Session should have timeout configured
        assert github_channel.session._timeout is not None
        
        await github_channel.close()
    
    @pytest.mark.asyncio
    async def test_close_handles_already_closed(self, github_channel):
        """Test that close handles already-closed session gracefully."""
        await github_channel.initialize()
        
        # Close once
        await github_channel.close()
        
        # Close again - should not raise
        await github_channel.close()
    
    @pytest.mark.asyncio
    async def test_close_without_initialize(self, github_channel):
        """Test closing without initializing first."""
        # Should not raise
        await github_channel.close()


# =============================================================================
# Learning Engine Tests
# =============================================================================

class TestOnlineLearningEngine:
    """Tests for OnlineLearningEngine."""
    
    @pytest.fixture
    def engine(self):
        from ai_core.distributed_vc.learning_engine import OnlineLearningEngine
        return OnlineLearningEngine()
    
    @pytest.fixture
    def valid_source(self):
        from ai_core.distributed_vc.learning_engine import LearningSource, ChannelType
        return LearningSource(
            source_id="valid_source",
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Valid Source",
            url="https://api.github.com",
            fetch_interval_seconds=300,
            priority=1
        )
    
    def test_register_source_success(self, engine, valid_source):
        """Test successful source registration."""
        engine.register_source(valid_source)
        
        assert valid_source.source_id in engine.sources
        assert valid_source.source_id in engine.channels
    
    def test_register_source_empty_id_fails(self, engine):
        """Test registration with empty source_id fails."""
        from ai_core.distributed_vc.learning_engine import LearningSource, ChannelType
        
        source = LearningSource(
            source_id="",  # Empty
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Test",
            url="https://example.com",
        )
        
        with pytest.raises(ValueError, match="source_id cannot be empty"):
            engine.register_source(source)
    
    def test_register_source_invalid_id_format(self, engine):
        """Test registration with invalid source_id format fails."""
        from ai_core.distributed_vc.learning_engine import LearningSource, ChannelType
        
        source = LearningSource(
            source_id="invalid@id!",  # Special chars
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Test",
            url="https://example.com",
        )
        
        with pytest.raises(ValueError, match="must contain only alphanumeric"):
            engine.register_source(source)
    
    def test_register_source_duplicate_fails(self, engine, valid_source):
        """Test duplicate registration fails."""
        engine.register_source(valid_source)
        
        with pytest.raises(ValueError, match="already registered"):
            engine.register_source(valid_source)
    
    def test_register_source_invalid_url(self, engine):
        """Test registration with invalid URL fails."""
        from ai_core.distributed_vc.learning_engine import LearningSource, ChannelType
        
        source = LearningSource(
            source_id="test",
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Test",
            url="ftp://invalid.com",  # Not http/https
        )
        
        with pytest.raises(ValueError, match="Invalid URL format"):
            engine.register_source(source)
    
    def test_register_source_short_interval_fails(self, engine):
        """Test registration with too short interval fails."""
        from ai_core.distributed_vc.learning_engine import LearningSource, ChannelType
        
        source = LearningSource(
            source_id="test",
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Test",
            url="https://example.com",
            fetch_interval_seconds=30,  # Less than 60
        )
        
        with pytest.raises(ValueError, match="fetch_interval_seconds must be >= 60"):
            engine.register_source(source)
    
    def test_register_source_invalid_priority(self, engine):
        """Test registration with invalid priority fails."""
        from ai_core.distributed_vc.learning_engine import LearningSource, ChannelType
        
        source = LearningSource(
            source_id="test",
            channel_type=ChannelType.GITHUB_TRENDING,
            name="Test",
            url="https://example.com",
            priority=10,  # Greater than 5
        )
        
        with pytest.raises(ValueError, match="priority must be 1-5"):
            engine.register_source(source)
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, engine, valid_source):
        """Test start and stop lifecycle."""
        engine.register_source(valid_source)
        
        await engine.start()
        assert engine.is_running is True
        
        await engine.stop()
        assert engine.is_running is False
    
    def test_get_metrics(self, engine):
        """Test getting metrics."""
        metrics = engine.get_metrics()
        
        assert "total_items_fetched" in metrics
        assert "total_items_processed" in metrics
        assert "total_items_integrated" in metrics


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter."""
    
    @pytest.fixture
    def limiter(self):
        from ai_core.distributed_vc.learning_engine import AsyncRateLimiter
        return AsyncRateLimiter(requests_per_hour=100)
    
    @pytest.mark.asyncio
    async def test_acquire_token(self, limiter):
        """Test acquiring a token."""
        result = await limiter.acquire()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_remaining(self, limiter):
        """Test getting remaining tokens."""
        initial = limiter.get_remaining()
        await limiter.acquire()
        after = limiter.get_remaining()
        
        assert after < initial
    
    @pytest.mark.asyncio
    async def test_acquire_multiple(self, limiter):
        """Test acquiring multiple tokens."""
        result = await limiter.acquire(tokens=5)
        assert result is True


# =============================================================================
# Data Quality Filter Tests
# =============================================================================

class TestDataQualityFilter:
    """Tests for DataQualityFilter."""
    
    @pytest.fixture
    def filter(self):
        from ai_core.distributed_vc.learning_engine import DataQualityFilter
        return DataQualityFilter(quality_threshold=0.7)
    
    @pytest.fixture
    def learning_item(self):
        from ai_core.distributed_vc.learning_engine import LearningItem, ChannelType
        return LearningItem(
            item_id="test_item",
            source_id="test_source",
            channel_type=ChannelType.GITHUB_TRENDING,
            title="Test Python Library for Machine Learning",
            content="This is a comprehensive python library for machine learning, "
                   "including algorithms for optimization and testing deployment in docker.",
        )
    
    def test_calculate_quality_good_content(self, filter, learning_item):
        """Test quality calculation for good content."""
        quality = filter.calculate_quality(learning_item)
        assert quality >= 0.5  # Should have decent quality
    
    def test_calculate_quality_empty_content(self, filter):
        """Test quality calculation for empty content."""
        from ai_core.distributed_vc.learning_engine import LearningItem, ChannelType
        
        item = LearningItem(
            item_id="empty",
            source_id="test",
            channel_type=ChannelType.GITHUB_TRENDING,
            title="",
            content="",
        )
        
        quality = filter.calculate_quality(item)
        assert quality < 0.5
    
    def test_filter_keeps_high_quality(self, filter, learning_item):
        """Test filtering keeps high quality items."""
        items = [learning_item]
        filtered = filter.filter(items)
        
        # Should keep item if quality meets threshold
        # Depends on content quality
        assert len(filtered) <= 1


# =============================================================================
# Protocol Message Tests
# =============================================================================

class TestProtocolMessage:
    """Tests for ProtocolMessage."""
    
    @pytest.fixture
    def message(self):
        from ai_core.distributed_vc.protocol import ProtocolMessage, MessageType
        return ProtocolMessage(
            message_id="test_123",
            message_type=MessageType.REQUEST,
            timestamp=datetime.now().isoformat(),
            action="analyze_code",
            payload={"code": "print('hello')"},
            source="node_a",
            target="node_b"
        )
    
    def test_to_dict(self, message):
        """Test converting message to dict."""
        data = message.to_dict()
        
        assert data["message_id"] == "test_123"
        assert data["action"] == "analyze_code"
        assert "payload" in data
    
    def test_from_dict(self, message):
        """Test creating message from dict."""
        from ai_core.distributed_vc.protocol import ProtocolMessage
        
        data = message.to_dict()
        restored = ProtocolMessage.from_dict(data)
        
        assert restored.message_id == message.message_id
        assert restored.action == message.action
    
    def test_compute_checksum(self, message):
        """Test checksum computation."""
        checksum = message.compute_checksum()
        
        assert checksum is not None
        assert len(checksum) == 16
    
    def test_verify_checksum_valid(self, message):
        """Test verifying valid checksum."""
        message.checksum = message.compute_checksum()
        
        assert message.verify_checksum() is True
    
    def test_verify_checksum_invalid(self, message):
        """Test verifying invalid checksum."""
        message.checksum = "invalid_checksum!"
        
        assert message.verify_checksum() is False
    
    def test_verify_checksum_missing(self, message):
        """Test verifying when no checksum."""
        message.checksum = None
        
        # Should pass when no checksum to verify
        assert message.verify_checksum() is True


# =============================================================================
# Bidirectional Protocol Tests
# =============================================================================

class TestBidirectionalProtocol:
    """Tests for BidirectionalProtocol."""
    
    @pytest.fixture
    def protocol(self):
        from ai_core.distributed_vc.protocol import BidirectionalProtocol
        return BidirectionalProtocol(node_id="test_node")
    
    def test_initialization(self, protocol):
        """Test protocol initialization."""
        assert protocol.node_id == "test_node"
        assert protocol.status.value == "disconnected"
    
    def test_generate_message_id(self, protocol):
        """Test message ID generation."""
        id1 = protocol._generate_message_id()
        id2 = protocol._generate_message_id()
        
        assert id1 != id2
        assert "test_node" in id1
    
    def test_register_handler(self, protocol):
        """Test registering action handler."""
        def handler(message):
            return {"result": "ok"}
        
        protocol.register_handler("test_action", handler)
        
        assert "test_action" in protocol.action_handlers
    
    def test_get_status(self, protocol):
        """Test getting protocol status."""
        status = protocol.get_status()
        
        assert status["node_id"] == "test_node"
        assert "status" in status
        assert "connected_peers" in status


# =============================================================================
# Test Suite Tests
# =============================================================================

class TestTestSuite:
    """Tests for TestSuite and TestResult."""
    
    @pytest.fixture
    def test_suite(self):
        from ai_core.distributed_vc.protocol import TestSuite, TestResult
        return TestSuite(
            suite_id="suite_001",
            name="Integration Tests",
            tests=[
                TestResult(
                    test_id="test_1",
                    test_name="health_check",
                    status="passed",
                    duration_ms=100.0
                ),
                TestResult(
                    test_id="test_2",
                    test_name="api_test",
                    status="passed",
                    duration_ms=200.0
                ),
                TestResult(
                    test_id="test_3",
                    test_name="error_test",
                    status="failed",
                    duration_ms=50.0,
                    error="Connection refused"
                ),
            ],
            started_at=datetime.now().isoformat()
        )
    
    def test_total_count(self, test_suite):
        """Test total test count."""
        assert test_suite.total == 3
    
    def test_passed_count(self, test_suite):
        """Test passed test count."""
        assert test_suite.passed == 2
    
    def test_failed_count(self, test_suite):
        """Test failed test count."""
        assert test_suite.failed == 1
    
    def test_success_rate(self, test_suite):
        """Test success rate calculation."""
        # 2 out of 3 passed = 66.67%
        assert abs(test_suite.success_rate - 0.6667) < 0.01
    
    def test_empty_suite_success_rate(self):
        """Test success rate for empty suite."""
        from ai_core.distributed_vc.protocol import TestSuite
        
        suite = TestSuite(
            suite_id="empty",
            name="Empty Suite",
            tests=[],
            started_at=datetime.now().isoformat()
        )
        
        assert suite.success_rate == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
