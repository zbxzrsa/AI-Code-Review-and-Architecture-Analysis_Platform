"""
Unit Tests for Auto Network Learning System
自动联网学习系统单元测试

Tests:
- V1/V3 Learning System lifecycle
- Quality assessment
- Rate limiting
- Data cleaning pipeline
- Technology elimination
- Data lifecycle management
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# V1/V3 Auto Learning System Tests
# =============================================================================

class TestV1V3AutoLearningSystem:
    """Tests for V1V3AutoLearningSystem."""
    
    @pytest.mark.asyncio
    async def test_v1_learning_system_start_stop(self):
        """Testing the startup and shutdown of the V1 learning system."""
        from ai_core.distributed_vc.auto_network_learning import (
            V1V3AutoLearningSystem,
            NetworkLearningConfig,
        )
        
        config = NetworkLearningConfig(
            v1_learning_interval_minutes=1,
            v2_push_enabled=False,  # Disable V2 push for testing
        )
        system = V1V3AutoLearningSystem("v1", config)
        
        await system.start()
        assert system._running is True
        assert len(system.connectors) > 0
        
        await system.stop()
        assert system._running is False
    
    @pytest.mark.asyncio
    async def test_v3_learning_system_start_stop(self):
        """Testing V3 learning system lifecycle."""
        from ai_core.distributed_vc.auto_network_learning import (
            V1V3AutoLearningSystem,
            NetworkLearningConfig,
        )
        
        config = NetworkLearningConfig(
            v3_learning_interval_minutes=1,
            v2_push_enabled=False,
        )
        system = V1V3AutoLearningSystem("v3", config)
        
        await system.start()
        assert system._running is True
        assert system.version == "v3"
        
        await system.stop()
        assert system._running is False
    
    @pytest.mark.asyncio
    async def test_invalid_version_raises_error(self):
        """Test that invalid version raises ValueError."""
        from ai_core.distributed_vc.auto_network_learning import (
            V1V3AutoLearningSystem,
        )
        
        with pytest.raises(ValueError, match="must be 'v1' or 'v3'"):
            V1V3AutoLearningSystem("v2")
    
    @pytest.mark.asyncio
    async def test_quality_assessment(self):
        """Testing quality assessment logic."""
        from ai_core.distributed_vc.auto_network_learning import (
            V1V3AutoLearningSystem,
            NetworkLearningConfig,
            LearningData,
            DataSource,
            QualityAssessor,
        )
        
        config = NetworkLearningConfig()
        assessor = QualityAssessor(config)
        
        # High quality item
        high_quality = LearningData(
            data_id="test_1",
            source=DataSource.GITHUB_TRENDING,
            title="Comprehensive Python Framework for Machine Learning",
            content=(
                "This is a comprehensive framework for machine learning and AI. "
                "It includes features for data processing, model training, and deployment. "
                "Built with Python, it supports TensorFlow, PyTorch, and JAX backends. "
                "The framework includes extensive documentation and examples. " * 10
            ),
            url="https://github.com/test/ml-framework",
            fetched_at=datetime.now(timezone.utc),
            tags=["python", "machine-learning", "ai"],
        )
        
        score = assessor.assess(high_quality)
        assert 0.0 <= score <= 1.0
        assert score >= 0.5  # Should be reasonably high quality
        
        # Low quality item
        low_quality = LearningData(
            data_id="test_2",
            source=DataSource.DEV_TO,
            title="Hi",
            content="Short",
            url="",
            fetched_at=datetime.now(timezone.utc),
        )
        
        low_score = assessor.assess(low_quality)
        assert low_score < score  # Should be lower than high quality
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test statistics retrieval."""
        from ai_core.distributed_vc.auto_network_learning import (
            V1V3AutoLearningSystem,
            NetworkLearningConfig,
        )
        
        config = NetworkLearningConfig(v2_push_enabled=False)
        system = V1V3AutoLearningSystem("v1", config)
        
        stats = system.get_stats()
        
        assert "version" in stats
        assert stats["version"] == "v1"
        assert "running" in stats
        assert "total_fetched" in stats
        assert "connectors" in stats


# =============================================================================
# Rate Limiter Tests
# =============================================================================

class TestAsyncRateLimiter:
    """Tests for AsyncRateLimiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test basic token acquisition."""
        from ai_core.distributed_vc.auto_network_learning import AsyncRateLimiter
        
        limiter = AsyncRateLimiter(max_requests=10, period_seconds=3600)
        
        # Should be able to acquire tokens
        for _ in range(10):
            assert await limiter.acquire() is True
        
        # 11th request should fail
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_get_remaining(self):
        """Test remaining token count."""
        from ai_core.distributed_vc.auto_network_learning import AsyncRateLimiter
        
        limiter = AsyncRateLimiter(max_requests=5, period_seconds=3600)
        
        assert limiter.get_remaining() == 5
        
        await limiter.acquire()
        assert limiter.get_remaining() == 4


# =============================================================================
# Data Cleansing Pipeline Tests
# =============================================================================

class TestDataCleansingPipeline:
    """Tests for DataCleansingPipeline."""
    
    @pytest.mark.asyncio
    async def test_pipeline_start_stop(self):
        """Test pipeline lifecycle."""
        from ai_core.distributed_vc.data_cleansing_pipeline import (
            DataCleansingPipeline,
            CleansingConfig,
        )
        
        config = CleansingConfig(v2_push_enabled=False)
        pipeline = DataCleansingPipeline(config)
        
        await pipeline.start()
        await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_pipeline_process_item(self):
        """Test processing a single item."""
        from ai_core.distributed_vc.data_cleansing_pipeline import (
            DataCleansingPipeline,
            CleansingConfig,
            CleansingStage,
        )
        
        config = CleansingConfig(
            min_content_length=10,
            min_final_quality=0.3,
            v2_push_enabled=False,
        )
        pipeline = DataCleansingPipeline(config)
        
        # Create mock item
        class MockItem:
            data_id = "test_1"
            title = "Test Title for Pipeline"
            content = "This is test content that is long enough to pass validation checks."
            source = "github"
            is_cleaned = False
            is_validated = False
            quality_score = 0.6
            metadata = {}
        
        item = MockItem()
        result = await pipeline.process_item(item)
        
        assert result.data_id == "test_1"
        assert result.stage in [CleansingStage.READY, CleansingStage.REJECTED]
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        """Test duplicate content detection."""
        from ai_core.distributed_vc.data_cleansing_pipeline import (
            DeduplicationCache,
        )
        
        cache = DeduplicationCache(max_size=100)
        
        content = "This is some test content for deduplication."
        
        # First check should not be duplicate
        is_dup, _ = cache.is_duplicate(content)
        assert is_dup is False
        
        # Add to cache
        cache.add(content, "item_1")
        
        # Second check should be duplicate
        is_dup, original_id = cache.is_duplicate(content)
        assert is_dup is True
        assert original_id == "item_1"
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test pipeline statistics."""
        from ai_core.distributed_vc.data_cleansing_pipeline import (
            DataCleansingPipeline,
            CleansingConfig,
        )
        
        config = CleansingConfig(v2_push_enabled=False)
        pipeline = DataCleansingPipeline(config)
        
        stats = pipeline.get_stats()
        
        assert "total_received" in stats
        assert "total_passed" in stats
        assert "total_rejected" in stats
        assert "pass_rate" in stats


# =============================================================================
# Infinite Learning Manager Tests
# =============================================================================

class TestInfiniteLearningManager:
    """Tests for InfiniteLearningManager."""
    
    @pytest.mark.asyncio
    async def test_manager_start_stop(self, tmp_path):
        """Test manager lifecycle."""
        from ai_core.distributed_vc.infinite_learning_manager import (
            InfiniteLearningManager,
            MemoryConfig,
        )
        
        config = MemoryConfig(
            max_memory_mb=100,
            checkpoint_interval_minutes=60,  # Long interval for testing
        )
        
        manager = InfiniteLearningManager(str(tmp_path / "data"), config)
        
        await manager.start()
        assert manager._running is True
        
        await manager.stop()
        assert manager._running is False
    
    @pytest.mark.asyncio
    async def test_add_learning_data(self, tmp_path):
        """Test adding learning data."""
        from ai_core.distributed_vc.infinite_learning_manager import (
            InfiniteLearningManager,
            MemoryConfig,
        )
        
        config = MemoryConfig(max_memory_mb=100)
        manager = InfiniteLearningManager(str(tmp_path / "data"), config)
        
        # Add data
        item_id = await manager.add_learning_data(
            {"content": "test data"},
            source="github",
        )
        
        assert item_id is not None
        assert manager.total_learned == 1
        assert len(manager._hot_data) == 1
    
    @pytest.mark.asyncio
    async def test_get_stats(self, tmp_path):
        """Test statistics retrieval."""
        from ai_core.distributed_vc.infinite_learning_manager import (
            InfiniteLearningManager,
            MemoryConfig,
        )
        
        config = MemoryConfig(max_memory_mb=100)
        manager = InfiniteLearningManager(str(tmp_path / "data"), config)
        
        await manager.add_learning_data({"content": "test"}, source="test")
        
        stats = manager.get_stats()
        
        assert stats["total_learned"] == 1
        assert stats["hot_data_count"] == 1


# =============================================================================
# Data Lifecycle Manager Tests
# =============================================================================

class TestDataLifecycleManager:
    """Tests for DataLifecycleManager."""
    
    @pytest.mark.asyncio
    async def test_manager_start_stop(self, tmp_path):
        """Test manager lifecycle."""
        from ai_core.distributed_vc.data_lifecycle_manager import (
            DataLifecycleManager,
            DataLifecycleConfig,
        )
        
        config = DataLifecycleConfig(cleanup_interval_hours=24)
        manager = DataLifecycleManager(str(tmp_path / "data"), config)
        
        await manager.start()
        assert manager._running is True
        
        await manager.stop()
        assert manager._running is False
    
    def test_register_data(self, tmp_path):
        """Test data registration."""
        from ai_core.distributed_vc.data_lifecycle_manager import (
            DataLifecycleManager,
            DataLifecycleConfig,
            DataState,
        )
        
        config = DataLifecycleConfig()
        manager = DataLifecycleManager(str(tmp_path / "data"), config)
        
        entry = manager.register_data("item_1", source="github", tech_id="langchain")
        
        assert entry.data_id == "item_1"
        assert entry.state == DataState.ACTIVE
        assert entry.source == "github"
        assert entry.tech_id == "langchain"
    
    def test_mark_obsolete(self, tmp_path):
        """Test marking data as obsolete."""
        from ai_core.distributed_vc.data_lifecycle_manager import (
            DataLifecycleManager,
            DataLifecycleConfig,
            DataState,
        )
        
        config = DataLifecycleConfig()
        manager = DataLifecycleManager(str(tmp_path / "data"), config)
        
        manager.register_data("item_1")
        manager.mark_obsolete("item_1", "Outdated content")
        
        entry = manager.get_entry("item_1")
        assert entry.state == DataState.OBSOLETE
        assert entry.obsolete_reason == "Outdated content"
    
    def test_mark_for_technology_elimination(self, tmp_path):
        """Test marking data for tech elimination."""
        from ai_core.distributed_vc.data_lifecycle_manager import (
            DataLifecycleManager,
            DataLifecycleConfig,
            DataState,
        )
        
        config = DataLifecycleConfig()
        manager = DataLifecycleManager(str(tmp_path / "data"), config)
        
        manager.register_data("item_1", tech_id="old_framework")
        manager.register_data("item_2", tech_id="old_framework")
        manager.register_data("item_3", tech_id="new_framework")
        
        count = manager.mark_for_technology_elimination("old_framework")
        
        assert count == 2
        assert manager.get_entry("item_1").state == DataState.OBSOLETE
        assert manager.get_entry("item_2").state == DataState.OBSOLETE
        assert manager.get_entry("item_3").state == DataState.ACTIVE
    
    def test_protect_item(self, tmp_path):
        """Test item protection."""
        from ai_core.distributed_vc.data_lifecycle_manager import (
            DataLifecycleManager,
            DataLifecycleConfig,
        )
        
        config = DataLifecycleConfig()
        manager = DataLifecycleManager(str(tmp_path / "data"), config)
        
        manager.register_data("critical_item")
        manager.protect_item("critical_item")
        
        assert "critical_item" in manager._protected_items
        
        manager.unprotect_item("critical_item")
        assert "critical_item" not in manager._protected_items
    
    def test_get_stats(self, tmp_path):
        """Test statistics retrieval."""
        from ai_core.distributed_vc.data_lifecycle_manager import (
            DataLifecycleManager,
            DataLifecycleConfig,
        )
        
        config = DataLifecycleConfig()
        manager = DataLifecycleManager(str(tmp_path / "data"), config)
        
        manager.register_data("item_1")
        manager.register_data("item_2")
        manager.mark_obsolete("item_2", "test")
        
        stats = manager.get_stats()
        
        assert stats["total_registered"] == 2
        assert stats["by_state"]["active"] == 1
        assert stats["by_state"]["obsolete"] == 1


# =============================================================================
# Technology Elimination Tests
# =============================================================================

class TestTechEliminationManager:
    """Tests for TechEliminationManager."""
    
    @pytest.mark.asyncio
    async def test_evaluate_technology(self):
        """Test technology evaluation."""
        from ai_core.three_version_cycle.spiral_evolution_manager import (
            TechEliminationManager,
            TechEliminationConfig,
        )
        
        config = TechEliminationConfig(
            min_accuracy_threshold=0.75,
            max_error_rate_threshold=0.15,
            consecutive_failures_to_eliminate=3,
        )
        
        # Mock version manager
        version_manager = MagicMock()
        version_manager.get_technology = AsyncMock(return_value={
            "name": "test_tech",
            "metrics": {
                "accuracy": 0.70,  # Below threshold
                "error_rate": 0.10,
            }
        })
        
        manager = TechEliminationManager(version_manager, config=config)
        
        result = await manager.evaluate_technology("test_tech_id")
        
        assert result["found"] is True
        assert result["should_eliminate"] is True
        assert result["consecutive_failures"] == 1
        assert len(result["reasons"]) > 0
    
    @pytest.mark.asyncio
    async def test_consecutive_failures_tracking(self):
        """Test consecutive failure tracking."""
        from ai_core.three_version_cycle.spiral_evolution_manager import (
            TechEliminationManager,
            TechEliminationConfig,
        )
        
        config = TechEliminationConfig(
            consecutive_failures_to_eliminate=3,
            auto_eliminate=False,  # Don't auto-eliminate for testing
        )
        
        version_manager = MagicMock()
        version_manager.get_technology = AsyncMock(return_value={
            "name": "failing_tech",
            "metrics": {"accuracy": 0.60, "error_rate": 0.20}
        })
        
        manager = TechEliminationManager(version_manager, config=config)
        
        # Evaluate 3 times
        for i in range(3):
            result = await manager.evaluate_technology("failing_tech")
            assert result["consecutive_failures"] == i + 1
    
    def test_get_at_risk_technologies(self):
        """Test at-risk technology retrieval."""
        from ai_core.three_version_cycle.spiral_evolution_manager import (
            TechEliminationManager,
            TechEliminationConfig,
        )
        
        manager = TechEliminationManager(config=TechEliminationConfig())
        
        # Manually set failure counts
        manager.failure_counts["tech_1"] = 1
        manager.failure_counts["tech_2"] = 2
        
        at_risk = manager.get_at_risk_technologies()
        
        assert len(at_risk) == 2
        # Most at risk should be first
        assert at_risk[0]["tech_id"] == "tech_2"
        assert at_risk[0]["risk_level"] == "medium"
    
    def test_get_elimination_status(self):
        """Test elimination status retrieval."""
        from ai_core.three_version_cycle.spiral_evolution_manager import (
            TechEliminationManager,
            TechEliminationConfig,
        )
        
        config = TechEliminationConfig(
            min_accuracy_threshold=0.75,
            auto_eliminate=True,
        )
        manager = TechEliminationManager(config=config)
        
        status = manager.get_elimination_status()
        
        assert "config" in status
        assert "statistics" in status
        assert status["config"]["min_accuracy"] == 0.75
        assert status["config"]["auto_eliminate"] is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_learning_to_cleansing_flow(self, tmp_path):
        """Test data flow from learning to cleansing."""
        from ai_core.distributed_vc.auto_network_learning import (
            V1V3AutoLearningSystem,
            NetworkLearningConfig,
            LearningData,
            DataSource,
        )
        from ai_core.distributed_vc.data_cleansing_pipeline import (
            DataCleansingPipeline,
            CleansingConfig,
        )
        
        # Track processed items
        processed_items = []
        
        async def on_data_ready(items):
            processed_items.extend(items)
        
        # Setup learning system
        learning_config = NetworkLearningConfig(
            v1_learning_interval_minutes=1,
            v2_push_enabled=False,
        )
        
        # Setup cleansing pipeline
        cleansing_config = CleansingConfig(
            min_content_length=10,
            min_final_quality=0.3,
            v2_push_enabled=False,
        )
        
        learning_system = V1V3AutoLearningSystem("v1", learning_config, on_data_ready)
        pipeline = DataCleansingPipeline(cleansing_config)
        
        # Simulate data
        test_item = LearningData(
            data_id="integration_test",
            source=DataSource.GITHUB_TRENDING,
            title="Integration Test Repository",
            content="This is a comprehensive test content for integration testing. " * 10,
            url="https://github.com/test/integration",
            fetched_at=datetime.now(timezone.utc),
            quality_score=0.8,
        )
        
        # Process through pipeline
        class MockItem:
            data_id = test_item.data_id
            title = test_item.title
            content = test_item.content
            source = test_item.source.value
            quality_score = test_item.quality_score
            is_cleaned = False
            is_validated = False
            metadata = {}
        
        results = await pipeline.process_batch([MockItem()])
        
        assert len(results) == 1
        # Item should pass or have valid result
        assert results[0].data_id == "integration_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
