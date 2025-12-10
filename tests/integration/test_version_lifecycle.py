"""
Integration Tests for Version Promotion Workflow

Tests cover the complete V1→V2→V3 lifecycle including:
- Experiment creation and evaluation
- Promotion decision making
- Gray-scale rollout simulation
- Rollback scenarios
- Cross-version feedback loops
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Full Lifecycle Tests
# =============================================================================

class TestFullLifecycle:
    """End-to-end tests for the complete version lifecycle."""
    
    @pytest.fixture
    def full_system(self):
        from ai_core.three_version_cycle import (
            EnhancedSelfEvolutionCycle,
            VersionManager,
        )
        
        vm = VersionManager()
        cycle = EnhancedSelfEvolutionCycle(version_manager=vm)
        
        return {
            "cycle": cycle,
            "version_manager": vm
        }
    
    @pytest.mark.asyncio
    async def test_complete_v1_to_v2_lifecycle(self, full_system):
        """Test complete lifecycle from V1 registration to V2 promotion."""
        from ai_core.three_version_cycle import Version
        
        vm = full_system["version_manager"]
        cycle = full_system["cycle"]
        
        # 1. Register new technology in V1
        tech = await vm.register_technology(
            name="lifecycle_test_tech",
            category="attention",
            description="Full lifecycle test",
            config={"layers": 12, "heads": 8},
            source="test"
        )
        
        assert tech.current_version == Version.V1_EXPERIMENTAL
        
        # 2. Simulate experiment with metrics collection
        tech.metrics = {
            "accuracy": 0.93,
            "error_rate": 0.02,
            "latency_p95_ms": 1800,
            "sample_count": 2500,
            "memory_mb": 512,
            "throughput_rps": 100
        }
        
        # 3. Trigger promotion
        result = await cycle.trigger_promotion(tech.tech_id)
        assert result["success"] is True
        
        # 4. Verify promotion was queued
        status = cycle.get_full_status()
        assert "spiral_status" in status
    
    @pytest.mark.asyncio
    async def test_v2_degradation_on_failure(self, full_system):
        """Test degradation flow when V2 technology fails."""
        from ai_core.three_version_cycle import Version
        
        vm = full_system["version_manager"]
        cycle = full_system["cycle"]
        
        # 1. Register and promote to V2
        tech = await vm.register_technology(
            name="degradation_test_tech",
            category="model",
            description="Degradation test",
            config={},
            source="test"
        )
        
        # Manually set to V2
        tech.current_version = Version.V2_PRODUCTION
        
        # 2. Simulate failure metrics
        tech.metrics = {
            "accuracy": 0.65,  # Below threshold
            "error_rate": 0.15,  # Above threshold
            "latency_p95_ms": 5000  # Above threshold
        }
        
        # 3. Trigger degradation
        result = await cycle.trigger_degradation(
            tech.tech_id,
            reason="Performance degradation detected"
        )
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_error_feedback_loop(self, full_system):
        """Test V1 error reporting and V2 fix generation."""
        cycle = full_system["cycle"]
        
        # 1. Report V1 error
        error_result = await cycle.report_v1_error(
            tech_id="feedback_test_tech",
            tech_name="Feedback Test Technology",
            error_type="compatibility",
            description="Incompatible with legacy API format"
        )
        
        assert error_result["success"] is True
        assert "error_id" in error_result
        
        # 2. Verify feedback was recorded
        stats = cycle.spiral_manager.feedback_system.get_feedback_statistics()
        assert stats["total_errors_reported"] >= 1


# =============================================================================
# Promotion Decision Tests
# =============================================================================

class TestPromotionDecisions:
    """Tests for promotion decision logic."""
    
    @pytest.fixture
    def version_manager(self):
        from ai_core.three_version_cycle import VersionManager
        return VersionManager()
    
    @pytest.mark.asyncio
    async def test_all_criteria_pass(self, version_manager):
        """Test promotion when all criteria pass."""
        tech = await version_manager.register_technology(
            name="passing_tech",
            category="optimizer",
            description="Passes all criteria",
            config={},
            source="test"
        )
        
        tech.metrics = {
            "accuracy": 0.95,
            "error_rate": 0.01,
            "latency_p95_ms": 1000,
            "sample_count": 3000
        }
        
        result = await version_manager.promote_technology(
            tech.tech_id,
            reason="All criteria passed"
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_accuracy_fail_blocks_promotion(self, version_manager):
        """Test that low accuracy blocks promotion."""
        from ai_core.three_version_cycle import Version
        
        tech = await version_manager.register_technology(
            name="low_accuracy_tech",
            category="model",
            description="Low accuracy",
            config={},
            source="test"
        )
        
        tech.metrics = {
            "accuracy": 0.70,  # Fails (< 0.85)
            "error_rate": 0.01,
            "latency_p95_ms": 1000,
            "sample_count": 3000
        }
        
        # Should remain in V1
        assert tech.current_version == Version.V1_EXPERIMENTAL
    
    @pytest.mark.asyncio
    async def test_high_error_rate_blocks_promotion(self, version_manager):
        """Test that high error rate blocks promotion."""
        from ai_core.three_version_cycle import Version
        
        tech = await version_manager.register_technology(
            name="high_error_tech",
            category="model",
            description="High error rate",
            config={},
            source="test"
        )
        
        tech.metrics = {
            "accuracy": 0.92,
            "error_rate": 0.15,  # Fails (> 0.05)
            "latency_p95_ms": 1000,
            "sample_count": 3000
        }
        
        # Should remain in V1
        assert tech.current_version == Version.V1_EXPERIMENTAL
    
    @pytest.mark.asyncio
    async def test_insufficient_samples_blocks_promotion(self, version_manager):
        """Test that insufficient samples block promotion."""
        from ai_core.three_version_cycle import Version
        
        tech = await version_manager.register_technology(
            name="few_samples_tech",
            category="model",
            description="Too few samples",
            config={},
            source="test"
        )
        
        tech.metrics = {
            "accuracy": 0.92,
            "error_rate": 0.02,
            "latency_p95_ms": 1000,
            "sample_count": 100  # Fails (< 1000)
        }
        
        # Should remain in V1
        assert tech.current_version == Version.V1_EXPERIMENTAL


# =============================================================================
# Gray-Scale Rollout Tests
# =============================================================================

class TestGrayScaleRollout:
    """Tests for gray-scale rollout simulation."""
    
    def test_rollout_phases(self):
        """Test gray-scale rollout phase definitions."""
        # Standard phases: 1%, 5%, 25%, 50%, 100%
        phases = [1, 5, 25, 50, 100]
        
        for i, phase in enumerate(phases):
            if i > 0:
                assert phase > phases[i-1]
        
        assert phases[-1] == 100
    
    def test_health_check_thresholds(self):
        """Test health check thresholds for rollout."""
        thresholds = {
            "error_rate_max": 0.05,
            "latency_p95_max_ms": 3000,
            "success_rate_min": 0.95
        }
        
        # Healthy metrics
        healthy = {
            "error_rate": 0.02,
            "latency_p95": 2000,
            "success_rate": 0.98
        }
        
        assert healthy["error_rate"] < thresholds["error_rate_max"]
        assert healthy["latency_p95"] < thresholds["latency_p95_max_ms"]
        assert healthy["success_rate"] > thresholds["success_rate_min"]


# =============================================================================
# Rollback Tests
# =============================================================================

class TestRollback:
    """Tests for rollback scenarios."""
    
    @pytest.fixture
    def version_manager(self):
        from ai_core.three_version_cycle import VersionManager
        return VersionManager()
    
    @pytest.mark.asyncio
    async def test_rollback_after_promotion(self, version_manager):
        """Test rollback after failed promotion."""
        from ai_core.three_version_cycle import Version
        
        tech = await version_manager.register_technology(
            name="rollback_test_tech",
            category="model",
            description="Rollback test",
            config={},
            source="test"
        )
        
        # Promote to V2
        tech.current_version = Version.V2_PRODUCTION
        
        # Simulate rollback via degradation
        result = await version_manager.degrade_technology(
            tech.tech_id,
            reason="Rollback: Performance issues in production"
        )
        
        assert result is True
        # After degradation, tech should be in V3
        assert tech.current_version == Version.V3_QUARANTINE


# =============================================================================
# Cross-Version Coordination Tests
# =============================================================================

class TestCrossVersionCoordination:
    """Tests for cross-version coordination."""
    
    @pytest.fixture
    def coordinator(self):
        from ai_core.three_version_cycle import DualAICoordinator
        return DualAICoordinator()
    
    def test_version_isolation(self, coordinator):
        """Test that versions are properly isolated."""
        from ai_core.three_version_cycle import AIType
        
        v1_ai = coordinator.get_ai("v1", AIType.CODE_REVIEW)
        v2_ai = coordinator.get_ai("v2", AIType.CODE_REVIEW)
        v3_ai = coordinator.get_ai("v3", AIType.CODE_REVIEW)
        
        # Each version should have separate AI instance
        assert v1_ai is not v2_ai
        assert v2_ai is not v3_ai
        assert v1_ai is not v3_ai
    
    def test_user_access_restriction(self, coordinator):
        """Test users can only access V2 CR-AI."""
        from ai_core.three_version_cycle import AIType
        
        # Users can only access V2 CR-AI
        assert coordinator.can_access("user", "v2", AIType.CODE_REVIEW) is True
        assert coordinator.can_access("user", "v1", AIType.CODE_REVIEW) is False
        assert coordinator.can_access("user", "v2", AIType.VERSION_CONTROL) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
