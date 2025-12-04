"""
Tests for Three-Version Self-Evolution Cycle

Tests cover:
- Version Manager
- Dual-AI Coordinator
- Cross-Version Feedback
- V3 Comparison Engine
- Spiral Evolution Manager
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Version Manager Tests
# =============================================================================

class TestVersionManager:
    """Tests for VersionManager class."""
    
    @pytest.fixture
    def version_manager(self):
        from ai_core.three_version_cycle import VersionManager
        return VersionManager()
    
    def test_initialization(self, version_manager):
        """Test VersionManager initializes with all three versions."""
        from ai_core.three_version_cycle import Version
        
        assert version_manager is not None
        assert Version.V1_EXPERIMENTAL in version_manager._states
        assert Version.V2_PRODUCTION in version_manager._states
        assert Version.V3_QUARANTINE in version_manager._states
    
    @pytest.mark.asyncio
    async def test_register_technology(self, version_manager):
        """Test registering a new technology."""
        tech = await version_manager.register_technology(
            name="test_attention",
            category="attention",
            description="Test attention mechanism",
            config={"heads": 8},
            source="test",
        )
        
        assert tech is not None
        assert tech.name == "test_attention"
        assert tech.category == "attention"
        assert tech.tech_id in version_manager._technologies
    
    @pytest.mark.asyncio
    async def test_promote_technology(self, version_manager):
        """Test promoting technology from V1 to V2."""
        # Register technology
        tech = await version_manager.register_technology(
            name="promotable_tech",
            category="model",
            description="Technology to promote",
            config={},
            source="test",
        )
        
        # Set good metrics
        tech.metrics = {
            "accuracy": 0.90,
            "error_rate": 0.02,
            "latency_p95_ms": 2000,
            "sample_count": 1500,
        }
        
        # Promote
        result = await version_manager.promote_technology(
            tech.tech_id,
            reason="Passed all criteria"
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_degrade_technology(self, version_manager):
        """Test degrading technology from V2 to V3."""
        from ai_core.three_version_cycle import Version
        
        # Register technology
        tech = await version_manager.register_technology(
            name="degradable_tech",
            category="model",
            description="Technology to degrade",
            config={},
            source="test",
        )
        
        # Move to V2 first
        tech.current_version = Version.V2_PRODUCTION
        
        # Degrade
        result = await version_manager.degrade_technology(
            tech.tech_id,
            reason="High error rate"
        )
        
        assert result is True
    
    def test_get_status_report(self, version_manager):
        """Test getting status report."""
        report = version_manager.get_status_report()
        
        assert "states" in report
        assert "metrics" in report
        assert "technologies" in report


# =============================================================================
# Dual-AI Coordinator Tests
# =============================================================================

class TestDualAICoordinator:
    """Tests for DualAICoordinator class."""
    
    @pytest.fixture
    def coordinator(self):
        from ai_core.three_version_cycle import DualAICoordinator
        return DualAICoordinator()
    
    def test_initialization(self, coordinator):
        """Test coordinator initializes with all version pairs."""
        assert "v1" in coordinator._version_pairs
        assert "v2" in coordinator._version_pairs
        assert "v3" in coordinator._version_pairs
    
    def test_get_ai(self, coordinator):
        """Test getting AI instances."""
        from ai_core.three_version_cycle import AIType
        
        v1_vc = coordinator.get_ai("v1", AIType.VERSION_CONTROL)
        v1_cr = coordinator.get_ai("v1", AIType.CODE_REVIEW)
        v2_vc = coordinator.get_ai("v2", AIType.VERSION_CONTROL)
        v2_cr = coordinator.get_ai("v2", AIType.CODE_REVIEW)
        
        assert v1_vc is not None
        assert v1_cr is not None
        assert v2_vc is not None
        assert v2_cr is not None
    
    def test_user_accessible_ai(self, coordinator):
        """Test that users can only access V2 CR-AI."""
        user_ai = coordinator.get_user_accessible_ai()
        
        assert user_ai is not None
        assert user_ai.version == "v2"
    
    def test_access_control_user(self, coordinator):
        """Test access control for regular users."""
        from ai_core.three_version_cycle import AIType
        
        # Users can only access V2 CR-AI
        assert coordinator.can_access("user", "v2", AIType.CODE_REVIEW) is True
        assert coordinator.can_access("user", "v1", AIType.CODE_REVIEW) is False
        assert coordinator.can_access("user", "v3", AIType.CODE_REVIEW) is False
        assert coordinator.can_access("user", "v2", AIType.VERSION_CONTROL) is False
    
    def test_access_control_admin(self, coordinator):
        """Test access control for admins."""
        from ai_core.three_version_cycle import AIType
        
        # Admins can access all versions
        assert coordinator.can_access("admin", "v1", AIType.CODE_REVIEW) is True
        assert coordinator.can_access("admin", "v2", AIType.CODE_REVIEW) is True
        assert coordinator.can_access("admin", "v3", AIType.CODE_REVIEW) is True
        assert coordinator.can_access("admin", "v1", AIType.VERSION_CONTROL) is True
        assert coordinator.can_access("admin", "v2", AIType.VERSION_CONTROL) is True
    
    @pytest.mark.asyncio
    async def test_route_request(self, coordinator):
        """Test request routing."""
        result = await coordinator.route_request(
            user_role="user",
            request_type="code_review",
            request_data={"code": "print('hello')"},
        )
        
        assert result["success"] is True
        assert result["version"] == "v2"
    
    def test_get_all_status(self, coordinator):
        """Test getting status of all AIs."""
        status = coordinator.get_all_status()
        
        assert "v1" in status
        assert "v2" in status
        assert "v3" in status
        assert "vc_ai" in status["v1"]
        assert "cr_ai" in status["v1"]


# =============================================================================
# Cross-Version Feedback Tests
# =============================================================================

class TestCrossVersionFeedback:
    """Tests for CrossVersionFeedbackSystem class."""
    
    @pytest.fixture
    def feedback_system(self):
        from ai_core.three_version_cycle import CrossVersionFeedbackSystem
        return CrossVersionFeedbackSystem()
    
    @pytest.mark.asyncio
    async def test_report_v1_error(self, feedback_system):
        """Test reporting a V1 error."""
        from ai_core.three_version_cycle import ErrorType
        
        error = await feedback_system.report_v1_error(
            technology_id="test_tech_1",
            technology_name="Test Technology",
            error_type=ErrorType.COMPATIBILITY,
            description="Incompatible with existing system",
        )
        
        assert error is not None
        assert error.error_id in feedback_system._errors
        assert error.technology_id == "test_tech_1"
    
    @pytest.mark.asyncio
    async def test_apply_fix(self, feedback_system):
        """Test applying a fix to V1."""
        from ai_core.three_version_cycle import ErrorType, V2Fix, FixStatus
        
        # Report error first
        error = await feedback_system.report_v1_error(
            technology_id="test_tech_2",
            technology_name="Test Technology 2",
            error_type=ErrorType.PERFORMANCE,
            description="Performance below threshold",
        )
        
        # Get the fix that was generated
        if feedback_system._fixes:
            fix_id = list(feedback_system._fixes.keys())[0]
            result = await feedback_system.apply_fix_to_v1(fix_id)
            assert result is True
    
    def test_get_feedback_statistics(self, feedback_system):
        """Test getting feedback statistics."""
        stats = feedback_system.get_feedback_statistics()
        
        assert "total_errors_reported" in stats
        assert "total_fixes_generated" in stats
        assert "fix_success_rate" in stats


# =============================================================================
# V3 Comparison Engine Tests
# =============================================================================

class TestV3ComparisonEngine:
    """Tests for V3ComparisonEngine class."""
    
    @pytest.fixture
    def comparison_engine(self):
        from ai_core.three_version_cycle import V3ComparisonEngine
        return V3ComparisonEngine()
    
    @pytest.mark.asyncio
    async def test_quarantine_technology(self, comparison_engine):
        """Test quarantining a technology."""
        from ai_core.three_version_cycle import ExclusionReason
        
        profile = await comparison_engine.quarantine_technology(
            tech_id="failed_tech_1",
            name="Failed Technology",
            category="model",
            source="v2",
            metrics={
                "accuracy": 0.60,
                "error_rate": 0.25,
                "latency_p95_ms": 5000,
            },
            reason=ExclusionReason.POOR_PERFORMANCE,
        )
        
        assert profile is not None
        assert profile.tech_id == "failed_tech_1"
        assert profile.tech_id in comparison_engine._profiles
    
    @pytest.mark.asyncio
    async def test_is_excluded(self, comparison_engine):
        """Test checking if technology is excluded."""
        from ai_core.three_version_cycle import ExclusionReason
        
        # Quarantine with permanent exclusion criteria
        await comparison_engine.quarantine_technology(
            tech_id="excluded_tech",
            name="Excluded Technology",
            category="model",
            source="v2",
            metrics={
                "accuracy": 0.40,  # Very low - should trigger permanent
                "error_rate": 0.60,
            },
            reason=ExclusionReason.SECURITY_VULNERABILITY,
        )
        
        is_excluded = await comparison_engine.is_excluded("excluded_tech")
        assert is_excluded is True
    
    @pytest.mark.asyncio
    async def test_compare_technology(self, comparison_engine):
        """Test comparing technology against baseline."""
        result = await comparison_engine.compare_technology(
            tech_id="new_tech",
            tech_metrics={
                "accuracy": 0.88,
                "error_rate": 0.03,
                "latency_p95_ms": 2500,
            },
        )
        
        assert result is not None
        assert "metrics_delta" in result.__dict__
        assert "recommendation" in result.__dict__
    
    def test_get_quarantine_statistics(self, comparison_engine):
        """Test getting quarantine statistics."""
        stats = comparison_engine.get_quarantine_statistics()
        
        assert "total_quarantined" in stats
        assert "permanent_exclusions" in stats
        assert "temporary_exclusions" in stats


# =============================================================================
# Spiral Evolution Manager Tests
# =============================================================================

class TestSpiralEvolutionManager:
    """Tests for SpiralEvolutionManager class."""
    
    @pytest.fixture
    def spiral_manager(self):
        from ai_core.three_version_cycle import SpiralEvolutionManager
        return SpiralEvolutionManager()
    
    def test_initialization(self, spiral_manager):
        """Test manager initialization."""
        assert spiral_manager.dual_ai is not None
        assert spiral_manager.feedback_system is not None
        assert spiral_manager.comparison_engine is not None
    
    @pytest.mark.asyncio
    async def test_trigger_promotion(self, spiral_manager):
        """Test triggering promotion."""
        result = await spiral_manager.trigger_promotion("tech_123")
        
        assert result["success"] is True
        assert "tech_123" in spiral_manager._pending_promotions
    
    @pytest.mark.asyncio
    async def test_trigger_degradation(self, spiral_manager):
        """Test triggering degradation."""
        result = await spiral_manager.trigger_degradation(
            "tech_456",
            reason="High error rate"
        )
        
        assert result["success"] is True
        assert "tech_456" in spiral_manager._pending_degradations
    
    @pytest.mark.asyncio
    async def test_request_reevaluation(self, spiral_manager):
        """Test requesting re-evaluation."""
        result = await spiral_manager.request_reevaluation("tech_789")
        
        assert result["success"] is True
        assert "tech_789" in spiral_manager._pending_reevaluations
    
    def test_get_cycle_status(self, spiral_manager):
        """Test getting cycle status."""
        status = spiral_manager.get_cycle_status()
        
        assert "running" in status
        assert "pending" in status
        assert "ai_status" in status


# =============================================================================
# Enhanced Self-Evolution Cycle Tests
# =============================================================================

class TestEnhancedSelfEvolutionCycle:
    """Tests for EnhancedSelfEvolutionCycle class."""
    
    @pytest.fixture
    def evolution_cycle(self):
        from ai_core.three_version_cycle import EnhancedSelfEvolutionCycle
        return EnhancedSelfEvolutionCycle()
    
    def test_initialization(self, evolution_cycle):
        """Test cycle initialization."""
        assert evolution_cycle.spiral_manager is not None
        assert evolution_cycle.version_manager is not None
    
    @pytest.mark.asyncio
    async def test_report_v1_error(self, evolution_cycle):
        """Test reporting V1 error through cycle."""
        result = await evolution_cycle.report_v1_error(
            tech_id="test_tech",
            tech_name="Test Technology",
            error_type="compatibility",
            description="Test error",
        )
        
        assert result["success"] is True
        assert "error_id" in result
    
    def test_get_full_status(self, evolution_cycle):
        """Test getting full status."""
        status = evolution_cycle.get_full_status()
        
        assert "running" in status
        assert "spiral_status" in status
    
    def test_get_dual_ai_status(self, evolution_cycle):
        """Test getting dual AI status."""
        status = evolution_cycle.get_dual_ai_status()
        
        assert "v1" in status
        assert "v2" in status
        assert "v3" in status


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete three-version cycle."""
    
    @pytest.fixture
    def full_system(self):
        from ai_core.three_version_cycle import (
            EnhancedSelfEvolutionCycle,
            VersionManager,
        )
        
        version_manager = VersionManager()
        cycle = EnhancedSelfEvolutionCycle(version_manager=version_manager)
        
        return {
            "cycle": cycle,
            "version_manager": version_manager,
        }
    
    @pytest.mark.asyncio
    async def test_full_promotion_flow(self, full_system):
        """Test full promotion flow from V1 to V2."""
        cycle = full_system["cycle"]
        vm = full_system["version_manager"]
        
        # Register technology in V1
        tech = await vm.register_technology(
            name="integration_test_tech",
            category="attention",
            description="Integration test technology",
            config={},
            source="test",
        )
        
        # Set passing metrics
        tech.metrics = {
            "accuracy": 0.92,
            "error_rate": 0.02,
            "latency_p95_ms": 1500,
            "sample_count": 2000,
        }
        
        # Trigger promotion
        result = await cycle.trigger_promotion(tech.tech_id)
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_error_remediation_flow(self, full_system):
        """Test error remediation flow (V1 error -> V2 fix)."""
        cycle = full_system["cycle"]
        
        # Report V1 error
        result = await cycle.report_v1_error(
            tech_id="error_test_tech",
            tech_name="Error Test Technology",
            error_type="compatibility",
            description="Test compatibility error",
        )
        
        assert result["success"] is True
        
        # Check that error was recorded
        stats = cycle.spiral_manager.feedback_system.get_feedback_statistics()
        assert stats["total_errors_reported"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
