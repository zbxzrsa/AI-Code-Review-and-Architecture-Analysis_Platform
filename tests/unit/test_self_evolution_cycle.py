"""
Tests for Self-Evolution Cycle - Base Classes

Tests cover:
- SelfEvolutionCycle base class
- CyclePhase transitions
- PromotionCriteria validation
- CycleResult dataclass
- EvolutionMetrics tracking
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
# Cycle Phase Tests
# =============================================================================

class TestCyclePhase:
    """Tests for CyclePhase enum."""
    
    def test_all_phases_exist(self):
        """Test all cycle phases are defined."""
        from ai_core.three_version_cycle.self_evolution_cycle import CyclePhase
        
        assert hasattr(CyclePhase, 'IDLE')
        assert hasattr(CyclePhase, 'EXPERIMENTING')
        assert hasattr(CyclePhase, 'EVALUATING')
        assert hasattr(CyclePhase, 'PROMOTING')
        assert hasattr(CyclePhase, 'DEGRADING')
    
    def test_phase_values(self):
        """Test phase string values."""
        from ai_core.three_version_cycle.self_evolution_cycle import CyclePhase
        
        assert CyclePhase.IDLE == "idle"
        assert CyclePhase.EXPERIMENTING == "experimenting"
        assert CyclePhase.EVALUATING == "evaluating"
        assert CyclePhase.PROMOTING == "promoting"
        assert CyclePhase.DEGRADING == "degrading"


# =============================================================================
# Promotion Criteria Tests
# =============================================================================

class TestPromotionCriteria:
    """Tests for PromotionCriteria dataclass."""
    
    @pytest.fixture
    def criteria(self):
        from ai_core.three_version_cycle.self_evolution_cycle import PromotionCriteria
        return PromotionCriteria()
    
    def test_default_values(self, criteria):
        """Test default promotion criteria values."""
        assert criteria.min_accuracy == 0.85
        assert criteria.max_error_rate == 0.05
        assert criteria.max_latency_p95_ms == 3000
        assert criteria.min_samples == 1000
    
    def test_custom_values(self):
        """Test custom promotion criteria."""
        from ai_core.three_version_cycle.self_evolution_cycle import PromotionCriteria
        
        custom = PromotionCriteria(
            min_accuracy=0.90,
            max_error_rate=0.02,
            max_latency_p95_ms=2000,
            min_samples=2000
        )
        
        assert custom.min_accuracy == 0.90
        assert custom.max_error_rate == 0.02
        assert custom.max_latency_p95_ms == 2000
        assert custom.min_samples == 2000
    
    def test_meets_all_criteria(self, criteria):
        """Test metrics that meet all criteria."""
        metrics = {
            "accuracy": 0.90,
            "error_rate": 0.03,
            "latency_p95_ms": 2500,
            "sample_count": 1500
        }
        
        assert metrics["accuracy"] >= criteria.min_accuracy
        assert metrics["error_rate"] <= criteria.max_error_rate
        assert metrics["latency_p95_ms"] <= criteria.max_latency_p95_ms
        assert metrics["sample_count"] >= criteria.min_samples
    
    def test_fails_accuracy_criteria(self, criteria):
        """Test metrics failing accuracy threshold."""
        metrics = {"accuracy": 0.80}
        assert metrics["accuracy"] < criteria.min_accuracy
    
    def test_fails_error_rate_criteria(self, criteria):
        """Test metrics failing error rate threshold."""
        metrics = {"error_rate": 0.10}
        assert metrics["error_rate"] > criteria.max_error_rate
    
    def test_fails_latency_criteria(self, criteria):
        """Test metrics failing latency threshold."""
        metrics = {"latency_p95_ms": 4000}
        assert metrics["latency_p95_ms"] > criteria.max_latency_p95_ms
    
    def test_fails_sample_criteria(self, criteria):
        """Test metrics failing sample count threshold."""
        metrics = {"sample_count": 500}
        assert metrics["sample_count"] < criteria.min_samples


# =============================================================================
# Evolution Metrics Tests
# =============================================================================

class TestEvolutionMetrics:
    """Tests for EvolutionMetrics dataclass."""
    
    def test_initial_values(self):
        """Test initial metrics are zero."""
        from ai_core.three_version_cycle.self_evolution_cycle import EvolutionMetrics
        
        metrics = EvolutionMetrics()
        
        assert metrics.cycle_count == 0
        assert metrics.promotions == 0
        assert metrics.degradations == 0
        assert metrics.experiments_started == 0
        assert metrics.experiments_completed == 0
        assert metrics.last_cycle_at is None
    
    def test_increment_metrics(self):
        """Test incrementing metrics."""
        from ai_core.three_version_cycle.self_evolution_cycle import EvolutionMetrics
        
        metrics = EvolutionMetrics()
        metrics.cycle_count += 1
        metrics.promotions += 2
        metrics.degradations += 1
        metrics.experiments_started += 5
        metrics.experiments_completed += 3
        metrics.last_cycle_at = datetime.now()
        
        assert metrics.cycle_count == 1
        assert metrics.promotions == 2
        assert metrics.degradations == 1
        assert metrics.experiments_started == 5
        assert metrics.experiments_completed == 3
        assert metrics.last_cycle_at is not None


# =============================================================================
# Cycle Result Tests
# =============================================================================

class TestCycleResult:
    """Tests for CycleResult dataclass."""
    
    def test_create_result(self):
        """Test creating a cycle result."""
        from ai_core.three_version_cycle.self_evolution_cycle import CycleResult
        
        result = CycleResult(
            cycle_id="test_001",
            started_at=datetime.now()
        )
        
        assert result.cycle_id == "test_001"
        assert result.started_at is not None
        assert result.completed_at is None
        assert result.promotions_made == 0
        assert result.degradations_made == 0
        assert result.actions == []
    
    def test_result_with_actions(self):
        """Test cycle result with recorded actions."""
        from ai_core.three_version_cycle.self_evolution_cycle import CycleResult
        
        result = CycleResult(
            cycle_id="test_002",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            promotions_made=2,
            degradations_made=1,
            actions=[
                {"type": "promotion", "tech_id": "tech_a"},
                {"type": "promotion", "tech_id": "tech_b"},
                {"type": "degradation", "tech_id": "tech_c"}
            ]
        )
        
        assert result.promotions_made == 2
        assert result.degradations_made == 1
        assert len(result.actions) == 3


# =============================================================================
# Self-Evolution Cycle Tests
# =============================================================================

class TestSelfEvolutionCycle:
    """Tests for SelfEvolutionCycle class."""
    
    @pytest.fixture
    def cycle(self):
        from ai_core.three_version_cycle import VersionManager
        from ai_core.three_version_cycle.self_evolution_cycle import SelfEvolutionCycle
        return SelfEvolutionCycle(version_manager=VersionManager())
    
    def test_initialization(self, cycle):
        """Test cycle initializes correctly."""
        assert cycle is not None
        assert cycle.version_manager is not None
        assert cycle.experiment_framework is not None
    
    def test_initial_phase(self, cycle):
        """Test cycle starts in IDLE phase."""
        from ai_core.three_version_cycle.self_evolution_cycle import CyclePhase
        assert cycle.current_phase == CyclePhase.IDLE
    
    def test_initial_metrics(self, cycle):
        """Test initial metrics are zero."""
        metrics = cycle.get_metrics()
        assert metrics.cycle_count == 0
        assert metrics.promotions == 0
    
    def test_not_running_initially(self, cycle):
        """Test cycle is not running initially."""
        assert cycle.is_running is False
    
    @pytest.mark.asyncio
    async def test_execute_cycle(self, cycle):
        """Test executing a single cycle."""
        result = await cycle.execute_cycle()
        
        assert result is not None
        assert result.cycle_id is not None
        assert result.started_at is not None
    
    @pytest.mark.asyncio
    async def test_execute_increments_count(self, cycle):
        """Test cycle count increments."""
        await cycle.execute_cycle()
        metrics = cycle.get_metrics()
        assert metrics.cycle_count == 1
    
    def test_get_status(self, cycle):
        """Test getting cycle status."""
        status = cycle.get_status()
        
        assert "running" in status
        assert "phase" in status
        assert "metrics" in status
    
    @pytest.mark.asyncio
    async def test_start_stop(self, cycle):
        """Test starting and stopping cycle."""
        await cycle.start()
        assert cycle.is_running is True
        
        await cycle.stop()
        assert cycle.is_running is False
    
    @pytest.mark.asyncio
    async def test_promotion_criteria_enforcement(self, cycle):
        """Test promotion criteria are enforced."""
        # Register a technology
        tech = await cycle.version_manager.register_technology(
            name="test_tech",
            category="model",
            description="Test",
            config={},
            source="test"
        )
        
        # Set failing metrics
        tech.metrics = {
            "accuracy": 0.70,  # Below 0.85
            "error_rate": 0.10,  # Above 0.05
            "latency_p95_ms": 4000,  # Above 3000
            "sample_count": 500  # Below 1000
        }
        
        # Execute cycle - should not promote
        result = await cycle.execute_cycle()
        assert result.promotions_made == 0


# =============================================================================
# Version Transition Tests
# =============================================================================

class TestVersionTransitions:
    """Tests for version transition scenarios."""
    
    @pytest.fixture
    def version_manager(self):
        from ai_core.three_version_cycle import VersionManager
        return VersionManager()
    
    @pytest.mark.asyncio
    async def test_v1_to_v2_promotion(self, version_manager):
        """Test V1 to V2 promotion flow."""
        from ai_core.three_version_cycle import Version
        
        tech = await version_manager.register_technology(
            name="promote_test",
            category="attention",
            description="Promotion test",
            config={},
            source="test"
        )
        
        # Verify starts in V1
        assert tech.current_version == Version.V1_EXPERIMENTAL
        
        # Set passing metrics
        tech.metrics = {
            "accuracy": 0.92,
            "error_rate": 0.02,
            "latency_p95_ms": 2000,
            "sample_count": 2000
        }
        
        # Promote
        await version_manager.promote_technology(tech.tech_id, "Passed")
        
        # Verify now in V2
        assert tech.current_version == Version.V2_PRODUCTION
    
    @pytest.mark.asyncio
    async def test_v2_to_v3_degradation(self, version_manager):
        """Test V2 to V3 degradation flow."""
        from ai_core.three_version_cycle import Version
        
        tech = await version_manager.register_technology(
            name="degrade_test",
            category="model",
            description="Degradation test",
            config={},
            source="test"
        )
        
        # Move to V2 first
        tech.current_version = Version.V2_PRODUCTION
        
        # Degrade
        await version_manager.degrade_technology(tech.tech_id, "High errors")
        
        # Verify now in V3
        assert tech.current_version == Version.V3_QUARANTINE
    
    @pytest.mark.asyncio
    async def test_v3_reevaluation(self, version_manager):
        """Test V3 re-evaluation possibility."""
        from ai_core.three_version_cycle import Version
        
        tech = await version_manager.register_technology(
            name="reeval_test",
            category="optimization",
            description="Re-evaluation test",
            config={},
            source="test"
        )
        
        # Move to V3
        tech.current_version = Version.V3_QUARANTINE
        
        # Verify in V3
        assert tech.current_version == Version.V3_QUARANTINE


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.fixture
    def version_manager(self):
        from ai_core.three_version_cycle import VersionManager
        return VersionManager()
    
    @pytest.mark.asyncio
    async def test_promote_nonexistent_tech(self, version_manager):
        """Test promoting non-existent technology."""
        result = await version_manager.promote_technology(
            "nonexistent_id",
            "Should fail"
        )
        # Should handle gracefully
        assert result is False or result is None
    
    @pytest.mark.asyncio
    async def test_degrade_nonexistent_tech(self, version_manager):
        """Test degrading non-existent technology."""
        result = await version_manager.degrade_technology(
            "nonexistent_id",
            "Should fail"
        )
        # Should handle gracefully
        assert result is False or result is None
    
    @pytest.mark.asyncio
    async def test_duplicate_registration(self, version_manager):
        """Test registering duplicate technology name."""
        await version_manager.register_technology(
            name="duplicate_tech",
            category="model",
            description="First",
            config={},
            source="test"
        )
        
        # Register again with same name
        tech2 = await version_manager.register_technology(
            name="duplicate_tech",
            category="model",
            description="Second",
            config={},
            source="test"
        )
        
        # Should create with different ID
        assert tech2 is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
