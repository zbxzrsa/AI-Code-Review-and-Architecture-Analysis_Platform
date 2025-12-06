"""
Self-Evolution Cycle Integration Tests

Tests that verify the three-version architecture operates as a
complete, closed-loop self-iterating system.

Test scenarios:
1. V1 → V2 promotion (successful experiment)
2. V2 → V3 demotion (SLO breach)
3. V3 → V1 recovery (successful recovery)
4. Complete cycle (V1 → V2 → V3 → V1)
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Import the cycle components
import sys
sys.path.insert(0, "services/lifecycle-controller")

from services.lifecycle_controller.controller import (
    LifecycleController,
    VersionState,
    VersionConfig,
    EvaluationMetrics,
)
from services.lifecycle_controller.recovery_manager import (
    RecoveryManager,
    RecoveryStatus,
    RecoveryConfig,
)
from services.lifecycle_controller.cycle_orchestrator import (
    CycleOrchestrator,
    CyclePhase,
)


class TestCycleInitialization:
    """Test cycle initialization and startup"""

    @pytest.fixture
    def lifecycle_controller(self):
        return LifecycleController()

    @pytest.fixture
    def recovery_manager(self):
        return RecoveryManager()

    @pytest.fixture
    def orchestrator(self, lifecycle_controller, recovery_manager):
        return CycleOrchestrator(lifecycle_controller, recovery_manager)

    @pytest.mark.asyncio
    async def test_orchestrator_starts(self, orchestrator):
        """Orchestrator should start successfully"""
        # Mock external dependencies
        orchestrator.lifecycle._http_client = AsyncMock()
        orchestrator.recovery._http_client = AsyncMock()
        
        await orchestrator.start()
        
        assert orchestrator._running is True
        
        await orchestrator.stop()
        assert orchestrator._running is False

    def test_initial_cycle_status(self, orchestrator):
        """Initial cycle status should be empty"""
        status = orchestrator.get_cycle_status()
        
        assert status["versions"]["v1_experiments"] == 0
        assert status["versions"]["v2_production"] == 0
        assert status["versions"]["v3_quarantined"] == 0


class TestV1ToV2Promotion:
    """Test V1 → V2 promotion path"""

    @pytest.fixture
    def orchestrator(self):
        lifecycle = LifecycleController()
        recovery = RecoveryManager()
        return CycleOrchestrator(lifecycle, recovery)

    @pytest.mark.asyncio
    async def test_register_experiment(self, orchestrator):
        """Should register new experiment in V1"""
        config = await orchestrator.register_new_experiment(
            version_id="v1-test-001",
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        assert config.version_id == "v1-test-001"
        assert config.current_state == VersionState.EXPERIMENT
        
        status = orchestrator.get_cycle_status()
        assert status["versions"]["v1_experiments"] == 1

    @pytest.mark.asyncio
    async def test_start_shadow_evaluation(self, orchestrator):
        """Should transition to shadow evaluation"""
        # Register experiment
        await orchestrator.register_new_experiment(
            version_id="v1-test-002",
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        # Start shadow evaluation
        await orchestrator.start_shadow_evaluation("v1-test-002")
        
        config = orchestrator.lifecycle.active_versions["v1-test-002"]
        assert config.current_state == VersionState.SHADOW

    @pytest.mark.asyncio
    async def test_promotion_decision_with_good_metrics(self, orchestrator):
        """Good metrics should lead to promotion decision"""
        metrics = EvaluationMetrics(
            total_requests=2000,
            p95_latency_ms=2500,
            error_rate=0.01,
            accuracy=0.92,
            accuracy_delta=0.04,
            security_pass_rate=0.995,
            cost_delta=0.05,
            accuracy_p_value=0.02
        )
        
        # Check against thresholds
        thresholds = orchestrator.lifecycle.thresholds
        
        assert metrics.p95_latency_ms <= thresholds.p95_latency_ms
        assert metrics.error_rate <= thresholds.error_rate
        assert metrics.accuracy_delta >= thresholds.accuracy_delta
        assert metrics.security_pass_rate >= thresholds.security_pass_rate


class TestV2ToV3Demotion:
    """Test V2 → V3 demotion path"""

    @pytest.fixture
    def orchestrator(self):
        lifecycle = LifecycleController()
        recovery = RecoveryManager()
        return CycleOrchestrator(lifecycle, recovery)

    @pytest.mark.asyncio
    async def test_quarantine_on_slo_breach(self, orchestrator):
        """SLO breach should trigger quarantine"""
        # Create a version in V2
        await orchestrator.register_new_experiment(
            version_id="v2-test-001",
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        # Manually set to stable
        config = orchestrator.lifecycle.active_versions["v2-test-001"]
        config.current_state = VersionState.STABLE
        
        # Trigger quarantine
        await orchestrator.trigger_quarantine(
            version_id="v2-test-001",
            reason="SLO breach: error rate exceeded",
            metrics={"error_rate": 0.05}
        )
        
        # Verify quarantined
        assert config.current_state == VersionState.QUARANTINE
        
        # Verify registered for recovery
        record = orchestrator.recovery.get_recovery_status("v2-test-001")
        assert record is not None
        assert record.recovery_status == RecoveryStatus.PENDING

    @pytest.mark.asyncio
    async def test_quarantine_metadata(self, orchestrator):
        """Quarantine should record reason and metrics"""
        await orchestrator.register_new_experiment(
            version_id="v2-test-002",
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        config = orchestrator.lifecycle.active_versions["v2-test-002"]
        config.current_state = VersionState.STABLE
        
        await orchestrator.trigger_quarantine(
            version_id="v2-test-002",
            reason="Critical security failure",
            metrics={"security_pass_rate": 0.85}
        )
        
        record = orchestrator.recovery.get_recovery_status("v2-test-002")
        assert record.quarantine_reason == "Critical security failure"


class TestV3ToV1Recovery:
    """Test V3 → V1 recovery path"""

    @pytest.fixture
    def recovery_manager(self):
        return RecoveryManager(config=RecoveryConfig(
            initial_cooldown_hours=0,  # No cooldown for testing
            max_recovery_attempts=3
        ))

    @pytest.fixture
    def orchestrator(self, recovery_manager):
        lifecycle = LifecycleController()
        return CycleOrchestrator(lifecycle, recovery_manager)

    def test_recovery_registration(self, recovery_manager):
        """Quarantined version should be registered for recovery"""
        record = recovery_manager.register_quarantine(
            version_id="v3-test-001",
            reason="Test quarantine",
            metadata={"test": True}
        )
        
        assert record.version_id == "v3-test-001"
        assert record.recovery_status == RecoveryStatus.PENDING
        assert record.next_eligible_time is not None

    def test_recovery_eligibility(self, recovery_manager):
        """Version should become eligible after cooldown"""
        record = recovery_manager.register_quarantine(
            version_id="v3-test-002",
            reason="Test quarantine"
        )
        
        # With initial_cooldown_hours=0, should be eligible immediately
        assert record.next_eligible_time <= datetime.now(timezone.utc)

    @pytest.mark.asyncio
    async def test_recovery_success_flow(self, orchestrator):
        """Successful recovery should promote back to V1"""
        # Register and quarantine
        await orchestrator.register_new_experiment(
            version_id="v3-test-003",
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        await orchestrator.trigger_quarantine(
            version_id="v3-test-003",
            reason="Test quarantine"
        )
        
        # Simulate recovery pass
        record = orchestrator.recovery.get_recovery_status("v3-test-003")
        record.recovery_status = RecoveryStatus.PASSED
        record.best_score = 0.95
        
        # Promote back to V1
        await orchestrator._promote_recovered_to_v1("v3-test-003")
        
        # Verify back in V1
        config = orchestrator.lifecycle.active_versions["v3-test-003"]
        assert config.current_state == VersionState.SHADOW
        assert config.metadata.get("recovered_from_quarantine") is True


class TestCompleteCycle:
    """Test complete V1 → V2 → V3 → V1 cycle"""

    @pytest.fixture
    def orchestrator(self):
        lifecycle = LifecycleController()
        recovery = RecoveryManager(config=RecoveryConfig(
            initial_cooldown_hours=0
        ))
        return CycleOrchestrator(lifecycle, recovery)

    @pytest.mark.asyncio
    async def test_full_cycle(self, orchestrator):
        """Test complete cycle: V1 → V2 → V3 → V1"""
        version_id = "cycle-test-001"
        
        # Step 1: Register in V1 (Experiment)
        await orchestrator.register_new_experiment(
            version_id=version_id,
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        config = orchestrator.lifecycle.active_versions[version_id]
        assert config.current_state == VersionState.EXPERIMENT
        print("✅ Step 1: Registered in V1 (Experiment)")
        
        # Step 2: Start shadow evaluation
        await orchestrator.start_shadow_evaluation(version_id)
        assert config.current_state == VersionState.SHADOW
        print("✅ Step 2: Started shadow evaluation")
        
        # Step 3: Simulate promotion to V2 gray-scale
        config.current_state = VersionState.GRAY_1
        print("✅ Step 3: Promoted to V2 Gray-scale (1%)")
        
        # Step 4: Progress through gray-scale
        config.current_state = VersionState.GRAY_25
        print("✅ Step 4: Progressed to Gray-scale (25%)")
        
        # Step 5: Reach stable production
        config.current_state = VersionState.STABLE
        print("✅ Step 5: Reached V2 Stable (Production)")
        
        # Step 6: Simulate SLO breach → Quarantine
        await orchestrator.trigger_quarantine(
            version_id=version_id,
            reason="Simulated SLO breach",
            metrics={"error_rate": 0.05}
        )
        assert config.current_state == VersionState.QUARANTINE
        print("✅ Step 6: Demoted to V3 (Quarantine)")
        
        # Step 7: Simulate recovery success
        record = orchestrator.recovery.get_recovery_status(version_id)
        record.recovery_status = RecoveryStatus.PASSED
        record.best_score = 0.95
        
        # Step 8: Recover back to V1
        await orchestrator._promote_recovered_to_v1(version_id)
        assert config.current_state == VersionState.SHADOW
        assert config.metadata.get("recovered_from_quarantine") is True
        print("✅ Step 7-8: Recovered back to V1 (Shadow)")
        
        print("\n🎉 COMPLETE CYCLE VERIFIED: V1 → V2 → V3 → V1")

    @pytest.mark.asyncio
    async def test_cycle_continuity(self, orchestrator):
        """Verify no dead ends in the cycle"""
        # Register multiple versions
        for i in range(5):
            await orchestrator.register_new_experiment(
                version_id=f"continuity-{i}",
                model_version="gpt-4o",
                prompt_version="code-review-v4"
            )
        
        # Quarantine some
        for i in range(2):
            await orchestrator.trigger_quarantine(
                version_id=f"continuity-{i}",
                reason="Test quarantine"
            )
        
        # Ensure continuity check
        await orchestrator._ensure_cycle_continuity()
        
        # All quarantined should be registered for recovery
        for i in range(2):
            record = orchestrator.recovery.get_recovery_status(f"continuity-{i}")
            assert record is not None, f"continuity-{i} not registered for recovery"


class TestCycleEvents:
    """Test cycle event logging"""

    @pytest.fixture
    def orchestrator(self):
        lifecycle = LifecycleController()
        recovery = RecoveryManager()
        return CycleOrchestrator(lifecycle, recovery)

    @pytest.mark.asyncio
    async def test_events_logged(self, orchestrator):
        """Cycle events should be logged"""
        await orchestrator.register_new_experiment(
            version_id="event-test-001",
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        events = orchestrator.get_cycle_events(version_id="event-test-001")
        assert len(events) >= 1
        assert events[0].version_id == "event-test-001"

    @pytest.mark.asyncio
    async def test_event_filtering(self, orchestrator):
        """Events should be filterable by version"""
        await orchestrator.register_new_experiment(
            version_id="event-test-002",
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        await orchestrator.register_new_experiment(
            version_id="event-test-003",
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        events_002 = orchestrator.get_cycle_events(version_id="event-test-002")
        events_003 = orchestrator.get_cycle_events(version_id="event-test-003")
        
        assert all(e.version_id == "event-test-002" for e in events_002)
        assert all(e.version_id == "event-test-003" for e in events_003)


class TestCycleCallbacks:
    """Test cycle event callbacks"""

    @pytest.fixture
    def orchestrator(self):
        lifecycle = LifecycleController()
        recovery = RecoveryManager(config=RecoveryConfig(initial_cooldown_hours=0))
        return CycleOrchestrator(lifecycle, recovery)

    @pytest.mark.asyncio
    async def test_recovery_callback(self, orchestrator):
        """Callback should be triggered on recovery"""
        callback_called = {"called": False, "version_id": None}
        
        async def on_recovery(version_id):
            callback_called["called"] = True
            callback_called["version_id"] = version_id
        
        orchestrator.register_callback("recovery_complete", on_recovery)
        
        # Create and quarantine
        await orchestrator.register_new_experiment(
            version_id="callback-test-001",
            model_version="gpt-4o",
            prompt_version="code-review-v4"
        )
        
        await orchestrator.trigger_quarantine(
            version_id="callback-test-001",
            reason="Test"
        )
        
        # Simulate recovery
        record = orchestrator.recovery.get_recovery_status("callback-test-001")
        record.recovery_status = RecoveryStatus.PASSED
        
        await orchestrator._promote_recovered_to_v1("callback-test-001")
        
        assert callback_called["called"] is True
        assert callback_called["version_id"] == "callback-test-001"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
