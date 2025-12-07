"""
PracticalDeploymentSystem 异步上下文管理器单元测试

模块功能描述:
    测试实用部署系统的异步上下文管理器功能。

测试覆盖:
    1. 正常操作流程
    2. 异常场景
    3. 资源清理验证
    4. 异步取消情况
    5. 嵌套上下文管理器
    6. 线程安全

测试工具类:
    - MockModel: 模拟模型
    - MockHealthChecker: 模拟健康检查器
    - MockRetrainingScheduler: 模拟重训练调度器

最后修改日期: 2024-12-07
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional

import torch
import torch.nn as nn


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

class MockModel(nn.Module):
    """Simple mock model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return {"logits": self.linear(x)}


class MockHealthChecker:
    """Mock health checker for testing."""
    def __init__(self, config=None):
        self.started = False
        self.stopped = False
        self._check_task = None
    
    async def start(self):
        self.started = True
    
    async def stop(self):
        self.stopped = True
    
    def get_status(self):
        return {"status": "healthy"}


class MockRetrainingScheduler:
    """Mock retraining scheduler for testing."""
    def __init__(self, *args, **kwargs):
        self.started = False
        self.stopped = False
        self._scheduler_task = None
        self.last_retraining = None
        self.next_retraining = None
        self.training_in_progress = False
    
    async def start(self):
        self.started = True
    
    async def stop(self):
        self.stopped = True
    
    def get_status(self):
        return {}


class MockFaultTolerance:
    """Mock fault tolerance manager."""
    def __init__(self, config=None):
        self.checkpoints_saved = 0
    
    async def save_checkpoint(self, *args, **kwargs):
        self.checkpoints_saved += 1
    
    def get_stats(self):
        return {"checkpoints_saved": self.checkpoints_saved}


# =============================================================================
# Test System State
# =============================================================================

class TestSystemState:
    """Tests for SystemState enum."""
    
    def test_system_state_values(self):
        """Test SystemState enum has expected values."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import SystemState
        
        assert SystemState.UNINITIALIZED == "uninitialized"
        assert SystemState.STARTING == "starting"
        assert SystemState.RUNNING == "running"
        assert SystemState.STOPPING == "stopping"
        assert SystemState.STOPPED == "stopped"
        assert SystemState.ERROR == "error"


class TestContextManagerState:
    """Tests for ContextManagerState dataclass."""
    
    def test_context_manager_state_defaults(self):
        """Test ContextManagerState default values."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import ContextManagerState
        
        state = ContextManagerState()
        
        assert state.nesting_level == 0
        assert state.entry_count == 0
        assert state.exit_count == 0
        assert state.last_exception is None
        assert state.cleanup_performed is False


# =============================================================================
# Test Normal Operation Flow
# =============================================================================

class TestNormalOperation:
    """Tests for normal context manager operation."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        return PracticalDeploymentConfig()
    
    @pytest.mark.asyncio
    async def test_context_manager_enter_exit(self):
        """Test basic enter and exit flow."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import (
                PracticalDeploymentSystem,
                SystemState,
            )
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        # Mock the components
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock) as mock_start:
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock) as mock_stop:
                system = PracticalDeploymentSystem(model, config)
                
                async with system as s:
                    assert s is system
                    assert system.state == SystemState.RUNNING
                    mock_start.assert_called_once()
                
                mock_stop.assert_called_once()
                assert system.state == SystemState.STOPPED
    
    @pytest.mark.asyncio
    async def test_context_manager_returns_instance(self):
        """Test that __aenter__ returns the system instance."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                async with system as returned_system:
                    assert returned_system is system
    
    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test state transitions during context manager lifecycle."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import (
                PracticalDeploymentSystem,
                SystemState,
            )
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                # Initial state
                assert system.state == SystemState.UNINITIALIZED
                
                async with system:
                    # Running state
                    assert system.state == SystemState.RUNNING
                
                # Stopped state
                assert system.state == SystemState.STOPPED


# =============================================================================
# Test Exception Scenarios
# =============================================================================

class TestExceptionScenarios:
    """Tests for exception handling in context manager."""
    
    @pytest.mark.asyncio
    async def test_exception_in_context_triggers_cleanup(self):
        """Test that exceptions trigger cleanup."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock) as mock_stop:
                system = PracticalDeploymentSystem(model, config)
                
                with pytest.raises(ValueError):
                    async with system:
                        raise ValueError("Test exception")
                
                # Cleanup should still be called
                mock_stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_exception_not_suppressed(self):
        """Test that exceptions are not suppressed."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                with pytest.raises(RuntimeError, match="Test error"):
                    async with system:
                        raise RuntimeError("Test error")
    
    @pytest.mark.asyncio
    async def test_startup_failure_handled(self):
        """Test that startup failures are handled gracefully."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import (
                PracticalDeploymentSystem,
                StartupError,
                SystemState,
            )
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(
            PracticalDeploymentSystem, 
            'start', 
            new_callable=AsyncMock,
            side_effect=RuntimeError("Startup failed")
        ):
            system = PracticalDeploymentSystem(model, config)
            
            with pytest.raises(StartupError):
                async with system:
                    pass
            
            assert system.state == SystemState.ERROR
    
    @pytest.mark.asyncio
    async def test_cleanup_error_logged_not_raised(self):
        """Test that cleanup errors are logged but don't raise."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import (
                PracticalDeploymentSystem,
                SystemState,
            )
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(
                PracticalDeploymentSystem, 
                'stop', 
                new_callable=AsyncMock,
                side_effect=RuntimeError("Cleanup failed")
            ):
                system = PracticalDeploymentSystem(model, config)
                
                # Should not raise even though cleanup fails
                async with system:
                    pass
                
                # State should still be STOPPED
                assert system.state == SystemState.STOPPED
    
    @pytest.mark.asyncio
    async def test_exception_info_recorded(self):
        """Test that exception info is recorded in context state."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                with pytest.raises(ValueError):
                    async with system:
                        raise ValueError("Test exception info")
                
                # Exception should be recorded
                assert system._context_state.last_exception is not None
                assert "Test exception info" in str(system._context_state.last_exception)


# =============================================================================
# Test Resource Cleanup Verification
# =============================================================================

class TestResourceCleanup:
    """Tests for resource cleanup."""
    
    @pytest.mark.asyncio
    async def test_cleanup_performed_flag(self):
        """Test cleanup_performed flag is set."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                assert not system._context_state.cleanup_performed
                
                async with system:
                    assert not system._context_state.cleanup_performed
                
                assert system._context_state.cleanup_performed
    
    @pytest.mark.asyncio
    async def test_stopped_at_timestamp_set(self):
        """Test stopped_at timestamp is set on exit."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                assert system.stopped_at is None
                
                async with system:
                    pass
                
                assert system.stopped_at is not None
                assert isinstance(system.stopped_at, datetime)
    
    @pytest.mark.asyncio
    async def test_context_info_property(self):
        """Test context_info property returns correct data."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                async with system:
                    info = system.context_info
                    
                    assert info["state"] == "running"
                    assert info["nesting_level"] == 1
                    assert info["entry_count"] == 1
                    assert info["cleanup_performed"] is False


# =============================================================================
# Test Nested Context Managers
# =============================================================================

class TestNestedContextManagers:
    """Tests for nested context manager support."""
    
    @pytest.mark.asyncio
    async def test_nested_context_reuses_system(self):
        """Test that nested contexts reuse the running system."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import (
                PracticalDeploymentSystem,
                SystemState,
            )
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock) as mock_start:
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock) as mock_stop:
                system = PracticalDeploymentSystem(model, config)
                
                async with system:
                    # Start called once
                    assert mock_start.call_count == 1
                    
                    async with system:
                        # Still only called once (reused)
                        assert mock_start.call_count == 1
                        assert system.state == SystemState.RUNNING
                    
                    # Still running after inner exit
                    assert system.state == SystemState.RUNNING
                    assert mock_stop.call_count == 0
                
                # Stop called after outer exit
                assert mock_stop.call_count == 1
    
    @pytest.mark.asyncio
    async def test_nesting_level_tracking(self):
        """Test nesting level is tracked correctly."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                assert system._context_state.nesting_level == 0
                
                async with system:
                    assert system._context_state.nesting_level == 1
                    
                    async with system:
                        assert system._context_state.nesting_level == 2
                        
                        async with system:
                            assert system._context_state.nesting_level == 3
                        
                        assert system._context_state.nesting_level == 2
                    
                    assert system._context_state.nesting_level == 1
                
                assert system._context_state.nesting_level == 0
    
    @pytest.mark.asyncio
    async def test_entry_exit_counts(self):
        """Test entry and exit counts are tracked."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                async with system:
                    assert system._context_state.entry_count == 1
                    
                    async with system:
                        assert system._context_state.entry_count == 2
                
                assert system._context_state.entry_count == 2
                assert system._context_state.exit_count == 2


# =============================================================================
# Test Async Cancellation
# =============================================================================

class TestAsyncCancellation:
    """Tests for async cancellation handling."""
    
    @pytest.mark.asyncio
    async def test_cancellation_during_operation(self):
        """Test cancellation during operation triggers cleanup."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        cleanup_called = False
        
        async def mock_stop():
            nonlocal cleanup_called
            cleanup_called = True
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', side_effect=mock_stop):
                system = PracticalDeploymentSystem(model, config)
                
                async def run_with_cancel():
                    async with system:
                        # Simulate long operation
                        await asyncio.sleep(10)
                
                task = asyncio.create_task(run_with_cancel())
                
                # Give time to enter context
                await asyncio.sleep(0.1)
                
                # Cancel the task
                task.cancel()
                
                with pytest.raises(asyncio.CancelledError):
                    await task
                
                # Cleanup should have been called
                assert cleanup_called


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""
    
    @pytest.mark.asyncio
    async def test_concurrent_context_entries(self):
        """Test concurrent context entries are handled safely."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        start_count = 0
        
        async def mock_start():
            nonlocal start_count
            start_count += 1
            await asyncio.sleep(0.1)  # Simulate startup time
        
        with patch.object(PracticalDeploymentSystem, 'start', side_effect=mock_start):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                async def enter_context():
                    async with system:
                        await asyncio.sleep(0.05)
                
                # Run multiple concurrent entries
                await asyncio.gather(
                    enter_context(),
                    enter_context(),
                    enter_context(),
                )
                
                # Start should only be called once due to lock
                assert start_count == 1


# =============================================================================
# Test Uptime Calculation
# =============================================================================

class TestUptimeCalculation:
    """Tests for uptime calculation."""
    
    @pytest.mark.asyncio
    async def test_uptime_calculated_correctly(self):
        """Test uptime is calculated correctly."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import PracticalDeploymentSystem
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                system = PracticalDeploymentSystem(model, config)
                
                async with system:
                    await asyncio.sleep(0.1)
                
                # Uptime should be approximately 0.1 seconds
                uptime = system._get_uptime_seconds()
                assert 0.05 <= uptime <= 0.5  # Allow for timing variance


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for context manager."""
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_with_processing(self):
        """Test full lifecycle including processing."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.deployment.system import (
                PracticalDeploymentSystem,
                SystemState,
            )
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        model = MockModel()
        config = PracticalDeploymentConfig()
        
        with patch.object(PracticalDeploymentSystem, 'start', new_callable=AsyncMock):
            with patch.object(PracticalDeploymentSystem, 'stop', new_callable=AsyncMock):
                with patch.object(
                    PracticalDeploymentSystem, 
                    'process', 
                    new_callable=AsyncMock,
                    return_value={"output": "test"}
                ) as mock_process:
                    system = PracticalDeploymentSystem(model, config)
                    
                    async with system as s:
                        result = await s.process("test input")
                        assert result["output"] == "test"
                        mock_process.assert_called_once_with("test input")
                    
                    assert system.state == SystemState.STOPPED
