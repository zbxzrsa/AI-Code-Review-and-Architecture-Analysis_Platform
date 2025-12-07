"""
Unit tests for the main deployment system orchestrator.
"""

import pytest
import asyncio
import torch
import torch.nn as nn
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
from ai_core.foundation_model.deployment.system import (
    PracticalDeploymentSystem,
    SystemState,
    ContextManagerState,
    StartupError,
    ShutdownError,
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)
    
    def forward(self, input_ids, **kwargs):
        return {"logits": torch.randn(input_ids.shape[0], 100)}


class TestSystemState:
    """Tests for SystemState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        assert SystemState.UNINITIALIZED == "uninitialized"
        assert SystemState.STARTING == "starting"
        assert SystemState.RUNNING == "running"
        assert SystemState.STOPPING == "stopping"
        assert SystemState.STOPPED == "stopped"
        assert SystemState.ERROR == "error"


class TestContextManagerState:
    """Tests for ContextManagerState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = ContextManagerState()
        
        assert state.nesting_level == 0
        assert state.entry_count == 0
        assert state.exit_count == 0
        assert state.last_exception is None
        assert state.cleanup_performed is False


class TestPracticalDeploymentSystemInit:
    """Tests for PracticalDeploymentSystem initialization."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleTestModel()

    @pytest.fixture
    def config(self):
        """Create test config."""
        return PracticalDeploymentConfig(
            lora_r=4,
            lora_alpha=16,
            lora_target_modules=["q_proj", "k_proj"],
            enable_rag=True,
            rag_use_faiss=False,
        )

    def test_initialization(self, model, config):
        """Test system initialization."""
        system = PracticalDeploymentSystem(model, config)
        
        assert system.base_model is model
        assert system.config is config
        assert system._state == SystemState.UNINITIALIZED
        assert system.is_running is False

    def test_components_initialized(self, model, config):
        """Test all components are initialized."""
        system = PracticalDeploymentSystem(model, config)
        
        assert system.adapter_manager is not None
        assert system.rag_system is not None
        assert system.data_collector is not None
        assert system.retraining_scheduler is not None
        assert system.quantizer is not None
        assert system.health_checker is not None
        assert system.fault_tolerance is not None
        assert system.cost_controller is not None


@pytest.mark.asyncio
class TestPracticalDeploymentSystemLifecycle:
    """Tests for system lifecycle methods."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleTestModel()

    @pytest.fixture
    def config(self):
        """Create test config with short intervals."""
        return PracticalDeploymentConfig(
            lora_target_modules=["q_proj"],
            health_check_interval_seconds=1,
            rag_use_faiss=False,
        )

    @pytest.fixture
    def system(self, model, config):
        """Create deployment system."""
        return PracticalDeploymentSystem(model, config)

    async def test_start_stop(self, system):
        """Test starting and stopping the system."""
        await system.start()
        
        assert system.is_running is True
        assert system.initialized_at is not None
        assert system._state == SystemState.UNINITIALIZED  # State managed by context manager
        
        await system.stop()
        
        assert system.is_running is False

    async def test_start_creates_default_adapter(self, system):
        """Test start creates and activates default adapter."""
        await system.start()
        
        assert "default" in system.adapter_manager.adapters
        assert system.adapter_manager.active_adapter == "default"
        
        await system.stop()

    async def test_double_start(self, system):
        """Test starting already running system."""
        await system.start()
        await system.start()  # Should not raise
        
        assert system.is_running is True
        
        await system.stop()


@pytest.mark.asyncio
class TestPracticalDeploymentSystemContextManager:
    """Tests for async context manager functionality."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleTestModel()

    @pytest.fixture
    def config(self):
        """Create test config."""
        return PracticalDeploymentConfig(
            lora_target_modules=["q_proj"],
            rag_use_faiss=False,
        )

    async def test_context_manager_basic(self, model, config):
        """Test basic context manager usage."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            assert system.is_running is True
            assert system._state == SystemState.RUNNING
        
        assert system._state == SystemState.STOPPED

    async def test_context_manager_cleanup_on_exception(self, model, config):
        """Test cleanup occurs even when exception raised."""
        system = PracticalDeploymentSystem(model, config)
        
        with pytest.raises(ValueError):
            async with system:
                assert system.is_running is True
                raise ValueError("Test error")
        
        # Cleanup should still have occurred
        assert system._state == SystemState.STOPPED

    async def test_context_manager_nesting(self, model, config):
        """Test nested context managers."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            assert system._context_state.nesting_level == 1
            
            async with system:
                assert system._context_state.nesting_level == 2
            
            # Inner context exit shouldn't cleanup
            assert system.is_running is True
            assert system._context_state.nesting_level == 1
        
        # Outer context exit should cleanup
        assert system._state == SystemState.STOPPED

    async def test_context_info(self, model, config):
        """Test context_info property."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            info = system.context_info
            
            assert info["state"] == "running"
            assert info["nesting_level"] == 1
            assert info["entry_count"] == 1


@pytest.mark.asyncio
class TestPracticalDeploymentSystemProcess:
    """Tests for process method."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleTestModel()

    @pytest.fixture
    def config(self):
        """Create test config."""
        return PracticalDeploymentConfig(
            lora_target_modules=["q_proj"],
            enable_rag=False,  # Disable RAG for simpler testing
            max_daily_tokens=1_000_000,
            rag_use_faiss=False,
        )

    async def test_process_basic(self, model, config):
        """Test basic processing."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            result = await system.process("Test input")
            
            assert "request_id" in result
            assert "input" in result
            assert "output" in result
            assert "timestamp" in result
            assert result["status"] == "success"

    async def test_process_increments_request_count(self, model, config):
        """Test request count incrementing."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            assert system.request_count == 0
            
            await system.process("Input 1")
            assert system.request_count == 1
            
            await system.process("Input 2")
            assert system.request_count == 2

    async def test_process_rate_limited(self, model, config):
        """Test rate limiting when limits exceeded."""
        config.max_daily_tokens = 10  # Very low limit
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            # First request might succeed
            result1 = await system.process("Test")
            
            # Exceed limit
            system.cost_controller.daily_tokens = 100
            
            result2 = await system.process("Test")
            
            assert result2["status"] == "rate_limited"

    async def test_process_with_rag(self, model, config):
        """Test processing with RAG enabled."""
        config.enable_rag = True
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            # Add knowledge
            system.add_knowledge("Python is great for AI")
            
            result = await system.process("Tell me about Python", use_rag=True)
            
            assert result["status"] == "success"
            assert "rag_context" in result

    async def test_process_batch(self, model, config):
        """Test batch processing."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            inputs = ["Input 1", "Input 2", "Input 3"]
            results = await system.process_batch(inputs)
            
            assert len(results) == 3
            assert all(r["status"] == "success" for r in results)


@pytest.mark.asyncio
class TestPracticalDeploymentSystemStatus:
    """Tests for status and information methods."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleTestModel()

    @pytest.fixture
    def config(self):
        """Create test config."""
        return PracticalDeploymentConfig(
            lora_target_modules=["q_proj"],
            rag_use_faiss=False,
        )

    async def test_get_status(self, model, config):
        """Test getting system status."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            status = system.get_status()
            
            assert "is_running" in status
            assert "initialized_at" in status
            assert "request_count" in status
            assert "health" in status
            assert "cost" in status
            assert "adapters" in status
            assert "retraining" in status
            
            assert status["is_running"] is True
            assert status["request_count"] == 0

    async def test_get_adapter_info(self, model, config):
        """Test getting adapter information."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            info = system.get_adapter_info("default")
            
            assert info is not None


@pytest.mark.asyncio
class TestPracticalDeploymentSystemAdapters:
    """Tests for adapter management."""

    @pytest.fixture
    def model(self):
        """Create test model."""
        return SimpleTestModel()

    @pytest.fixture
    def config(self):
        """Create test config."""
        return PracticalDeploymentConfig(
            lora_target_modules=["q_proj", "k_proj"],
            rag_use_faiss=False,
        )

    async def test_create_adapter(self, model, config):
        """Test creating new adapter."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            await system.create_adapter("custom_adapter")
            
            assert "custom_adapter" in system.adapter_manager.adapters

    async def test_switch_adapter(self, model, config):
        """Test switching adapters."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            await system.create_adapter("adapter2")
            system.switch_adapter("adapter2")
            
            assert system.adapter_manager.active_adapter == "adapter2"

    async def test_merge_adapters(self, model, config):
        """Test merging adapters."""
        system = PracticalDeploymentSystem(model, config)
        
        async with system:
            await system.create_adapter("adapter1")
            await system.create_adapter("adapter2")
            
            await system.merge_adapters(
                ["default", "adapter1"],
                weights=[0.6, 0.4],
                new_name="merged"
            )
            
            assert "merged" in system.adapter_manager.adapters


class TestExceptions:
    """Tests for custom exceptions."""

    def test_startup_error(self):
        """Test StartupError exception."""
        error = StartupError("Failed to start")
        assert str(error) == "Failed to start"

    def test_shutdown_error(self):
        """Test ShutdownError exception."""
        error = ShutdownError("Failed to shutdown")
        assert str(error) == "Failed to shutdown"
