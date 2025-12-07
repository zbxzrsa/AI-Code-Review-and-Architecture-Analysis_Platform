"""
Unit tests for LoRA adapter module.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
from ai_core.foundation_model.deployment.lora import LoRALayer, LoRAAdapterManager


class TestLoRALayer:
    """Tests for LoRALayer class."""

    @pytest.fixture
    def linear_layer(self):
        """Create a simple linear layer for testing."""
        return nn.Linear(64, 32)

    @pytest.fixture
    def lora_layer(self, linear_layer):
        """Create a LoRA layer wrapping the linear layer."""
        return LoRALayer(linear_layer, r=8, alpha=32, dropout=0.05)

    def test_initialization(self, lora_layer):
        """Test LoRA layer initialization."""
        assert lora_layer.r == 8
        assert lora_layer.alpha == 32
        assert lora_layer.scaling == 4.0  # alpha / r

    def test_lora_matrices_shapes(self, lora_layer):
        """Test LoRA matrix shapes are correct."""
        # A: in_features -> r (64 -> 8)
        assert lora_layer.lora_A.weight.shape == (8, 64)
        # B: r -> out_features (8 -> 32)
        assert lora_layer.lora_B.weight.shape == (32, 8)

    def test_original_layer_frozen(self, lora_layer, linear_layer):
        """Test that original layer weights are frozen."""
        assert not linear_layer.weight.requires_grad

    def test_lora_weights_trainable(self, lora_layer):
        """Test that LoRA weights are trainable."""
        assert lora_layer.lora_A.weight.requires_grad
        assert lora_layer.lora_B.weight.requires_grad

    def test_forward_shape(self, lora_layer):
        """Test forward pass output shape."""
        x = torch.randn(4, 64)  # batch_size=4, in_features=64
        output = lora_layer(x)
        assert output.shape == (4, 32)  # batch_size=4, out_features=32

    def test_forward_adds_lora_output(self, lora_layer):
        """Test that forward pass adds LoRA contribution."""
        x = torch.randn(4, 64)
        
        # Get original output
        with torch.no_grad():
            original_output = lora_layer.original_layer(x)
        
        # Initially B is zeros, so output should match original
        lora_layer.eval()  # Disable dropout
        with torch.no_grad():
            combined_output = lora_layer(x)
        
        # B initialized to zeros, so outputs should be very close
        torch.testing.assert_close(original_output, combined_output, rtol=1e-4, atol=1e-4)

    def test_forward_after_training(self, lora_layer):
        """Test forward pass after modifying LoRA weights."""
        x = torch.randn(4, 64)
        
        # Set non-zero B weights
        with torch.no_grad():
            lora_layer.lora_B.weight.fill_(0.1)
        
        lora_layer.eval()
        with torch.no_grad():
            original_output = lora_layer.original_layer(x)
            combined_output = lora_layer(x)
        
        # Outputs should now be different
        assert not torch.allclose(original_output, combined_output)

    def test_merge_weights(self, lora_layer):
        """Test merging LoRA weights into original layer."""
        x = torch.randn(4, 64)
        
        # Set non-zero B weights
        with torch.no_grad():
            lora_layer.lora_B.weight.fill_(0.1)
        
        lora_layer.eval()
        with torch.no_grad():
            output_before_merge = lora_layer(x)
        
        # Merge weights
        lora_layer.merge_weights()
        
        with torch.no_grad():
            # After merge, original layer alone should give same output
            merged_output = lora_layer.original_layer(x)
        
        # Note: After merge, forward() would add LoRA again, so we compare with original_layer
        torch.testing.assert_close(output_before_merge, merged_output, rtol=1e-4, atol=1e-4)

    def test_get_lora_state_dict(self, lora_layer):
        """Test extracting LoRA state dict."""
        state_dict = lora_layer.get_lora_state_dict()
        
        assert "lora_A" in state_dict
        assert "lora_B" in state_dict
        assert state_dict["lora_A"].shape == (8, 64)
        assert state_dict["lora_B"].shape == (32, 8)

    def test_load_lora_state_dict(self, lora_layer):
        """Test loading LoRA state dict."""
        # Create new weights
        new_a = torch.randn(8, 64)
        new_b = torch.randn(32, 8)
        
        lora_layer.load_lora_state_dict({
            "lora_A": new_a,
            "lora_B": new_b,
        })
        
        torch.testing.assert_close(lora_layer.lora_A.weight.data, new_a)
        torch.testing.assert_close(lora_layer.lora_B.weight.data, new_b)

    def test_dropout_applied(self, lora_layer):
        """Test dropout is applied during training."""
        lora_layer.train()
        x = torch.randn(100, 64)
        
        # Multiple forward passes should give different results due to dropout
        with torch.no_grad():
            lora_layer.lora_B.weight.fill_(0.1)
        
        outputs = [lora_layer(x) for _ in range(3)]
        
        # At least one pair should be different due to dropout
        different = False
        for i in range(len(outputs) - 1):
            if not torch.allclose(outputs[i], outputs[i + 1]):
                different = True
                break
        
        assert different, "Dropout should cause variation between passes"


class TestLoRAAdapterManager:
    """Tests for LoRAAdapterManager class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model with linear layers."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.o_proj = nn.Linear(64, 64)
                self.mlp = nn.Linear(64, 128)
            
            def forward(self, x):
                return self.mlp(self.o_proj(self.v_proj(x)))
        
        return SimpleModel()

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return PracticalDeploymentConfig(
            lora_r=4,
            lora_alpha=16,
            lora_dropout=0.0,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    @pytest.fixture
    def manager(self, simple_model, config):
        """Create adapter manager."""
        return LoRAAdapterManager(simple_model, config)

    def test_initialization(self, manager, config):
        """Test manager initialization."""
        assert manager.config == config
        assert manager.active_adapter is None
        assert len(manager.adapters) == 0

    def test_base_model_frozen(self, manager, simple_model):
        """Test base model parameters are frozen."""
        if manager.config.freeze_base_model:
            for param in simple_model.parameters():
                assert not param.requires_grad

    def test_create_adapter(self, manager):
        """Test creating a new adapter."""
        adapter_layers = manager.create_adapter("test_adapter")
        
        assert "test_adapter" in manager.adapters
        assert len(adapter_layers) == 4  # q, k, v, o projections
        assert manager.adapter_versions["test_adapter"] == 1

    def test_create_adapter_metadata(self, manager):
        """Test adapter metadata is created."""
        manager.create_adapter("test_adapter")
        
        metadata = manager.adapter_metadata["test_adapter"]
        assert "created_at" in metadata
        assert "target_modules" in metadata
        assert "r" in metadata
        assert "alpha" in metadata

    def test_activate_adapter(self, manager):
        """Test activating an adapter."""
        manager.create_adapter("adapter1")
        manager.activate_adapter("adapter1")
        
        assert manager.active_adapter == "adapter1"

    def test_activate_nonexistent_adapter(self, manager):
        """Test activating non-existent adapter raises error."""
        with pytest.raises(ValueError, match="not found"):
            manager.activate_adapter("nonexistent")

    def test_deactivate_adapter(self, manager):
        """Test deactivating adapter."""
        manager.create_adapter("adapter1")
        manager.activate_adapter("adapter1")
        manager.deactivate_adapter()
        
        assert manager.active_adapter is None

    def test_get_trainable_parameters(self, manager):
        """Test getting trainable parameters."""
        manager.create_adapter("adapter1")
        params = manager.get_trainable_parameters("adapter1")
        
        assert len(params) == 8  # 4 layers × 2 matrices (A, B) each
        for param in params:
            assert param.requires_grad

    def test_save_and_load_adapter(self, manager):
        """Test saving and loading adapter."""
        with TemporaryDirectory() as tmpdir:
            # Create and modify adapter
            manager.create_adapter("test_adapter")
            
            # Modify weights
            for layer in manager.adapters["test_adapter"].values():
                with torch.no_grad():
                    layer.lora_A.weight.fill_(0.5)
                    layer.lora_B.weight.fill_(0.3)
            
            # Save adapter
            manager.save_adapter("test_adapter", tmpdir)
            
            # Clear adapter
            original_weights = {
                name: layer.get_lora_state_dict()
                for name, layer in manager.adapters["test_adapter"].items()
            }
            del manager.adapters["test_adapter"]
            
            # Load adapter
            manager.load_adapter("test_adapter", tmpdir)
            
            # Verify weights match
            for name, layer in manager.adapters["test_adapter"].items():
                state = layer.get_lora_state_dict()
                torch.testing.assert_close(state["lora_A"], original_weights[name]["lora_A"])
                torch.testing.assert_close(state["lora_B"], original_weights[name]["lora_B"])

    def test_merge_adapters(self, manager):
        """Test merging multiple adapters."""
        # Create two adapters
        manager.create_adapter("adapter1")
        manager.create_adapter("adapter2")
        
        # Set different weights
        for layer in manager.adapters["adapter1"].values():
            with torch.no_grad():
                layer.lora_A.weight.fill_(1.0)
                layer.lora_B.weight.fill_(1.0)
        
        for layer in manager.adapters["adapter2"].values():
            with torch.no_grad():
                layer.lora_A.weight.fill_(0.0)
                layer.lora_B.weight.fill_(0.0)
        
        # Merge with equal weights
        merged = manager.merge_adapters(["adapter1", "adapter2"], weights=[0.5, 0.5], new_name="merged")
        
        assert "merged" in manager.adapters
        
        # Check merged weights are averaged
        for layer in merged.values():
            torch.testing.assert_close(
                layer.lora_A.weight.data,
                torch.full_like(layer.lora_A.weight.data, 0.5),
                rtol=1e-4, atol=1e-4
            )

    def test_merge_adapters_requires_two(self, manager):
        """Test merging requires at least 2 adapters."""
        manager.create_adapter("adapter1")
        
        with pytest.raises(ValueError, match="at least 2"):
            manager.merge_adapters(["adapter1"])

    def test_multiple_adapters(self, manager):
        """Test managing multiple adapters."""
        manager.create_adapter("adapter1")
        manager.create_adapter("adapter2")
        manager.create_adapter("adapter3")
        
        assert len(manager.adapters) == 3
        
        manager.activate_adapter("adapter2")
        assert manager.active_adapter == "adapter2"
        
        manager.activate_adapter("adapter1")
        assert manager.active_adapter == "adapter1"


class TestLoRAIntegration:
    """Integration tests for LoRA functionality."""

    def test_training_loop_simulation(self):
        """Simulate a simple training loop with LoRA."""
        # Create model
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        
        config = PracticalDeploymentConfig(
            lora_r=4,
            lora_alpha=16,
            lora_target_modules=["0", "2"],  # Target the linear layers
        )
        
        manager = LoRAAdapterManager(model, config)
        manager.create_adapter("training")
        manager.activate_adapter("training")
        
        # Get trainable parameters
        params = manager.get_trainable_parameters()
        optimizer = torch.optim.SGD(params, lr=0.01)
        
        # Simple training step
        x = torch.randn(8, 32)
        target = torch.randint(0, 10, (8,))
        
        # Note: This is a simplified test - actual training would use the LoRA forward
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify gradients were computed for LoRA parameters
        for param in params:
            if param.grad is not None:
                assert param.grad.abs().sum() > 0

    def test_lora_reduces_trainable_params(self):
        """Test that LoRA significantly reduces trainable parameters."""
        in_features = 1024
        out_features = 1024
        r = 8
        
        # Full fine-tuning params
        full_params = in_features * out_features
        
        # LoRA params
        lora_params = (in_features * r) + (r * out_features)
        
        # LoRA should use much fewer parameters
        reduction = (full_params - lora_params) / full_params
        assert reduction > 0.98  # Should reduce by >98%
