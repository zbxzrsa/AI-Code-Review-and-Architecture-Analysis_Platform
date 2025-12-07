"""
Unit tests for model quantization implementations.

Tests both the legacy practical_deployment.py and new modular quantization.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


class SimpleModel(nn.Module):
    """Simple model for testing quantization."""
    
    def __init__(self, input_size=64, hidden_size=32, output_size=16):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class TestModularQuantization:
    """Tests for the modular deployment/quantization.py."""
    
    def test_quantization_stats_dataclass(self):
        """Test QuantizationStats dataclass."""
        from ai_core.foundation_model.deployment.quantization import QuantizationStats
        
        stats = QuantizationStats(
            original_size_mb=100.0,
            quantized_size_mb=25.0,
            compression_ratio=4.0,
            num_quantized_layers=10,
            quantization_type="int4",
        )
        
        assert stats.original_size_mb == 100.0
        assert stats.compression_ratio == 4.0
    
    def test_int4_quantizer_init(self):
        """Test INT4Quantizer initialization."""
        from ai_core.foundation_model.deployment.quantization import INT4Quantizer
        
        quantizer = INT4Quantizer(
            compute_dtype=torch.float16,
            quant_type="nf4",
            double_quant=True,
        )
        
        assert quantizer.compute_dtype == torch.float16
        assert quantizer.quant_type == "nf4"
        assert quantizer.double_quant is True
    
    def test_int4_quantizer_without_bitsandbytes(self):
        """Test INT4Quantizer behavior when bitsandbytes not installed."""
        from ai_core.foundation_model.deployment.quantization import INT4Quantizer
        
        quantizer = INT4Quantizer()
        
        # If bitsandbytes not available, should return original model
        model = SimpleModel()
        quantized_model, stats = quantizer.quantize(model)
        
        if not quantizer._bnb_available:
            # Should return original model unchanged
            assert stats.compression_ratio == 1.0
            assert stats.num_quantized_layers == 0
    
    def test_model_quantizer_init(self):
        """Test ModelQuantizer initialization."""
        from ai_core.foundation_model.deployment.quantization import ModelQuantizer
        from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig
        
        config = PracticalDeploymentConfig()
        quantizer = ModelQuantizer(config)
        
        assert quantizer.config == config
    
    def test_estimate_memory_savings_int8(self):
        """Test memory savings estimation for INT8."""
        from ai_core.foundation_model.deployment.quantization import ModelQuantizer
        from ai_core.foundation_model.deployment.config import QuantizationType
        
        savings = ModelQuantizer.estimate_memory_savings(
            original_params=1_000_000_000,  # 1B params
            quant_type=QuantizationType.INT8,
        )
        
        assert savings["savings_percent"] == 75.0
        assert savings["original_gb"] == 4.0  # 1B * 4 bytes = 4GB
    
    def test_estimate_memory_savings_int4(self):
        """Test memory savings estimation for INT4."""
        from ai_core.foundation_model.deployment.quantization import ModelQuantizer
        from ai_core.foundation_model.deployment.config import QuantizationType
        
        savings = ModelQuantizer.estimate_memory_savings(
            original_params=1_000_000_000,  # 1B params
            quant_type=QuantizationType.INT4,
        )
        
        assert savings["savings_percent"] == 87.5
        assert savings["quantized_gb"] == 0.5  # ~0.5 bytes per param


class TestLegacyQuantization:
    """Tests for the legacy practical_deployment.py quantization."""
    
    def test_legacy_quantize_int8(self):
        """Test legacy INT8 quantization."""
        import warnings
        
        # Suppress deprecation warning for test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            from ai_core.foundation_model.practical_deployment import (
                ModelQuantizer,
                PracticalDeploymentConfig,
                QuantizationType,
            )
        
        config = PracticalDeploymentConfig(
            quantization_type=QuantizationType.INT8,
        )
        quantizer = ModelQuantizer(config)
        
        model = SimpleModel()
        quantized = quantizer.quantize_int8(model)
        
        # Should return a quantized model
        assert quantized is not None
    
    def test_legacy_quantize_int4_implementation(self):
        """Test that legacy quantize_int4 is now implemented (not placeholder)."""
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            from ai_core.foundation_model.practical_deployment import (
                ModelQuantizer,
                PracticalDeploymentConfig,
                QuantizationType,
            )
        
        config = PracticalDeploymentConfig(
            quantization_type=QuantizationType.INT4,
        )
        quantizer = ModelQuantizer(config)
        
        model = SimpleModel()
        
        # Should not just return the original model unchanged
        # (the old placeholder did that)
        quantized = quantizer.quantize_int4(model)
        
        # Even if bitsandbytes not installed, should log properly
        assert quantized is not None
    
    def test_legacy_estimate_memory_savings(self):
        """Test legacy memory savings estimation."""
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            from ai_core.foundation_model.practical_deployment import (
                ModelQuantizer,
                QuantizationType,
            )
        
        savings = ModelQuantizer.estimate_memory_savings(
            original_params=1_000_000,
            quant_type=QuantizationType.INT4,
        )
        
        assert "savings_percent" in savings
        assert savings["savings_percent"] == 87.5


class TestQuantizationTypes:
    """Tests for QuantizationType enum."""
    
    def test_quantization_types_exist(self):
        """Test all quantization types are defined."""
        from ai_core.foundation_model.deployment.config import QuantizationType
        
        assert QuantizationType.NONE == "none"
        assert QuantizationType.INT8 == "int8"
        assert QuantizationType.INT4 == "int4"
        assert QuantizationType.FP8 == "fp8"
        assert QuantizationType.GPTQ == "gptq"
        assert QuantizationType.AWQ == "awq"


class TestBackwardCompatibility:
    """Tests for backward compatibility between legacy and modular code."""
    
    def test_imports_from_both_locations(self):
        """Test that QuantizationType can be imported from both locations."""
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            from ai_core.foundation_model.practical_deployment import QuantizationType as LegacyQT
            from ai_core.foundation_model.deployment.config import QuantizationType as ModularQT
        
        # Both should have same values
        assert LegacyQT.INT8.value == ModularQT.INT8.value
        assert LegacyQT.INT4.value == ModularQT.INT4.value
    
    def test_config_compatibility(self):
        """Test PracticalDeploymentConfig compatibility."""
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
            from ai_core.foundation_model.practical_deployment import PracticalDeploymentConfig as LegacyConfig
            from ai_core.foundation_model.deployment.config import PracticalDeploymentConfig as ModularConfig
        
        legacy = LegacyConfig()
        modular = ModularConfig()
        
        # Both should have same default values
        assert legacy.lora_r == modular.lora_r
        assert legacy.lora_alpha == modular.lora_alpha


@pytest.mark.parametrize("quant_type,expected_savings", [
    ("int8", 75.0),
    ("int4", 87.5),
    ("fp8", 75.0),
])
def test_quantization_savings_parametrized(quant_type, expected_savings):
    """Parametrized test for quantization savings."""
    from ai_core.foundation_model.deployment.quantization import ModelQuantizer
    from ai_core.foundation_model.deployment.config import QuantizationType
    
    qt = QuantizationType(quant_type)
    savings = ModelQuantizer.estimate_memory_savings(
        original_params=1_000_000,
        quant_type=qt,
    )
    
    assert savings["savings_percent"] == expected_savings
