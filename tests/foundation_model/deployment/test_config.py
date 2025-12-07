"""
Unit tests for deployment configuration module.
"""

import pytest
from dataclasses import fields

from ai_core.foundation_model.deployment.config import (
    QuantizationType,
    RetrainingFrequency,
    PracticalDeploymentConfig,
)


class TestQuantizationType:
    """Tests for QuantizationType enum."""

    def test_enum_values(self):
        """Test all quantization type values exist."""
        assert QuantizationType.NONE == "none"
        assert QuantizationType.INT8 == "int8"
        assert QuantizationType.INT4 == "int4"
        assert QuantizationType.FP8 == "fp8"
        assert QuantizationType.GPTQ == "gptq"
        assert QuantizationType.AWQ == "awq"

    def test_enum_is_string(self):
        """Test quantization types are strings."""
        for qt in QuantizationType:
            assert isinstance(qt.value, str)

    def test_enum_count(self):
        """Test expected number of quantization types."""
        assert len(QuantizationType) == 6


class TestRetrainingFrequency:
    """Tests for RetrainingFrequency enum."""

    def test_enum_values(self):
        """Test all retraining frequency values exist."""
        assert RetrainingFrequency.HOURLY == "hourly"
        assert RetrainingFrequency.DAILY == "daily"
        assert RetrainingFrequency.WEEKLY == "weekly"
        assert RetrainingFrequency.MONTHLY == "monthly"
        assert RetrainingFrequency.ON_DEMAND == "on_demand"

    def test_enum_is_string(self):
        """Test retraining frequencies are strings."""
        for rf in RetrainingFrequency:
            assert isinstance(rf.value, str)


class TestPracticalDeploymentConfig:
    """Tests for PracticalDeploymentConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PracticalDeploymentConfig()
        
        # Base model defaults
        assert config.base_model_path == "models/base"
        assert config.freeze_base_model is True
        
        # LoRA defaults
        assert config.lora_r == 8
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert len(config.lora_target_modules) == 7
        
        # RAG defaults
        assert config.enable_rag is True
        assert config.rag_top_k == 5
        assert config.rag_similarity_threshold == 0.7
        
        # Retraining defaults
        assert config.retraining_frequency == RetrainingFrequency.WEEKLY
        assert config.min_samples_for_retraining == 1000
        
        # Quantization defaults
        assert config.quantization_type == QuantizationType.INT8
        
        # Cost control defaults
        assert config.max_daily_tokens == 100_000_000
        assert config.max_monthly_cost_usd == 100_000
        
        # Fault tolerance defaults
        assert config.checkpoint_interval_minutes == 30
        assert config.max_retries == 3
        assert config.health_check_interval_seconds == 60

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = PracticalDeploymentConfig(
            lora_r=16,
            lora_alpha=64,
            enable_rag=False,
            quantization_type=QuantizationType.INT4,
            max_daily_tokens=50_000_000,
        )
        
        assert config.lora_r == 16
        assert config.lora_alpha == 64
        assert config.enable_rag is False
        assert config.quantization_type == QuantizationType.INT4
        assert config.max_daily_tokens == 50_000_000
        
        # Defaults should remain for unspecified fields
        assert config.freeze_base_model is True
        assert config.rag_top_k == 5

    def test_lora_target_modules_default(self):
        """Test default LoRA target modules."""
        config = PracticalDeploymentConfig()
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        assert config.lora_target_modules == expected_modules

    def test_lora_target_modules_custom(self):
        """Test custom LoRA target modules."""
        custom_modules = ["attention", "mlp"]
        config = PracticalDeploymentConfig(lora_target_modules=custom_modules)
        assert config.lora_target_modules == custom_modules

    def test_is_dataclass(self):
        """Test that config is a proper dataclass."""
        config = PracticalDeploymentConfig()
        field_names = [f.name for f in fields(config)]
        
        assert "base_model_path" in field_names
        assert "lora_r" in field_names
        assert "enable_rag" in field_names
        assert "quantization_type" in field_names

    def test_int4_specific_config(self):
        """Test INT4 quantization specific configuration."""
        config = PracticalDeploymentConfig(
            quantization_type=QuantizationType.INT4,
            int4_compute_dtype="float16",
            int4_quant_type="nf4",
            int4_double_quant=True,
        )
        
        assert config.quantization_type == QuantizationType.INT4
        assert config.int4_compute_dtype == "float16"
        assert config.int4_quant_type == "nf4"
        assert config.int4_double_quant is True

    def test_rag_faiss_config(self):
        """Test RAG FAISS configuration."""
        config = PracticalDeploymentConfig()
        assert config.rag_use_faiss is True
        
        config_no_faiss = PracticalDeploymentConfig(rag_use_faiss=False)
        assert config_no_faiss.rag_use_faiss is False


class TestConfigValidation:
    """Tests for configuration validation scenarios."""

    def test_lora_alpha_greater_than_r(self):
        """Test typical LoRA alpha > r configuration."""
        config = PracticalDeploymentConfig(lora_r=8, lora_alpha=32)
        assert config.lora_alpha / config.lora_r == 4.0

    def test_threshold_ranges(self):
        """Test threshold values are in valid ranges."""
        config = PracticalDeploymentConfig()
        assert 0.0 <= config.rag_similarity_threshold <= 1.0
        assert 0.0 <= config.uncertainty_threshold <= 1.0
        assert 0.0 <= config.lora_dropout <= 1.0

    def test_positive_values(self):
        """Test that numeric values are positive."""
        config = PracticalDeploymentConfig()
        assert config.lora_r > 0
        assert config.lora_alpha > 0
        assert config.rag_top_k > 0
        assert config.min_samples_for_retraining > 0
        assert config.max_daily_tokens > 0
        assert config.max_monthly_cost_usd > 0
        assert config.checkpoint_interval_minutes > 0
        assert config.max_retries > 0
        assert config.health_check_interval_seconds > 0
