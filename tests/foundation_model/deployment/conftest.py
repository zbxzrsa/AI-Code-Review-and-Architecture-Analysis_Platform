"""
Shared pytest fixtures for deployment module tests.
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_core.foundation_model.deployment.config import (
    PracticalDeploymentConfig,
    QuantizationType,
    RetrainingFrequency,
)


class SimpleLinearModel(nn.Module):
    """Simple linear model for testing."""
    
    def __init__(self, in_features=64, hidden_features=128, out_features=32):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerLikeModel(nn.Module):
    """Model with transformer-like projections for LoRA testing."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, input_ids, **kwargs):
        # Simplified transformer-like forward
        x = torch.randn(input_ids.shape[0], 64)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Simple attention (not proper, just for testing)
        attn_output = self.o_proj(v)
        
        # MLP
        gate = self.gate_proj(attn_output)
        up = self.up_proj(attn_output)
        output = self.down_proj(gate * up)
        
        return {"logits": output}


@pytest.fixture
def simple_model():
    """Provide a simple linear model."""
    return SimpleLinearModel()


@pytest.fixture
def transformer_model():
    """Provide a transformer-like model."""
    return TransformerLikeModel()


@pytest.fixture
def default_config():
    """Provide default configuration."""
    return PracticalDeploymentConfig()


@pytest.fixture
def minimal_config():
    """Provide minimal configuration for fast testing."""
    return PracticalDeploymentConfig(
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=["q_proj", "k_proj"],
        enable_rag=False,
        rag_use_faiss=False,
        quantization_type=QuantizationType.NONE,
        max_daily_tokens=1_000_000,
        max_monthly_cost_usd=1000.0,
        health_check_interval_seconds=1,
        checkpoint_interval_minutes=1,
    )


@pytest.fixture
def int4_config():
    """Provide INT4 quantization configuration."""
    return PracticalDeploymentConfig(
        quantization_type=QuantizationType.INT4,
        int4_compute_dtype="float16",
        int4_quant_type="nf4",
        int4_double_quant=True,
    )


@pytest.fixture
def rag_config():
    """Provide RAG-focused configuration."""
    return PracticalDeploymentConfig(
        enable_rag=True,
        rag_top_k=5,
        rag_similarity_threshold=0.5,
        rag_use_faiss=False,  # Use NumPy for testing
    )


@pytest.fixture
def random_embedding():
    """Generate random embedding."""
    import numpy as np
    return np.random.randn(768).astype(np.float32)


@pytest.fixture
def sample_documents():
    """Provide sample documents for RAG testing."""
    return [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with many layers",
        "Python is commonly used for data science",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret images",
    ]


# Markers for conditional test execution
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
