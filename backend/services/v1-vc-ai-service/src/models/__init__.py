"""
Model architecture module for V1 VC-AI.

Contains custom implementations for:
- Attention mechanisms (sparse, cross-layer, GQA)
- Custom tokenizer for code/commit vocabulary
- Mixture-of-Experts layers
- Speculative decoding components
"""

from .attention import (
    SparseAttention,
    CrossLayerAttention,
    GroupedQueryAttention,
    FlashAttentionWrapper,
)
from .tokenizer import (
    CodeCommitTokenizer,
    SpecialTokens,
)
from .moe import (
    MixtureOfExperts,
    ExpertRouter,
    VersionControlExpert,
)
from .architecture import (
    VersionControlAIModel,
    ModelFactory,
)

__all__ = [
    "SparseAttention",
    "CrossLayerAttention",
    "GroupedQueryAttention",
    "FlashAttentionWrapper",
    "CodeCommitTokenizer",
    "SpecialTokens",
    "MixtureOfExperts",
    "ExpertRouter",
    "VersionControlExpert",
    "VersionControlAIModel",
    "ModelFactory",
]
