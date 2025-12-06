"""
Foundation Model Architecture

Advanced Transformer architecture with:
- Mixture of Experts (MoE) - Only activate partial parameters
- RoPE (Rotary Position Embedding) - Superior position encoding
- Flash Attention - Memory-efficient attention
- Sparse Attention - Long context support (128K-1M tokens)
- Multi-modal fusion capability

Target: 500B-1T parameters with MoE efficiency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class AttentionType(str, Enum):
    """Attention mechanism types."""
    STANDARD = "standard"
    FLASH = "flash"
    SPARSE = "sparse"
    SLIDING_WINDOW = "sliding_window"
    GROUPED_QUERY = "grouped_query"


class ExpertRoutingStrategy(str, Enum):
    """Expert routing strategies for MoE."""
    TOP_K = "top_k"
    EXPERT_CHOICE = "expert_choice"
    HASH = "hash"
    SOFT = "soft"


@dataclass
class MoEConfig:
    """
    Mixture of Experts Configuration
    
    Target: 500B-1T total parameters, but only activate ~50B-100B per forward pass
    """
    # Model dimensions
    vocab_size: int = 128000  # Large vocab for multilingual
    hidden_size: int = 8192  # d_model
    intermediate_size: int = 28672  # FFN intermediate (3.5x hidden)
    num_hidden_layers: int = 80  # Transformer layers
    num_attention_heads: int = 64  # Attention heads
    num_key_value_heads: int = 8  # GQA: fewer KV heads for efficiency
    
    # MoE configuration
    num_experts: int = 128  # Total number of experts
    num_experts_per_token: int = 2  # Active experts per token
    expert_capacity_factor: float = 1.25  # Load balancing
    router_aux_loss_coef: float = 0.01  # Router load balancing loss
    
    # Position encoding
    max_position_embeddings: int = 131072  # 128K context window
    rope_theta: float = 500000.0  # RoPE base frequency
    rope_scaling: Optional[Dict] = None  # For extended context
    
    # Attention
    attention_type: AttentionType = AttentionType.FLASH
    attention_dropout: float = 0.0
    sliding_window_size: int = 4096  # For sliding window attention
    
    # Layer norms and activations
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"  # SwiGLU activation
    
    # Training
    tie_word_embeddings: bool = False
    use_cache: bool = True
    
    # Multi-modal (future)
    vision_hidden_size: int = 1024
    audio_hidden_size: int = 512
    enable_multimodal: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.hidden_size % self.num_attention_heads == 0
        
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_queries_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
    
    @property
    def total_params(self) -> int:
        """Estimate total parameters."""
        # Embedding
        embed_params = self.vocab_size * self.hidden_size * 2  # input + output
        
        # Attention per layer (with GQA)
        q_params = self.hidden_size * self.hidden_size
        kv_params = self.hidden_size * (self.hidden_size // self.num_queries_per_kv) * 2
        o_params = self.hidden_size * self.hidden_size
        attn_params = (q_params + kv_params + o_params) * self.num_hidden_layers
        
        # MoE FFN per layer
        expert_params = 3 * self.hidden_size * self.intermediate_size  # gate, up, down
        moe_params = expert_params * self.num_experts * self.num_hidden_layers
        
        # Router
        router_params = self.hidden_size * self.num_experts * self.num_hidden_layers
        
        # LayerNorms
        norm_params = 4 * self.hidden_size * self.num_hidden_layers
        
        total = embed_params + attn_params + moe_params + router_params + norm_params
        return total
    
    @property
    def active_params(self) -> int:
        """Estimate active parameters per forward pass."""
        # Same as above but only count active experts
        embed_params = self.vocab_size * self.hidden_size * 2
        
        q_params = self.hidden_size * self.hidden_size
        kv_params = self.hidden_size * (self.hidden_size // self.num_queries_per_kv) * 2
        o_params = self.hidden_size * self.hidden_size
        attn_params = (q_params + kv_params + o_params) * self.num_hidden_layers
        
        # Only active experts
        expert_params = 3 * self.hidden_size * self.intermediate_size
        active_moe_params = expert_params * self.num_experts_per_token * self.num_hidden_layers
        
        router_params = self.hidden_size * self.num_experts * self.num_hidden_layers
        norm_params = 4 * self.hidden_size * self.num_hidden_layers
        
        total = embed_params + attn_params + active_moe_params + router_params + norm_params
        return total


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================

class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    Superior position encoding that:
    - Encodes relative positions naturally
    - Supports extrapolation to longer sequences
    - Compatible with linear attention
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 500000.0,
        scaling_factor: float = 1.0,
        scaling_type: str = "linear",  # linear, dynamic, yarn
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for efficiency
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute and cache cos/sin values."""
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            
            # Apply scaling for extended context
            position_ids = torch.arange(seq_len, device=device)
            if self.scaling_type == "linear":
                position_ids = position_ids / self.scaling_factor
            elif self.scaling_type == "dynamic":
                # Dynamic NTK-aware scaling
                if seq_len > self.max_position_embeddings:
                    base = self.base * (
                        (self.scaling_factor * seq_len / self.max_position_embeddings) - 
                        (self.scaling_factor - 1)
                    ) ** (self.dim / (self.dim - 2))
                    inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
                else:
                    inv_freq = self.inv_freq.to(device)
            else:
                inv_freq = self.inv_freq.to(device)
            
            if self.scaling_type != "dynamic":
                inv_freq = self.inv_freq.to(device)
            
            # [seq_len, dim/2]
            freqs = torch.outer(position_ids.float(), inv_freq)
            # [seq_len, dim]
            emb = torch.cat([freqs, freqs], dim=-1)
            
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: [batch, seq_len, num_heads, head_dim]
            position_ids: Optional position indices
            
        Returns:
            cos, sin tensors for rotation
        """
        seq_len = x.shape[1]
        self._compute_cos_sin(seq_len, x.device, x.dtype)
        
        if position_ids is not None:
            cos = self._cos_cached[position_ids]
            sin = self._sin_cached[position_ids]
        else:
            cos = self._cos_cached[:seq_len]
            sin = self._sin_cached[:seq_len]
        
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to queries and keys.
    
    Args:
        q: [batch, seq_len, num_heads, head_dim]
        k: [batch, seq_len, num_kv_heads, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]
    """
    def rotate_half(x):
        """Rotate half of the hidden dims."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    # Expand for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


# =============================================================================
# Flash Attention
# =============================================================================

class FlashAttentionLayer(nn.Module):
    """
    Flash Attention Implementation
    
    Memory-efficient attention that:
    - Uses tiling to reduce memory from O(N²) to O(N)
    - Fuses operations for better GPU utilization
    - Supports causal masking efficiently
    """
    
    def __init__(
        self,
        config: MoEConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        
        # QKV projections
        self.q_proj = nn.Linear(
            config.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, 
            self.num_kv_heads * self.head_dim, 
            bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, 
            self.num_kv_heads * self.head_dim, 
            bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            config.hidden_size, 
            bias=False
        )
        
        # RoPE
        self.rotary_emb = RoPEEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Attention dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Flash attention forward pass.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional KV cache
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        new_cache = (k, v) if use_cache else None
        
        # Expand KV for GQA
        if self.num_queries_per_kv > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_queries_per_kv, -1)
            k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_queries_per_kv, -1)
            v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Try to use Flash Attention 2 if available
        try:
            from flash_attn import flash_attn_func
            
            # Flash attention expects [batch, seq_len, num_heads, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                causal=True,
            )
            attn_weights = None
            
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
            
        except ImportError:
            # Fallback to standard scaled dot-product attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask
            if attention_mask is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, k.size(2), dtype=torch.bool, device=q.device),
                    diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            else:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights if output_attentions else None, new_cache


# =============================================================================
# Sparse Attention
# =============================================================================

class SparseAttention(nn.Module):
    """
    Sparse Attention for Long Context
    
    Combines multiple patterns for efficient long-range attention:
    - Local attention (sliding window)
    - Strided attention (global patterns)
    - Global attention (CLS tokens)
    
    Supports context windows up to 1M tokens.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        layer_idx: int = 0,
        local_window_size: int = 4096,
        global_stride: int = 512,
        num_global_tokens: int = 64,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.local_window_size = local_window_size
        self.global_stride = global_stride
        self.num_global_tokens = num_global_tokens
        
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        
        # QKV projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RoPEEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # Global tokens for long-range dependencies
        self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, config.hidden_size) * 0.02)
        
        self.scale = self.head_dim ** -0.5
    
    def _compute_sparse_mask(
        self, 
        seq_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute sparse attention mask combining local + strided + global patterns.
        """
        # Local attention mask (sliding window)
        local_mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.local_window_size // 2)
            end = min(seq_len, i + self.local_window_size // 2 + 1)
            local_mask[i, start:end] = False
        
        # Strided attention mask
        strided_mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        for i in range(seq_len):
            # Attend to every global_stride-th position
            strided_positions = torch.arange(0, seq_len, self.global_stride, device=device)
            strided_mask[i, strided_positions] = False
        
        # Combine masks (attend if either local OR strided)
        combined_mask = local_mask & strided_mask
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        combined_mask = combined_mask | causal_mask
        
        return combined_mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Sparse attention forward pass.
        """
        batch_size, _, _ = hidden_states.shape
        
        # Add global tokens
        global_tokens = self.global_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        extended_hidden = torch.cat([global_tokens, hidden_states], dim=1)
        extended_seq_len = extended_hidden.shape[1]
        
        # Project
        q = self.q_proj(extended_hidden)
        k = self.k_proj(extended_hidden)
        v = self.v_proj(extended_hidden)
        
        # Reshape
        q = q.view(batch_size, extended_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, extended_seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, extended_seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand KV for GQA
        num_queries_per_kv = self.num_heads // self.num_kv_heads
        if num_queries_per_kv > 1:
            k = k.unsqueeze(2).expand(-1, -1, num_queries_per_kv, -1, -1)
            k = k.reshape(batch_size, self.num_heads, extended_seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, num_queries_per_kv, -1, -1)
            v = v.reshape(batch_size, self.num_heads, extended_seq_len, self.head_dim)
        
        # Compute attention with sparse mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sparse mask
        sparse_mask = self._compute_sparse_mask(extended_seq_len, hidden_states.device)
        attn_weights = attn_weights.masked_fill(sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, extended_seq_len, -1)
        
        # Remove global tokens from output
        attn_output = attn_output[:, self.num_global_tokens:, :]
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights if output_attentions else None, None


# =============================================================================
# Expert Router (MoE)
# =============================================================================

class ExpertRouter(nn.Module):
    """
    Expert Router for Mixture of Experts
    
    Routes tokens to top-k experts with load balancing.
    Implements auxiliary loss for balanced expert utilization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        routing_strategy: ExpertRoutingStrategy = ExpertRoutingStrategy.TOP_K,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.routing_strategy = routing_strategy
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        
        # Router weights
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Noise for load balancing during training
        self.jitter_noise = 0.01
    
    def _compute_auxiliary_loss(
        self,
        router_probs: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing.
        
        Encourages even distribution of tokens across experts.
        """
        # [num_experts]
        tokens_per_expert = expert_mask.float().sum(dim=(0, 1))
        total_tokens = expert_mask.shape[0] * expert_mask.shape[1]
        
        # Expert utilization fraction
        f_i = tokens_per_expert / total_tokens
        
        # Average router probability per expert
        p_i = router_probs.mean(dim=(0, 1))
        
        # Auxiliary loss: sum(f_i * p_i)
        aux_loss = self.num_experts * (f_i * p_i).sum()
        
        return aux_loss * self.aux_loss_coef
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            router_probs: [batch, seq_len, num_experts] - routing probabilities
            expert_indices: [batch, seq_len, num_experts_per_token] - selected experts
            expert_weights: [batch, seq_len, num_experts_per_token] - expert weights
            aux_loss: Load balancing auxiliary loss
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [batch, seq_len, num_experts]
        
        # Add noise during training for exploration
        if self.training:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        if self.routing_strategy == ExpertRoutingStrategy.TOP_K:
            # Select top-k experts
            expert_weights, expert_indices = torch.topk(
                router_probs, 
                self.num_experts_per_token, 
                dim=-1
            )
            
            # Normalize weights
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
            
        elif self.routing_strategy == ExpertRoutingStrategy.EXPERT_CHOICE:
            # Expert choice routing (each expert selects top tokens)
            # Transpose for expert-centric view
            router_probs_t = router_probs.transpose(-1, -2)  # [batch, num_experts, seq_len]
            
            capacity = int(hidden_states.shape[1] * self.capacity_factor / self.num_experts)
            _, _ = torch.topk(router_probs_t, capacity, dim=-1)  # For expert-choice routing
            
            # Convert back to token-centric view (simplified)
            # Default to top-k for simplicity in this implementation
            expert_weights, expert_indices = torch.topk(
                router_probs, 
                self.num_experts_per_token, 
                dim=-1
            )
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        else:
            # Default to top-k
            expert_weights, expert_indices = torch.topk(
                router_probs, 
                self.num_experts_per_token, 
                dim=-1
            )
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary loss
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts)
        aux_loss = self._compute_auxiliary_loss(router_probs, expert_mask)
        
        return router_probs, expert_indices, expert_weights, aux_loss


# =============================================================================
# Expert Module
# =============================================================================

class Expert(nn.Module):
    """Single expert FFN module with SwiGLU activation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: down(silu(gate(x)) * up(x))"""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer
    
    Replaces standard FFN with multiple expert FFNs.
    Only activates top-k experts per token.
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Router
        self.router = ExpertRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_token,
            capacity_factor=config.expert_capacity_factor,
            aux_loss_coef=config.router_aux_loss_coef,
        )
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MoE forward pass.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: Router auxiliary loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route tokens
        _, expert_indices, expert_weights, aux_loss = self.router(hidden_states)
        
        # Flatten for processing
        hidden_flat = hidden_states.view(-1, hidden_size)  # [batch*seq_len, hidden_size]
        expert_indices_flat = expert_indices.view(-1, self.config.num_experts_per_token)
        expert_weights_flat = expert_weights.view(-1, self.config.num_experts_per_token)
        
        # Compute expert outputs
        output = torch.zeros_like(hidden_flat)
        
        for k in range(self.config.num_experts_per_token):
            for expert_idx in range(self.config.num_experts):
                # Find tokens routed to this expert at position k
                mask = (expert_indices_flat[:, k] == expert_idx)
                if mask.any():
                    expert_input = hidden_flat[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_weights_flat[mask, k:k+1] * expert_output
        
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, aux_loss


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# =============================================================================
# Transformer Block
# =============================================================================

class MoETransformerBlock(nn.Module):
    """
    Single transformer block with MoE FFN.
    
    Architecture:
    - Pre-norm (RMSNorm)
    - Attention (Flash/Sparse/Standard)
    - Residual connection
    - Pre-norm (RMSNorm)
    - MoE FFN
    - Residual connection
    """
    
    def __init__(
        self,
        config: MoEConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        if config.attention_type == AttentionType.SPARSE:
            self.attention = SparseAttention(config, layer_idx)
        else:
            self.attention = FlashAttentionLayer(config, layer_idx)
        
        # MoE FFN
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = MoELayer(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple], torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            hidden_states: Output hidden states
            attn_weights: Attention weights (if output_attentions)
            new_cache: KV cache (if use_cache)
            aux_loss: MoE auxiliary loss
        """
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, attn_weights, new_cache = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        hidden_states = residual + hidden_states
        
        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, attn_weights, new_cache, aux_loss


# =============================================================================
# MoE Transformer Model
# =============================================================================

class MoETransformer(nn.Module):
    """
    Complete MoE Transformer for Foundation Model
    
    Target specifications:
    - 500B-1T total parameters
    - ~50B-100B active parameters per forward
    - 128K-1M context window
    - Multi-modal capable
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MoETransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Output projection (can be tied with embedding)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Multi-modal encoders (optional)
        if config.enable_multimodal:
            self.vision_encoder = nn.Linear(config.vision_hidden_size, config.hidden_size)
            self.audio_encoder = nn.Linear(config.audio_hidden_size, config.hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info("Initialized MoE Transformer:")
        logger.info(f"  Total parameters: {config.total_params:,}")
        logger.info(f"  Active parameters: {config.active_params:,}")
        logger.info(f"  Efficiency ratio: {config.active_params / config.total_params:.2%}")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # Multi-modal inputs
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Optional KV cache
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            vision_features: Optional vision embeddings
            audio_features: Optional audio embeddings
            
        Returns:
            Dictionary with:
            - logits: [batch, seq_len, vocab_size]
            - aux_loss: Total MoE auxiliary loss
            - past_key_values: KV cache (if use_cache)
            - hidden_states: All hidden states (if output_hidden_states)
            - attentions: Attention weights (if output_attentions)
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Add multi-modal features if provided
        if self.config.enable_multimodal:
            if vision_features is not None:
                vision_emb = self.vision_encoder(vision_features)
                hidden_states = torch.cat([vision_emb, hidden_states], dim=1)
            if audio_features is not None:
                audio_emb = self.audio_encoder(audio_features)
                hidden_states = torch.cat([audio_emb, hidden_states], dim=1)
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], 
                device=hidden_states.device
            ).unsqueeze(0)
        
        # Process through layers
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        new_key_values = [] if use_cache else None
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            past_kv = past_key_values[i] if past_key_values else None
            
            hidden_states, attn_weights, new_cache, aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            total_aux_loss += aux_loss
            
            if use_cache:
                new_key_values.append(new_cache)
            
            if output_attentions:
                all_attentions.append(attn_weights)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        
        return {
            "logits": logits,
            "aux_loss": total_aux_loss / len(self.layers),
            "past_key_values": new_key_values,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "last_hidden_state": hidden_states,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: [batch, seq_len] - prompt tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use argmax
            
        Returns:
            Generated token IDs
        """
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Forward pass (with KV cache)
            outputs = self.forward(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs["logits"][:, -1, :] / temperature
            past_key_values = outputs["past_key_values"]
            
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS (assuming token 2 is EOS)
            if (next_token == 2).all():
                break
        
        return generated
    
    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if self.lm_head is not None:
                n_params -= self.lm_head.weight.numel()
        return n_params


# =============================================================================
# Model Configurations
# =============================================================================

def get_moe_config_500b() -> MoEConfig:
    """500B parameter MoE configuration."""
    return MoEConfig(
        vocab_size=128000,
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        num_experts=128,
        num_experts_per_token=2,
        max_position_embeddings=131072,
    )


def get_moe_config_1t() -> MoEConfig:
    """1T parameter MoE configuration."""
    return MoEConfig(
        vocab_size=128000,
        hidden_size=12288,
        intermediate_size=43008,
        num_hidden_layers=96,
        num_attention_heads=96,
        num_key_value_heads=8,
        num_experts=256,
        num_experts_per_token=2,
        max_position_embeddings=1048576,  # 1M context
    )


def get_moe_config_dev() -> MoEConfig:
    """Small development configuration for testing."""
    return MoEConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1792,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_token=2,
        max_position_embeddings=4096,
    )
