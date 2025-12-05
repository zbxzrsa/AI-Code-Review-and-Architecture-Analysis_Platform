"""
Custom Attention Mechanisms for V1 VC-AI

Implements various attention patterns for experimentation:
- Sparse attention for long commit histories
- Cross-layer attention routing for semantic change tracking
- Grouped Query Attention (GQA) variants
- Flash Attention 2 wrapper
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionOutput:
    """Output from attention computation"""
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


class SparseAttention(nn.Module):
    """
    Sparse attention for processing long commit histories efficiently.
    
    Uses a combination of:
    - Local attention (sliding window)
    - Global attention (selected positions)
    - Random attention (for expressivity)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int = 64,
        num_local_blocks: int = 3,
        num_global_blocks: int = 1,
        num_random_blocks: int = 1,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.num_local_blocks = num_local_blocks
        self.num_global_blocks = num_global_blocks
        self.num_random_blocks = num_random_blocks
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _create_sparse_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create sparse attention mask"""
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Initialize mask (1 = attend, 0 = mask)
        mask = torch.zeros(num_blocks, num_blocks, device=device)
        
        for i in range(num_blocks):
            # Local attention
            local_start = max(0, i - self.num_local_blocks // 2)
            local_end = min(num_blocks, i + self.num_local_blocks // 2 + 1)
            mask[i, local_start:local_end] = 1
            
            # Global attention (first and last blocks)
            mask[i, :self.num_global_blocks] = 1
            mask[i, -self.num_global_blocks:] = 1
            
            # Random attention
            if self.num_random_blocks > 0:
                random_indices = torch.randperm(num_blocks)[:self.num_random_blocks]
                mask[i, random_indices] = 1
        
        # Expand to sequence length
        full_mask = mask.repeat_interleave(self.block_size, dim=0)
        full_mask = full_mask.repeat_interleave(self.block_size, dim=1)
        full_mask = full_mask[:seq_len, :seq_len]
        
        return full_mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key values for caching
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        new_past_key_value = (k, v)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply sparse mask
        sparse_mask = self._create_sparse_mask(k.size(2), hidden_states.device)
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return AttentionOutput(
            hidden_states=attn_output,
            attention_weights=attn_weights if output_attentions else None,
            past_key_value=new_past_key_value,
        )


class CrossLayerAttention(nn.Module):
    """
    Cross-layer attention routing for semantic change tracking.
    
    Routes information between non-adjacent layers to capture
    long-range dependencies in code changes.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        source_layers: List[int],
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.source_layers = source_layers
        
        # Projections for cross-layer attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in source_layers
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in source_layers
        ])
        self.o_proj = nn.Linear(hidden_size * len(source_layers), hidden_size, bias=False)
        
        # Layer gating
        self.layer_gate = nn.Linear(hidden_size, len(source_layers))
        
        self.dropout = nn.Dropout(attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        source_hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project query from current layer
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention from each source layer
        cross_outputs = []
        for i, (source_hidden, k_proj, v_proj) in enumerate(
            zip(source_hidden_states, self.k_projs, self.v_projs)
        ):
            k = k_proj(source_hidden)
            v = v_proj(source_hidden)
            
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention scores
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
            
            cross_outputs.append(attn_output)
        
        # Combine outputs from different source layers
        combined = torch.cat(cross_outputs, dim=-1)
        output = self.o_proj(combined)
        
        # Apply layer gating
        _ = F.softmax(self.layer_gate(hidden_states), dim=-1)  # noqa: F841 - gate_weights for future
        # Weighted combination could be applied here for more sophisticated gating
        
        return output


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation.
    
    Reduces memory bandwidth by sharing key-value heads across
    multiple query heads, while maintaining model quality.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        attention_dropout: float = 0.0,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = hidden_size // num_heads
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Rotary position embeddings
        self._init_rope(rope_theta, max_position_embeddings)
        
    def _init_rope(self, theta: float, max_position_embeddings: int):
        """Initialize rotary position embeddings"""
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cos/sin cache
        t = torch.arange(max_position_embeddings)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to queries and keys"""
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        
        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Repeat key-value heads to match query heads"""
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim
        )
        return hidden_states.reshape(batch, self.num_heads, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        q, k = self._apply_rotary_pos_emb(q, k, position_ids)
        
        # Handle past key values
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        new_past_key_value = (k, v)
        
        # Repeat KV heads to match query heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return AttentionOutput(
            hidden_states=attn_output,
            attention_weights=attn_weights if output_attentions else None,
            past_key_value=new_past_key_value,
        )


class FlashAttentionWrapper(nn.Module):
    """
    Wrapper for Flash Attention 2 integration.
    
    Provides memory-efficient attention computation with
    significant speedups for long sequences.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        use_alibi: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_alibi = use_alibi
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout_p = attention_dropout
        
        # Check for flash attention availability
        self._flash_attention_available = self._check_flash_attention()
        
    def _check_flash_attention(self) -> bool:
        """Check if flash attention is available"""
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Run flash attention"""
        from flash_attn import flash_attn_func
        
        # Flash attention expects (batch, seq, heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=causal,
        )
        
        return output.transpose(1, 2)
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fallback to standard attention"""
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
        return torch.matmul(attn_weights, v)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        new_past_key_value = (k, v) if use_cache else None
        
        # Compute attention
        if self._flash_attention_available and not output_attentions:
            attn_output = self._flash_attention(q, k, v, causal=True)
            attn_weights = None
        else:
            attn_output = self._standard_attention(q, k, v, attention_mask)
            attn_weights = None  # Would need separate computation for weights
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return AttentionOutput(
            hidden_states=attn_output,
            attention_weights=attn_weights,
            past_key_value=new_past_key_value,
        )
