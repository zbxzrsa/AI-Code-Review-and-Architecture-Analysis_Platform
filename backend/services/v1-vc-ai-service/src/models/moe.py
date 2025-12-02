"""
Mixture of Experts (MoE) for V1 VC-AI

Implements sparse MoE layers for specialized version control tasks:
- Semantic analysis expert
- Impact prediction expert
- Change type classification expert
- Code structure expert
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertType(str, Enum):
    """Types of specialized experts"""
    SEMANTIC_ANALYSIS = "semantic_analysis"
    IMPACT_PREDICTION = "impact_prediction"
    CHANGE_CLASSIFICATION = "change_classification"
    CODE_STRUCTURE = "code_structure"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION = "documentation"
    SECURITY = "security"


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts"""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_experts: int = 8
    num_experts_per_token: int = 2       # Top-k routing
    expert_capacity: float = 1.25        # Capacity factor
    router_jitter_noise: float = 0.01
    router_aux_loss_coef: float = 0.01   # Load balancing loss
    router_z_loss_coef: float = 0.001    # Router z-loss


class ExpertRouter(nn.Module):
    """
    Router that assigns tokens to experts.
    
    Uses top-k gating with auxiliary load balancing loss.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        jitter_noise: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.jitter_noise = jitter_noise
        
        # Router weights
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            
        Returns:
            router_probs: (batch, seq_len, num_experts)
            expert_indices: (batch, seq_len, num_experts_per_token)
            expert_weights: (batch, seq_len, num_experts_per_token)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten batch and sequence dimensions
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            hidden_states_flat = hidden_states_flat + torch.randn_like(hidden_states_flat) * self.jitter_noise
        
        # Compute router logits
        router_logits = self.gate(hidden_states_flat)
        
        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )
        
        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Reshape back
        router_probs = router_probs.view(batch_size, seq_len, self.num_experts)
        expert_indices = expert_indices.view(batch_size, seq_len, self.num_experts_per_token)
        expert_weights = expert_weights.view(batch_size, seq_len, self.num_experts_per_token)
        
        return router_probs, expert_indices, expert_weights
    
    def compute_auxiliary_loss(
        self,
        router_probs: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        
        Encourages uniform distribution of tokens across experts.
        """
        # expert_mask: (batch * seq_len, num_experts) binary mask
        # router_probs: (batch * seq_len, num_experts)
        
        # Fraction of tokens routed to each expert
        tokens_per_expert = expert_mask.float().mean(dim=0)
        
        # Average routing probability for each expert
        router_prob_per_expert = router_probs.mean(dim=0)
        
        # Load balancing loss
        aux_loss = (tokens_per_expert * router_prob_per_expert).sum() * self.num_experts
        
        return aux_loss


class VersionControlExpert(nn.Module):
    """
    Single expert module specialized for a version control task.
    
    Uses standard FFN architecture with optional task-specific heads.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        expert_type: ExpertType = ExpertType.SEMANTIC_ANALYSIS,
        activation: str = "silu",
    ):
        super().__init__()
        self.expert_type = expert_type
        
        # FFN layers
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        # Activation
        if activation == "silu":
            self.act_fn = nn.SiLU()
        elif activation == "gelu":
            self.act_fn = nn.GELU()
        elif activation == "relu":
            self.act_fn = nn.ReLU()
        else:
            self.act_fn = nn.SiLU()
        
        # Optional task-specific projection
        self._init_task_specific_layers()
        
    def _init_task_specific_layers(self):
        """Initialize task-specific layers based on expert type"""
        # Could add specialized layers for different expert types
        pass
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Uses SwiGLU-style gating: down(act(gate(x)) * up(x))
        """
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts layer for version control tasks.
    
    Routes tokens to specialized experts based on input semantics.
    """
    
    def __init__(
        self,
        config: MoEConfig,
        expert_types: Optional[List[ExpertType]] = None,
    ):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        
        # Default expert types
        if expert_types is None:
            expert_types = list(ExpertType)[:config.num_experts]
        
        # Router
        self.router = ExpertRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_token,
            jitter_noise=config.router_jitter_noise,
        )
        
        # Experts
        self.experts = nn.ModuleList([
            VersionControlExpert(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                expert_type=expert_type,
            )
            for expert_type in expert_types
        ])
        
        # For tracking expert utilization
        self.register_buffer(
            "expert_utilization",
            torch.zeros(config.num_experts),
            persistent=False,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            output_router_logits: Whether to return routing probabilities
            
        Returns:
            output: (batch, seq_len, hidden_size)
            router_logits: Optional routing probabilities
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing decisions
        router_probs, expert_indices, expert_weights = self.router(hidden_states)
        
        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Flatten for easier indexing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        expert_indices_flat = expert_indices.view(-1, self.num_experts_per_token)
        expert_weights_flat = expert_weights.view(-1, self.num_experts_per_token)
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            token_mask = (expert_indices_flat == expert_idx).any(dim=-1)
            
            if not token_mask.any():
                continue
            
            # Get tokens for this expert
            expert_tokens = hidden_states_flat[token_mask]
            
            # Get weights for this expert
            expert_weight_mask = (expert_indices_flat[token_mask] == expert_idx)
            token_weights = expert_weights_flat[token_mask]
            weights = (token_weights * expert_weight_mask.float()).sum(dim=-1, keepdim=True)
            
            # Process through expert
            expert_output = expert(expert_tokens)
            
            # Weight and accumulate
            weighted_output = expert_output * weights
            final_hidden_states.view(-1, hidden_size)[token_mask] += weighted_output
            
            # Track utilization
            if self.training:
                self.expert_utilization[expert_idx] += token_mask.sum().float()
        
        # Compute auxiliary loss for load balancing
        aux_loss = None
        if self.training:
            expert_mask = F.one_hot(
                expert_indices.view(-1, self.num_experts_per_token),
                num_classes=self.num_experts
            ).sum(dim=1)
            aux_loss = self.router.compute_auxiliary_loss(
                router_probs.view(-1, self.num_experts),
                expert_mask,
            )
        
        if output_router_logits:
            return final_hidden_states, router_probs, aux_loss
        
        return final_hidden_states, aux_loss
    
    def get_expert_utilization(self) -> Dict[str, float]:
        """Get normalized expert utilization statistics"""
        total = self.expert_utilization.sum()
        if total == 0:
            return {f"expert_{i}": 0.0 for i in range(self.num_experts)}
        
        normalized = self.expert_utilization / total
        return {f"expert_{i}": normalized[i].item() for i in range(self.num_experts)}
    
    def reset_utilization_stats(self):
        """Reset expert utilization tracking"""
        self.expert_utilization.zero_()


class SparseMoEBlock(nn.Module):
    """
    Complete sparse MoE transformer block.
    
    Combines attention with MoE FFN layer.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        moe_config: MoEConfig,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Attention (using standard for simplicity, could use custom)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        
        # MoE layer
        self.moe = MixtureOfExperts(moe_config)
        
        # Dropout
        self.dropout = nn.Dropout(hidden_dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MoE FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states, output_router_logits=False)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, aux_loss


class MoEModel(nn.Module):
    """
    Full MoE-based model for version control analysis.
    
    Stacks multiple MoE blocks with shared or separate experts.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        moe_config: MoEConfig,
        max_position_embeddings: int = 4096,
        moe_layer_frequency: int = 2,  # Use MoE every N layers
    ):
        super().__init__()
        
        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % moe_layer_frequency == 0:
                # MoE layer
                self.layers.append(SparseMoEBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    moe_config=moe_config,
                ))
            else:
                # Standard FFN layer (for efficiency)
                self.layers.append(self._create_standard_block(
                    hidden_size, num_attention_heads, moe_config.intermediate_size
                ))
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Output projection
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def _create_standard_block(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
    ) -> nn.Module:
        """Create a standard transformer block (non-MoE)"""
        return nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
        
        # Process through layers
        total_aux_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, SparseMoEBlock):
                hidden_states, aux_loss = layer(hidden_states, attention_mask)
                if aux_loss is not None:
                    total_aux_loss += aux_loss
            else:
                hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        return logits, total_aux_loss
