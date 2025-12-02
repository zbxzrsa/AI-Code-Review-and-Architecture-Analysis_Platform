"""
Main Model Architecture for V1 VC-AI

Integrates all components into a unified version control AI model:
- Custom attention mechanisms
- BPE tokenization
- Mixture of Experts
- LoRA fine-tuning support
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, LoRAConfig, QuantizationConfig
from .attention import GroupedQueryAttention, SparseAttention, FlashAttentionWrapper
from .moe import MixtureOfExperts, MoEConfig, ExpertType


logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Supported task types"""
    COMMIT_MESSAGE_GENERATION = "commit_message_generation"
    CHANGE_TYPE_CLASSIFICATION = "change_type_classification"
    IMPACT_PREDICTION = "impact_prediction"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CODE_REVIEW = "code_review"


@dataclass
class ModelOutput:
    """Output from the version control AI model"""
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    
    # Task-specific outputs
    change_type_logits: Optional[torch.Tensor] = None
    impact_logits: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None


class VersionControlAIModel(nn.Module):
    """
    Main Version Control AI model architecture.
    
    Supports multiple configurations:
    - Standard transformer with GQA/Flash Attention
    - Sparse attention for long sequences
    - MoE layers for specialized processing
    - Multi-task outputs
    """
    
    # Change type classes
    CHANGE_TYPES = ["bug_fix", "feature", "refactor", "optimization", "docs", "test", "chore"]
    
    # Impact levels
    IMPACT_LEVELS = ["low", "medium", "high", "critical"]
    
    def __init__(
        self,
        config: ModelConfig,
        use_moe: bool = False,
        moe_config: Optional[MoEConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.use_moe = use_moe
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = self._create_position_embeddings(config)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.layers.append(self._create_layer(i, config, moe_config))
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Output heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Multi-task heads
        self.change_type_head = nn.Linear(config.hidden_size, len(self.CHANGE_TYPES))
        self.impact_head = nn.Linear(config.hidden_size, len(self.IMPACT_LEVELS))
        
        # Pooler for sequence-level tasks
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_position_embeddings(self, config: ModelConfig) -> nn.Module:
        """Create position embeddings (learned or RoPE)"""
        # Using learned position embeddings for simplicity
        # RoPE is applied in attention layers
        return nn.Embedding(config.max_position_embeddings, config.hidden_size)
    
    def _create_layer(
        self,
        layer_idx: int,
        config: ModelConfig,
        moe_config: Optional[MoEConfig],
    ) -> nn.Module:
        """Create a single transformer layer"""
        return TransformerLayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.attention.num_key_value_heads,
            attention_type=config.attention.attention_type.value,
            use_moe=self.use_moe and layer_idx % 2 == 0,  # MoE every other layer
            moe_config=moe_config,
            use_flash_attention=config.attention.use_flash_attention,
            rope_theta=config.attention.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        labels: Optional[torch.Tensor] = None,
        change_type_labels: Optional[torch.Tensor] = None,
        impact_labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[ModelOutput, Tuple]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        if position_ids is None:
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = 0
            position_ids = torch.arange(
                past_length, past_length + seq_len, device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states + self.embed_positions(position_ids)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)
        
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # Process through layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        new_past_key_values = () if use_cache else None
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                new_past_key_values += (layer_outputs[-1],)
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)
            
            # Accumulate MoE auxiliary loss
            if hasattr(layer, 'aux_loss') and layer.aux_loss is not None:
                total_aux_loss += layer.aux_loss
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Multi-task outputs (using pooled representation)
        pooled_output = self.pooler(hidden_states[:, 0, :])  # CLS token
        change_type_logits = self.change_type_head(pooled_output)
        impact_logits = self.impact_head(pooled_output)
        
        # Compute losses
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        if change_type_labels is not None:
            change_type_loss = F.cross_entropy(change_type_logits, change_type_labels)
            loss = loss + change_type_loss if loss is not None else change_type_loss
        
        if impact_labels is not None:
            impact_loss = F.cross_entropy(impact_logits, impact_labels)
            loss = loss + impact_loss if loss is not None else impact_loss
        
        # Add auxiliary loss
        if total_aux_loss > 0:
            aux_loss = torch.tensor(total_aux_loss, device=device)
            if loss is not None:
                loss = loss + self.config.lora.lora_alpha * aux_loss / 1000
        else:
            aux_loss = None
        
        if return_dict:
            return ModelOutput(
                logits=logits,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                past_key_values=new_past_key_values,
                loss=loss,
                aux_loss=aux_loss,
                change_type_logits=change_type_logits,
                impact_logits=impact_logits,
                embeddings=pooled_output,
            )
        
        return (logits, loss)
    
    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device),
            diagonal=1,
        )
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _gradient_checkpointing_func(self, layer, *args):
        """Wrapper for gradient checkpointing"""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(layer),
            *args,
            use_reentrant=False,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text autoregressively"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=generated[:, -1:] if past_key_values else generated,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
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
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated


class TransformerLayer(nn.Module):
    """Single transformer layer with configurable attention and FFN"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        attention_type: str = "flash_attention_2",
        use_moe: bool = False,
        moe_config: Optional[MoEConfig] = None,
        use_flash_attention: bool = True,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Attention
        num_kv_heads = num_key_value_heads or num_attention_heads
        if attention_type == "grouped_query" and num_kv_heads != num_attention_heads:
            self.self_attn = GroupedQueryAttention(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_key_value_heads=num_kv_heads,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
            )
        elif use_flash_attention:
            self.self_attn = FlashAttentionWrapper(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
            )
        else:
            self.self_attn = GroupedQueryAttention(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_key_value_heads=num_attention_heads,
            )
        
        # FFN or MoE
        if use_moe and moe_config is not None:
            self.mlp = MixtureOfExperts(moe_config)
            self.use_moe = True
        else:
            self.mlp = MLP(hidden_size, intermediate_size)
            self.use_moe = False
        
        self.aux_loss = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        
        hidden_states = residual + attn_output.hidden_states
        
        # MLP / MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.use_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
            self.aux_loss = aux_loss
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_output.attention_weights,)
        outputs += (attn_output.past_key_value,)
        
        return outputs


class MLP(nn.Module):
    """Standard MLP with SwiGLU activation"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ModelFactory:
    """Factory for creating and loading models"""
    
    @staticmethod
    def create_model(
        config: ModelConfig,
        use_moe: bool = False,
        moe_config: Optional[MoEConfig] = None,
    ) -> VersionControlAIModel:
        """Create a new model from config"""
        return VersionControlAIModel(config, use_moe, moe_config)
    
    @staticmethod
    def from_pretrained(
        model_path: str,
        config: Optional[ModelConfig] = None,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        quantization_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> VersionControlAIModel:
        """Load a pretrained model with optional quantization and LoRA"""
        logger.info(f"Loading model from {model_path}")
        
        # This would integrate with HuggingFace transformers
        # For now, return a placeholder
        try:
            from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
            from peft import get_peft_model, LoraConfig as PeftLoraConfig
            
            # Load base model config
            if config is None:
                hf_config = AutoConfig.from_pretrained(model_path)
                config = ModelConfig(
                    hidden_size=hf_config.hidden_size,
                    intermediate_size=hf_config.intermediate_size,
                    num_hidden_layers=hf_config.num_hidden_layers,
                    num_attention_heads=hf_config.num_attention_heads,
                    vocab_size=hf_config.vocab_size,
                )
            
            # Prepare quantization
            bnb_config = None
            if quantization_config:
                bnb_config = BitsAndBytesConfig(**quantization_config.to_bnb_config())
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=getattr(torch, torch_dtype),
                trust_remote_code=True,
            )
            
            # Apply LoRA if configured
            if lora_config:
                peft_config = PeftLoraConfig(**lora_config.to_peft_config())
                model = get_peft_model(model, peft_config)
                logger.info(f"Applied LoRA with rank {lora_config.r}")
            
            return model
            
        except ImportError:
            logger.warning("HuggingFace transformers not available, using custom model")
            return VersionControlAIModel(config or ModelConfig())
    
    @staticmethod
    def save_model(
        model: VersionControlAIModel,
        save_path: str,
        save_lora_only: bool = False,
    ):
        """Save model to disk"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        if save_lora_only:
            # Save only LoRA weights
            try:
                model.save_pretrained(save_path)
                logger.info(f"Saved LoRA weights to {save_path}")
            except AttributeError:
                torch.save(model.state_dict(), f"{save_path}/model.pt")
        else:
            # Save full model
            torch.save(model.state_dict(), f"{save_path}/model.pt")
            logger.info(f"Saved full model to {save_path}")
