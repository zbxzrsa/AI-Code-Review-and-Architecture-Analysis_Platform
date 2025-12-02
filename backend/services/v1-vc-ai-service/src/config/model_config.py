"""
Model Configuration for V1 VC-AI

Defines model architecture, quantization, and attention configurations
for LLaMA 2 13B / Mistral 7B with aggressive optimizations.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum


class ModelType(str, Enum):
    """Supported base models"""
    LLAMA2_13B = "meta-llama/Llama-2-13b-hf"
    LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
    MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"
    CODELLAMA_13B = "codellama/CodeLlama-13b-hf"
    CODELLAMA_7B = "codellama/CodeLlama-7b-hf"


class AttentionType(str, Enum):
    """Attention mechanism types"""
    STANDARD = "standard"
    MULTI_QUERY = "multi_query"           # MQA
    GROUPED_QUERY = "grouped_query"       # GQA
    FLASH_ATTENTION_2 = "flash_attention_2"
    SPARSE = "sparse"
    CROSS_LAYER = "cross_layer"


class QuantizationType(str, Enum):
    """Quantization methods"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"                          # 4-bit NormalFloat
    FP4 = "fp4"                          # 4-bit Floating Point


@dataclass
class QuantizationConfig:
    """
    Quantization configuration for model compression.
    
    Uses BitsAndBytes for 4-bit quantization with QLoRA compatibility.
    """
    bits: int = 4
    quant_type: QuantizationType = QuantizationType.NF4
    use_double_quant: bool = True        # Nested quantization for extra savings
    compute_dtype: str = "float16"       # Compute in fp16 for accuracy
    
    # Memory optimization
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    llm_int8_enable_fp32_cpu_offload: bool = False
    
    def to_bnb_config(self) -> dict:
        """Convert to BitsAndBytes config dict"""
        return {
            "load_in_4bit": self.bits == 4,
            "load_in_8bit": self.bits == 8,
            "bnb_4bit_quant_type": self.quant_type.value if self.bits == 4 else None,
            "bnb_4bit_use_double_quant": self.use_double_quant if self.bits == 4 else False,
            "bnb_4bit_compute_dtype": self.compute_dtype,
            "llm_int8_threshold": self.llm_int8_threshold,
            "llm_int8_skip_modules": self.llm_int8_skip_modules,
        }


@dataclass
class LoRAConfig:
    """
    Low-Rank Adaptation (LoRA) configuration.
    
    High rank (128) for maximum expressiveness in experimentation.
    """
    r: int = 128                         # High LoRA rank for expressiveness
    lora_alpha: int = 256                # Scaling factor (2x rank)
    lora_dropout: float = 0.05           # Light dropout for regularization
    bias: Literal["none", "all", "lora_only"] = "none"
    
    # Target modules for adaptation
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",    # Query projection
        "k_proj",    # Key projection  
        "v_proj",    # Value projection
        "o_proj",    # Output projection
        "gate_proj", # MLP gate
        "up_proj",   # MLP up projection
        "down_proj", # MLP down projection
    ])
    
    # Task-specific LoRA
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[List[str]] = None
    
    # Fan-in/fan-out initialization
    fan_in_fan_out: bool = False
    init_lora_weights: bool = True
    
    def to_peft_config(self) -> dict:
        """Convert to PEFT LoraConfig dict"""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "target_modules": self.target_modules,
            "task_type": self.task_type,
            "modules_to_save": self.modules_to_save,
            "fan_in_fan_out": self.fan_in_fan_out,
            "init_lora_weights": self.init_lora_weights,
        }


@dataclass
class AttentionConfig:
    """
    Attention mechanism configuration.
    
    Supports multiple attention patterns for experimentation.
    """
    attention_type: AttentionType = AttentionType.FLASH_ATTENTION_2
    
    # Multi-Query / Grouped-Query Attention
    num_key_value_heads: Optional[int] = None  # For GQA/MQA
    
    # Flash Attention settings
    use_flash_attention: bool = True
    use_triton_flash_attention: bool = True
    flash_attention_recompute: bool = False
    
    # Sparse attention for long sequences
    sparse_attention_config: dict = field(default_factory=lambda: {
        "block_size": 64,
        "num_local_blocks": 3,
        "num_global_blocks": 1,
        "attention_dropout": 0.0,
    })
    
    # Cross-layer attention routing
    cross_layer_attention: bool = False
    cross_layer_interval: int = 4        # Every 4th layer
    
    # RoPE (Rotary Position Embedding) settings
    rope_scaling_type: str = "dynamic"   # dynamic, linear, or none
    rope_scaling_factor: float = 2.0     # 2x context extension
    rope_theta: float = 10000.0          # Base frequency
    
    # Sliding window (for Mistral-like models)
    sliding_window: Optional[int] = 4096


@dataclass
class TokenizerConfig:
    """
    Custom tokenizer configuration for code/commit vocabulary.
    """
    vocab_size: int = 32000
    
    # Special tokens for version control
    special_tokens: dict = field(default_factory=lambda: {
        "commit_token": "<COMMIT>",
        "diff_token": "<DIFF>",
        "change_type_token": "<CHANGE_TYPE>",
        "impact_level_token": "<IMPACT_LEVEL>",
        "file_start_token": "<FILE>",
        "file_end_token": "</FILE>",
        "hunk_token": "<HUNK>",
        "addition_token": "<ADD>",
        "deletion_token": "<DEL>",
        "context_token": "<CTX>",
        "semantic_break_token": "<BREAK>",
    })
    
    # BPE training settings
    bpe_dropout: float = 0.0
    min_frequency: int = 2
    
    # Code-specific preprocessing
    preserve_whitespace: bool = True
    split_on_punctuation: bool = True
    lowercase: bool = False              # Case-sensitive for code


@dataclass
class ModelConfig:
    """
    Complete model configuration for V1 VC-AI.
    
    Primary: LLaMA 2 13B or Mistral 7B with aggressive quantization.
    """
    # Base model selection
    model_type: ModelType = ModelType.MISTRAL_7B
    model_name_or_path: Optional[str] = None  # Override for custom path
    
    # Model architecture
    hidden_size: int = 4096
    intermediate_size: int = 14336       # Mistral default
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    max_position_embeddings: int = 32768 # Extended context
    
    # Vocabulary
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Sub-configurations
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    
    # Model loading options
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    use_cache: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = True
    use_memory_efficient_attention: bool = True
    
    def get_model_path(self) -> str:
        """Get the model path/identifier"""
        return self.model_name_or_path or self.model_type.value
    
    def to_hf_config(self) -> dict:
        """Convert to HuggingFace AutoConfig compatible dict"""
        return {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "use_cache": self.use_cache,
            "rope_scaling": {
                "type": self.attention.rope_scaling_type,
                "factor": self.attention.rope_scaling_factor,
            } if self.attention.rope_scaling_type != "none" else None,
        }


# Default configuration instances
DEFAULT_MISTRAL_CONFIG = ModelConfig(
    model_type=ModelType.MISTRAL_7B,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    attention=AttentionConfig(
        attention_type=AttentionType.GROUPED_QUERY,
        num_key_value_heads=8,  # GQA with 8 KV heads
        sliding_window=4096,
    ),
)

DEFAULT_LLAMA2_CONFIG = ModelConfig(
    model_type=ModelType.LLAMA2_13B,
    hidden_size=5120,
    intermediate_size=13824,
    num_hidden_layers=40,
    num_attention_heads=40,
    attention=AttentionConfig(
        attention_type=AttentionType.FLASH_ATTENTION_2,
        num_key_value_heads=40,  # Standard MHA
    ),
)

DEFAULT_CODELLAMA_CONFIG = ModelConfig(
    model_type=ModelType.CODELLAMA_13B,
    hidden_size=5120,
    intermediate_size=13824,
    num_hidden_layers=40,
    num_attention_heads=40,
    max_position_embeddings=16384,  # Extended for code
    attention=AttentionConfig(
        attention_type=AttentionType.FLASH_ATTENTION_2,
        rope_scaling_factor=2.0,
    ),
)
