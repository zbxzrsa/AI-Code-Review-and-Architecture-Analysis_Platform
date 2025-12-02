"""
Model Configuration for V1 Code Review AI

Defines model architecture for Mistral 7B / CodeLLaMA-7B
with multi-dimensional code analysis capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class BaseModelType(str, Enum):
    """Supported base models for code review"""
    MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
    MISTRAL_7B_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"
    CODELLAMA_7B = "codellama/CodeLlama-7b-hf"
    CODELLAMA_7B_INSTRUCT = "codellama/CodeLlama-7b-Instruct-hf"
    CODELLAMA_13B = "codellama/CodeLlama-13b-hf"
    DEEPSEEK_CODER_7B = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    STARCODER2_7B = "bigcode/starcoder2-7b"


class TaskType(str, Enum):
    """Code review task types for multi-task learning"""
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    ARCHITECTURE = "architecture"
    TESTING = "testing"


@dataclass
class QuantizationConfig:
    """INT4 quantization configuration"""
    bits: int = 4
    quant_type: str = "nf4"
    use_double_quant: bool = True
    compute_dtype: str = "float16"
    
    def to_bnb_config(self) -> dict:
        return {
            "load_in_4bit": self.bits == 4,
            "load_in_8bit": self.bits == 8,
            "bnb_4bit_quant_type": self.quant_type,
            "bnb_4bit_use_double_quant": self.use_double_quant,
            "bnb_4bit_compute_dtype": self.compute_dtype,
        }


@dataclass
class LoRAConfig:
    """LoRA configuration for efficient fine-tuning"""
    r: int = 96                          # LoRA rank
    lora_alpha: int = 192                # Scaling factor (2x rank)
    lora_dropout: float = 0.1
    bias: str = "none"
    
    # Target modules for code review
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Task-specific LoRA adapters
    task_adapters: Dict[str, bool] = field(default_factory=lambda: {
        TaskType.CORRECTNESS.value: True,
        TaskType.SECURITY.value: True,
        TaskType.PERFORMANCE.value: True,
        TaskType.MAINTAINABILITY.value: True,
        TaskType.ARCHITECTURE.value: True,
        TaskType.TESTING.value: True,
    })
    
    def to_peft_config(self) -> dict:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "target_modules": self.target_modules,
            "task_type": "CAUSAL_LM",
        }


@dataclass
class SpecialTokensConfig:
    """Special tokens for code review tasks"""
    code_block_start: str = "<CODE_BLOCK>"
    code_block_end: str = "</CODE_BLOCK>"
    review_start: str = "<REVIEW_START>"
    review_end: str = "</REVIEW_END>"
    finding_start: str = "<FINDING>"
    finding_end: str = "</FINDING>"
    
    # Bug type labels
    bug_type_security: str = "<BUG_TYPE:security>"
    bug_type_performance: str = "<BUG_TYPE:performance>"
    bug_type_correctness: str = "<BUG_TYPE:correctness>"
    bug_type_maintainability: str = "<BUG_TYPE:maintainability>"
    
    # Severity labels
    severity_critical: str = "<SEVERITY:critical>"
    severity_high: str = "<SEVERITY:high>"
    severity_medium: str = "<SEVERITY:medium>"
    severity_low: str = "<SEVERITY:low>"
    
    # Reasoning tokens
    reasoning_start: str = "<REASONING>"
    reasoning_end: str = "</REASONING>"
    step_token: str = "<STEP>"
    
    def all_tokens(self) -> List[str]:
        return [
            self.code_block_start, self.code_block_end,
            self.review_start, self.review_end,
            self.finding_start, self.finding_end,
            self.bug_type_security, self.bug_type_performance,
            self.bug_type_correctness, self.bug_type_maintainability,
            self.severity_critical, self.severity_high,
            self.severity_medium, self.severity_low,
            self.reasoning_start, self.reasoning_end, self.step_token,
        ]


@dataclass
class ModelConfig:
    """Complete model configuration for V1 CR-AI"""
    # Base model
    model_type: BaseModelType = BaseModelType.CODELLAMA_7B_INSTRUCT
    model_name_or_path: Optional[str] = None
    
    # Architecture parameters
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    max_position_embeddings: int = 4096
    vocab_size: int = 32000
    
    # Sub-configurations
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    special_tokens: SpecialTokensConfig = field(default_factory=SpecialTokensConfig)
    
    # Model loading
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    use_cache: bool = True
    
    # Multi-task settings
    use_task_adapters: bool = True
    task_embedding_dim: int = 64
    
    # Memory optimization
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    def get_model_path(self) -> str:
        return self.model_name_or_path or self.model_type.value


# Pre-configured model profiles
CODELLAMA_7B_CONFIG = ModelConfig(
    model_type=BaseModelType.CODELLAMA_7B_INSTRUCT,
    max_position_embeddings=16384,  # Extended context for code
)

MISTRAL_7B_CONFIG = ModelConfig(
    model_type=BaseModelType.MISTRAL_7B_INSTRUCT,
    max_position_embeddings=32768,
)

DEEPSEEK_CODER_CONFIG = ModelConfig(
    model_type=BaseModelType.DEEPSEEK_CODER_7B,
    max_position_embeddings=16384,
)
