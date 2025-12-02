"""
Inference Configuration for V1 VC-AI

Real-time inference parameters optimized for high throughput
and creative exploration in the experimentation environment.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
from enum import Enum


class DecodingStrategy(str, Enum):
    """Text generation decoding strategies"""
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    SAMPLING = "sampling"
    TOP_K = "top_k"
    TOP_P = "top_p"
    CONTRASTIVE = "contrastive"
    SPECULATIVE = "speculative"


class CacheEvictionPolicy(str, Enum):
    """Cache eviction policies for KV cache"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"


@dataclass
class GenerationConfig:
    """
    Text generation configuration.
    
    Higher temperature for creative exploration in V1.
    """
    # Sampling parameters
    temperature: float = 0.8             # High for creativity
    top_p: float = 0.95                  # Nucleus sampling
    top_k: int = 50                      # Top-k sampling
    
    # Length control
    max_tokens: int = 2048
    min_tokens: int = 10
    max_new_tokens: Optional[int] = None
    
    # Repetition control
    repetition_penalty: float = 1.2
    length_penalty: float = 0.6
    no_repeat_ngram_size: int = 3
    
    # Beam search (for higher quality)
    num_beams: int = 3
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    early_stopping: bool = True
    
    # Contrastive decoding
    penalty_alpha: float = 0.6          # For contrastive search
    
    # Output control
    do_sample: bool = True
    num_return_sequences: int = 1
    output_scores: bool = False
    return_dict_in_generate: bool = True
    
    # Special tokens
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Stopping criteria
    stop_sequences: List[str] = field(default_factory=list)


@dataclass  
class OptimizationConfig:
    """Inference optimization settings"""
    # Attention optimization
    use_flash_attention: bool = True
    use_triton_flash_attention: bool = True
    
    # KV cache
    use_cache: bool = True
    past_key_values_length: int = 0
    
    # Device and memory
    device_map: str = "auto"
    offload_folder: Optional[str] = None
    max_memory: Optional[dict] = None
    
    # Quantization for inference
    quantization_method: str = "int8"
    
    # RoPE scaling for extended context
    rope_scaling_factor: float = 2.0
    
    # Tensor parallelism
    tensor_parallel_size: int = 1


@dataclass
class BatchingConfig:
    """
    Dynamic batching configuration for high throughput.
    """
    # Batch sizes
    batch_size: int = 32                 # Aggressive batch processing
    prefill_batch_size: int = 64         # Larger for prefill phase
    decode_batch_size: int = 256         # Larger for decode phase
    
    # Dynamic batching
    dynamic_batching: bool = True
    max_batch_wait_time_ms: int = 50     # Max wait for batch formation
    
    # Request handling
    max_concurrent_requests: int = 1000
    request_timeout_seconds: int = 60
    
    # Memory management
    max_batch_total_tokens: int = 32768
    max_input_length: int = 4096
    max_total_tokens: int = 8192


@dataclass
class CachingConfig:
    """
    Caching configuration for inference acceleration.
    """
    # KV cache
    cache_size_gb: float = 16.0
    use_prefix_caching: bool = True
    cache_eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    
    # Prompt caching
    enable_prompt_cache: bool = True
    prompt_cache_size: int = 10000
    prompt_cache_ttl_seconds: int = 3600
    
    # Result caching
    enable_result_cache: bool = True
    result_cache_size: int = 50000
    result_cache_ttl_seconds: int = 300
    
    # Semantic deduplication
    enable_semantic_dedup: bool = True
    semantic_similarity_threshold: float = 0.95


@dataclass
class SpeculativeDecodingConfig:
    """
    Speculative decoding configuration.
    
    Uses a smaller draft model to generate candidates,
    verified by the main model for faster inference.
    """
    enabled: bool = True
    
    # Draft model settings
    draft_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    draft_model_device: str = "cuda:0"
    
    # Speculation parameters
    num_speculative_tokens: int = 5      # Tokens to speculate per step
    max_speculation_length: int = 64
    
    # Acceptance criteria
    acceptance_threshold: float = 0.1    # Min probability ratio
    use_typical_acceptance: bool = True
    
    # Performance tuning
    draft_batch_size: int = 8
    verify_batch_size: int = 4


@dataclass
class InferenceConfig:
    """
    Complete inference configuration for V1 VC-AI.
    
    Optimized for high throughput and creative exploration.
    """
    # Sub-configurations
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    speculative: SpeculativeDecodingConfig = field(default_factory=SpeculativeDecodingConfig)
    
    # Model settings
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Health checks
    health_check_interval_seconds: int = 30
    warmup_requests: int = 10
    
    def to_vllm_config(self) -> dict:
        """Convert to vLLM serving configuration"""
        return {
            "model": self.model_path,
            "tokenizer": self.tokenizer_path,
            "tensor_parallel_size": self.optimization.tensor_parallel_size,
            "max_num_seqs": self.batching.max_concurrent_requests,
            "max_num_batched_tokens": self.batching.max_batch_total_tokens,
            "max_model_len": self.batching.max_total_tokens,
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
            "dtype": "float16",
        }
    
    def to_tgi_config(self) -> dict:
        """Convert to Text Generation Inference configuration"""
        return {
            "model_id": self.model_path,
            "max_concurrent_requests": self.batching.max_concurrent_requests,
            "max_input_length": self.batching.max_input_length,
            "max_total_tokens": self.batching.max_total_tokens,
            "max_batch_prefill_tokens": self.batching.max_batch_total_tokens,
            "waiting_served_ratio": 0.3,
            "max_waiting_tokens": 20,
        }


# Pre-configured inference profiles
HIGH_THROUGHPUT_CONFIG = InferenceConfig(
    generation=GenerationConfig(
        temperature=0.7,
        num_beams=1,
        do_sample=True,
    ),
    batching=BatchingConfig(
        batch_size=64,
        dynamic_batching=True,
        max_batch_wait_time_ms=100,
    ),
    speculative=SpeculativeDecodingConfig(enabled=True),
)

HIGH_QUALITY_CONFIG = InferenceConfig(
    generation=GenerationConfig(
        temperature=0.3,
        num_beams=5,
        do_sample=False,
        repetition_penalty=1.5,
    ),
    batching=BatchingConfig(
        batch_size=16,
        dynamic_batching=False,
    ),
    speculative=SpeculativeDecodingConfig(enabled=False),
)

EXPERIMENTAL_CONFIG = InferenceConfig(
    generation=GenerationConfig(
        temperature=1.0,
        top_p=0.98,
        top_k=100,
        num_beams=3,
        do_sample=True,
        diversity_penalty=0.5,
    ),
    batching=BatchingConfig(
        batch_size=32,
        dynamic_batching=True,
    ),
    speculative=SpeculativeDecodingConfig(
        enabled=True,
        num_speculative_tokens=8,
    ),
)
