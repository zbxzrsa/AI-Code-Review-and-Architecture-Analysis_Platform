"""
Training Configuration for V1 Code Review AI

Comprehensive data pipeline and training settings for
instruction-tuning with chain-of-thought examples.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class DataSourceType(str, Enum):
    """Types of training data sources"""
    REAL_REVIEWS = "real_reviews"
    SYNTHETIC_BUGS = "synthetic_bugs"
    SYNTHETIC_PERF = "synthetic_performance"
    SYNTHETIC_ARCH = "synthetic_architecture"
    BUG_BOUNTY = "bug_bounty"


class BugPattern(str, Enum):
    """Known bug patterns for synthetic injection"""
    OFF_BY_ONE = "off_by_one_errors"
    NULL_POINTER = "null_pointer_dereferences"
    BUFFER_OVERFLOW = "buffer_overflows"
    INTEGER_OVERFLOW = "integer_overflows"
    SQL_INJECTION = "sql_injections"
    XSS = "xss_vulnerabilities"
    RACE_CONDITION = "race_conditions"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    CSRF = "csrf_vulnerabilities"


class PerformanceIssue(str, Enum):
    """Performance issue patterns for synthetic injection"""
    NESTED_LOOPS = "nested_loops_instead_of_binary_search"
    STRING_CONCAT = "repeated_string_concatenation"
    LIST_COPIES = "unnecessary_list_copies"
    MISSING_CACHE = "missing_caching"
    INEFFICIENT_QUERY = "n_plus_one_queries"
    MEMORY_LEAK = "memory_leaks"


class ArchitecturalProblem(str, Enum):
    """Architectural problems for synthetic injection"""
    TIGHT_COUPLING = "tight_coupling"
    GOD_OBJECT = "god_objects"
    CIRCULAR_DEPS = "circular_dependencies"
    MISSING_ABSTRACTION = "missing_abstractions"
    LEAKY_ABSTRACTION = "leaky_abstractions"


@dataclass
class RealReviewsConfig:
    """Configuration for real code review data sources"""
    github_repos: List[str] = field(default_factory=lambda: [
        "tensorflow/tensorflow",
        "pytorch/pytorch",
        "huggingface/transformers",
        "kubernetes/kubernetes",
        "microsoft/vscode",
        "facebook/react",
        "angular/angular",
        "django/django",
        "pallets/flask",
        "apache/spark",
    ])
    
    review_sources: List[str] = field(default_factory=lambda: [
        "github_pr_reviews",
        "gerrit_logs",
        "reviewboard_archives",
    ])
    
    target_size: int = 500000  # 500k+ code review pairs


@dataclass
class SyntheticBugConfig:
    """Configuration for synthetic bug injection"""
    enabled: bool = True
    target_count: int = 100000  # 100k+ synthetic bugs
    
    patterns: List[BugPattern] = field(default_factory=lambda: list(BugPattern))
    
    injection_strategies: Dict[str, str] = field(default_factory=lambda: {
        "off_by_one": "Modify loop bounds by ±1",
        "null_pointer": "Remove null checks before access",
        "sql_injection": "Replace parameterized queries with string formatting",
        "xss": "Remove HTML escaping",
        "command_injection": "Use shell=True with user input",
    })


@dataclass
class SyntheticPerfConfig:
    """Configuration for synthetic performance issues"""
    enabled: bool = True
    target_count: int = 50000  # 50k+ synthetic issues
    
    issues: List[PerformanceIssue] = field(default_factory=lambda: list(PerformanceIssue))
    
    injection_strategies: Dict[str, str] = field(default_factory=lambda: {
        "nested_loops": "Replace O(log n) with O(n²) algorithms",
        "string_concat": "Use += in loops instead of join()",
        "list_copies": "Add unnecessary list() calls",
        "missing_cache": "Remove @cache/@lru_cache decorators",
    })


@dataclass
class SyntheticArchConfig:
    """Configuration for synthetic architectural problems"""
    enabled: bool = True
    target_count: int = 30000  # 30k+ synthetic patterns
    
    problems: List[ArchitecturalProblem] = field(default_factory=lambda: list(ArchitecturalProblem))


@dataclass
class DataProcessingConfig:
    """Data processing and cleaning configuration"""
    # Normalization
    remove_pii: bool = True
    normalize_paths: bool = True
    standardize_formatting: bool = True
    
    # Deduplication
    dedup_method: str = "semantic_embeddings"
    similarity_threshold: float = 0.95
    expected_dedup_ratio: float = 0.10  # ~10% duplicates
    
    # Quality filtering
    quality_criteria: List[str] = field(default_factory=lambda: [
        "Review has clear, actionable feedback",
        "Code and review are semantically related",
        "No offensive or irrelevant content",
        "Reproducible findings",
        "Minimum review length: 20 characters",
        "Maximum review length: 2000 characters",
    ])


@dataclass
class TokenizationConfig:
    """Tokenization settings"""
    use_native_tokenizer: bool = True
    max_tokens: int = 2048
    sliding_window_overlap: int = 256
    
    # Truncation strategy
    truncation_strategy: str = "longest_first"
    padding_strategy: str = "max_length"


@dataclass
class LossFunctionConfig:
    """Loss function configuration"""
    primary: str = "cross_entropy"
    
    auxiliary_losses: Dict[str, float] = field(default_factory=lambda: {
        "contrastive": 0.1,  # Separate correct vs incorrect reviews
        "focal": 0.05,       # Weight hard negative examples
        "ranking": 0.05,     # Prefer better reviews over mediocre
    })
    
    # Focal loss parameters
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # Contrastive loss parameters
    contrastive_margin: float = 0.5
    contrastive_temperature: float = 0.07


@dataclass
class RegularizationConfig:
    """Regularization settings"""
    dropout: float = 0.1
    layer_dropout: float = 0.1
    mixup_alpha: float = 0.1
    label_smoothing: float = 0.1
    
    # Weight decay
    weight_decay: float = 0.01
    
    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class DataPipelineConfig:
    """Complete data pipeline configuration"""
    real_reviews: RealReviewsConfig = field(default_factory=RealReviewsConfig)
    synthetic_bugs: SyntheticBugConfig = field(default_factory=SyntheticBugConfig)
    synthetic_perf: SyntheticPerfConfig = field(default_factory=SyntheticPerfConfig)
    synthetic_arch: SyntheticArchConfig = field(default_factory=SyntheticArchConfig)
    processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)


@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Basic training parameters
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    num_epochs: int = 2
    max_steps: int = -1
    
    # Learning rate
    learning_rate: float = 1.5e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 1000
    
    # Early stopping (disabled for exploration)
    early_stopping: bool = False
    early_stopping_patience: int = 5
    
    # Sub-configurations
    data_pipeline: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    loss: LossFunctionConfig = field(default_factory=LossFunctionConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    
    # Distributed training
    dataloader_num_workers: int = 4
    ddp_find_unused_parameters: bool = False
    
    # Reproducibility
    seed: int = 42
    
    # Output
    output_dir: str = "./outputs/v1-cr-ai"
    run_name: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard", "wandb"])
    
    def get_effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# Pre-configured training profiles
FAST_TRAINING_CONFIG = TrainingConfig(
    batch_size=128,
    num_epochs=1,
    eval_steps=200,
    learning_rate=3e-4,
)

FULL_TRAINING_CONFIG = TrainingConfig(
    batch_size=256,
    num_epochs=2,
    eval_steps=500,
)

ABLATION_CONFIG = TrainingConfig(
    batch_size=64,
    num_epochs=1,
    early_stopping=True,
    early_stopping_patience=3,
)
