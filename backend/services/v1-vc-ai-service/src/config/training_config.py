"""
Training Configuration for V1 VC-AI

Comprehensive training and fine-tuning configuration including:
- Data pipeline settings
- Optimization parameters
- Curriculum learning stages
- Multi-task learning weights
- Contrastive learning configuration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum


class LRSchedulerType(str, Enum):
    """Learning rate scheduler types"""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_WARMUP = "cosine_with_warmup"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    ONE_CYCLE = "one_cycle"


class DataSourceType(str, Enum):
    """Training data source types"""
    GIT_REPOSITORY = "git_repository"
    SYNTHETIC = "synthetic"
    AUGMENTED = "augmented"
    ADVERSARIAL = "adversarial"


@dataclass
class DataSourceConfig:
    """Configuration for a single data source"""
    source_type: DataSourceType
    path_or_url: str
    weight: float = 1.0
    max_samples: Optional[int] = None
    
    # Preprocessing options
    deduplicate: bool = True
    filter_noise: bool = True
    validate_alignment: bool = True


@dataclass
class DataConfig:
    """
    Data preparation pipeline configuration.
    
    Sources include major open-source repositories for diverse
    commit patterns and coding styles.
    """
    # Primary data sources
    primary_sources: List[DataSourceConfig] = field(default_factory=lambda: [
        DataSourceConfig(
            source_type=DataSourceType.GIT_REPOSITORY,
            path_or_url="tensorflow/tensorflow",
            weight=1.0,
        ),
        DataSourceConfig(
            source_type=DataSourceType.GIT_REPOSITORY,
            path_or_url="pytorch/pytorch",
            weight=1.0,
        ),
        DataSourceConfig(
            source_type=DataSourceType.GIT_REPOSITORY,
            path_or_url="huggingface/transformers",
            weight=1.2,  # Prioritize ML-focused repos
        ),
        DataSourceConfig(
            source_type=DataSourceType.GIT_REPOSITORY,
            path_or_url="kubernetes/kubernetes",
            weight=0.8,
        ),
        DataSourceConfig(
            source_type=DataSourceType.GIT_REPOSITORY,
            path_or_url="microsoft/vscode",
            weight=0.9,
        ),
        DataSourceConfig(
            source_type=DataSourceType.GIT_REPOSITORY,
            path_or_url="facebook/react",
            weight=0.8,
        ),
    ])
    
    # Data split ratios
    train_split: float = 0.80
    validation_split: float = 0.10
    test_split: float = 0.10
    
    # Augmentation settings
    augmentation: dict = field(default_factory=lambda: {
        "synthetic_commits": {
            "enabled": True,
            "ratio": 0.1,  # 10% synthetic data
            "change_types": ["bug_fix", "feature", "refactor", "optimization", "docs"],
        },
        "semantic_variation": {
            "enabled": True,
            "ratio": 0.05,  # 5% rephrased messages
            "techniques": ["paraphrase", "simplify", "expand"],
        },
        "adversarial_samples": {
            "enabled": True,
            "ratio": 0.02,  # 2% edge cases
            "types": ["empty_diff", "giant_commit", "binary_files", "merge_commits"],
        },
    })
    
    # Quality gates
    quality_gates: dict = field(default_factory=lambda: {
        "deduplication": {
            "enabled": True,
            "method": "minhash_lsh",
            "similarity_threshold": 0.85,
        },
        "noise_filtering": {
            "enabled": True,
            "min_diff_lines": 1,
            "max_diff_lines": 10000,
            "exclude_patterns": [
                r"^Merge branch",
                r"^Revert",
                r"auto-generated",
            ],
        },
        "semantic_validation": {
            "enabled": True,
            "min_message_length": 10,
            "require_verb": True,
            "check_diff_message_alignment": True,
        },
    })
    
    # Preprocessing
    max_sequence_length: int = 4096
    max_diff_lines: int = 500
    include_file_context: bool = True
    context_lines: int = 3
    
    # Caching
    cache_dir: str = ".cache/training_data"
    use_memory_mapping: bool = True


@dataclass
class OptimizationConfig:
    """Optimizer and learning rate configuration"""
    optimizer: str = "adamw"
    learning_rate: float = 2e-4          # Higher for faster convergence
    weight_decay: float = 0.01
    
    # AdamW specific
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Learning rate scheduling
    lr_scheduler: LRSchedulerType = LRSchedulerType.COSINE_WITH_WARMUP
    warmup_steps: int = 500
    warmup_ratio: float = 0.0            # Alternative to warmup_steps
    num_cycles: float = 0.5              # For cosine scheduler
    
    # Gradient management
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8  # Effective batch = batch_size * 8


@dataclass
class CurriculumStage:
    """Single stage in curriculum learning"""
    name: str
    description: str
    max_diff_lines: int
    max_files_changed: int
    epochs: int
    learning_rate_multiplier: float = 1.0


@dataclass
class CurriculumConfig:
    """
    Curriculum learning configuration.
    
    Progressive complexity: simple → medium → complex → cross-module
    """
    enabled: bool = True
    
    stages: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage(
            name="stage_1_simple",
            description="Simple commits (10 lines, 1 file)",
            max_diff_lines=10,
            max_files_changed=1,
            epochs=1,
            learning_rate_multiplier=1.0,
        ),
        CurriculumStage(
            name="stage_2_medium",
            description="Medium commits (50 lines, 3 files)",
            max_diff_lines=50,
            max_files_changed=3,
            epochs=1,
            learning_rate_multiplier=0.8,
        ),
        CurriculumStage(
            name="stage_3_complex",
            description="Complex commits (500 lines, 10+ files)",
            max_diff_lines=500,
            max_files_changed=15,
            epochs=1,
            learning_rate_multiplier=0.5,
        ),
        CurriculumStage(
            name="stage_4_cross_module",
            description="Cross-module refactorings (1000+ lines)",
            max_diff_lines=2000,
            max_files_changed=50,
            epochs=1,
            learning_rate_multiplier=0.3,
        ),
    ])
    
    # Difficulty scoring
    difficulty_weights: dict = field(default_factory=lambda: {
        "diff_lines": 0.3,
        "files_changed": 0.2,
        "cyclomatic_complexity": 0.2,
        "cross_module_changes": 0.3,
    })


@dataclass
class TaskConfig:
    """Configuration for a single training task"""
    name: str
    weight: float
    loss_type: str = "cross_entropy"
    output_dim: Optional[int] = None


@dataclass
class MultiTaskConfig:
    """
    Multi-task learning configuration.
    
    Primary tasks with weighted loss contribution.
    """
    enabled: bool = True
    
    tasks: List[TaskConfig] = field(default_factory=lambda: [
        TaskConfig(
            name="commit_message_generation",
            weight=0.40,
            loss_type="cross_entropy",
        ),
        TaskConfig(
            name="change_type_classification",
            weight=0.30,
            loss_type="cross_entropy",
            output_dim=5,  # bug_fix, feature, refactor, optimization, docs
        ),
        TaskConfig(
            name="impact_prediction",
            weight=0.20,
            loss_type="cross_entropy",
            output_dim=4,  # low, medium, high, critical
        ),
        TaskConfig(
            name="semantic_similarity",
            weight=0.10,
            loss_type="cosine_embedding",
        ),
    ])
    
    # Task balancing
    use_uncertainty_weighting: bool = True  # Learn task weights
    gradient_blending: bool = True
    task_sampling_strategy: str = "proportional"  # proportional, round_robin, annealing


@dataclass
class ContrastiveLearningConfig:
    """
    Contrastive learning configuration.
    
    Uses InfoNCE loss for learning semantic embeddings.
    """
    enabled: bool = True
    
    # Pair construction
    positive_pair_strategy: str = "semantic_similarity"  # Same change type
    negative_pair_strategy: str = "different_change_type"
    hard_negative_mining: bool = True
    
    # Loss configuration
    temperature: float = 0.07
    loss_function: str = "InfoNCE"
    
    # Training settings
    embedding_dim: int = 768
    projection_dim: int = 256
    num_negatives: int = 32
    
    # Memory bank for efficient negative sampling
    use_memory_bank: bool = True
    memory_bank_size: int = 65536
    momentum: float = 0.999


@dataclass
class TrainingConfig:
    """
    Complete training configuration for V1 VC-AI.
    
    Aggressive batch sizes and exploration-focused settings.
    """
    # Basic training parameters
    batch_size: int = 256                # Aggressive batch size
    num_epochs: int = 3
    max_steps: int = -1                  # -1 means use epochs
    
    # Evaluation and checkpointing
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Early stopping (disabled for exploration)
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    
    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = True                    # Mixed precision
    bf16: bool = False                   # Alternative to fp16
    
    # Distributed training
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    ddp_find_unused_parameters: bool = False
    
    # Reproducibility
    seed: int = 42
    data_seed: Optional[int] = None
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    multi_task: MultiTaskConfig = field(default_factory=MultiTaskConfig)
    contrastive: ContrastiveLearningConfig = field(default_factory=ContrastiveLearningConfig)
    
    # Output settings
    output_dir: str = "./outputs"
    run_name: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard", "wandb"])
    
    # Experiment tracking
    experiment_name: str = "v1-vc-ai"
    experiment_tags: List[str] = field(default_factory=lambda: ["v1", "experimental"])
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation"""
        return self.batch_size * self.optimization.gradient_accumulation_steps
    
    def to_hf_training_args(self) -> dict:
        """Convert to HuggingFace TrainingArguments compatible dict"""
        return {
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "num_train_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.optimization.gradient_accumulation_steps,
            "learning_rate": self.optimization.learning_rate,
            "weight_decay": self.optimization.weight_decay,
            "adam_beta1": self.optimization.adam_beta1,
            "adam_beta2": self.optimization.adam_beta2,
            "adam_epsilon": self.optimization.adam_epsilon,
            "max_grad_norm": self.optimization.max_grad_norm,
            "lr_scheduler_type": self.optimization.lr_scheduler.value,
            "warmup_steps": self.optimization.warmup_steps,
            "warmup_ratio": self.optimization.warmup_ratio,
            "evaluation_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "seed": self.seed,
            "data_seed": self.data_seed or self.seed,
            "report_to": self.report_to,
            "load_best_model_at_end": not self.early_stopping,
            "ddp_find_unused_parameters": self.ddp_find_unused_parameters,
        }


# Pre-configured training profiles
FAST_ITERATION_CONFIG = TrainingConfig(
    batch_size=128,
    num_epochs=1,
    eval_steps=50,
    save_steps=100,
    optimization=OptimizationConfig(
        learning_rate=5e-4,
        warmup_steps=100,
    ),
    curriculum=CurriculumConfig(enabled=False),
)

FULL_TRAINING_CONFIG = TrainingConfig(
    batch_size=256,
    num_epochs=3,
    eval_steps=100,
    save_steps=200,
    curriculum=CurriculumConfig(enabled=True),
    multi_task=MultiTaskConfig(enabled=True),
    contrastive=ContrastiveLearningConfig(enabled=True),
)

ABLATION_STUDY_CONFIG = TrainingConfig(
    batch_size=64,
    num_epochs=1,
    early_stopping=True,
    early_stopping_patience=3,
    optimization=OptimizationConfig(
        learning_rate=1e-4,
    ),
)
