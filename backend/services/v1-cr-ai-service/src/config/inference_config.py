"""
Inference Configuration for V1 Code Review AI

Multi-strategy review pipeline with:
- Baseline direct review
- Chain-of-thought reasoning
- Few-shot in-context learning
- Contrastive analysis
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class ReviewStrategy(str, Enum):
    """Available review strategies"""
    BASELINE = "baseline"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    CONTRASTIVE = "contrastive"
    ENSEMBLE = "ensemble"


class ExampleSelectionMethod(str, Enum):
    """Methods for selecting few-shot examples"""
    RANDOM = "random"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TASK_SPECIFIC = "task_specific"
    DIVERSE = "diverse"


@dataclass
class CodeParsingConfig:
    """Configuration for code preprocessing"""
    extract_ast: bool = True
    build_dependency_graph: bool = True
    extract_signatures: bool = True
    resolve_imports: bool = True
    
    # Supported languages
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "typescript", "javascript", "java",
        "go", "rust", "cpp", "c", "csharp",
    ])


@dataclass
class ContextEnrichmentConfig:
    """Configuration for context enrichment"""
    include_surrounding_code: bool = True
    context_lines: int = 500  # ±500 lines
    include_called_functions: bool = True
    include_type_hints: bool = True
    include_documentation: bool = True
    include_test_files: bool = True


@dataclass
class BaselineStrategyConfig:
    """Configuration for baseline direct review"""
    prompt_template: str = """Review the following code for {dimension} issues.

<CODE_BLOCK>
{code}
</CODE_BLOCK>

Provide your review in JSON format with findings."""
    
    output_format: str = "structured_json"
    max_tokens: int = 1024


@dataclass
class ChainOfThoughtConfig:
    """Configuration for chain-of-thought reasoning"""
    enabled: bool = True
    
    prompt_template: str = """Think step-by-step to review this code:

<CODE_BLOCK>
{code}
</CODE_BLOCK>

<REASONING>
<STEP>1. First, understand the code's purpose and logic flow</STEP>
<STEP>2. Identify the function signatures and contracts</STEP>
<STEP>3. Check for {dimension} issues</STEP>
<STEP>4. Analyze edge cases and boundary conditions</STEP>
<STEP>5. Formulate actionable suggestions</STEP>
</REASONING>

Now provide your detailed findings:"""
    
    reasoning_steps: List[str] = field(default_factory=lambda: [
        "Understand the code's purpose",
        "Identify function signatures and contracts",
        "Check for issues in the target dimension",
        "Analyze edge cases",
        "Suggest improvements",
    ])
    
    output_format: str = "reasoning_steps_plus_findings"
    max_tokens: int = 2048


@dataclass
class FewShotConfig:
    """Configuration for few-shot in-context learning"""
    enabled: bool = True
    num_examples: int = 3
    selection_method: ExampleSelectionMethod = ExampleSelectionMethod.SEMANTIC_SIMILARITY
    
    prompt_template: str = """Here are examples of good code reviews:

{examples}

Now review this code following the same pattern:

<CODE_BLOCK>
{code}
</CODE_BLOCK>

Provide your review:"""
    
    example_template: str = """Example {n}:
Code:
```
{example_code}
```
Review:
{example_review}
---"""
    
    # Example selection parameters
    similarity_threshold: float = 0.7
    diversity_factor: float = 0.3  # Balance similarity vs diversity
    
    max_tokens: int = 2048


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive analysis"""
    enabled: bool = True
    
    prompt_template: str = """Compare these two versions of the code:

Original (correct):
<CODE_BLOCK>
{code_correct}
</CODE_BLOCK>

Modified (potentially buggy):
<CODE_BLOCK>
{code_buggy}
</CODE_BLOCK>

Analyze:
1. What differences do you see?
2. What issues were introduced?
3. What is the root cause?
4. How should it be fixed?"""
    
    output_format: str = "diff_analysis_plus_root_cause"
    max_tokens: int = 1536


@dataclass
class EnsembleConfig:
    """Configuration for ensemble review"""
    enabled: bool = True
    
    # Strategies to ensemble
    strategies: List[ReviewStrategy] = field(default_factory=lambda: [
        ReviewStrategy.BASELINE,
        ReviewStrategy.CHAIN_OF_THOUGHT,
        ReviewStrategy.FEW_SHOT,
    ])
    
    # Voting mechanism
    voting_method: str = "weighted_average"  # or "majority", "unanimous"
    
    # Strategy weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        ReviewStrategy.BASELINE.value: 0.3,
        ReviewStrategy.CHAIN_OF_THOUGHT.value: 0.4,
        ReviewStrategy.FEW_SHOT.value: 0.3,
    })
    
    # Minimum agreement threshold
    min_agreement: float = 0.5


@dataclass
class BatchingConfig:
    """Configuration for inference batching"""
    dynamic_batching: bool = True
    batch_size: int = 64
    max_wait_time_ms: int = 100
    max_batch_tokens: int = 32768


@dataclass
class CachingConfig:
    """Configuration for review caching"""
    enabled: bool = True
    cache_ttl_minutes: int = 1440  # 24 hours
    cache_hit_rate_target: float = 0.30  # >= 30%
    
    # Cache key strategy
    cache_key_includes: List[str] = field(default_factory=lambda: [
        "code_hash",
        "dimension",
        "strategy",
        "model_version",
    ])


@dataclass
class EarlyExitConfig:
    """Configuration for early exit conditions"""
    enabled: bool = True
    
    # Exit on critical security issue
    exit_on_critical_security: bool = True
    
    # Exit on high confidence
    high_confidence_threshold: float = 0.95
    min_findings_for_exit: int = 3


@dataclass
class StrategyConfig:
    """Configuration for all review strategies"""
    baseline: BaselineStrategyConfig = field(default_factory=BaselineStrategyConfig)
    chain_of_thought: ChainOfThoughtConfig = field(default_factory=ChainOfThoughtConfig)
    few_shot: FewShotConfig = field(default_factory=FewShotConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)


@dataclass
class GenerationConfig:
    """Generation parameters"""
    temperature: float = 0.3  # Lower for more consistent reviews
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repetition_penalty: float = 1.1
    
    # For critical reviews, use even lower temperature
    critical_temperature: float = 0.1


@dataclass
class InferenceConfig:
    """Complete inference configuration"""
    # Preprocessing
    code_parsing: CodeParsingConfig = field(default_factory=CodeParsingConfig)
    context_enrichment: ContextEnrichmentConfig = field(default_factory=ContextEnrichmentConfig)
    
    # Strategies
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    default_strategy: ReviewStrategy = ReviewStrategy.CHAIN_OF_THOUGHT
    
    # Optimization
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    early_exit: EarlyExitConfig = field(default_factory=EarlyExitConfig)
    
    # Generation
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Model settings
    model_path: Optional[str] = None
    device_map: str = "auto"
    
    # Service settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


# Pre-configured inference profiles
FAST_INFERENCE_CONFIG = InferenceConfig(
    default_strategy=ReviewStrategy.BASELINE,
    generation=GenerationConfig(
        temperature=0.2,
        max_tokens=1024,
    ),
)

THOROUGH_INFERENCE_CONFIG = InferenceConfig(
    default_strategy=ReviewStrategy.ENSEMBLE,
    generation=GenerationConfig(
        temperature=0.3,
        max_tokens=2048,
    ),
)

SECURITY_FOCUSED_CONFIG = InferenceConfig(
    default_strategy=ReviewStrategy.CHAIN_OF_THOUGHT,
    generation=GenerationConfig(
        temperature=0.1,  # Very low for consistency
        max_tokens=2048,
    ),
)
