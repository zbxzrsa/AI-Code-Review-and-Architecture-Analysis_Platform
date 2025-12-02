"""
Evaluation Configuration for V1 VC-AI

Comprehensive evaluation framework with performance, innovation,
and efficiency metrics for experiment assessment and promotion decisions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum


class MetricType(str, Enum):
    """Types of evaluation metrics"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    INNOVATION = "innovation"
    EFFICIENCY = "efficiency"


class PromotionDecision(str, Enum):
    """Possible promotion outcomes"""
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class FailureSeverity(str, Enum):
    """Failure severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a single evaluation metric"""
    name: str
    description: str
    measurement: str
    threshold: str
    weight: float
    higher_is_better: bool = True


@dataclass
class PerformanceMetrics:
    """
    Core performance metrics configuration.
    """
    accuracy: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="accuracy",
        description="Semantic correctness of version tracking",
        measurement="f1_score on gold-standard commits",
        threshold=">= 0.92",
        weight=0.25,
        higher_is_better=True,
    ))
    
    latency: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="latency",
        description="End-to-end processing time per commit",
        measurement="p99_latency_ms",
        threshold="<= 500ms",
        weight=0.15,
        higher_is_better=False,
    ))
    
    throughput: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="throughput",
        description="Commits processed per second",
        measurement="requests_per_second",
        threshold=">= 100 RPS",
        weight=0.10,
        higher_is_better=True,
    ))
    
    precision: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="precision",
        description="Precision of change type classification",
        measurement="precision_score",
        threshold=">= 0.90",
        weight=0.10,
        higher_is_better=True,
    ))
    
    recall: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="recall",
        description="Recall of change type classification",
        measurement="recall_score",
        threshold=">= 0.88",
        weight=0.10,
        higher_is_better=True,
    ))


@dataclass
class InnovationMetrics:
    """
    Innovation-focused metrics for V1 experimentation.
    """
    new_technique_impact: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="new_technique_impact",
        description="Performance improvement from novel approaches",
        measurement="accuracy_delta_vs_v2_baseline",
        threshold=">= 0.05 (5% improvement)",
        weight=0.30,
        higher_is_better=True,
    ))
    
    experimentation_coverage: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="experimentation_coverage",
        description="Breadth of techniques tested",
        measurement="unique_architectures_tried",
        threshold=">= 5 per month",
        weight=0.12,
        higher_is_better=True,
    ))
    
    risk_tolerance: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="risk_tolerance",
        description="Balance between innovation and stability",
        measurement="failure_rate vs improvement_magnitude",
        threshold="failure_rate < improvement_rate * 2",
        weight=0.08,
        higher_is_better=True,
    ))


@dataclass
class EfficiencyMetrics:
    """
    Resource efficiency metrics.
    """
    cost_per_request: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="cost_per_request",
        description="API + compute + storage cost",
        measurement="dollars_per_1000_requests",
        threshold="<= $0.10",
        weight=0.12,
        higher_is_better=False,
    ))
    
    model_size: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="model_size",
        description="Disk/memory footprint",
        measurement="total_size_gb",
        threshold="<= 15GB (quantized)",
        weight=0.08,
        higher_is_better=False,
    ))
    
    memory_utilization: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="memory_utilization",
        description="GPU memory usage during inference",
        measurement="peak_memory_gb",
        threshold="<= 24GB",
        weight=0.05,
        higher_is_better=False,
    ))


@dataclass
class MetricThresholds:
    """
    Threshold values for promotion decisions.
    """
    # Must-pass thresholds
    min_accuracy: float = 0.92
    max_latency_p99_ms: int = 500
    max_error_rate: float = 0.02
    
    # Improvement thresholds (vs V2 baseline)
    min_accuracy_improvement: float = 0.05      # 5% improvement
    min_cost_reduction: float = 0.20            # 20% cost reduction
    min_latency_improvement: float = 0.25       # 25% faster
    
    # Failure thresholds
    max_low_priority_failures: int = 2          # Per 1000 experiments
    max_high_priority_failures: int = 0         # Absolute block
    
    # Innovation thresholds
    min_techniques_per_month: int = 5
    max_failure_to_improvement_ratio: float = 2.0


@dataclass
class PromotionCriteria:
    """
    Criteria for V1 -> V2 promotion decisions.
    """
    # Must pass all of these
    must_pass: List[str] = field(default_factory=lambda: [
        "accuracy >= 0.92",
        "latency_p99 <= 500ms",
        "no_regressions_on_test_suite",
        "security_audit_passed",
        "no_new_critical_failures",
    ])
    
    # Must meet improvement threshold OR one of alternatives
    improvement_threshold: dict = field(default_factory=lambda: {
        "primary": "accuracy_improvement >= 5% over V2",
        "alternatives": [
            "cost_reduction >= 20%",
            "latency_improvement >= 25%",
            "new_capability_enabled",
        ],
    })
    
    # Failure tolerance
    failure_tolerance: dict = field(default_factory=lambda: {
        "low_priority": "<= 2 per 1000 experiments",
        "high_priority": "0 (absolute block)",
        "recovery_capability": "must be demonstrable",
    })


@dataclass
class PromotionConfig:
    """
    Complete promotion decision configuration.
    """
    criteria: PromotionCriteria = field(default_factory=PromotionCriteria)
    thresholds: MetricThresholds = field(default_factory=MetricThresholds)
    
    # Decision outcomes
    decision_actions: dict = field(default_factory=lambda: {
        PromotionDecision.APPROVED: "move_to_v2_validation_queue",
        PromotionDecision.CONDITIONAL: "requires_human_review_or_minor_fixes",
        PromotionDecision.REJECTED: "send_to_v3_quarantine_and_analyze_failure",
        PromotionDecision.BLOCKED: "prevent_from_retrying_same_technique",
    })
    
    # Review settings
    require_human_review: bool = True
    min_reviewers: int = 2
    auto_approve_minor_improvements: bool = False
    
    # Timing
    validation_period_hours: int = 24
    rollback_window_hours: int = 2


@dataclass
class EvaluationConfig:
    """
    Complete evaluation configuration for V1 VC-AI.
    """
    # Metric categories
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    innovation: InnovationMetrics = field(default_factory=InnovationMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    
    # Promotion settings
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    
    # Evaluation settings
    evaluation_dataset: str = "test"     # train, validation, test
    num_eval_samples: int = 1000
    eval_batch_size: int = 32
    
    # Baseline comparison
    v2_baseline_path: Optional[str] = None
    baseline_metrics_cache: Optional[str] = None
    
    # Reporting
    report_format: str = "json"          # json, html, markdown
    include_visualizations: bool = True
    save_predictions: bool = True
    
    # A/B testing
    enable_ab_testing: bool = True
    ab_test_traffic_percentage: float = 0.1
    ab_test_min_samples: int = 1000
    statistical_significance_level: float = 0.05
    
    def get_all_metrics(self) -> Dict[str, MetricDefinition]:
        """Get all metric definitions as a dictionary"""
        metrics = {}
        for category in [self.performance, self.innovation, self.efficiency]:
            for field_name in category.__dataclass_fields__:
                metric = getattr(category, field_name)
                if isinstance(metric, MetricDefinition):
                    metrics[metric.name] = metric
        return metrics
    
    def get_metric_weights(self) -> Dict[str, float]:
        """Get normalized metric weights"""
        metrics = self.get_all_metrics()
        total_weight = sum(m.weight for m in metrics.values())
        return {name: m.weight / total_weight for name, m in metrics.items()}


# Pre-configured evaluation profiles
STRICT_EVALUATION_CONFIG = EvaluationConfig(
    promotion=PromotionConfig(
        thresholds=MetricThresholds(
            min_accuracy=0.95,
            max_latency_p99_ms=300,
            min_accuracy_improvement=0.10,
        ),
    ),
    num_eval_samples=5000,
    ab_test_min_samples=5000,
)

EXPERIMENTAL_EVALUATION_CONFIG = EvaluationConfig(
    promotion=PromotionConfig(
        thresholds=MetricThresholds(
            min_accuracy=0.85,
            max_latency_p99_ms=1000,
            min_accuracy_improvement=0.02,
        ),
        require_human_review=False,
        auto_approve_minor_improvements=True,
    ),
    num_eval_samples=500,
    ab_test_min_samples=500,
)

ABLATION_EVALUATION_CONFIG = EvaluationConfig(
    num_eval_samples=200,
    enable_ab_testing=False,
    include_visualizations=True,
    save_predictions=True,
)
