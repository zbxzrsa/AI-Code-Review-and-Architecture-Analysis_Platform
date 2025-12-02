"""
Evaluation Configuration for V1 Code Review AI

Comprehensive metrics for accuracy, efficiency, quality, and reliability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class MetricCategory(str, Enum):
    """Categories of evaluation metrics"""
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    RELIABILITY = "reliability"
    INNOVATION = "innovation"


@dataclass
class MetricDefinition:
    """Definition of a single metric"""
    name: str
    definition: str
    measurement: str
    target: str
    weight: float = 1.0


@dataclass
class AccuracyMetrics:
    """Accuracy-focused metrics"""
    precision: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="precision",
        definition="Of findings reported, how many are correct?",
        measurement="# correct_findings / # total_findings",
        target=">= 95%",
        weight=0.25,
    ))
    
    recall: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="recall",
        definition="Of actual bugs, how many were found?",
        measurement="# found_bugs / # injected_bugs",
        target=">= 90%",
        weight=0.25,
    ))
    
    f1_score: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="f1_score",
        definition="Harmonic mean of precision and recall",
        measurement="2 * (precision * recall) / (precision + recall)",
        target=">= 0.92",
        weight=0.20,
    ))
    
    # Per-dimension accuracy targets
    per_dimension_targets: Dict[str, float] = field(default_factory=lambda: {
        "correctness": 0.93,
        "security": 0.95,
        "performance": 0.87,
        "maintainability": 0.85,
        "architecture": 0.83,
        "testing": 0.80,
    })


@dataclass
class EfficiencyMetrics:
    """Efficiency-focused metrics"""
    latency_p50: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="latency_p50",
        definition="Median review latency",
        measurement="wall-clock time per review (p50)",
        target="<= 300ms",
        weight=0.15,
    ))
    
    latency_p99: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="latency_p99",
        definition="99th percentile review latency",
        measurement="wall-clock time per review (p99)",
        target="<= 1000ms",
        weight=0.15,
    ))
    
    throughput: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="throughput",
        definition="Reviews per second",
        measurement="requests_per_second",
        target=">= 50 RPS",
        weight=0.10,
    ))
    
    concurrent_capacity: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="concurrent_capacity",
        definition="Max concurrent requests",
        measurement="max_concurrent_requests",
        target=">= 500",
        weight=0.05,
    ))


@dataclass
class QualityMetrics:
    """Quality-focused metrics"""
    actionability: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="actionability",
        definition="Are suggestions concrete and implementable?",
        measurement="% developers who can immediately implement",
        target=">= 90%",
        weight=0.15,
    ))
    
    clarity: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="clarity",
        definition="Are explanations clear and helpful?",
        measurement="expert_reviewer_clarity_score (1-5)",
        target=">= 4.2",
        weight=0.10,
    ))
    
    novelty: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="novelty",
        definition="Does review catch issues missed by humans?",
        measurement="% reviews with novel insights",
        target=">= 20%",
        weight=0.10,
    ))


@dataclass
class ReliabilityMetrics:
    """Reliability-focused metrics"""
    consistency: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="consistency",
        definition="Same code gets same review every time",
        measurement="review_comparison_similarity",
        target=">= 0.95",
        weight=0.15,
    ))
    
    false_positive_rate: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="false_positive_rate",
        definition="% reported issues that aren't real",
        measurement="1 - precision",
        target="<= 5%",
        weight=0.10,
    ))
    
    false_negative_rate: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="false_negative_rate",
        definition="% of real issues that were missed",
        measurement="1 - recall",
        target="<= 10%",
        weight=0.10,
    ))
    
    hallucination_rate: MetricDefinition = field(default_factory=lambda: MetricDefinition(
        name="hallucination_rate",
        definition="% reviews containing fabricated information",
        measurement="hallucination_detection_rate",
        target="<= 2%",
        weight=0.15,
    ))


@dataclass
class InnovationScore:
    """Innovation score calculation"""
    formula: str = "accuracy * 0.5 + efficiency_improvement * 0.2 + novelty_detection * 0.3"
    comparison_baseline: str = "V2 production model"
    target_improvement: float = 0.08  # >= V2 + 8%


@dataclass
class MetricThresholds:
    """Threshold values for metrics"""
    # Accuracy thresholds
    min_precision: float = 0.95
    min_recall: float = 0.90
    min_f1: float = 0.92
    
    # Efficiency thresholds
    max_latency_p50_ms: int = 300
    max_latency_p99_ms: int = 1000
    min_throughput_rps: int = 50
    
    # Quality thresholds
    min_actionability: float = 0.90
    min_clarity_score: float = 4.2
    min_novelty_rate: float = 0.20
    
    # Reliability thresholds
    min_consistency: float = 0.95
    max_false_positive_rate: float = 0.05
    max_false_negative_rate: float = 0.10
    max_hallucination_rate: float = 0.02
    
    # Innovation threshold
    min_innovation_improvement: float = 0.08


@dataclass
class HallucinationDetectionConfig:
    """Configuration for hallucination detection"""
    consistency_check_enabled: bool = True
    consistency_runs: int = 3
    consistency_stddev_threshold: float = 0.2
    
    fact_checking_enabled: bool = True
    fact_checks: List[str] = field(default_factory=lambda: [
        "line_existence",
        "error_message_validity",
        "fix_syntax_validity",
    ])
    
    confidence_threshold: float = 0.5
    min_avg_confidence: float = 0.75


@dataclass
class EvaluationConfig:
    """Complete evaluation configuration"""
    # Metric categories
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    reliability: ReliabilityMetrics = field(default_factory=ReliabilityMetrics)
    
    # Thresholds
    thresholds: MetricThresholds = field(default_factory=MetricThresholds)
    
    # Innovation
    innovation: InnovationScore = field(default_factory=InnovationScore)
    
    # Hallucination detection
    hallucination: HallucinationDetectionConfig = field(default_factory=HallucinationDetectionConfig)
    
    # Evaluation settings
    eval_dataset: str = "test"
    num_eval_samples: int = 5000
    eval_batch_size: int = 64
    
    # A/B testing
    enable_ab_testing: bool = True
    ab_test_traffic: float = 0.1
    statistical_significance: float = 0.05
    
    # Reporting
    report_format: str = "json"
    include_examples: bool = True
    save_predictions: bool = True


# Pre-configured evaluation profiles
STRICT_EVAL_CONFIG = EvaluationConfig(
    thresholds=MetricThresholds(
        min_precision=0.97,
        min_recall=0.93,
        max_hallucination_rate=0.01,
    ),
    num_eval_samples=10000,
)

QUICK_EVAL_CONFIG = EvaluationConfig(
    num_eval_samples=1000,
    enable_ab_testing=False,
)

SECURITY_EVAL_CONFIG = EvaluationConfig(
    thresholds=MetricThresholds(
        min_precision=0.99,  # Very high for security
        min_recall=0.95,
        max_false_negative_rate=0.05,
    ),
)
