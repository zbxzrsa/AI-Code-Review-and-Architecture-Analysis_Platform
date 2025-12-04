"""
Evaluation Pipeline Service

Provides objective evaluation for the three-version self-evolution cycle:
- Shadow traffic comparison (V1 vs V2)
- Gold-set evaluation (for promotion and recovery)
- Statistical significance testing

Usage:
    from services.evaluation_pipeline import (
        ShadowComparator,
        GoldSetEvaluator,
    )
    
    # Shadow comparison for V1 → V2
    comparator = ShadowComparator()
    await comparator.start()
    
    # Record outputs from shadow traffic
    comparator.record_v1_output(v1_output)
    comparator.record_v2_output(v2_output)
    
    # Get promotion recommendation
    recommendation = comparator.evaluate_promotion("v1-exp-001")
    if recommendation.recommend_promotion:
        # Start gray-scale rollout
        pass
    
    # Gold-set evaluation for recovery
    evaluator = GoldSetEvaluator()
    await evaluator.start()
    
    report = await evaluator.evaluate(
        version_id="v3-quarantine-001",
        evaluation_type="recovery"
    )
    
    if report.passed:
        # Promote back to V1
        pass
"""

from .shadow_comparator import (
    ShadowComparator,
    AnalysisOutput,
    ComparisonPair,
    ComparisonMetrics,
    PromotionRecommendation,
)

from .gold_set_evaluator import (
    GoldSetEvaluator,
    GoldSetTestCase,
    TestResult,
    EvaluationReport,
    TestCategory,
)

__all__ = [
    # Shadow comparison
    "ShadowComparator",
    "AnalysisOutput",
    "ComparisonPair",
    "ComparisonMetrics",
    "PromotionRecommendation",
    
    # Gold-set evaluation
    "GoldSetEvaluator",
    "GoldSetTestCase",
    "TestResult",
    "EvaluationReport",
    "TestCategory",
]

__version__ = "1.0.0"
