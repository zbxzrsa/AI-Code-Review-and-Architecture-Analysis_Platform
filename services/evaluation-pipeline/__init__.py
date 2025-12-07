"""
评估管道服务 (Evaluation Pipeline Service)

模块功能描述:
    为三版本自演化循环提供客观评估。

主要功能:
    - 影子流量比较（V1 vs V2）
    - 金标集评估（用于升级和恢复）
    - 统计显著性测试

主要组件:
    - ShadowComparator: 影子流量比较器
    - GoldSetEvaluator: 金标集评估器
    - PromotionRecommendation: 升级建议

使用示例:
    from services.evaluation_pipeline import (
        ShadowComparator,
        GoldSetEvaluator,
    )
    
    # V1 → V2 影子比较
    comparator = ShadowComparator()
    await comparator.start()
    
    # 记录影子流量输出
    comparator.record_v1_output(v1_output)
    comparator.record_v2_output(v2_output)
    
    # 获取升级建议
    recommendation = comparator.evaluate_promotion("v1-exp-001")
    if recommendation.recommend_promotion:
        # 开始灰度发布
        pass
    
    # 恢复的金标集评估
    evaluator = GoldSetEvaluator()
    await evaluator.start()
    
    report = await evaluator.evaluate(
        version_id="v3-quarantine-001",
        evaluation_type="recovery"
    )
    
    if report.passed:
        # 提升回 V1
        pass

最后修改日期: 2024-12-07
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
