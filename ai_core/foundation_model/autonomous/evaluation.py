"""
自评估模块 (Self-Evaluation Module)

模块功能描述:
    提供自动基准测试和知识差距检测功能。

主要功能:
    - 定期基准测试评估
    - 知识差距识别
    - 性能跟踪
    - 学习触发建议

主要组件:
    - SelfEvaluationSystem: 自评估系统主类
    - EvaluationConfig: 评估配置

最后修改日期: 2024-12-07
"""

import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import BenchmarkResult, KnowledgeGap

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """
    自评估配置数据类
    
    功能描述:
        配置自评估系统的各项参数。
    
    配置项:
        - eval_interval_steps: 评估间隔步数
        - min_eval_interval_hours: 最小评估间隔（小时）
        - benchmark_suite: 基准测试套件
        - min_benchmark_score: 最低基准分数
        - knowledge_gap_threshold: 知识差距阈值
    """
    # Evaluation frequency
    eval_interval_steps: int = 1000
    min_eval_interval_hours: float = 1.0
    
    # Benchmarks
    benchmark_suite: List[str] = field(default_factory=lambda: [
        "code_review_accuracy",
        "bug_detection_precision",
        "security_scan_recall",
        "suggestion_quality",
    ])
    
    # Thresholds
    min_benchmark_score: float = 0.8
    knowledge_gap_threshold: float = 0.1
    significant_degradation: float = 0.05
    
    # Learning triggers
    auto_trigger_learning: bool = True
    min_gap_severity_to_trigger: float = 0.3


class SelfEvaluationSystem:
    """
    自主学习自评估系统
    
    功能描述:
        提供自动化的模型评估和知识差距检测。
    
    主要特性:
        - 定期基准测试执行
        - 知识差距检测
        - 性能趋势跟踪
        - 学习建议
    
    使用示例:
        eval_system = SelfEvaluationSystem(config)
        
        # 注册基准测试
        eval_system.register_benchmark("code_review", my_benchmark_fn)
        
        # 运行评估
        results = await eval_system.run_evaluation()
        
        # 检测差距
        gaps = eval_system.detect_knowledge_gaps()
        
        # 检查是否应触发学习
        should_learn, gap = eval_system.should_trigger_learning()
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        初始化自评估系统
        
        参数:
            config: 评估配置对象
        """
        self.config = config or EvaluationConfig()
        
        # Registered benchmarks
        self.benchmarks: Dict[str, Callable] = {}
        
        # Results history
        self.results_history: List[BenchmarkResult] = []
        self.gap_history: List[KnowledgeGap] = []
        
        # Current state
        self._last_evaluation = datetime.now(timezone.utc) - timedelta(hours=2)
        self._current_scores: Dict[str, float] = {}
        self._baseline_scores: Dict[str, float] = {}
        
        # Statistics
        self.total_evaluations = 0
        self.gaps_detected = 0
        self.gaps_resolved = 0
    
    def register_benchmark(
        self,
        name: str,
        benchmark_fn: Callable[[], float],
    ):
        """
        Register a benchmark function.
        
        Args:
            name: Benchmark name
            benchmark_fn: Function that returns a score (0-1)
        """
        self.benchmarks[name] = benchmark_fn
        logger.info(f"Registered benchmark: {name}")
    
    async def run_evaluation(self) -> List[BenchmarkResult]:
        """
        Run all registered benchmarks.
        
        Returns:
            List of benchmark results
        """
        logger.info("Running self-evaluation")
        results = []
        
        for name, benchmark_fn in self.benchmarks.items():
            try:
                # Run benchmark
                score = await self._run_benchmark(name, benchmark_fn)
                
                result = BenchmarkResult(
                    benchmark_name=name,
                    score=score,
                    timestamp=datetime.now(timezone.utc),
                    details={"previous_score": self._current_scores.get(name)},
                )
                
                results.append(result)
                self.results_history.append(result)
                
                # Update current scores
                self._current_scores[name] = score
                
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")
        
        self._last_evaluation = datetime.now(timezone.utc)
        self.total_evaluations += 1
        
        logger.info(f"Evaluation complete: {len(results)} benchmarks run")
        return results
    
    async def _run_benchmark(
        self,
        name: str,
        benchmark_fn: Callable,
    ) -> float:
        """Run a single benchmark."""
        import asyncio
        
        if asyncio.iscoroutinefunction(benchmark_fn):
            return await benchmark_fn()
        return benchmark_fn()
    
    def detect_knowledge_gaps(self) -> List[KnowledgeGap]:
        """
        Detect knowledge gaps based on benchmark performance.
        
        Returns:
            List of detected knowledge gaps
        """
        gaps = []
        
        for name, score in self._current_scores.items():
            # Check against threshold
            if score < self.config.min_benchmark_score:
                severity = 1.0 - score  # Higher severity for lower scores
                
                gap = KnowledgeGap(
                    gap_id=str(uuid.uuid4())[:8],
                    domain=name,
                    description=f"Below threshold: {score:.2%} < {self.config.min_benchmark_score:.2%}",
                    severity=severity,
                    detected_at=datetime.now(timezone.utc),
                    evidence=[f"Benchmark score: {score:.2%}"],
                )
                
                gaps.append(gap)
                self.gap_history.append(gap)
                self.gaps_detected += 1
            
            # Check for degradation
            baseline = self._baseline_scores.get(name)
            if baseline and score < baseline - self.config.significant_degradation:
                severity = (baseline - score) * 2  # Scale severity
                
                gap = KnowledgeGap(
                    gap_id=str(uuid.uuid4())[:8],
                    domain=f"{name}_degradation",
                    description=f"Performance degraded: {baseline:.2%} → {score:.2%}",
                    severity=min(severity, 1.0),
                    detected_at=datetime.now(timezone.utc),
                    evidence=[
                        f"Baseline: {baseline:.2%}",
                        f"Current: {score:.2%}",
                        f"Degradation: {(baseline - score):.2%}",
                    ],
                )
                
                gaps.append(gap)
                self.gap_history.append(gap)
                self.gaps_detected += 1
        
        return gaps
    
    def should_trigger_learning(self) -> Tuple[bool, Optional[KnowledgeGap]]:
        """
        Check if learning should be triggered.
        
        Returns:
            Tuple of (should_trigger, gap_to_address)
        """
        if not self.config.auto_trigger_learning:
            return False, None
        
        # Get unresolved gaps
        unresolved = self.get_unresolved_gaps()
        
        # Filter by severity
        significant_gaps = [
            gap for gap in unresolved
            if gap.severity >= self.config.min_gap_severity_to_trigger
        ]
        
        if significant_gaps:
            # Return most severe gap
            most_severe = max(significant_gaps, key=lambda g: g.severity)
            return True, most_severe
        
        return False, None
    
    def get_unresolved_gaps(self) -> List[KnowledgeGap]:
        """Get all unresolved knowledge gaps."""
        return [gap for gap in self.gap_history if not gap.resolved]
    
    def resolve_gap(self, gap_id: str, resolution_data: Dict[str, Any]):
        """Mark a knowledge gap as resolved."""
        for gap in self.gap_history:
            if gap.gap_id == gap_id:
                gap.resolved = True
                gap.resolution_data = resolution_data
                self.gaps_resolved += 1
                logger.info(f"Gap resolved: {gap.domain}")
                break
    
    def set_baseline(self, scores: Optional[Dict[str, float]] = None):
        """
        Set baseline scores for degradation detection.
        
        Args:
            scores: Baseline scores (uses current if None)
        """
        if scores:
            self._baseline_scores = scores.copy()
        else:
            self._baseline_scores = self._current_scores.copy()
        
        logger.info(f"Baseline set: {self._baseline_scores}")
    
    def should_evaluate(self, steps_since_last: int) -> bool:
        """Check if evaluation should run based on steps."""
        return steps_since_last >= self.config.eval_interval_steps
    
    def time_since_last_evaluation(self) -> timedelta:
        """Get time since last evaluation."""
        return datetime.now(timezone.utc) - self._last_evaluation
    
    def get_current_scores(self) -> Dict[str, float]:
        """Get current benchmark scores."""
        return self._current_scores.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        avg_score = (
            sum(self._current_scores.values()) / len(self._current_scores)
            if self._current_scores else 0.0
        )
        
        return {
            "total_evaluations": self.total_evaluations,
            "last_evaluation": self._last_evaluation.isoformat(),
            "current_scores": self._current_scores,
            "average_score": round(avg_score, 3),
            "gaps_detected": self.gaps_detected,
            "gaps_resolved": self.gaps_resolved,
            "unresolved_gaps": len(self.get_unresolved_gaps()),
            "benchmarks_registered": len(self.benchmarks),
        }
    
    def get_recent_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent benchmark results."""
        return [r.to_dict() for r in self.results_history[-limit:]]
    
    # =========================================================================
    # Mock Benchmarks for Testing
    # =========================================================================
    
    def register_mock_benchmarks(self):
        """Register mock benchmarks for testing."""
        for benchmark_name in self.config.benchmark_suite:
            self.register_benchmark(
                benchmark_name,
                lambda: random.uniform(0.7, 0.95),
            )
        logger.info(f"Registered {len(self.config.benchmark_suite)} mock benchmarks")
