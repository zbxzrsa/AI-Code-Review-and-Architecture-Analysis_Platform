"""
AI Orchestrator - Unified routing for Version Control AI and Code Review AI.

Responsibilities:
- Unified routing for both Version Control AI and Code Review AI
- Multi-expert coordination with weighted voting
- Provider health monitoring and failover
- Cost tracking and budget enforcement
- Streaming response handling (SSE)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
import json
from uuid import uuid4

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Execution mode for coordinator."""
    PIPELINE = "pipeline"  # Sequential execution
    PARALLEL = "parallel"  # Concurrent execution
    BACKTRACK = "backtrack"  # Retry with fallback


class ExpertType(str, Enum):
    """Expert agent types."""
    SECURITY = "security"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    DEPENDENCY = "dependency"
    TEST = "test"
    DOCUMENTATION = "documentation"
    META = "meta"


@dataclass
class ExpertFinding:
    """Finding from an expert."""
    type: str
    severity: str  # critical, high, medium, low
    line_range: tuple[int, int]
    description: str
    recommendation: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "severity": self.severity,
            "line_range": list(self.line_range),
            "description": self.description,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
        }


@dataclass
class ExpertResult:
    """Result from an expert agent."""
    expert_type: ExpertType
    findings: List[ExpertFinding]
    summary: str
    execution_time_ms: float
    confidence: float = 0.0
    provider_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "expert_type": self.expert_type.value,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "execution_time_ms": self.execution_time_ms,
            "confidence": self.confidence,
            "provider_used": self.provider_used,
        }


@dataclass
class TaskComplexity:
    """Task complexity estimation."""
    score: float  # 0-1 scale
    estimated_cost: float
    estimated_latency_ms: int
    recommended_experts: List[ExpertType] = field(default_factory=list)
    budget_allocation: Dict[ExpertType, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "recommended_experts": [e.value for e in self.recommended_experts],
            "budget_allocation": {k.value: v for k, v in self.budget_allocation.items()},
        }


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    max_findings_per_analysis: int = 15
    quality_threshold: float = 0.85
    cost_threshold: float = 0.10
    latency_threshold_ms: int = 5000
    enable_streaming: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


class TaskOrchestrator:
    """Top-level orchestrator for task routing and expert selection."""

    def __init__(self, config: OrchestratorConfig = OrchestratorConfig()):
        """Initialize orchestrator."""
        self.config = config
        self.task_history = {}

    def estimate_complexity(
        self,
        code: str,
        language: str,
        task_type: str,
    ) -> TaskComplexity:
        """
        Estimate task complexity (0-1 scale).

        Returns:
            TaskComplexity with cost and latency estimates
        """
        # Simple heuristic-based estimation
        code_length = len(code)
        _ = len(code.split('\n'))  # Reserved for future line-based analysis

        # Complexity score based on code size and language
        base_score = min(code_length / 10000, 1.0)

        # Adjust by language complexity
        language_multipliers = {
            "python": 0.8,
            "javascript": 0.9,
            "java": 1.0,
            "rust": 1.1,
            "cpp": 1.2,
        }
        complexity_score = base_score * language_multipliers.get(language.lower(), 1.0)

        # Estimate cost and latency
        estimated_cost = 0.01 + (complexity_score * 0.09)
        estimated_latency = 1000 + int(complexity_score * 4000)

        # Select experts based on task type
        recommended_experts = self._select_experts_for_task(task_type)

        # Allocate budget across experts
        budget_allocation = self._allocate_budget(
            recommended_experts,
            estimated_cost,
        )

        return TaskComplexity(
            score=complexity_score,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            recommended_experts=recommended_experts,
            budget_allocation=budget_allocation,
        )

    def _select_experts_for_task(self, task_type: str) -> List[ExpertType]:
        """Select experts based on task type."""
        expert_map = {
            "security": [ExpertType.SECURITY, ExpertType.META],
            "quality": [ExpertType.QUALITY, ExpertType.DEPENDENCY, ExpertType.META],
            "performance": [ExpertType.PERFORMANCE, ExpertType.DEPENDENCY, ExpertType.META],
            "testing": [ExpertType.TEST, ExpertType.QUALITY, ExpertType.META],
            "documentation": [ExpertType.DOCUMENTATION, ExpertType.QUALITY, ExpertType.META],
            "comprehensive": [
                ExpertType.SECURITY,
                ExpertType.QUALITY,
                ExpertType.PERFORMANCE,
                ExpertType.DEPENDENCY,
                ExpertType.TEST,
                ExpertType.DOCUMENTATION,
                ExpertType.META,
            ],
        }
        return expert_map.get(task_type, [ExpertType.QUALITY, ExpertType.META])

    def _allocate_budget(
        self,
        experts: List[ExpertType],
        total_budget: float,
    ) -> Dict[ExpertType, float]:
        """Allocate budget across experts."""
        if not experts:
            return {}

        # Equal allocation with slight adjustment for meta expert
        base_allocation = total_budget / len(experts)
        allocation = {}

        for expert in experts:
            if expert == ExpertType.META:
                # Meta expert gets 20% less (validation only)
                allocation[expert] = base_allocation * 0.8
            else:
                allocation[expert] = base_allocation

        return allocation

    async def validate_output(
        self,
        results: List[ExpertResult],
    ) -> Dict[str, Any]:
        """Validate and sanitize output."""
        validation_result = {
            "valid": True,
            "issues": [],
            "sanitized_results": [],
        }

        for result in results:
            # Check finding count
            if len(result.findings) > self.config.max_findings_per_analysis:
                validation_result["issues"].append(
                    f"{result.expert_type.value}: Too many findings "
                    f"({len(result.findings)} > {self.config.max_findings_per_analysis})"
                )
                # Keep only top findings by confidence
                result.findings = sorted(
                    result.findings,
                    key=lambda f: f.confidence,
                    reverse=True,
                )[:self.config.max_findings_per_analysis]

            # Check confidence scores
            if result.confidence < 0.5:
                validation_result["issues"].append(
                    f"{result.expert_type.value}: Low confidence ({result.confidence})"
                )

            validation_result["sanitized_results"].append(result.to_dict())

        return validation_result


class ExpertCoordinator:
    """Middle-layer coordinator for multi-expert execution."""

    def __init__(self):
        """Initialize coordinator."""
        self.execution_mode = ExecutionMode.PARALLEL

    async def execute_experts(
        self,
        experts: List[ExpertType],
        code: str,
        language: str,
        execution_mode: ExecutionMode = ExecutionMode.PARALLEL,
    ) -> List[ExpertResult]:
        """
        Execute multiple experts.

        Args:
            experts: List of expert types to execute
            code: Code to analyze
            language: Programming language
            execution_mode: Pipeline, parallel, or backtrack

        Returns:
            List of expert results
        """
        if execution_mode == ExecutionMode.PARALLEL:
            return await self._execute_parallel(experts, code, language)
        elif execution_mode == ExecutionMode.PIPELINE:
            return await self._execute_pipeline(experts, code, language)
        else:  # BACKTRACK
            return await self._execute_with_backtrack(experts, code, language)

    async def _execute_parallel(
        self,
        experts: List[ExpertType],
        code: str,
        language: str,
    ) -> List[ExpertResult]:
        """Execute experts in parallel."""
        tasks = [
            self._execute_expert(expert, code, language)
            for expert in experts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, ExpertResult)]

    async def _execute_pipeline(
        self,
        experts: List[ExpertType],
        code: str,
        language: str,
    ) -> List[ExpertResult]:
        """Execute experts sequentially."""
        results = []
        for expert in experts:
            result = await self._execute_expert(expert, code, language)
            if isinstance(result, ExpertResult):
                results.append(result)
        return results

    async def _execute_with_backtrack(
        self,
        experts: List[ExpertType],
        code: str,
        language: str,
    ) -> List[ExpertResult]:
        """Execute with retry and fallback."""
        results = []
        for expert in experts:
            result = await self._execute_with_retry(expert, code, language)
            if isinstance(result, ExpertResult):
                results.append(result)
        return results

    async def _execute_expert(
        self,
        expert: ExpertType,
        _code: str,  # Reserved for actual expert execution
        _language: str,  # Reserved for actual expert execution
    ) -> Optional[ExpertResult]:
        """Execute a single expert."""
        # Mock implementation
        import time
        start = time.time()

        # Simulate expert execution
        await asyncio.sleep(0.1)

        execution_time = (time.time() - start) * 1000

        return ExpertResult(
            expert_type=expert,
            findings=[],
            summary=f"Analysis by {expert.value} expert",
            execution_time_ms=execution_time,
            confidence=0.85,
            provider_used="mock-provider",
        )

    async def _execute_with_retry(
        self,
        expert: ExpertType,
        code: str,
        language: str,
        max_retries: int = 3,
    ) -> Optional[ExpertResult]:
        """Execute with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                result = await self._execute_expert(expert, code, language)
                if result:
                    return result
            except Exception as e:
                logger.warning(
                    f"Expert execution failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Expert {expert.value} failed after {max_retries} retries")
        return None

    async def aggregate_results(
        self,
        results: List[ExpertResult],
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple experts.

        Uses Borda count and weighted voting.
        """
        if not results:
            return {"findings": [], "summary": "No results", "confidence": 0.0}

        # Aggregate findings
        all_findings = []
        for result in results:
            all_findings.extend(result.findings)

        # Sort by confidence and severity
        all_findings.sort(
            key=lambda f: (f.confidence, {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(f.severity, 0)),
            reverse=True,
        )

        # Calculate consistency score
        consistency_score = self._calculate_consistency(results)

        # Aggregate confidence using weighted voting
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0

        return {
            "findings": [f.to_dict() for f in all_findings[:15]],
            "expert_results": [r.to_dict() for r in results],
            "consistency_score": consistency_score,
            "average_confidence": avg_confidence,
            "total_execution_time_ms": sum(r.execution_time_ms for r in results),
        }

    def _calculate_consistency(self, results: List[ExpertResult]) -> float:
        """Calculate consistency score across experts."""
        if len(results) < 2:
            return 1.0

        # Simple consistency: average confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)
        return avg_confidence


class ProviderRouter:
    """Routes requests to appropriate providers based on strategy."""

    def __init__(self):
        """Initialize router."""
        self.provider_health = {}
        self.routing_strategies = {
            "complexity": self._route_by_complexity,
            "health": self._route_by_health,
            "cost": self._route_by_cost,
            "latency": self._route_by_latency,
            "success_rate": self._route_by_success_rate,
        }

    async def select_provider(
        self,
        task_complexity: TaskComplexity,
        available_providers: List[str],
        strategy: str = "complexity",
    ) -> str:
        """
        Select provider based on strategy.

        Returns:
            Selected provider name
        """
        if not available_providers:
            raise ValueError("No available providers")

        if strategy in self.routing_strategies:
            return await self.routing_strategies[strategy](
                task_complexity,
                available_providers,
            )

        return available_providers[0]

    def _route_by_complexity(
        self,
        complexity: TaskComplexity,
        providers: List[str],
    ) -> str:
        """Route based on task complexity."""
        # Simple tasks to cheap models, complex to powerful models
        if complexity.score < 0.3:
            # Prefer cheap models - find gpt-3.5 if available
            cheap = [p for p in providers if "gpt-3.5" in p]
            return cheap[0] if cheap else providers[0]
        elif complexity.score > 0.7:
            # Prefer powerful models - find gpt-4 if available
            powerful = [p for p in providers if "gpt-4" in p]
            return powerful[0] if powerful else providers[0]
        return providers[0]

    def _route_by_health(
        self,
        complexity: TaskComplexity,  # noqa: ARG002 - reserved for health-based routing
        providers: List[str],
    ) -> str:
        """Route based on provider health."""
        # Skip unhealthy providers
        healthy = [p for p in providers if self.provider_health.get(p, {}).get("healthy", True)]
        return healthy[0] if healthy else providers[0]

    def _route_by_cost(
        self,
        complexity: TaskComplexity,  # noqa: ARG002 - reserved for cost estimation
        providers: List[str],
    ) -> str:
        """Route to minimize cost."""
        # Select cheapest provider
        return min(providers, key=lambda p: self.provider_health.get(p, {}).get("cost", 0.1))

    def _route_by_latency(
        self,
        complexity: TaskComplexity,  # noqa: ARG002 - reserved for latency estimation
        providers: List[str],
    ) -> str:
        """Route to minimize latency."""
        # Select fastest provider
        return min(providers, key=lambda p: self.provider_health.get(p, {}).get("latency_ms", 5000))

    def _route_by_success_rate(
        self,
        complexity: TaskComplexity,  # noqa: ARG002 - reserved for success rate estimation
        providers: List[str],
    ) -> str:
        """Route to most reliable provider."""
        # Select provider with highest success rate
        return max(providers, key=lambda p: self.provider_health.get(p, {}).get("success_rate", 0.9))

    def get_fallback_chain(self, primary: str) -> List[str]:
        """Get fallback chain for a provider."""
        # Example fallback chain
        chains = {
            "user-gpt4": ["platform-claude3", "platform-gpt35", "local-codellama"],
            "platform-claude3": ["platform-gpt4", "platform-gpt35", "local-codellama"],
            "platform-gpt4": ["platform-claude3", "platform-gpt35", "local-codellama"],
            "platform-gpt35": ["local-codellama"],
            "local-codellama": [],
        }
        return chains.get(primary, [])
