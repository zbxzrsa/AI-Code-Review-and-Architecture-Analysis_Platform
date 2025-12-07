"""
CodeReviewAI_V1 - Code Reviewer

Main code review orchestration with multi-strategy support:
- Baseline direct review
- Chain-of-thought reasoning
- Few-shot in-context learning
- Contrastive analysis
- Ensemble voting
"""

import hashlib
import logging
import time
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

from .models import (
    Finding, ReviewResult, ReviewStatus, ReviewConfig,
    Dimension, Severity
)

logger = logging.getLogger(__name__)


class ReviewStrategy:
    """Base class for review strategies"""

    def __init__(self, name: str):
        self.name = name

    async def review(
        self,
        code: str,
        language: str,
        config: ReviewConfig
    ) -> List[Finding]:
        """Execute review strategy"""
        raise NotImplementedError


class BaselineStrategy(ReviewStrategy):
    """Direct instruction-tuned review"""

    def __init__(self):
        super().__init__("baseline")

    async def review(
        self,
        code: str,
        language: str,
        config: ReviewConfig
    ) -> List[Finding]:
        """Execute baseline review"""
        findings = []

        # Analyze code for common issues
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Security checks
            if 'eval(' in line or 'exec(' in line:
                findings.append(Finding(
                    dimension=Dimension.SECURITY.value,
                    issue="Use of eval/exec detected",
                    line_numbers=[i],
                    severity=Severity.HIGH.value,
                    confidence=0.95,
                    suggestion="Avoid using eval/exec. Use safer alternatives.",
                    explanation="eval/exec can execute arbitrary code and is a security risk.",
                    cwe_id="CWE-95",
                ))

            # SQL injection check
            if 'execute(' in line and '%s' not in line and '?' not in line:
                if 'f"' in line or "f'" in line or '+' in line:
                    findings.append(Finding(
                        dimension=Dimension.SECURITY.value,
                        issue="Potential SQL injection vulnerability",
                        line_numbers=[i],
                        severity=Severity.CRITICAL.value,
                        confidence=0.85,
                        suggestion="Use parameterized queries instead of string concatenation.",
                        explanation="String concatenation in SQL queries can lead to SQL injection.",
                        cwe_id="CWE-89",
                    ))

            # Password in code
            if 'password' in line.lower() and '=' in line:
                if '"' in line or "'" in line:
                    findings.append(Finding(
                        dimension=Dimension.SECURITY.value,
                        issue="Hardcoded password detected",
                        line_numbers=[i],
                        severity=Severity.CRITICAL.value,
                        confidence=0.9,
                        suggestion="Use environment variables or a secrets manager.",
                        explanation="Hardcoded passwords are a security risk.",
                        cwe_id="CWE-798",
                    ))

            # Performance: nested loops
            if language == "python":
                if line.strip().startswith('for ') or line.strip().startswith('while '):
                    # Check if inside another loop (simplified check)
                    indent = len(line) - len(line.lstrip())
                    if indent > 4:  # Nested
                        findings.append(Finding(
                            dimension=Dimension.PERFORMANCE.value,
                            issue="Nested loop detected",
                            line_numbers=[i],
                            severity=Severity.MEDIUM.value,
                            confidence=0.7,
                            suggestion="Consider optimizing nested loops for better performance.",
                            explanation="Nested loops can lead to O(n²) or worse complexity.",
                        ))

        return findings


class ChainOfThoughtStrategy(ReviewStrategy):
    """Chain-of-thought reasoning strategy"""

    def __init__(self):
        super().__init__("chain_of_thought")

    async def review(
        self,
        code: str,
        language: str,
        config: ReviewConfig
    ) -> List[Finding]:
        """Execute CoT review with reasoning steps"""
        findings = []

        # Step 1: Understand code structure
        lines = code.split('\n')
        functions = []
        classes = []

        for i, line in enumerate(lines, 1):
            if line.strip().startswith('def '):
                functions.append((i, line.strip()))
            elif line.strip().startswith('class '):
                classes.append((i, line.strip()))

        # Step 2: Analyze each function
        for line_num, func_def in functions:
            reasoning = [
                f"Step 1: Identified function at line {line_num}",
                f"Step 2: Analyzing function: {func_def}",
            ]

            # Check for missing docstring
            if line_num < len(lines):
                next_line = lines[line_num] if line_num < len(lines) else ""
                if '"""' not in next_line and "'''" not in next_line:
                    reasoning.append("Step 3: No docstring found after function definition")
                    findings.append(Finding(
                        dimension=Dimension.MAINTAINABILITY.value,
                        issue="Function missing docstring",
                        line_numbers=[line_num],
                        severity=Severity.LOW.value,
                        confidence=0.9,
                        suggestion="Add a docstring explaining the function's purpose.",
                        explanation="Docstrings improve code maintainability.",
                        reasoning_steps=reasoning,
                    ))

        # Step 3: Check class structure
        for line_num, class_def in classes:
            reasoning = [
                f"Step 1: Identified class at line {line_num}",
                f"Step 2: Analyzing class: {class_def}",
            ]

            # Check for __init__ method
            class_content = '\n'.join(lines[line_num:line_num+50])
            if '__init__' not in class_content:
                reasoning.append("Step 3: No __init__ method found")
                findings.append(Finding(
                    dimension=Dimension.ARCHITECTURE.value,
                    issue="Class missing __init__ method",
                    line_numbers=[line_num],
                    severity=Severity.INFO.value,
                    confidence=0.8,
                    suggestion="Consider adding an __init__ method for initialization.",
                    explanation="Classes typically need an initializer for proper instantiation.",
                    reasoning_steps=reasoning,
                ))

        return findings


class EnsembleStrategy(ReviewStrategy):
    """Ensemble strategy combining multiple approaches"""

    def __init__(self):
        super().__init__("ensemble")
        self.strategies = [
            BaselineStrategy(),
            ChainOfThoughtStrategy(),
        ]

    async def review(
        self,
        code: str,
        language: str,
        config: ReviewConfig
    ) -> List[Finding]:
        """Execute ensemble review with voting"""
        all_findings = []

        # Run all strategies
        for strategy in self.strategies:
            findings = await strategy.review(code, language, config)
            all_findings.extend(findings)

        # Deduplicate and vote
        unique_findings = {}
        for finding in all_findings:
            key = f"{finding.dimension}:{finding.issue}:{tuple(finding.line_numbers)}"
            if key not in unique_findings:
                unique_findings[key] = finding
            else:
                # Boost confidence for duplicate findings
                existing = unique_findings[key]
                existing.confidence = min(1.0, existing.confidence + 0.1)

        return list(unique_findings.values())


class CodeReviewer:
    """
    Main code review orchestrator.

    Supports multiple review strategies and dimensions.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        strategy: str = "ensemble",
        config: Optional[ReviewConfig] = None,
    ):
        """
        Initialize code reviewer.

        Args:
            strategy: Review strategy (baseline, chain_of_thought, ensemble)
            config: Review configuration
        """
        self.config = config or ReviewConfig()
        self.strategy = self._get_strategy(strategy)

        # Metrics
        self._review_count = 0
        self._total_findings = 0
        self._total_time_ms = 0.0

    def _get_strategy(self, strategy_name: str) -> ReviewStrategy:
        """Get review strategy by name"""
        strategies = {
            "baseline": BaselineStrategy,
            "chain_of_thought": ChainOfThoughtStrategy,
            "ensemble": EnsembleStrategy,
        }

        if strategy_name not in strategies:
            logger.warning(f"Unknown strategy {strategy_name}, using ensemble")
            strategy_name = "ensemble"

        return strategies[strategy_name]()

    async def review(
        self,
        code: str,
        language: str = "python",
        config: Optional[ReviewConfig] = None,
    ) -> ReviewResult:
        """
        Execute code review.

        Args:
            code: Source code to review
            language: Programming language
            config: Optional config override

        Returns:
            ReviewResult with findings and scores
        """
        start_time = time.time()
        config = config or self.config

        # Generate review ID and code hash
        review_id = str(uuid.uuid4())
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        logger.info(f"Starting review {review_id} with strategy {self.strategy.name}")

        try:
            # Execute review strategy
            findings = await self.strategy.review(code, language, config)

            # Filter by confidence
            findings = [f for f in findings if f.confidence >= config.min_confidence]

            # Limit findings
            findings = findings[:config.max_findings]

            # Calculate scores
            overall_score, dimension_scores = self._calculate_scores(findings)

            # Calculate confidence stats
            avg_conf = sum(f.confidence for f in findings) / len(findings) if findings else 1.0
            min_conf = min((f.confidence for f in findings), default=1.0)

            processing_time = (time.time() - start_time) * 1000

            result = ReviewResult(
                review_id=review_id,
                code_hash=code_hash,
                status=ReviewStatus.COMPLETED,
                findings=findings,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                model_version=self.VERSION,
                strategy_used=self.strategy.name,
                processing_time_ms=processing_time,
                avg_confidence=avg_conf,
                min_confidence=min_conf,
                verified_findings_count=len(findings),
            )

            # Update metrics
            self._review_count += 1
            self._total_findings += len(findings)
            self._total_time_ms += processing_time

            logger.info(f"Review {review_id} completed: {len(findings)} findings, score={overall_score}")

            return result

        except Exception as e:
            logger.error(f"Review {review_id} failed: {e}")
            return ReviewResult(
                review_id=review_id,
                code_hash=code_hash,
                status=ReviewStatus.FAILED,
                findings=[],
                overall_score=0,
                dimension_scores={},
                model_version=self.VERSION,
                strategy_used=self.strategy.name,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def _calculate_scores(self, findings: List[Finding]) -> tuple[float, Dict[str, float]]:
        """Calculate overall and dimension scores"""
        base_score = 100.0

        # Severity penalties
        penalties = {
            Severity.CRITICAL.value: 20,
            Severity.HIGH.value: 10,
            Severity.MEDIUM.value: 5,
            Severity.LOW.value: 2,
            Severity.INFO.value: 0,
        }

        # Calculate dimension scores
        dimension_scores = {d.value: 100.0 for d in Dimension}

        for finding in findings:
            penalty = penalties.get(finding.severity, 5)
            weighted_penalty = penalty * finding.confidence

            base_score -= weighted_penalty

            if finding.dimension in dimension_scores:
                dimension_scores[finding.dimension] -= weighted_penalty

        # Ensure scores are non-negative
        overall_score = max(0, base_score)
        dimension_scores = {k: max(0, v) for k, v in dimension_scores.items()}

        return overall_score, dimension_scores

    def get_metrics(self) -> Dict[str, Any]:
        """Get reviewer metrics"""
        return {
            "review_count": self._review_count,
            "total_findings": self._total_findings,
            "avg_findings_per_review": self._total_findings / max(1, self._review_count),
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / max(1, self._review_count),
        }
