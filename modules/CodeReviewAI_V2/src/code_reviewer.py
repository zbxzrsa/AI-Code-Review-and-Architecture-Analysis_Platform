"""
CodeReviewAI_V2 - Production Code Reviewer

Enhanced code review with:
- Hallucination detection
- Production-grade error handling
- SLO enforcement
- Caching and batching
"""

import hashlib
import logging
import time
import uuid
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from functools import lru_cache

from .models import (
    Finding, ReviewResult, ReviewStatus, ReviewConfig,
    Dimension, Severity, VerificationStatus
)
from .hallucination_detector import HallucinationDetector

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker for external service calls"""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        if self.state == "closed":
            return True

        if self.state == "open":
            if time.time() - (self.last_failure_time or 0) > self.reset_timeout:
                self.state = "half-open"
                return True
            return False

        return True  # half-open

    def record_success(self):
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning("Circuit breaker opened due to failures")


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
        raise NotImplementedError


class ProductionStrategy(ReviewStrategy):
    """Production-optimized review strategy"""

    def __init__(self):
        super().__init__("production")

    async def review(
        self,
        code: str,
        language: str,
        config: ReviewConfig
    ) -> List[Finding]:
        """Execute production review with comprehensive checks"""
        findings = []
        lines = code.split('\n')

        # Security checks (OWASP Top 10 aligned)
        security_patterns = [
            (r'eval\s*\(', 'eval() usage', Severity.CRITICAL, 'CWE-95'),
            (r'exec\s*\(', 'exec() usage', Severity.CRITICAL, 'CWE-95'),
            (r'pickle\.loads?\s*\(', 'Unsafe pickle', Severity.HIGH, 'CWE-502'),
            (r'yaml\.load\s*\([^)]*\)', 'Unsafe YAML load', Severity.HIGH, 'CWE-502'),
            (r'subprocess.*shell\s*=\s*True', 'Shell injection risk', Severity.HIGH, 'CWE-78'),
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password', Severity.CRITICAL, 'CWE-798'),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key', Severity.CRITICAL, 'CWE-798'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token', Severity.HIGH, 'CWE-798'),
        ]

        import re
        for pattern, issue, severity, cwe in security_patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_num = code[:match.start()].count('\n') + 1
                findings.append(Finding(
                    dimension=Dimension.SECURITY.value,
                    issue=issue,
                    line_numbers=[line_num],
                    severity=severity.value,
                    confidence=0.9,
                    suggestion=f"Address security concern: {issue}",
                    explanation="This pattern may indicate a security vulnerability.",
                    cwe_id=cwe,
                    rule_id=f"SEC-{cwe}",
                    code_snippet=lines[line_num - 1] if line_num <= len(lines) else "",
                ))

        # Performance checks
        perf_patterns = [
            (r'for\s+\w+\s+in\s+.*:\s*\n\s+for\s+\w+\s+in', 'Nested loop', Severity.MEDIUM),
            (r'time\.sleep\s*\(', 'Blocking sleep', Severity.MEDIUM),
            (r'\.\s*append\s*\([^)]+\)\s*$', 'Append in loop (potential)', Severity.LOW),
        ]

        for pattern, issue, severity in perf_patterns:
            for match in re.finditer(pattern, code, re.MULTILINE):
                line_num = code[:match.start()].count('\n') + 1
                findings.append(Finding(
                    dimension=Dimension.PERFORMANCE.value,
                    issue=issue,
                    line_numbers=[line_num],
                    severity=severity.value,
                    confidence=0.75,
                    suggestion=f"Consider optimizing: {issue}",
                    explanation="This pattern may impact performance.",
                    rule_id=f"PERF-{issue[:10].upper().replace(' ', '')}",
                    code_snippet=lines[line_num - 1] if line_num <= len(lines) else "",
                ))

        # Maintainability checks
        maint_patterns = [
            (r'#\s*(TODO|FIXME|HACK|XXX)', 'TODO/FIXME comment', Severity.INFO),
            (r'except\s*:\s*\n\s*(pass|\.\.\.)', 'Bare except', Severity.HIGH),
            (r'def\s+\w+\s*\([^)]{100,}\)', 'Too many parameters', Severity.MEDIUM),
        ]

        for pattern, issue, severity in maint_patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE):
                line_num = code[:match.start()].count('\n') + 1
                findings.append(Finding(
                    dimension=Dimension.MAINTAINABILITY.value,
                    issue=issue,
                    line_numbers=[line_num],
                    severity=severity.value,
                    confidence=0.85,
                    suggestion=f"Address maintainability issue: {issue}",
                    explanation="This pattern affects code maintainability.",
                    rule_id=f"MAINT-{issue[:10].upper().replace(' ', '')}",
                    code_snippet=lines[line_num - 1] if line_num <= len(lines) else "",
                ))

        return findings


class CodeReviewer:
    """
    Production Code Reviewer with enhanced reliability.

    Features:
    - Hallucination detection
    - Circuit breaker pattern
    - Caching
    - SLO enforcement
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        strategy: str = "production",
        config: Optional[ReviewConfig] = None,
        enable_hallucination_check: bool = True,
        slo_timeout_ms: int = 3000,
    ):
        self.config = config or ReviewConfig()
        self.strategy = ProductionStrategy()
        self.enable_hallucination_check = enable_hallucination_check
        self.slo_timeout_ms = slo_timeout_ms

        # Components
        self.hallucination_detector = HallucinationDetector()
        self.circuit_breaker = CircuitBreaker()

        # Cache
        self._cache: Dict[str, ReviewResult] = {}
        self._cache_ttl = 3600  # 1 hour

        # Metrics
        self._review_count = 0
        self._cache_hits = 0
        self._slo_violations = 0
        self._total_time_ms = 0.0

    async def review(
        self,
        code: str,
        language: str = "python",
        config: Optional[ReviewConfig] = None,
    ) -> ReviewResult:
        """
        Execute production code review with SLO enforcement.

        Args:
            code: Source code to review
            language: Programming language
            config: Optional config override

        Returns:
            ReviewResult with verified findings
        """
        start_time = time.time()
        config = config or self.config

        # Generate identifiers
        review_id = str(uuid.uuid4())
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        # Check cache
        if config.enable_caching and code_hash in self._cache:
            cached = self._cache[code_hash]
            cached.from_cache = True
            self._cache_hits += 1
            logger.info(f"Cache hit for review {review_id}")
            return cached

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker open, returning empty result")
            return self._create_error_result(
                review_id, code_hash, "Circuit breaker open", start_time
            )

        try:
            # Execute review with timeout
            review_task = self._execute_review(code, language, config)

            try:
                findings = await asyncio.wait_for(
                    review_task,
                    timeout=self.slo_timeout_ms / 1000.0
                )
            except asyncio.TimeoutError:
                self._slo_violations += 1
                logger.warning(f"Review {review_id} timed out (SLO violation)")
                return self._create_timeout_result(review_id, code_hash, start_time)

            # Hallucination check
            verified_count = len(findings)
            rejected_count = 0

            if self.enable_hallucination_check and findings:
                original_count = len(findings)
                findings = await self.hallucination_detector.verify_findings(findings, code)
                verified_count = len(findings)
                rejected_count = original_count - verified_count

            # Filter by confidence
            findings = [f for f in findings if f.confidence >= config.min_confidence]
            findings = findings[:config.max_findings]

            # Calculate scores
            overall_score, dimension_scores = self._calculate_scores(findings)

            # Calculate confidence stats
            avg_conf = sum(f.confidence for f in findings) / len(findings) if findings else 1.0
            min_conf = min((f.confidence for f in findings), default=1.0)

            processing_time = (time.time() - start_time) * 1000
            slo_met = processing_time <= self.slo_timeout_ms

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
                hallucination_check_passed=rejected_count == 0,
                verified_findings_count=verified_count,
                rejected_findings_count=rejected_count,
                slo_met=slo_met,
            )

            # Update cache
            if config.enable_caching:
                self._cache[code_hash] = result

            # Update metrics
            self._review_count += 1
            self._total_time_ms += processing_time
            self.circuit_breaker.record_success()

            if not slo_met:
                self._slo_violations += 1

            logger.info(
                f"Review {review_id} completed: {len(findings)} findings, "
                f"score={overall_score:.1f}, SLO={'met' if slo_met else 'violated'}"
            )

            return result

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Review {review_id} failed: {e}")
            return self._create_error_result(review_id, code_hash, str(e), start_time)

    async def _execute_review(
        self,
        code: str,
        language: str,
        config: ReviewConfig,
    ) -> List[Finding]:
        """Execute the actual review"""
        return await self.strategy.review(code, language, config)

    def _calculate_scores(self, findings: List[Finding]) -> tuple[float, Dict[str, float]]:
        """Calculate quality scores"""
        base_score = 100.0

        penalties = {
            Severity.CRITICAL.value: 20,
            Severity.HIGH.value: 10,
            Severity.MEDIUM.value: 5,
            Severity.LOW.value: 2,
            Severity.INFO.value: 0,
        }

        dimension_scores = {d.value: 100.0 for d in Dimension}

        for finding in findings:
            penalty = penalties.get(finding.severity, 5)
            weighted_penalty = penalty * finding.confidence

            base_score -= weighted_penalty

            if finding.dimension in dimension_scores:
                dimension_scores[finding.dimension] -= weighted_penalty

        overall_score = max(0, base_score)
        dimension_scores = {k: max(0, v) for k, v in dimension_scores.items()}

        return overall_score, dimension_scores

    def _create_error_result(
        self,
        review_id: str,
        code_hash: str,
        error: str,
        start_time: float,
    ) -> ReviewResult:
        """Create error result"""
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
            slo_met=False,
        )

    def _create_timeout_result(
        self,
        review_id: str,
        code_hash: str,
        start_time: float,
    ) -> ReviewResult:
        """Create timeout result"""
        return ReviewResult(
            review_id=review_id,
            code_hash=code_hash,
            status=ReviewStatus.TIMEOUT,
            findings=[],
            overall_score=0,
            dimension_scores={},
            model_version=self.VERSION,
            strategy_used=self.strategy.name,
            processing_time_ms=(time.time() - start_time) * 1000,
            slo_met=False,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get reviewer metrics"""
        return {
            "review_count": self._review_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._review_count + self._cache_hits),
            "slo_violations": self._slo_violations,
            "slo_compliance_rate": 1 - (self._slo_violations / max(1, self._review_count)),
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": self._total_time_ms / max(1, self._review_count),
            "circuit_breaker_state": self.circuit_breaker.state,
            "hallucination_metrics": self.hallucination_detector.get_metrics(),
        }

    def clear_cache(self):
        """Clear review cache"""
        self._cache.clear()
