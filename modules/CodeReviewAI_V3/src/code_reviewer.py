"""
CodeReviewAI_V3 - Legacy Code Reviewer

Read-only reviewer for baseline comparison.
"""

import hashlib
import logging
import time
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

from .models import Finding, ReviewResult, ReviewStatus, Dimension, Severity

logger = logging.getLogger(__name__)


class CodeReviewer:
    """
    Legacy code reviewer (read-only mode).

    Used for:
    - Baseline comparison
    - Re-evaluation
    - Historical analysis
    """

    VERSION = "3.0.0"
    MODE = "quarantine"

    def __init__(self):
        """Initialize legacy reviewer"""
        self._review_count = 0
        self._archived_results: Dict[str, ReviewResult] = {}

        logger.warning("CodeReviewAI_V3 is in quarantine mode - use for comparison only")

    async def review(
        self,
        code: str,
        language: str = "python",
    ) -> ReviewResult:
        """
        Execute legacy review for comparison.

        Note: This is for baseline comparison only.
        """
        start_time = time.time()

        review_id = str(uuid.uuid4())
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        # Check archived results first
        if code_hash in self._archived_results:
            logger.info(f"Returning archived result for {code_hash}")
            return self._archived_results[code_hash]

        # Basic detection (legacy rules)
        findings = self._detect_legacy(code, language)

        # Calculate scores
        overall_score, dimension_scores = self._calculate_scores(findings)

        processing_time = (time.time() - start_time) * 1000

        result = ReviewResult(
            review_id=review_id,
            code_hash=code_hash,
            status=ReviewStatus.QUARANTINED,
            findings=findings,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            model_version=self.VERSION,
            processing_time_ms=processing_time,
            quarantine_reason="Legacy version",
            quarantined_at=datetime.now(timezone.utc),
        )

        # Archive result
        self._archived_results[code_hash] = result
        self._review_count += 1

        logger.info(f"Legacy review {review_id}: {len(findings)} findings, score={overall_score}")

        return result

    def _detect_legacy(self, code: str, language: str) -> List[Finding]:
        """Legacy detection rules"""
        findings = []
        lines = code.split('\n')

        # Simple pattern matching (legacy rules)
        patterns = [
            ('eval(', 'eval() usage', Severity.CRITICAL, Dimension.SECURITY, 'CWE-95'),
            ('exec(', 'exec() usage', Severity.CRITICAL, Dimension.SECURITY, 'CWE-95'),
            ('password =', 'Hardcoded password', Severity.HIGH, Dimension.SECURITY, 'CWE-798'),
        ]

        for i, line in enumerate(lines, 1):
            for pattern, issue, severity, dimension, cwe in patterns:
                if pattern in line:
                    findings.append(Finding(
                        dimension=dimension.value,
                        issue=issue,
                        line_numbers=[i],
                        severity=severity.value,
                        confidence=0.8,
                        suggestion=f"Address: {issue}",
                        explanation="Legacy detection rule",
                        cwe_id=cwe,
                        rule_id=f"V3-{cwe}",
                        code_snippet=line.strip(),
                    ))

        return findings

    def _calculate_scores(self, findings: List[Finding]) -> tuple[float, Dict[str, float]]:
        """Calculate legacy scores"""
        base_score = 100.0
        dimension_scores = {d.value: 100.0 for d in Dimension}

        penalties = {
            Severity.CRITICAL.value: 20,
            Severity.HIGH.value: 10,
            Severity.MEDIUM.value: 5,
            Severity.LOW.value: 2,
        }

        for finding in findings:
            penalty = penalties.get(finding.severity, 5)
            base_score -= penalty
            if finding.dimension in dimension_scores:
                dimension_scores[finding.dimension] -= penalty

        return max(0, base_score), {k: max(0, v) for k, v in dimension_scores.items()}

    def get_archived_count(self) -> int:
        """Get count of archived results"""
        return len(self._archived_results)

    def clear_archive(self):
        """Clear archived results (admin only)"""
        self._archived_results.clear()
        logger.warning("V3 archive cleared")
