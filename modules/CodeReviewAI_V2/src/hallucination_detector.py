"""
CodeReviewAI_V2 - Hallucination Detector

Detects and filters hallucinated findings:
- Consistency checking across multiple runs
- Fact verification against actual code
- Confidence scoring and filtering
"""

import logging
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from .models import Finding, VerificationStatus

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of finding verification"""
    finding_id: str
    status: VerificationStatus
    confidence: float
    method: str
    details: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "status": self.status.value,
            "confidence": self.confidence,
            "method": self.method,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class HallucinationDetector:
    """
    Detects and filters hallucinated findings.

    Methods:
    - Code fact verification
    - Consistency checking
    - Line number validation
    - Issue existence verification
    """

    def __init__(
        self,
        consistency_threshold: float = 0.7,
        min_verification_confidence: float = 0.8,
    ):
        """
        Initialize hallucination detector.

        Args:
            consistency_threshold: Threshold for consistency check
            min_verification_confidence: Minimum confidence for verification
        """
        self.consistency_threshold = consistency_threshold
        self.min_verification_confidence = min_verification_confidence

        # Cache for consistency checking
        self._consistency_cache: Dict[str, List[Finding]] = {}

        # Metrics
        self._verified_count = 0
        self._rejected_count = 0
        self._total_checked = 0

    async def verify_finding(
        self,
        finding: Finding,
        code: str,
    ) -> VerificationResult:
        """
        Verify a single finding against the code.

        Args:
            finding: Finding to verify
            code: Source code to verify against

        Returns:
            VerificationResult with status and confidence
        """
        self._total_checked += 1

        verification_checks = []

        # Check 1: Line number validation
        line_valid, line_conf = self._verify_line_numbers(finding, code)
        verification_checks.append(("line_numbers", line_valid, line_conf))

        # Check 2: Code snippet existence
        snippet_valid, snippet_conf = self._verify_code_snippet(finding, code)
        verification_checks.append(("code_snippet", snippet_valid, snippet_conf))

        # Check 3: Issue pattern verification
        pattern_valid, pattern_conf = self._verify_issue_pattern(finding, code)
        verification_checks.append(("issue_pattern", pattern_valid, pattern_conf))

        # Check 4: Semantic consistency
        semantic_valid, semantic_conf = self._verify_semantic_consistency(finding, code)
        verification_checks.append(("semantic", semantic_valid, semantic_conf))

        # Calculate overall verification
        valid_checks = [c for c in verification_checks if c[1]]
        total_confidence = sum(c[2] for c in verification_checks) / len(verification_checks)

        # Determine status
        if len(valid_checks) >= 3 and total_confidence >= self.min_verification_confidence:
            status = VerificationStatus.VERIFIED
            self._verified_count += 1
        elif len(valid_checks) <= 1 or total_confidence < 0.5:
            status = VerificationStatus.REJECTED
            self._rejected_count += 1
        else:
            status = VerificationStatus.UNCERTAIN

        result = VerificationResult(
            finding_id=finding.rule_id or "unknown",
            status=status,
            confidence=total_confidence,
            method="multi_check",
            details={
                "checks": [
                    {"name": name, "passed": valid, "confidence": conf}
                    for name, valid, conf in verification_checks
                ],
                "valid_check_count": len(valid_checks),
            },
            timestamp=datetime.now(timezone.utc),
        )

        logger.debug(f"Verification result for {finding.rule_id}: {status.value} ({total_confidence:.2f})")

        return result

    def _verify_line_numbers(
        self,
        finding: Finding,
        code: str,
    ) -> Tuple[bool, float]:
        """Verify that referenced line numbers exist"""
        lines = code.split('\n')
        total_lines = len(lines)

        if not finding.line_numbers:
            return False, 0.0

        valid_lines = sum(
            1 for ln in finding.line_numbers
            if 1 <= ln <= total_lines
        )

        confidence = valid_lines / len(finding.line_numbers) if finding.line_numbers else 0
        is_valid = confidence >= 0.8

        return is_valid, confidence

    def _verify_code_snippet(
        self,
        finding: Finding,
        code: str,
    ) -> Tuple[bool, float]:
        """Verify that code snippet exists in source"""
        if not finding.code_snippet:
            # No snippet to verify - neutral
            return True, 0.5

        snippet = finding.code_snippet.strip()

        # Direct match
        if snippet in code:
            return True, 1.0

        # Partial match (lines)
        snippet_lines = [l.strip() for l in snippet.split('\n') if l.strip()]
        code_lines = [l.strip() for l in code.split('\n')]

        matched = sum(1 for sl in snippet_lines if sl in code_lines)
        confidence = matched / len(snippet_lines) if snippet_lines else 0

        return confidence >= 0.6, confidence

    def _verify_issue_pattern(
        self,
        finding: Finding,
        code: str,
    ) -> Tuple[bool, float]:
        """Verify that the reported issue pattern exists"""
        issue_patterns = {
            "eval": ["eval(", "eval ("],
            "exec": ["exec(", "exec ("],
            "password": ["password", "passwd", "pwd"],
            "sql injection": ["execute(", "cursor.", "SELECT", "INSERT", "UPDATE", "DELETE"],
            "shell": ["shell=True", "subprocess"],
            "pickle": ["pickle.load", "pickle.loads"],
        }

        issue_lower = finding.issue.lower()

        for keyword, patterns in issue_patterns.items():
            if keyword in issue_lower:
                # Check if any pattern exists in code
                found = any(p.lower() in code.lower() for p in patterns)
                if found:
                    return True, 0.9
                else:
                    return False, 0.1

        # Unknown issue type - can't verify pattern
        return True, 0.5

    def _verify_semantic_consistency(
        self,
        finding: Finding,
        code: str,
    ) -> Tuple[bool, float]:
        """Verify semantic consistency between finding and code"""
        lines = code.split('\n')

        if not finding.line_numbers:
            return True, 0.5

        # Get referenced lines
        ref_lines = []
        for ln in finding.line_numbers:
            if 1 <= ln <= len(lines):
                ref_lines.append(lines[ln - 1])

        if not ref_lines:
            return False, 0.0

        ref_code = '\n'.join(ref_lines).lower()

        # Check if issue keywords appear in referenced lines
        issue_words = finding.issue.lower().split()
        relevant_words = [w for w in issue_words if len(w) > 3 and w.isalpha()]

        if not relevant_words:
            return True, 0.5

        matched = sum(1 for w in relevant_words if w in ref_code)
        confidence = matched / len(relevant_words)

        return confidence >= 0.3, min(confidence + 0.3, 1.0)

    async def verify_findings(
        self,
        findings: List[Finding],
        code: str,
    ) -> List[Finding]:
        """
        Verify multiple findings and filter hallucinations.

        Args:
            findings: List of findings to verify
            code: Source code

        Returns:
            Filtered list of verified findings
        """
        verified_findings = []

        for finding in findings:
            result = await self.verify_finding(finding, code)

            # Update finding with verification info
            finding.verification_status = result.status.value
            finding.verification_confidence = result.confidence
            finding.verified_at = result.timestamp
            finding.verification_method = result.method

            # Only include verified or uncertain findings
            if result.status != VerificationStatus.REJECTED:
                verified_findings.append(finding)
            else:
                logger.info(f"Rejected hallucinated finding: {finding.issue}")

        logger.info(
            f"Verified {len(verified_findings)}/{len(findings)} findings "
            f"({len(findings) - len(verified_findings)} rejected)"
        )

        return verified_findings

    async def consistency_check(
        self,
        findings: List[Finding],
        code_hash: str,
        run_count: int = 3,
    ) -> Dict[str, float]:
        """
        Check consistency of findings across multiple runs.

        Args:
            findings: Findings from current run
            code_hash: Hash of the code being reviewed
            run_count: Number of runs to compare

        Returns:
            Consistency scores per finding
        """
        # Store findings for this code
        if code_hash not in self._consistency_cache:
            self._consistency_cache[code_hash] = []

        self._consistency_cache[code_hash].append(findings)

        # Limit cache size
        if len(self._consistency_cache[code_hash]) > run_count:
            self._consistency_cache[code_hash] = self._consistency_cache[code_hash][-run_count:]

        # Calculate consistency
        if len(self._consistency_cache[code_hash]) < 2:
            return {}

        # Build finding signature map
        all_runs = self._consistency_cache[code_hash]
        finding_counts: Dict[str, int] = {}

        for run_findings in all_runs:
            for finding in run_findings:
                sig = self._finding_signature(finding)
                finding_counts[sig] = finding_counts.get(sig, 0) + 1

        # Calculate consistency score
        num_runs = len(all_runs)
        consistency_scores = {
            sig: count / num_runs
            for sig, count in finding_counts.items()
        }

        return consistency_scores

    def _finding_signature(self, finding: Finding) -> str:
        """Generate unique signature for a finding"""
        sig_parts = [
            finding.dimension,
            finding.issue[:50],
            str(sorted(finding.line_numbers)),
            finding.severity,
        ]
        sig_str = "|".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()[:12]

    def get_metrics(self) -> Dict[str, Any]:
        """Get detector metrics"""
        return {
            "total_checked": self._total_checked,
            "verified_count": self._verified_count,
            "rejected_count": self._rejected_count,
            "verification_rate": self._verified_count / max(1, self._total_checked),
            "rejection_rate": self._rejected_count / max(1, self._total_checked),
        }

    def clear_cache(self):
        """Clear consistency cache"""
        self._consistency_cache.clear()
