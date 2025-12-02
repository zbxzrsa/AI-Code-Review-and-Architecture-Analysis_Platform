"""
Hallucination Detector for V1 Code Review AI

Detects hallucinations through:
- Consistency checking (multiple runs)
- Fact verification (against actual code)
- Confidence scoring
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging
import re
import statistics

logger = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Result from hallucination detection"""
    hallucination_detected: bool
    confidence: float
    problematic_findings: List[int]  # Indices of problematic findings
    explanation: str
    
    # Detailed results
    consistency_score: float
    fact_check_results: Dict[int, bool]
    confidence_scores: Dict[int, float]
    
    # Mitigation
    mitigations_applied: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hallucination_detected": self.hallucination_detected,
            "confidence": self.confidence,
            "problematic_findings": self.problematic_findings,
            "explanation": self.explanation,
            "consistency_score": self.consistency_score,
            "fact_check_results": self.fact_check_results,
            "confidence_scores": self.confidence_scores,
            "mitigations_applied": self.mitigations_applied,
        }


class ConsistencyChecker:
    """
    Checks consistency of reviews across multiple runs.
    
    Hallucinations often produce inconsistent results when
    the same code is reviewed multiple times.
    """
    
    def __init__(self, num_runs: int = 3, stddev_threshold: float = 0.2):
        self.num_runs = num_runs
        self.stddev_threshold = stddev_threshold
    
    async def check_consistency(
        self,
        review_engine: Any,
        code: str,
        dimensions: List[str],
        strategy: str,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Run multiple reviews and check consistency.
        
        Returns:
            consistency_score: 0-1 score (higher = more consistent)
            run_results: Results from each run
        """
        run_results = []
        
        for i in range(self.num_runs):
            result = await review_engine.review(
                code=code,
                dimensions=dimensions,
                strategy=strategy,
                use_cache=False,
            )
            
            run_results.append({
                "run": i,
                "findings_count": len(result.findings),
                "overall_score": result.overall_score,
                "finding_hashes": self._hash_findings(result.findings),
            })
        
        # Calculate consistency
        scores = [r["overall_score"] for r in run_results]
        finding_counts = [r["findings_count"] for r in run_results]
        
        # Score consistency
        if len(set(scores)) == 1:
            score_consistency = 1.0
        else:
            score_stddev = statistics.stdev(scores) if len(scores) > 1 else 0
            score_consistency = max(0, 1 - (score_stddev / 50))  # Normalize
        
        # Finding count consistency
        if len(set(finding_counts)) == 1:
            count_consistency = 1.0
        else:
            count_stddev = statistics.stdev(finding_counts) if len(finding_counts) > 1 else 0
            max_count = max(finding_counts) if finding_counts else 1
            count_consistency = max(0, 1 - (count_stddev / max(1, max_count)))
        
        # Finding overlap (Jaccard similarity)
        all_hashes = [set(r["finding_hashes"]) for r in run_results]
        if all_hashes and any(all_hashes):
            union = set.union(*all_hashes)
            intersection = set.intersection(*all_hashes) if len(all_hashes) > 1 else all_hashes[0]
            overlap_consistency = len(intersection) / max(1, len(union))
        else:
            overlap_consistency = 1.0
        
        # Combined consistency score
        consistency_score = (
            score_consistency * 0.3 +
            count_consistency * 0.3 +
            overlap_consistency * 0.4
        )
        
        return consistency_score, run_results
    
    def _hash_findings(self, findings: List[Any]) -> List[str]:
        """Create hashes for findings for comparison"""
        hashes = []
        for f in findings:
            key = f"{f.dimension}:{f.issue}:{','.join(map(str, f.line_numbers))}"
            hashes.append(key)
        return hashes


class FactChecker:
    """
    Verifies findings against actual code.
    
    Checks:
    - Do referenced lines exist?
    - Are error messages valid?
    - Are suggested fixes syntactically correct?
    """
    
    def __init__(self):
        self.checks = [
            self._check_line_existence,
            self._check_code_snippet_match,
            self._check_fix_syntax,
        ]
    
    def check_facts(
        self,
        code: str,
        findings: List[Any],
    ) -> Dict[int, bool]:
        """
        Check each finding against the code.
        
        Returns:
            Dict mapping finding index to validity (True = valid)
        """
        lines = code.split('\n')
        results = {}
        
        for i, finding in enumerate(findings):
            # Run all fact checks
            all_passed = True
            
            for check in self.checks:
                if not check(code, lines, finding):
                    all_passed = False
                    break
            
            results[i] = all_passed
        
        return results
    
    def _check_line_existence(
        self,
        code: str,
        lines: List[str],
        finding: Any,
    ) -> bool:
        """Check if referenced lines exist"""
        for line_num in finding.line_numbers:
            if line_num < 1 or line_num > len(lines):
                logger.warning(f"Finding references non-existent line {line_num}")
                return False
        return True
    
    def _check_code_snippet_match(
        self,
        code: str,
        lines: List[str],
        finding: Any,
    ) -> bool:
        """Check if code snippet matches actual code"""
        if not finding.code_snippet:
            return True
        
        snippet = finding.code_snippet.strip()
        
        # Check if snippet exists in code
        if snippet in code:
            return True
        
        # Check if it's in referenced lines
        for line_num in finding.line_numbers:
            if line_num >= 1 and line_num <= len(lines):
                if snippet in lines[line_num - 1]:
                    return True
        
        logger.warning(f"Code snippet not found: {snippet[:50]}...")
        return False
    
    def _check_fix_syntax(
        self,
        code: str,
        lines: List[str],
        finding: Any,
    ) -> bool:
        """Check if suggested fix is syntactically valid"""
        if not finding.fix_snippet:
            return True
        
        # Basic syntax check for Python
        fix = finding.fix_snippet.strip()
        
        # Check for basic syntax issues
        if fix.count('(') != fix.count(')'):
            logger.warning("Fix has unbalanced parentheses")
            return False
        
        if fix.count('[') != fix.count(']'):
            logger.warning("Fix has unbalanced brackets")
            return False
        
        if fix.count('{') != fix.count('}'):
            logger.warning("Fix has unbalanced braces")
            return False
        
        return True


class HallucinationDetector:
    """
    Main hallucination detection orchestrator.
    
    Combines multiple detection mechanisms:
    1. Consistency checking
    2. Fact verification
    3. Confidence scoring
    """
    
    # Hallucination triggers
    TRIGGERS = [
        "invented_vulnerabilities",
        "false_performance_claims",
        "impossible_error_messages",
        "contradictory_feedback",
        "non_existent_lines",
        "invalid_syntax_suggestions",
    ]
    
    def __init__(
        self,
        consistency_runs: int = 3,
        consistency_threshold: float = 0.8,
        confidence_threshold: float = 0.5,
        min_avg_confidence: float = 0.75,
    ):
        self.consistency_checker = ConsistencyChecker(num_runs=consistency_runs)
        self.fact_checker = FactChecker()
        self.consistency_threshold = consistency_threshold
        self.confidence_threshold = confidence_threshold
        self.min_avg_confidence = min_avg_confidence
    
    async def detect(
        self,
        code: str,
        review_result: Any,
        review_engine: Optional[Any] = None,
        run_consistency_check: bool = True,
    ) -> HallucinationResult:
        """
        Detect hallucinations in a review.
        
        Args:
            code: Original code that was reviewed
            review_result: The review result to check
            review_engine: Engine for consistency checks (optional)
            run_consistency_check: Whether to run multiple passes
            
        Returns:
            HallucinationResult with detection details
        """
        problematic_findings = []
        mitigations_applied = []
        explanations = []
        
        # 1. Fact checking
        fact_results = self.fact_checker.check_facts(code, review_result.findings)
        
        for idx, valid in fact_results.items():
            if not valid:
                problematic_findings.append(idx)
                explanations.append(f"Finding {idx} failed fact verification")
        
        # 2. Confidence scoring
        confidence_scores = {}
        low_confidence_findings = []
        
        for i, finding in enumerate(review_result.findings):
            confidence_scores[i] = finding.confidence
            if finding.confidence < self.confidence_threshold:
                low_confidence_findings.append(i)
                if i not in problematic_findings:
                    problematic_findings.append(i)
        
        # Average confidence check
        if review_result.findings:
            avg_confidence = sum(f.confidence for f in review_result.findings) / len(review_result.findings)
        else:
            avg_confidence = 1.0
        
        if avg_confidence < self.min_avg_confidence:
            explanations.append(f"Average confidence ({avg_confidence:.2f}) below threshold")
        
        # 3. Consistency checking
        consistency_score = 1.0
        if run_consistency_check and review_engine:
            consistency_score, _ = await self.consistency_checker.check_consistency(
                review_engine=review_engine,
                code=code,
                dimensions=list(set(f.dimension for f in review_result.findings)),
                strategy=review_result.strategy_used,
            )
            
            if consistency_score < self.consistency_threshold:
                explanations.append(f"Low consistency score: {consistency_score:.2f}")
        
        # Determine if hallucination detected
        hallucination_detected = (
            len(problematic_findings) > 0 or
            avg_confidence < self.min_avg_confidence or
            consistency_score < self.consistency_threshold
        )
        
        # Calculate overall confidence in detection
        detection_confidence = self._calculate_detection_confidence(
            fact_results, confidence_scores, consistency_score
        )
        
        # Generate explanation
        if hallucination_detected:
            if problematic_findings:
                explanation = f"Detected {len(problematic_findings)} problematic finding(s). "
            else:
                explanation = "Review shows signs of potential hallucination. "
            explanation += "; ".join(explanations)
        else:
            explanation = "No hallucinations detected. Review appears reliable."
        
        # Apply mitigations
        if hallucination_detected:
            mitigations_applied = self._apply_mitigations(problematic_findings, review_result)
        
        return HallucinationResult(
            hallucination_detected=hallucination_detected,
            confidence=detection_confidence,
            problematic_findings=problematic_findings,
            explanation=explanation,
            consistency_score=consistency_score,
            fact_check_results=fact_results,
            confidence_scores=confidence_scores,
            mitigations_applied=mitigations_applied,
        )
    
    def _calculate_detection_confidence(
        self,
        fact_results: Dict[int, bool],
        confidence_scores: Dict[int, float],
        consistency_score: float,
    ) -> float:
        """Calculate confidence in hallucination detection"""
        # Higher confidence = more certain about detection result
        
        # Fact check confidence
        if fact_results:
            fact_pass_rate = sum(fact_results.values()) / len(fact_results)
        else:
            fact_pass_rate = 1.0
        
        # Confidence score average
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            avg_confidence = 1.0
        
        # Combined detection confidence
        detection_confidence = (
            fact_pass_rate * 0.4 +
            avg_confidence * 0.3 +
            consistency_score * 0.3
        )
        
        return detection_confidence
    
    def _apply_mitigations(
        self,
        problematic_indices: List[int],
        review_result: Any,
    ) -> List[str]:
        """Apply mitigations for detected hallucinations"""
        mitigations = []
        
        # Mark problematic findings
        for idx in problematic_indices:
            if idx < len(review_result.findings):
                finding = review_result.findings[idx]
                finding.confidence *= 0.5  # Reduce confidence
                mitigations.append(f"Reduced confidence for finding {idx}")
        
        # Could also:
        # - Request human review
        # - Re-run with lower temperature
        # - Cross-validate with static analysis
        
        return mitigations
    
    async def validate_and_filter(
        self,
        code: str,
        review_result: Any,
        review_engine: Optional[Any] = None,
    ) -> Tuple[Any, HallucinationResult]:
        """
        Validate review and filter out hallucinated findings.
        
        Returns:
            Filtered review result and detection result
        """
        detection = await self.detect(
            code=code,
            review_result=review_result,
            review_engine=review_engine,
            run_consistency_check=True,
        )
        
        if detection.hallucination_detected:
            # Filter out problematic findings
            valid_indices = set(range(len(review_result.findings))) - set(detection.problematic_findings)
            review_result.findings = [
                review_result.findings[i] for i in sorted(valid_indices)
            ]
            
            # Recalculate scores
            if review_result.findings:
                review_result.avg_confidence = sum(
                    f.confidence for f in review_result.findings
                ) / len(review_result.findings)
                review_result.min_confidence = min(
                    f.confidence for f in review_result.findings
                )
        
        return review_result, detection
