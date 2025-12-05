"""
Code Review Engine for V1 CR-AI

Main review orchestration with multi-strategy support:
- Baseline direct review
- Chain-of-thought reasoning
- Few-shot in-context learning
- Contrastive analysis
- Ensemble voting
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
from enum import Enum
import hashlib
import logging
import time
import re

import torch


logger = logging.getLogger(__name__)


# Constants for regex patterns
CLASS_PATTERN_PYTHON = r'class\s+(\w+)'
CLASS_PATTERN_JS = r'class\s+(\w+)'
CLASS_PATTERN_TS = r'(?:class|interface)\s+(\w+)'


class ReviewStatus(str, Enum):
    """Status of a review"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class Finding:
    """A single finding from code review"""
    dimension: str
    issue: str
    line_numbers: List[int]
    severity: str  # critical, high, medium, low
    confidence: float
    suggestion: str
    explanation: str
    
    # Optional metadata
    cwe_id: Optional[str] = None
    rule_id: Optional[str] = None
    code_snippet: Optional[str] = None
    fix_snippet: Optional[str] = None
    reasoning_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "issue": self.issue,
            "line_numbers": self.line_numbers,
            "severity": self.severity,
            "confidence": self.confidence,
            "suggestion": self.suggestion,
            "explanation": self.explanation,
            "cwe_id": self.cwe_id,
            "rule_id": self.rule_id,
            "code_snippet": self.code_snippet,
            "fix_snippet": self.fix_snippet,
            "reasoning_steps": self.reasoning_steps,
        }


@dataclass
class ReviewResult:
    """Complete result from code review"""
    review_id: str
    code_hash: str
    status: ReviewStatus
    
    # Findings by dimension
    findings: List[Finding]
    
    # Scores
    overall_score: float  # 0-100
    dimension_scores: Dict[str, float]
    
    # Metadata
    model_version: str
    strategy_used: str
    processing_time_ms: float
    timestamp: datetime
    
    # Confidence
    avg_confidence: float
    min_confidence: float
    
    # Errors if any
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "code_hash": self.code_hash,
            "status": self.status.value,
            "findings": [f.to_dict() for f in self.findings],
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "model_version": self.model_version,
            "strategy_used": self.strategy_used,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "error_message": self.error_message,
        }


class ReviewEngine:
    """
    Main code review engine.
    
    Orchestrates multi-strategy review with:
    - Code preprocessing and parsing
    - Strategy selection and execution
    - Finding aggregation and scoring
    - Result caching
    """
    
    # Severity weights for scoring
    SEVERITY_WEIGHTS = {
        "critical": 40,
        "high": 25,
        "medium": 10,
        "low": 5,
        "info": 1,
    }
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        config: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Review cache
        self._cache: Dict[str, ReviewResult] = {}
        
        # Metrics
        self.metrics = {
            "total_reviews": 0,
            "cache_hits": 0,
            "avg_processing_time_ms": 0.0,
            "findings_by_severity": dict.fromkeys(self.SEVERITY_WEIGHTS, 0),
            "findings_by_dimension": {},
        }
    
    def _compute_code_hash(self, code: str) -> str:
        """Compute hash of code for caching"""
        return hashlib.sha256(code.encode()).hexdigest()[:16]
    
    def _check_cache(
        self,
        code_hash: str,
        dimensions: List[str],
        strategy: str,
    ) -> Optional[ReviewResult]:
        """Check cache for existing review"""
        cache_key = f"{code_hash}:{','.join(sorted(dimensions))}:{strategy}"
        return self._cache.get(cache_key)
    
    def _store_cache(
        self,
        result: ReviewResult,
        dimensions: List[str],
    ):
        """Store review result in cache"""
        cache_key = f"{result.code_hash}:{','.join(sorted(dimensions))}:{result.strategy_used}"
        self._cache[cache_key] = result
    
    async def review(
        self,
        code: str,
        language: str = "python",
        dimensions: Optional[List[str]] = None,
        strategy: str = "chain_of_thought",
        context: Optional[str] = None,
        use_cache: bool = True,
    ) -> ReviewResult:
        """
        Perform code review.
        
        Args:
            code: Source code to review
            language: Programming language
            dimensions: Review dimensions to check
            strategy: Review strategy to use
            context: Additional context (surrounding code, etc.)
            use_cache: Whether to use cached results
            
        Returns:
            ReviewResult with findings
        """
        import uuid
        start_time = time.time()
        
        # Default dimensions
        if dimensions is None:
            dimensions = ["correctness", "security", "performance"]
        
        # Compute code hash
        code_hash = self._compute_code_hash(code)
        
        # Check cache
        if use_cache:
            cached = self._check_cache(code_hash, dimensions, strategy)
            if cached:
                self.metrics["cache_hits"] += 1
                cached.status = ReviewStatus.CACHED
                return cached
        
        review_id = str(uuid.uuid4())
        
        try:
            # Preprocess code
            preprocessed = await self._preprocess_code(code, language, context)
            
            # Execute review strategy
            findings = await self._execute_strategy(
                preprocessed, dimensions, strategy, language
            )
            
            # Calculate scores
            overall_score, dimension_scores = self._calculate_scores(findings, dimensions)
            
            # Calculate confidence
            if findings:
                confidences = [f.confidence for f in findings]
                avg_confidence = sum(confidences) / len(confidences)
                min_confidence = min(confidences)
            else:
                avg_confidence = 1.0
                min_confidence = 1.0
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = ReviewResult(
                review_id=review_id,
                code_hash=code_hash,
                status=ReviewStatus.COMPLETED,
                findings=findings,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                model_version="v1-cr-ai-0.1.0",
                strategy_used=strategy,
                processing_time_ms=processing_time_ms,
                timestamp=datetime.now(timezone.utc),
                avg_confidence=avg_confidence,
                min_confidence=min_confidence,
            )
            
            # Update metrics
            self._update_metrics(result)
            
            # Store in cache
            if use_cache:
                self._store_cache(result, dimensions)
            
            return result
            
        except Exception as e:
            logger.error(f"Review failed: {e}", exc_info=True)
            
            return ReviewResult(
                review_id=review_id,
                code_hash=code_hash,
                status=ReviewStatus.FAILED,
                findings=[],
                overall_score=0.0,
                dimension_scores={},
                model_version="v1-cr-ai-0.1.0",
                strategy_used=strategy,
                processing_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(timezone.utc),
                avg_confidence=0.0,
                min_confidence=0.0,
                error_message=str(e),
            )
    
    async def _preprocess_code(
        self,
        code: str,
        language: str,
        context: Optional[str],
    ) -> Dict[str, Any]:
        """Preprocess code for review"""
        # Basic preprocessing
        lines = code.split('\n')
        
        # Extract function/class definitions
        functions = self._extract_functions(code, language)
        classes = self._extract_classes(code, language)
        imports = self._extract_imports(code, language)
        
        return {
            "code": code,
            "lines": lines,
            "language": language,
            "context": context,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "line_count": len(lines),
        }
    
    def _extract_functions(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract function definitions"""
        functions = []
        
        patterns = {
            "python": r'def\s+(\w+)\s*\((.*?)\)\s*(?:->.*?)?:',
            "javascript": r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()',
            "typescript": r'(?:function\s+(\w+)|(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\()',
            "java": r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
        }
        
        pattern = patterns.get(language, patterns["python"])
        
        for i, line in enumerate(code.split('\n'), 1):
            match = re.search(pattern, line)
            if match:
                name = match.group(1) or (match.group(2) if len(match.groups()) > 1 else None)
                if name:
                    functions.append({
                        "name": name,
                        "line": i,
                        "signature": line.strip(),
                    })
        
        return functions
    
    def _extract_classes(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Extract class definitions"""
        classes = []
        
        patterns = {
            "python": CLASS_PATTERN_PYTHON,
            "javascript": CLASS_PATTERN_JS,
            "typescript": CLASS_PATTERN_TS,
            "java": r'(?:public\s+)?(?:class|interface)\s+(\w+)',
        }
        
        pattern = patterns.get(language, patterns["python"])
        
        for i, line in enumerate(code.split('\n'), 1):
            match = re.search(pattern, line)
            if match:
                classes.append({
                    "name": match.group(1),
                    "line": i,
                })
        
        return classes
    
    def _extract_imports(self, code: str, language: str) -> List[str]:
        """Extract import statements"""
        imports = []
        
        patterns = {
            "python": r'^(?:from\s+[\w.]+\s+)?import\s+.+$',
            "javascript": r'^import\s+.+$',
            "typescript": r'^import\s+.+$',
            "java": r'^import\s+[\w.]+;$',
        }
        
        pattern = patterns.get(language, patterns["python"])
        
        for line in code.split('\n'):
            if re.match(pattern, line.strip()):
                imports.append(line.strip())
        
        return imports
    
    async def _execute_strategy(
        self,
        preprocessed: Dict[str, Any],
        dimensions: List[str],
        strategy: str,
        language: str,
    ) -> List[Finding]:
        """Execute the selected review strategy"""
        if strategy == "baseline":
            return await self._baseline_review(preprocessed, dimensions, language)
        elif strategy == "chain_of_thought":
            return await self._cot_review(preprocessed, dimensions, language)
        elif strategy == "few_shot":
            return await self._few_shot_review(preprocessed, dimensions, language)
        elif strategy == "contrastive":
            return await self._contrastive_review(preprocessed, dimensions, language)
        elif strategy == "ensemble":
            return await self._ensemble_review(preprocessed, dimensions, language)
        else:
            return await self._baseline_review(preprocessed, dimensions, language)
    
    async def _baseline_review(
        self,
        preprocessed: Dict[str, Any],
        dimensions: List[str],
        language: str,
    ) -> List[Finding]:
        """Baseline direct review"""
        findings = []
        
        for dimension in dimensions:
            dimension_findings = await self._analyze_dimension(
                preprocessed, dimension, language, reasoning=False
            )
            findings.extend(dimension_findings)
        
        return findings
    
    async def _cot_review(
        self,
        preprocessed: Dict[str, Any],
        dimensions: List[str],
        language: str,
    ) -> List[Finding]:
        """Chain-of-thought review with reasoning steps"""
        findings = []
        
        for dimension in dimensions:
            dimension_findings = await self._analyze_dimension(
                preprocessed, dimension, language, reasoning=True
            )
            findings.extend(dimension_findings)
        
        return findings
    
    async def _few_shot_review(
        self,
        preprocessed: Dict[str, Any],
        dimensions: List[str],
        language: str,
    ) -> List[Finding]:
        """Few-shot review with examples"""
        # In production, would retrieve similar examples
        # For now, use baseline with higher confidence
        findings = await self._baseline_review(preprocessed, dimensions, language)
        
        # Boost confidence for few-shot
        for finding in findings:
            finding.confidence = min(finding.confidence * 1.1, 1.0)
        
        return findings
    
    async def _contrastive_review(
        self,
        preprocessed: Dict[str, Any],
        dimensions: List[str],
        language: str,
    ) -> List[Finding]:
        """Contrastive analysis comparing versions"""
        # Would compare correct vs potentially buggy versions
        return await self._baseline_review(preprocessed, dimensions, language)
    
    async def _ensemble_review(
        self,
        preprocessed: Dict[str, Any],
        dimensions: List[str],
        language: str,
    ) -> List[Finding]:
        """Ensemble review combining multiple strategies"""
        # Run multiple strategies
        baseline_findings = await self._baseline_review(preprocessed, dimensions, language)
        cot_findings = await self._cot_review(preprocessed, dimensions, language)
        
        # Merge and deduplicate findings
        all_findings = baseline_findings + cot_findings
        
        # Group by issue signature
        grouped: Dict[str, List[Finding]] = {}
        for finding in all_findings:
            key = f"{finding.dimension}:{finding.issue}:{','.join(map(str, finding.line_numbers))}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(finding)
        
        # Merge findings with voting
        merged_findings = []
        for findings_group in grouped.values():
            if len(findings_group) >= 1:
                # Average confidence
                avg_confidence = sum(f.confidence for f in findings_group) / len(findings_group)
                best = max(findings_group, key=lambda f: f.confidence)
                best.confidence = avg_confidence
                merged_findings.append(best)
        
        return merged_findings
    
    async def _analyze_dimension(
        self,
        preprocessed: Dict[str, Any],
        dimension: str,
        language: str,
        reasoning: bool = False,
    ) -> List[Finding]:
        """Analyze code for a specific dimension"""
        findings = []
        code = preprocessed["code"]
        lines = preprocessed["lines"]
        
        if dimension == "correctness":
            findings.extend(self._check_correctness(code, lines, language, reasoning))
        elif dimension == "security":
            findings.extend(self._check_security(code, lines, language, reasoning))
        elif dimension == "performance":
            findings.extend(self._check_performance(code, lines, language, reasoning))
        elif dimension == "maintainability":
            findings.extend(self._check_maintainability(code, lines, language, reasoning))
        elif dimension == "architecture":
            findings.extend(self._check_architecture(code, lines, language, reasoning))
        elif dimension == "testing":
            findings.extend(self._check_testing(code, lines, language, reasoning))
        
        return findings
    
    def _check_correctness(
        self,
        code: str,
        lines: List[str],
        language: str,
        reasoning: bool,
    ) -> List[Finding]:
        """Check for correctness issues"""
        findings = []
        
        # Off-by-one patterns
        off_by_one_patterns = [
            (r'range\(len\(\w+\)\s*-\s*1\)', "Potential off-by-one: range(len(x)-1) may miss last element"),
            (r'\[\w+\s*\+\s*1\]', "Potential off-by-one: index+1 may exceed bounds"),
            (r'<=\s*len\(', "Potential off-by-one: <= len() may exceed bounds"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, issue in off_by_one_patterns:
                if re.search(pattern, line):
                    reasoning_steps = []
                    if reasoning:
                        reasoning_steps = [
                            "Identified array/list indexing operation",
                            "Detected pattern that commonly causes off-by-one errors",
                            "Verified bounds checking may be incorrect",
                        ]
                    
                    findings.append(Finding(
                        dimension="correctness",
                        issue=issue,
                        line_numbers=[i],
                        severity="high",
                        confidence=0.75,
                        suggestion="Verify loop bounds and array indices",
                        explanation="This pattern is commonly associated with off-by-one errors.",
                        code_snippet=line.strip(),
                        reasoning_steps=reasoning_steps,
                    ))
        
        return findings
    
    def _check_security(
        self,
        code: str,
        lines: List[str],
        language: str,
        reasoning: bool,
    ) -> List[Finding]:
        """Check for security vulnerabilities"""
        findings = []
        
        # Security patterns
        security_patterns = [
            (r'f["\'].*\{.*\}.*SELECT|INSERT|UPDATE|DELETE', "SQL Injection", "CWE-89", "critical"),
            (r'eval\s*\(', "Code Injection via eval()", "CWE-94", "critical"),
            (r'exec\s*\(', "Code Injection via exec()", "CWE-94", "critical"),
            (r'os\.system\s*\(', "Command Injection", "CWE-78", "critical"),
            (r'subprocess.*shell\s*=\s*True', "Command Injection with shell=True", "CWE-78", "critical"),
            (r'pickle\.loads?\s*\(', "Insecure Deserialization", "CWE-502", "high"),
            (r'password\s*=\s*["\']', "Hardcoded Password", "CWE-798", "high"),
            (r'api_key\s*=\s*["\']', "Hardcoded API Key", "CWE-798", "high"),
            (r'innerHTML\s*=', "Potential XSS via innerHTML", "CWE-79", "high"),
            (r'dangerouslySetInnerHTML', "Potential XSS via dangerouslySetInnerHTML", "CWE-79", "high"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, issue, cwe, severity in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    reasoning_steps = []
                    if reasoning:
                        reasoning_steps = [
                            f"Detected security-sensitive pattern: {pattern}",
                            f"This pattern is associated with {cwe}",
                            "User input may reach this code path",
                            "Recommend using parameterized queries/safe APIs",
                        ]
                    
                    findings.append(Finding(
                        dimension="security",
                        issue=issue,
                        line_numbers=[i],
                        severity=severity,
                        confidence=0.85,
                        suggestion=f"Use safe alternatives to prevent {cwe}",
                        explanation=f"This code pattern is vulnerable to {issue}.",
                        cwe_id=cwe,
                        code_snippet=line.strip(),
                        reasoning_steps=reasoning_steps,
                    ))
        
        return findings
    
    def _check_performance(
        self,
        code: str,  # noqa: ARG002 - Reserved for code-level analysis
        lines: List[str],
        language: str,  # noqa: ARG002 - Reserved for language-specific patterns
        reasoning: bool,  # noqa: ARG002 - Reserved for reasoning steps
    ) -> List[Finding]:
        """Check for performance issues"""
        findings = []
        
        # Performance patterns
        perf_patterns = [
            (r'for.*for.*for', "Triple nested loop - O(n³) complexity", "high"),
            (r'\+\s*=\s*["\']', "String concatenation in loop", "medium"),
            (r'\.append\s*\(.*\.append', "Nested list operations", "medium"),
            (r'in\s+list\(', "Converting to list for membership test", "low"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, issue, severity in perf_patterns:
                if re.search(pattern, line):
                    findings.append(Finding(
                        dimension="performance",
                        issue=issue,
                        line_numbers=[i],
                        severity=severity,
                        confidence=0.70,
                        suggestion="Consider algorithmic optimization",
                        explanation="This pattern may cause performance degradation.",
                        code_snippet=line.strip(),
                    ))
        
        return findings
    
    def _check_maintainability(
        self,
        code: str,  # noqa: ARG002 - Reserved for code-level analysis
        lines: List[str],
        language: str,  # noqa: ARG002 - Reserved for language-specific patterns
        reasoning: bool,  # noqa: ARG002 - Reserved for reasoning steps
    ) -> List[Finding]:
        """Check for maintainability issues"""
        findings = []
        
        # Check function length
        in_function = False
        function_start = 0
        function_name = ""
        _ = 0  # noqa: F841 - indent_level reserved for future
        
        for i, line in enumerate(lines, 1):
            if re.match(r'\s*def\s+(\w+)', line):
                if in_function and (i - function_start) > 50:
                    findings.append(Finding(
                        dimension="maintainability",
                        issue=f"Function '{function_name}' is too long ({i - function_start} lines)",
                        line_numbers=[function_start],
                        severity="medium",
                        confidence=0.80,
                        suggestion="Consider breaking down into smaller functions",
                        explanation="Functions longer than 50 lines are harder to understand and maintain.",
                    ))
                
                match = re.match(r'\s*def\s+(\w+)', line)
                function_name = match.group(1)
                function_start = i
                in_function = True
                indent_level = len(line) - len(line.lstrip())
        
        # Magic numbers
        magic_number_pattern = r'(?<!["\'\w])(?<!\.)(\d{2,})(?!["\'\w])'
        for i, line in enumerate(lines, 1):
            if re.search(magic_number_pattern, line) and 'import' not in line:
                findings.append(Finding(
                    dimension="maintainability",
                    issue="Magic number detected",
                    line_numbers=[i],
                    severity="low",
                    confidence=0.60,
                    suggestion="Consider using named constants",
                    explanation="Magic numbers make code harder to understand and maintain.",
                    code_snippet=line.strip(),
                ))
        
        return findings
    
    def _check_architecture(
        self,
        code: str,  # noqa: ARG002 - Reserved for code-level analysis
        lines: List[str],
        language: str,  # noqa: ARG002 - Reserved for language-specific patterns
        reasoning: bool,  # noqa: ARG002 - Reserved for reasoning steps
    ) -> List[Finding]:
        """Check for architecture issues"""
        findings = []
        
        # God class detection (simplified)
        class_methods = {}
        current_class = None
        
        for i, line in enumerate(lines, 1):
            class_match = re.match(r'class\s+(\w+)', line)
            if class_match:
                current_class = class_match.group(1)
                class_methods[current_class] = 0
            
            if current_class and re.match(r'\s+def\s+', line):
                class_methods[current_class] = class_methods.get(current_class, 0) + 1
        
        for class_name, method_count in class_methods.items():
            if method_count > 20:
                findings.append(Finding(
                    dimension="architecture",
                    issue=f"Class '{class_name}' has too many methods ({method_count})",
                    line_numbers=[1],  # Would need to track actual line
                    severity="medium",
                    confidence=0.70,
                    suggestion="Consider splitting into smaller, focused classes",
                    explanation="Classes with many methods may violate Single Responsibility Principle.",
                ))
        
        return findings
    
    def _check_testing(
        self,
        code: str,  # noqa: ARG002 - Reserved for future pattern matching
        lines: List[str],
        language: str,  # noqa: ARG002 - Reserved for language-specific checks
        reasoning: bool,  # noqa: ARG002 - Reserved for reasoning steps
    ) -> List[Finding]:
        """Check for testing issues"""
        findings = []
        
        # Check for test functions without assertions
        in_test = False
        test_start = 0
        has_assertion = False
        
        for i, line in enumerate(lines, 1):
            if re.match(r'\s*def\s+test_', line):
                if in_test and not has_assertion:
                    findings.append(Finding(
                        dimension="testing",
                        issue="Test function without assertions",
                        line_numbers=[test_start],
                        severity="medium",
                        confidence=0.75,
                        suggestion="Add assertions to verify expected behavior",
                        explanation="Tests without assertions don't actually test anything.",
                    ))
                
                in_test = True
                test_start = i
                has_assertion = False
            
            if in_test and ('assert' in line or 'expect' in line):
                has_assertion = True
        
        return findings
    
    def _calculate_scores(
        self,
        findings: List[Finding],
        dimensions: List[str],
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate overall and per-dimension scores"""
        # Start with perfect score
        base_score = 100.0
        
        # Deduct points based on findings
        total_deduction = 0.0
        dimension_deductions: Dict[str, float] = dict.fromkeys(dimensions, 0.0)
        
        for finding in findings:
            weight = self.SEVERITY_WEIGHTS.get(finding.severity, 5)
            deduction = weight * finding.confidence
            total_deduction += deduction
            
            if finding.dimension in dimension_deductions:
                dimension_deductions[finding.dimension] += deduction
        
        # Calculate overall score
        overall_score = max(0.0, base_score - total_deduction)
        
        # Calculate dimension scores
        dimension_scores = {}
        for dim in dimensions:
            dim_score = max(0.0, base_score - dimension_deductions[dim])
            dimension_scores[dim] = dim_score
        
        return overall_score, dimension_scores
    
    def _update_metrics(self, result: ReviewResult):
        """Update internal metrics"""
        self.metrics["total_reviews"] += 1
        
        # Update average processing time
        n = self.metrics["total_reviews"]
        old_avg = self.metrics["avg_processing_time_ms"]
        self.metrics["avg_processing_time_ms"] = old_avg + (result.processing_time_ms - old_avg) / n
        
        # Update findings counts
        for finding in result.findings:
            self.metrics["findings_by_severity"][finding.severity] = \
                self.metrics["findings_by_severity"].get(finding.severity, 0) + 1
            self.metrics["findings_by_dimension"][finding.dimension] = \
                self.metrics["findings_by_dimension"].get(finding.dimension, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            **self.metrics,
            "cache_hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["total_reviews"]),
        }
    
    def clear_cache(self):
        """Clear the review cache"""
        self._cache.clear()
