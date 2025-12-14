"""
Machine Learning Pattern Recognition

Provides ML-based code pattern recognition for:
- Error pattern detection
- Code quality prediction
- Similar issue clustering
- Auto-fix suggestion ranking

Integrates with the three-version system:
- V1: Trains on experimental patterns
- V2: Uses validated models for production
- V3: Learns from failed patterns
"""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Pattern Recognition Data Classes
# =============================================================================

@dataclass
class CodePattern:
    """Represents a learned code pattern."""
    pattern_id: str
    name: str
    description: str
    pattern_type: str  # error, quality, style, security
    
    # Pattern definition
    regex_pattern: Optional[str] = None
    ast_signature: Optional[Dict[str, Any]] = None
    token_sequence: Optional[List[str]] = None
    
    # Classification
    categories: List[str] = field(default_factory=list)
    severity: str = "medium"
    
    # Learning data
    examples_positive: List[str] = field(default_factory=list)
    examples_negative: List[str] = field(default_factory=list)
    
    # Performance metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    detection_count: int = 0
    false_positive_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "v1"


@dataclass
class PatternMatch:
    """Result of pattern matching."""
    pattern_id: str
    pattern_name: str
    match_text: str
    confidence: float
    location: Dict[str, int]
    context: str
    suggested_fix: Optional[str] = None


@dataclass
class CodeEmbedding:
    """Vector embedding for code similarity."""
    embedding_id: str
    code_hash: str
    embedding_vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimilarCode:
    """Similar code finding."""
    code: str
    similarity_score: float
    issue_type: Optional[str] = None
    fix_applied: Optional[str] = None


# =============================================================================
# Pattern Learning Engine
# =============================================================================

class PatternLearningEngine:
    """
    Engine for learning code patterns from examples.
    
    Uses:
    - N-gram analysis for token patterns
    - TF-IDF for code similarity
    - Pattern clustering for issue grouping
    """
    
    def __init__(self):
        self.patterns: Dict[str, CodePattern] = {}
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)  # category -> pattern_ids
        
        # Token statistics
        self.token_frequencies: Dict[str, int] = defaultdict(int)
        self.pattern_cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Initialize built-in patterns
        self._initialize_builtin_patterns()
        
        logger.info("Pattern Learning Engine initialized")
    
    def _initialize_builtin_patterns(self) -> None:
        """Initialize patterns built from expert knowledge."""
        builtin_patterns = [
            # Security patterns
            CodePattern(
                pattern_id="builtin-sec-001",
                name="hardcoded_credential",
                description="Hardcoded password, API key, or secret",
                pattern_type="security",
                regex_pattern=r'(password|secret|api_key|token)\s*[=:]\s*["\'][^"\']{4,}["\']',
                categories=["security", "credentials"],
                severity="critical",
                version="v2",  # Production-ready
            ),
            CodePattern(
                pattern_id="builtin-sec-002",
                name="sql_injection",
                description="Potential SQL injection vulnerability",
                pattern_type="security",
                regex_pattern=r'(execute|query)\s*\([^)]*\+\s*([\w]+|["\'][^"\']*["\'])',
                categories=["security", "injection"],
                severity="critical",
                version="v2",
            ),
            CodePattern(
                pattern_id="builtin-sec-003",
                name="eval_exec",
                description="Dangerous eval/exec usage",
                pattern_type="security",
                regex_pattern=r'\b(eval|exec)\s*\([^)]+\)',
                categories=["security", "code_execution"],
                severity="high",
                version="v2",
            ),
            CodePattern(
                pattern_id="builtin-sec-004",
                name="command_injection",
                description="Potential command injection",
                pattern_type="security",
                regex_pattern=r'(os\.system|subprocess\.call|subprocess\.run)\s*\([^)]*[+%]',
                categories=["security", "injection"],
                severity="critical",
                version="v2",
            ),
            
            # Performance patterns
            CodePattern(
                pattern_id="builtin-perf-001",
                name="n_plus_one",
                description="Potential N+1 query pattern",
                pattern_type="performance",
                regex_pattern=r'for\s+\w+\s+in\s+\w+:\s*\n\s*.*\.(query|execute|find)',
                categories=["performance", "database"],
                severity="medium",
                version="v2",
            ),
            CodePattern(
                pattern_id="builtin-perf-002",
                name="string_concat_loop",
                description="Inefficient string concatenation in loop",
                pattern_type="performance",
                regex_pattern=r'for\s+.*:\s*\n\s*\w+\s*\+=\s*["\']',
                categories=["performance", "string"],
                severity="low",
                version="v2",
            ),
            
            # Error patterns
            CodePattern(
                pattern_id="builtin-err-001",
                name="bare_except",
                description="Bare except clause catches all exceptions",
                pattern_type="error",
                regex_pattern=r'except\s*:',
                categories=["error_handling", "python"],
                severity="medium",
                version="v2",
            ),
            CodePattern(
                pattern_id="builtin-err-002",
                name="uninitialized_variable",
                description="Potentially uninitialized variable",
                pattern_type="error",
                regex_pattern=r'^\s*if\s+.*:\s*\n\s*\w+\s*=.*\n(?!.*else)',
                categories=["error", "logic"],
                severity="medium",
                version="v2",
            ),
            
            # Style patterns
            CodePattern(
                pattern_id="builtin-sty-001",
                name="magic_number",
                description="Magic number without named constant",
                pattern_type="style",
                regex_pattern=r'[^\w](\d{3,})[^\d]',  # 3+ digit numbers
                categories=["style", "maintainability"],
                severity="low",
                version="v2",
            ),
            CodePattern(
                pattern_id="builtin-sty-002",
                name="long_function",
                description="Function exceeds recommended length",
                pattern_type="style",
                categories=["style", "complexity"],
                severity="low",
                version="v2",
            ),
        ]
        
        for pattern in builtin_patterns:
            self.patterns[pattern.pattern_id] = pattern
            for category in pattern.categories:
                self.pattern_index[category].add(pattern.pattern_id)
    
    def learn_pattern(
        self,
        name: str,
        positive_examples: List[str],
        negative_examples: Optional[List[str]] = None,
        pattern_type: str = "error",
        categories: Optional[List[str]] = None,
    ) -> CodePattern:
        """
        Learn a new pattern from examples.
        
        Args:
            name: Pattern name
            positive_examples: Code snippets that contain the pattern
            negative_examples: Code snippets that don't contain the pattern
            pattern_type: Type of pattern
            categories: Categories for the pattern
            
        Returns:
            The learned pattern
        """
        pattern_id = f"learned-{uuid.uuid4().hex[:8]}"
        
        # Extract common pattern from positive examples
        regex_pattern = self._extract_regex_pattern(positive_examples)
        token_sequence = self._extract_token_sequence(positive_examples)
        
        pattern = CodePattern(
            pattern_id=pattern_id,
            name=name,
            description=f"Learned pattern: {name}",
            pattern_type=pattern_type,
            regex_pattern=regex_pattern,
            token_sequence=token_sequence,
            categories=categories or [pattern_type],
            examples_positive=positive_examples[:10],  # Keep top 10
            examples_negative=negative_examples[:10] if negative_examples else [],
            version="v1",  # Experimental
        )
        
        # Validate pattern on examples
        if positive_examples:
            true_positives = sum(
                1 for ex in positive_examples
                if self._match_pattern(pattern, ex)
            )
            pattern.precision = true_positives / len(positive_examples)
        
        if negative_examples:
            false_positives = sum(
                1 for ex in negative_examples
                if self._match_pattern(pattern, ex)
            )
            if negative_examples:
                pattern.precision = (
                    pattern.precision * len(positive_examples)
                ) / (len(positive_examples) + false_positives) if positive_examples else 0
        
        # Calculate F1
        if pattern.precision > 0 and pattern.recall > 0:
            pattern.f1_score = (
                2 * pattern.precision * pattern.recall
            ) / (pattern.precision + pattern.recall)
        
        self.patterns[pattern_id] = pattern
        for category in pattern.categories:
            self.pattern_index[category].add(pattern_id)
        
        logger.info(f"Learned new pattern: {name} (precision={pattern.precision:.2f})")
        
        return pattern
    
    def _extract_regex_pattern(self, examples: List[str]) -> Optional[str]:
        """Extract common regex pattern from examples."""
        if not examples:
            return None
        
        # Simple approach: find longest common substring
        if len(examples) == 1:
            # Escape special chars and create pattern
            escaped = re.escape(examples[0].strip())
            return escaped[:50]  # Limit length
        
        # Find common substrings
        common = examples[0]
        for example in examples[1:]:
            common = self._longest_common_substring(common, example)
            if len(common) < 5:  # Too short to be useful
                return None
        
        if len(common) >= 5:
            return re.escape(common)
        
        return None
    
    def _longest_common_substring(self, s1: str, s2: str) -> str:
        """Find longest common substring."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0
        end_idx = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_len:
                        max_len = dp[i][j]
                        end_idx = i
        
        return s1[end_idx - max_len:end_idx]
    
    def _extract_token_sequence(self, examples: List[str]) -> List[str]:
        """Extract common token sequence from examples."""
        if not examples:
            return []
        
        # Simple tokenization
        token_lists = []
        for example in examples:
            tokens = re.findall(r'\w+|[^\w\s]', example)
            token_lists.append(tokens)
        
        if not token_lists:
            return []
        
        # Find common token subsequence
        common = token_lists[0]
        for tokens in token_lists[1:]:
            common = self._common_subsequence(common, tokens)
        
        return common[:20]  # Limit length
    
    def _common_subsequence(self, seq1: List[str], seq2: List[str]) -> List[str]:
        """Find common subsequence of tokens."""
        # Simple approach: keep tokens that appear in both
        set2 = set(seq2)
        return [t for t in seq1 if t in set2]
    
    def _match_pattern(self, pattern: CodePattern, code: str) -> bool:
        """Check if pattern matches code."""
        if pattern.regex_pattern:
            try:
                if re.search(pattern.regex_pattern, code, re.IGNORECASE | re.MULTILINE):
                    return True
            except re.error:
                pass
        
        if pattern.token_sequence:
            tokens = re.findall(r'\w+|[^\w\s]', code)
            if all(t in tokens for t in pattern.token_sequence):
                return True
        
        return False
    
    def detect_patterns(
        self,
        code: str,
        categories: Optional[List[str]] = None,
    ) -> List[PatternMatch]:
        """
        Detect patterns in code.
        
        Args:
            code: Source code to analyze
            categories: Optional filter by categories
            
        Returns:
            List of pattern matches
        """
        matches = []
        
        # Filter patterns by category
        pattern_ids = set(self.patterns.keys())
        if categories:
            pattern_ids = set()
            for category in categories:
                pattern_ids.update(self.pattern_index.get(category, set()))
        
        for pattern_id in pattern_ids:
            pattern = self.patterns.get(pattern_id)
            if not pattern:
                continue
            
            # Try regex matching
            if pattern.regex_pattern:
                try:
                    for match in re.finditer(
                        pattern.regex_pattern,
                        code,
                        re.IGNORECASE | re.MULTILINE
                    ):
                        # Calculate line number
                        line_start = code[:match.start()].count('\n') + 1
                        
                        matches.append(PatternMatch(
                            pattern_id=pattern.pattern_id,
                            pattern_name=pattern.name,
                            match_text=match.group(),
                            confidence=pattern.precision if pattern.precision > 0 else 0.8,
                            location={
                                "line": line_start,
                                "start": match.start(),
                                "end": match.end(),
                            },
                            context=self._get_context(code, match.start(), match.end()),
                        ))
                        
                        pattern.detection_count += 1
                except re.error:
                    continue
        
        return matches
    
    def _get_context(self, code: str, start: int, end: int, context_lines: int = 2) -> str:
        """Get context around a match."""
        lines = code.split('\n')
        
        # Find line numbers
        line_start = code[:start].count('\n')
        line_end = code[:end].count('\n')
        
        # Get context
        context_start = max(0, line_start - context_lines)
        context_end = min(len(lines), line_end + context_lines + 1)
        
        return '\n'.join(lines[context_start:context_end])
    
    def promote_pattern(self, pattern_id: str) -> bool:
        """Promote pattern from V1 to V2 after validation."""
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return False
        
        # Check if pattern meets quality threshold
        if pattern.precision >= 0.85 and pattern.detection_count >= 10:
            pattern.version = "v2"
            pattern.updated_at = datetime.now(timezone.utc)
            logger.info(f"Promoted pattern {pattern_id} to V2")
            return True
        
        return False
    
    def quarantine_pattern(self, pattern_id: str, reason: str) -> bool:
        """Move pattern to V3 quarantine."""
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return False
        
        pattern.version = "v3"
        pattern.updated_at = datetime.now(timezone.utc)
        logger.warning(f"Quarantined pattern {pattern_id}: {reason}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        v1_patterns = sum(1 for p in self.patterns.values() if p.version == "v1")
        v2_patterns = sum(1 for p in self.patterns.values() if p.version == "v2")
        v3_patterns = sum(1 for p in self.patterns.values() if p.version == "v3")
        
        return {
            "total_patterns": len(self.patterns),
            "v1_experimental": v1_patterns,
            "v2_production": v2_patterns,
            "v3_quarantine": v3_patterns,
            "categories": list(self.pattern_index.keys()),
            "total_detections": sum(p.detection_count for p in self.patterns.values()),
        }


# =============================================================================
# Code Similarity Engine
# =============================================================================

class CodeSimilarityEngine:
    """
    Engine for finding similar code patterns.
    
    Uses token-based similarity for fast comparison.
    """
    
    def __init__(self):
        self.code_index: Dict[str, Dict[str, Any]] = {}  # hash -> code data
        self.issue_associations: Dict[str, List[str]] = defaultdict(list)  # issue -> code hashes
        
        logger.info("Code Similarity Engine initialized")
    
    def add_code(
        self,
        code: str,
        issue_type: Optional[str] = None,
        fix_applied: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add code to the similarity index.
        
        Returns:
            Code hash
        """
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        # Tokenize and store
        tokens = self._tokenize(code)
        
        self.code_index[code_hash] = {
            "code": code,
            "tokens": tokens,
            "token_set": set(tokens),
            "issue_type": issue_type,
            "fix_applied": fix_applied,
            "metadata": metadata or {},
        }
        
        if issue_type:
            self.issue_associations[issue_type].append(code_hash)
        
        return code_hash
    
    def find_similar(
        self,
        code: str,
        threshold: float = 0.7,
        limit: int = 10,
    ) -> List[SimilarCode]:
        """
        Find similar code in the index.
        
        Args:
            code: Code to find similar matches for
            threshold: Minimum similarity threshold
            limit: Maximum results to return
            
        Returns:
            List of similar code findings
        """
        if not self.code_index:
            return []
        
        tokens = self._tokenize(code)
        token_set = set(tokens)
        
        similarities = []
        
        for code_hash, data in self.code_index.items():
            indexed_tokens = data["token_set"]
            
            # Jaccard similarity
            intersection = len(token_set & indexed_tokens)
            union = len(token_set | indexed_tokens)
            
            if union > 0:
                similarity = intersection / union
                
                if similarity >= threshold:
                    similarities.append(SimilarCode(
                        code=data["code"],
                        similarity_score=similarity,
                        issue_type=data.get("issue_type"),
                        fix_applied=data.get("fix_applied"),
                    ))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return similarities[:limit]
    
    def find_by_issue_type(
        self,
        issue_type: str,
        limit: int = 10,
    ) -> List[SimilarCode]:
        """Find code examples with a specific issue type."""
        code_hashes = self.issue_associations.get(issue_type, [])
        
        results = []
        for code_hash in code_hashes[:limit]:
            data = self.code_index.get(code_hash)
            if data:
                results.append(SimilarCode(
                    code=data["code"],
                    similarity_score=1.0,
                    issue_type=data.get("issue_type"),
                    fix_applied=data.get("fix_applied"),
                ))
        
        return results
    
    def _tokenize(self, code: str) -> List[str]:
        """Tokenize code for comparison."""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Tokenize
        tokens = re.findall(r'\w+|[^\w\s]', code)
        
        return tokens
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "indexed_code_count": len(self.code_index),
            "issue_types": list(self.issue_associations.keys()),
            "associations_count": sum(len(v) for v in self.issue_associations.values()),
        }


# =============================================================================
# Unified ML Recognition System
# =============================================================================

class MLPatternRecognition:
    """
    Unified ML-based pattern recognition system.
    
    Integrates:
    - Pattern learning engine
    - Code similarity engine
    - Confidence scoring
    - Three-version integration
    """
    
    def __init__(self):
        self.pattern_engine = PatternLearningEngine()
        self.similarity_engine = CodeSimilarityEngine()
        
        # Prediction history for confidence calibration
        self.prediction_history: List[Dict[str, Any]] = []
        
        logger.info("ML Pattern Recognition System initialized")
    
    async def analyze_code(
        self,
        code: str,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze code using ML pattern recognition.
        
        Returns comprehensive analysis including:
        - Detected patterns
        - Similar code examples
        - Confidence scores
        - Suggested fixes
        """
        # Detect patterns
        patterns = self.pattern_engine.detect_patterns(code, categories)
        
        # Find similar code
        similar = self.similarity_engine.find_similar(code, threshold=0.6, limit=5)
        
        # Aggregate analysis
        analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patterns_detected": len(patterns),
            "patterns": [
                {
                    "name": p.pattern_name,
                    "confidence": p.confidence,
                    "location": p.location,
                    "match": p.match_text[:100],
                }
                for p in patterns
            ],
            "similar_code_found": len(similar),
            "similar_examples": [
                {
                    "similarity": s.similarity_score,
                    "issue_type": s.issue_type,
                    "has_fix": s.fix_applied is not None,
                }
                for s in similar
            ],
            "risk_score": self._calculate_risk_score(patterns),
            "recommendations": self._generate_recommendations(patterns, similar),
        }
        
        return analysis
    
    def _calculate_risk_score(self, patterns: List[PatternMatch]) -> float:
        """Calculate overall risk score from patterns."""
        if not patterns:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2,
        }
        
        total_weight = sum(
            severity_weights.get(
                self.pattern_engine.patterns.get(p.pattern_id, CodePattern("", "", "", "")).severity,
                0.3
            ) * p.confidence
            for p in patterns
        )
        
        # Normalize to 0-1
        return min(1.0, total_weight / max(len(patterns), 1))
    
    def _generate_recommendations(
        self,
        patterns: List[PatternMatch],
        similar: List[SimilarCode],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Pattern-based recommendations
        severe_patterns = [
            p for p in patterns
            if self.pattern_engine.patterns.get(p.pattern_id, CodePattern("", "", "", "")).severity in ["critical", "high"]
        ]
        
        if severe_patterns:
            recommendations.append(
                f"Fix {len(severe_patterns)} high-severity pattern(s) before deployment"
            )
        
        # Similar code recommendations
        fixed_examples = [s for s in similar if s.fix_applied]
        if fixed_examples:
            recommendations.append(
                "Similar issues have been fixed before - consider applying proven solutions"
            )
        
        if not recommendations:
            recommendations.append("No critical issues detected - code looks good!")
        
        return recommendations
    
    def learn_from_fix(
        self,
        buggy_code: str,
        fixed_code: str,
        issue_type: str,
    ) -> None:
        """Learn from a successful fix."""
        # Add to similarity index
        self.similarity_engine.add_code(
            code=buggy_code,
            issue_type=issue_type,
            fix_applied=fixed_code,
        )
        
        logger.info(f"Learned from fix: {issue_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            "pattern_engine": self.pattern_engine.get_statistics(),
            "similarity_engine": self.similarity_engine.get_statistics(),
            "prediction_history_size": len(self.prediction_history),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data classes
    "CodePattern",
    "PatternMatch",
    "CodeEmbedding",
    "SimilarCode",
    # Engines
    "PatternLearningEngine",
    "CodeSimilarityEngine",
    # Main system
    "MLPatternRecognition",
]
