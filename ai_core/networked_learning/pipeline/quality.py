"""
Quality Assessment for Collected Data

Evaluates content quality based on:
- Content integrity (completeness, structure)
- Technical relevance (keywords, code ratio)
- Timeliness (freshness, recency)
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from ..collectors.base import CollectedItem, ContentType
from ..config import QualityThresholds

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """
    Quality score breakdown for a collected item.
    
    Attributes:
        overall: Overall quality score (0.0-1.0)
        integrity: Content integrity score
        relevance: Technical relevance score
        timeliness: Content freshness score
        passed: Whether item passed quality threshold
        reasons: Reasons for score or rejection
    """
    overall: float
    integrity: float
    relevance: float
    timeliness: float
    passed: bool
    reasons: List[str]
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "overall": round(self.overall, 3),
            "integrity": round(self.integrity, 3),
            "relevance": round(self.relevance, 3),
            "timeliness": round(self.timeliness, 3),
            "passed": self.passed,
            "reasons": self.reasons,
        }


class QualityAssessor:
    """
    Assesses quality of collected items.
    
    Quality Score Components:
    1. Integrity (40%): Content completeness and structure
    2. Relevance (40%): Technical relevance to software engineering
    3. Timeliness (20%): Content freshness
    
    Threshold: >= 0.8 for acceptance
    """
    
    # Technical keywords for relevance scoring
    TECH_KEYWORDS = {
        "high": {
            "algorithm", "api", "architecture", "async", "authentication",
            "backend", "benchmark", "binary", "cache", "class",
            "compile", "concurrency", "container", "database", "debug",
            "deploy", "docker", "encryption", "framework", "function",
            "git", "http", "interface", "json", "kubernetes",
            "library", "linux", "microservice", "module", "network",
            "optimization", "package", "parallel", "pattern", "performance",
            "protocol", "python", "query", "refactor", "rest",
            "security", "server", "sql", "syntax", "test",
            "thread", "type", "variable", "version", "virtual",
        },
        "medium": {
            "build", "code", "config", "data", "dev",
            "error", "file", "fix", "install", "log",
            "method", "object", "output", "process", "program",
            "project", "run", "script", "service", "software",
            "system", "tool", "update", "user", "web",
        },
    }
    
    # Code patterns
    CODE_PATTERNS = [
        r'```[\s\S]*?```',  # Markdown code blocks
        r'`[^`]+`',  # Inline code
        r'def\s+\w+\s*\(',  # Python function
        r'function\s+\w+\s*\(',  # JavaScript function
        r'class\s+\w+',  # Class definition
        r'import\s+[\w.]+',  # Import statements
        r'\w+\s*=\s*\{',  # Object/dict assignment
        r'if\s*\(.+\)\s*\{',  # If statements
        r'for\s*\(.+\)\s*\{',  # For loops
        r'=>',  # Arrow functions
    ]
    
    def __init__(self, thresholds: QualityThresholds):
        self.thresholds = thresholds
        self._code_regex = re.compile("|".join(self.CODE_PATTERNS), re.IGNORECASE)
    
    def assess(self, item: CollectedItem) -> QualityScore:
        """
        Assess quality of a collected item.
        
        Args:
            item: CollectedItem to assess
            
        Returns:
            QualityScore with detailed breakdown
        """
        reasons = []
        
        # Calculate component scores
        integrity = self._assess_integrity(item, reasons)
        relevance = self._assess_relevance(item, reasons)
        timeliness = self._assess_timeliness(item, reasons)
        
        # Calculate overall score
        overall = self.thresholds.calculate_score(integrity, relevance, timeliness)
        passed = overall >= self.thresholds.min_quality_score
        
        if not passed:
            reasons.append(f"Overall score {overall:.2f} below threshold {self.thresholds.min_quality_score}")
        
        return QualityScore(
            overall=overall,
            integrity=integrity,
            relevance=relevance,
            timeliness=timeliness,
            passed=passed,
            reasons=reasons,
        )
    
    def _assess_integrity(
        self,
        item: CollectedItem,
        reasons: List[str],
    ) -> float:
        """Assess content integrity."""
        score = 1.0
        content = item.content
        
        # Check content length
        if len(content) < self.thresholds.min_content_length:
            score *= 0.3
            reasons.append(f"Content too short ({len(content)} chars)")
        elif len(content) > self.thresholds.max_content_length:
            score *= 0.7
            reasons.append("Content exceeds max length")
        
        # Check for empty/placeholder content
        if not content.strip():
            return 0.0
        
        # Check title presence
        if not item.title or len(item.title) < 5:
            score *= 0.8
            reasons.append("Missing or short title")
        
        # Check for structured content (headings, lists)
        has_structure = any([
            re.search(r'^#+\s', content, re.MULTILINE),  # Markdown headings
            re.search(r'^[-*]\s', content, re.MULTILINE),  # Lists
            re.search(r'^\d+\.\s', content, re.MULTILINE),  # Numbered lists
        ])
        
        if has_structure:
            score = min(1.0, score * 1.1)
        
        # Check for excessive whitespace/formatting issues
        whitespace_ratio = len(re.findall(r'\s', content)) / max(len(content), 1)
        if whitespace_ratio > 0.5:
            score *= 0.8
            reasons.append("Excessive whitespace")
        
        return min(1.0, max(0.0, score))
    
    def _assess_relevance(
        self,
        item: CollectedItem,
        reasons: List[str],
    ) -> float:
        """Assess technical relevance."""
        score = 0.5  # Base score
        content_lower = item.content.lower()
        title_lower = (item.title or "").lower()
        combined = content_lower + " " + title_lower
        
        # Check for technical keywords
        high_keyword_count = sum(1 for kw in self.TECH_KEYWORDS["high"] if kw in combined)
        medium_keyword_count = sum(1 for kw in self.TECH_KEYWORDS["medium"] if kw in combined)
        
        # Keyword scoring
        keyword_score = min(0.3, high_keyword_count * 0.03 + medium_keyword_count * 0.01)
        score += keyword_score
        
        # Check for code content
        code_matches = self._code_regex.findall(item.content)
        if code_matches:
            code_ratio = len("".join(code_matches)) / max(len(item.content), 1)
            if item.content_type == ContentType.CODE:
                # Code content should have high code ratio
                if code_ratio >= self.thresholds.min_code_ratio:
                    score += 0.2
                else:
                    score -= 0.1
            else:
                # Documentation/articles with code examples
                if code_ratio > 0.05:
                    score += 0.1
        
        # Check tags relevance
        tech_tags = {"programming", "coding", "development", "software", "devops",
                     "backend", "frontend", "api", "database", "cloud"}
        matching_tags = set(t.lower() for t in item.tags) & tech_tags
        score += len(matching_tags) * 0.02
        
        # Penalize advertisement-like content
        ad_patterns = [
            r'click here', r'buy now', r'subscribe', r'free trial',
            r'limited time', r'discount', r'coupon', r'affiliate',
        ]
        ad_count = sum(1 for p in ad_patterns if re.search(p, content_lower))
        if ad_count > 2:
            score *= 0.5
            reasons.append("Excessive promotional content")
        
        return min(1.0, max(0.0, score))
    
    def _assess_timeliness(
        self,
        item: CollectedItem,
        reasons: List[str],
    ) -> float:
        """Assess content freshness."""
        now = datetime.now(timezone.utc)
        
        # Use updated_at or created_at
        content_date = item.updated_at or item.created_at
        
        if not content_date:
            # Unknown date - moderate penalty
            reasons.append("Missing date information")
            return 0.6
        
        # Calculate age in days
        age_days = (now - content_date).days
        
        if age_days < 0:
            # Future date - data error
            return 0.3
        elif age_days <= 7:
            return 1.0  # Very fresh
        elif age_days <= 30:
            return 0.9  # Recent
        elif age_days <= 90:
            return 0.8  # Within quarter
        elif age_days <= 365:
            return 0.7  # Within year
        elif age_days <= 730:
            return 0.5  # 1-2 years
        else:
            reasons.append(f"Content is {age_days // 365} years old")
            return max(0.2, 0.5 - (age_days - 730) / 3650)
    
    def batch_assess(
        self,
        items: List[CollectedItem],
    ) -> Dict[str, QualityScore]:
        """
        Assess quality of multiple items.
        
        Args:
            items: List of items to assess
            
        Returns:
            Dictionary mapping item unique_id to QualityScore
        """
        return {item.unique_id: self.assess(item) for item in items}
