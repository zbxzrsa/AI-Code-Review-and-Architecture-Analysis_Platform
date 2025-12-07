"""
Code Review Business Logic Service

Handles all code review operations including:
- Code analysis
- Issue detection
- Fix suggestions
- Review history

Module Size: ~250 lines (target < 2000)
"""

import re
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from ..config import logger, MOCK_MODE


class IssueSeverity(str, Enum):
    """Issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class IssueCategory(str, Enum):
    """Issue categories."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    BUG = "bug"
    MAINTAINABILITY = "maintainability"
    BEST_PRACTICE = "best_practice"


@dataclass
class CodeIssue:
    """Represents a code issue found during review."""
    id: str
    line: int
    column: int
    end_line: Optional[int]
    end_column: Optional[int]
    severity: IssueSeverity
    category: IssueCategory
    message: str
    rule_id: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class ReviewResult:
    """Code review result."""
    review_id: str
    file_path: str
    language: str
    issues: List[CodeIssue]
    score: float  # 0-100
    summary: str
    reviewed_at: str
    duration_ms: int
    lines_analyzed: int


class CodeReviewService:
    """
    Service for code review operations.
    
    Provides:
    - Static code analysis
    - Pattern-based issue detection
    - AI-powered suggestions (when available)
    - Review history tracking
    """
    
    def __init__(self):
        self.review_history: Dict[str, ReviewResult] = {}
        self._pattern_rules = self._load_pattern_rules()
    
    def _load_pattern_rules(self) -> List[Dict[str, Any]]:
        """Load pattern-based detection rules."""
        return [
            {
                "id": "SEC001",
                "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
                "category": IssueCategory.SECURITY,
                "severity": IssueSeverity.ERROR,
                "message": "Hardcoded password detected",
                "suggestion": "Use environment variables or secrets management",
            },
            {
                "id": "SEC002",
                "pattern": r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
                "category": IssueCategory.SECURITY,
                "severity": IssueSeverity.ERROR,
                "message": "Hardcoded API key detected",
                "suggestion": "Store API keys in environment variables",
            },
            {
                "id": "SEC003",
                "pattern": r"eval\s*\(",
                "category": IssueCategory.SECURITY,
                "severity": IssueSeverity.WARNING,
                "message": "Use of eval() is dangerous",
                "suggestion": "Avoid eval() or use safer alternatives",
            },
            {
                "id": "PERF001",
                "pattern": r"SELECT\s+\*\s+FROM",
                "category": IssueCategory.PERFORMANCE,
                "severity": IssueSeverity.WARNING,
                "message": "SELECT * can be inefficient",
                "suggestion": "Select only needed columns",
            },
            {
                "id": "STYLE001",
                "pattern": r"^\s{0,3}(if|for|while|def|class)\s+.+:\s*$",
                "category": IssueCategory.STYLE,
                "severity": IssueSeverity.INFO,
                "message": "Consider adding docstring",
                "suggestion": "Add documentation for better maintainability",
            },
            {
                "id": "BUG001",
                "pattern": r"except\s*:\s*$",
                "category": IssueCategory.BUG,
                "severity": IssueSeverity.WARNING,
                "message": "Bare except clause catches all exceptions",
                "suggestion": "Specify the exception type to catch",
            },
        ]
    
    async def analyze_code(
        self,
        code: str,
        language: str,
        file_path: str = "untitled",
        include_ai: bool = True,
    ) -> ReviewResult:
        """
        Analyze code and return review results.
        
        Args:
            code: Source code to analyze
            language: Programming language
            file_path: File path for context
            include_ai: Include AI-powered analysis
            
        Returns:
            ReviewResult with issues and score
        """
        import time
        start_time = time.time()
        
        review_id = hashlib.md5(f"{code}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        issues: List[CodeIssue] = []
        
        # Pattern-based analysis
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for rule in self._pattern_rules:
                if re.search(rule["pattern"], line, re.IGNORECASE):
                    issue = CodeIssue(
                        id=f"{review_id}-{len(issues)+1}",
                        line=line_num,
                        column=1,
                        end_line=line_num,
                        end_column=len(line),
                        severity=rule["severity"],
                        category=rule["category"],
                        message=rule["message"],
                        rule_id=rule["id"],
                        suggestion=rule.get("suggestion"),
                        auto_fixable=rule.get("auto_fixable", False),
                    )
                    issues.append(issue)
        
        # Mock AI analysis if enabled
        if include_ai and MOCK_MODE:
            issues.extend(self._mock_ai_issues(review_id, len(lines)))
        
        # Calculate score
        score = self._calculate_score(issues, len(lines))
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        result = ReviewResult(
            review_id=review_id,
            file_path=file_path,
            language=language,
            issues=issues,
            score=score,
            summary=self._generate_summary(issues, score),
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            lines_analyzed=len(lines),
        )
        
        # Store in history
        self.review_history[review_id] = result
        
        logger.info(f"Code review completed: {review_id}, {len(issues)} issues, score: {score}")
        
        return result
    
    def _mock_ai_issues(self, review_id: str, total_lines: int) -> List[CodeIssue]:
        """Generate mock AI-detected issues."""
        return [
            CodeIssue(
                id=f"{review_id}-ai-1",
                line=min(15, total_lines),
                column=1,
                end_line=min(15, total_lines),
                end_column=50,
                severity=IssueSeverity.INFO,
                category=IssueCategory.MAINTAINABILITY,
                message="Consider extracting this logic into a separate function",
                rule_id="AI-MAINT-001",
                suggestion="Create a helper function to improve readability",
            ),
        ]
    
    def _calculate_score(self, issues: List[CodeIssue], lines: int) -> float:
        """Calculate review score based on issues."""
        if lines == 0:
            return 100.0
        
        penalty = 0
        for issue in issues:
            if issue.severity == IssueSeverity.ERROR:
                penalty += 15
            elif issue.severity == IssueSeverity.WARNING:
                penalty += 5
            elif issue.severity == IssueSeverity.INFO:
                penalty += 1
        
        score = max(0, 100 - penalty)
        return round(score, 1)
    
    def _generate_summary(self, issues: List[CodeIssue], score: float) -> str:
        """Generate human-readable summary."""
        if not issues:
            return "No issues found. Code looks good!"
        
        error_count = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
        warning_count = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)
        
        parts = []
        if error_count > 0:
            parts.append(f"{error_count} error(s)")
        if warning_count > 0:
            parts.append(f"{warning_count} warning(s)")
        
        return f"Found {', '.join(parts)}. Score: {score}/100"
    
    def get_review(self, review_id: str) -> Optional[ReviewResult]:
        """Get review result by ID."""
        return self.review_history.get(review_id)
    
    def list_reviews(self, limit: int = 10) -> List[ReviewResult]:
        """List recent reviews."""
        reviews = list(self.review_history.values())
        reviews.sort(key=lambda r: r.reviewed_at, reverse=True)
        return reviews[:limit]
