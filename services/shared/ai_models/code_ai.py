"""
Code AI - User-Facing AI for Code Review and Analysis

This AI is responsible for:
- Code review and analysis
- Bug detection
- Security vulnerability scanning
- Code optimization suggestions
- Documentation generation

Users can customize their Code AI models through the API
"""

import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json

from .base_ai import BaseAI, AIConfig, VersionType, ModelProvider

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of code analysis"""
    REVIEW = "review"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    REFACTOR = "refactor"
    TEST_GENERATION = "test_generation"


@dataclass
class CodeIssue:
    """A detected code issue"""
    issue_id: str
    type: str
    severity: str  # 'error', 'warning', 'info', 'hint'
    line_start: int
    line_end: int
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    message: str = ""
    suggestion: Optional[str] = None
    fix_available: bool = False
    fix_code: Optional[str] = None


@dataclass
class AnalysisResult:
    """Result of code analysis"""
    analysis_id: str
    analysis_type: AnalysisType
    timestamp: str
    language: str
    issues: List[CodeIssue]
    summary: str
    score: float  # 0-100
    metrics: Dict[str, Any]
    suggestions: List[str]


@dataclass
class UserModelPreference:
    """User's custom model preference"""
    user_id: str
    preferred_provider: ModelProvider
    preferred_model: str
    custom_config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class CodeAI(BaseAI):
    """
    Code AI - User-facing AI for code review and analysis

    Features:
    - Multi-language code analysis
    - Customizable analysis types
    - User-configurable model selection
    - Streaming responses
    - Fix suggestions
    """

    # Analysis prompts for different types
    ANALYSIS_PROMPTS = {
        AnalysisType.REVIEW: """Perform a comprehensive code review. Identify:
- Code quality issues
- Potential bugs
- Logic errors
- Best practice violations
Provide specific line numbers and actionable suggestions.""",

        AnalysisType.SECURITY: """Perform a security analysis. Identify:
- SQL injection vulnerabilities
- XSS vulnerabilities
- Authentication issues
- Data exposure risks
- Input validation problems
Rate severity as critical, high, medium, or low.""",

        AnalysisType.PERFORMANCE: """Analyze code for performance issues:
- Time complexity problems
- Memory inefficiencies
- Unnecessary computations
- Database query optimization
- Caching opportunities""",

        AnalysisType.STYLE: """Check code style and formatting:
- Naming conventions
- Code organization
- Comment quality
- Consistency issues
- Readability improvements""",

        AnalysisType.DOCUMENTATION: """Analyze documentation quality:
- Missing documentation
- Outdated comments
- Incomplete docstrings
- API documentation gaps
Generate improved documentation where needed.""",

        AnalysisType.REFACTOR: """Suggest refactoring improvements:
- Code duplication
- Long methods
- Complex conditionals
- Design pattern opportunities
- SOLID principle violations""",

        AnalysisType.TEST_GENERATION: """Generate comprehensive tests:
- Unit tests for functions
- Edge cases
- Error handling tests
- Integration test suggestions
Use appropriate testing framework for the language."""
    }

    def __init__(
        self,
        config: AIConfig,
        version_type: VersionType,
        user_preference: Optional[UserModelPreference] = None
    ):
        super().__init__(config, version_type)
        self.user_preference = user_preference
        self.analysis_history: List[AnalysisResult] = []

    def set_user_preference(self, preference: UserModelPreference) -> None:
        """Set user's model preference"""
        self.user_preference = preference
        logger.info(f"Set model preference for user {preference.user_id}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response for code-related queries"""
        self.record_request(tokens_used=150, success=True)

        # In production, this would call the actual AI model
        # with user preferences applied
        return f"Code AI ({self.version_type.value}) analysis response"

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response for code analysis"""
        response = await self.generate(prompt, system_prompt, **kwargs)
        for chunk in response.split():
            yield chunk + " "
            await asyncio.sleep(0.02)

    async def analyze_code(
        self,
        code: str,
        language: str,
        analysis_type: str = "review"
    ) -> Dict[str, Any]:
        """
        Analyze code and return structured results

        Args:
            code: Source code to analyze
            language: Programming language
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results with issues and suggestions
        """
        try:
            atype = AnalysisType(analysis_type)
        except ValueError:
            atype = AnalysisType.REVIEW

        # Get analysis prompt
        system_prompt = self.ANALYSIS_PROMPTS.get(atype, self.ANALYSIS_PROMPTS[AnalysisType.REVIEW])

        # Build analysis prompt
        prompt = f"""Language: {language}

Code to analyze:
```{language}
{code}
```

{system_prompt}

Provide your analysis in JSON format with:
- issues: array of detected issues
- summary: brief summary
- score: quality score 0-100
- suggestions: improvement suggestions"""

        # Generate analysis
        response = await self.generate(prompt, system_prompt)

        # Parse and structure results (simplified)
        result = AnalysisResult(
            analysis_id=f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            analysis_type=atype,
            timestamp=datetime.now().isoformat(),
            language=language,
            issues=[],
            summary="Analysis completed",
            score=85.0,
            metrics={
                'lines_analyzed': len(code.split('\n')),
                'language': language,
                'analysis_type': atype.value
            },
            suggestions=["Consider adding more comprehensive tests"]
        )

        self.analysis_history.append(result)
        self.record_request(tokens_used=200, success=True)

        return {
            'analysis_id': result.analysis_id,
            'analysis_type': result.analysis_type.value,
            'timestamp': result.timestamp,
            'language': result.language,
            'issues': [vars(i) for i in result.issues],
            'summary': result.summary,
            'score': result.score,
            'metrics': result.metrics,
            'suggestions': result.suggestions
        }

    async def review_code(
        self,
        code: str,
        language: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive code review"""
        return await self.analyze_code(code, language, "review")

    async def scan_security(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """Scan code for security vulnerabilities"""
        return await self.analyze_code(code, language, "security")

    async def suggest_refactoring(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """Suggest code refactoring improvements"""
        return await self.analyze_code(code, language, "refactor")

    async def generate_tests(
        self,
        code: str,
        language: str,
        framework: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate tests for the given code"""
        result = await self.analyze_code(code, language, "test_generation")
        result['test_framework'] = framework or self._detect_test_framework(language)
        return result

    async def generate_documentation(
        self,
        code: str,
        language: str,
        _style: str = "docstring"  # Reserved for style-specific generation
    ) -> Dict[str, Any]:
        """Generate documentation for code"""
        return await self.analyze_code(code, language, "documentation")

    async def explain_code(
        self,
        code: str,
        language: str,
        detail_level: str = "medium"
    ) -> str:
        """Explain what the code does"""
        prompt = f"""Explain the following {language} code in {detail_level} detail:

```{language}
{code}
```

Provide a clear explanation suitable for developers."""

        return await self.generate(prompt)

    async def fix_issue(
        self,
        code: str,
        issue: CodeIssue,
        language: str
    ) -> Dict[str, Any]:
        """Generate a fix for a specific issue"""
        prompt = f"""Fix the following issue in this {language} code:

Issue: {issue.message}
Location: Line {issue.line_start}-{issue.line_end}
Severity: {issue.severity}

Code:
```{language}
{code}
```

Provide the corrected code and explain the fix."""

        response = await self.generate(prompt)

        return {
            'original_issue': vars(issue),
            'fix_applied': True,
            'explanation': response,
            'corrected_code': code  # In production, parse and return actual fix
        }

    def _detect_test_framework(self, language: str) -> str:
        """Detect appropriate test framework for language"""
        frameworks = {
            'python': 'pytest',
            'javascript': 'jest',
            'typescript': 'jest',
            'java': 'junit',
            'go': 'testing',
            'rust': 'cargo test',
            'csharp': 'xunit',
            'ruby': 'rspec'
        }
        return frameworks.get(language.lower(), 'unittest')

    def get_analysis_history(
        self,
        limit: int = 10,
        analysis_type: Optional[AnalysisType] = None
    ) -> List[Dict[str, Any]]:
        """Get recent analysis history"""
        history = self.analysis_history

        if analysis_type:
            history = [h for h in history if h.analysis_type == analysis_type]

        history = sorted(history, key=lambda x: x.timestamp, reverse=True)[:limit]

        return [
            {
                'analysis_id': h.analysis_id,
                'analysis_type': h.analysis_type.value,
                'timestamp': h.timestamp,
                'language': h.language,
                'score': h.score,
                'issue_count': len(h.issues)
            }
            for h in history
        ]

    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages"""
        return [
            'python', 'javascript', 'typescript', 'java', 'go',
            'rust', 'csharp', 'cpp', 'c', 'ruby', 'php', 'swift',
            'kotlin', 'scala', 'haskell', 'elixir', 'sql', 'shell'
        ]

    def get_supported_analysis_types(self) -> List[str]:
        """Get list of supported analysis types"""
        return [t.value for t in AnalysisType]
