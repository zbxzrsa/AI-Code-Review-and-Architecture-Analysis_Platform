"""
Code Review AI Prompts

Comprehensive prompts for user-facing code analysis.
"""

from typing import List, Optional

# System prompt for Code Review AI
CODE_REVIEW_SYSTEM_PROMPT = """
ROLE: Code Review AI - User-Facing Code Analysis Engine

You analyze code for security vulnerabilities, quality issues, performance bottlenecks, 
and architectural problems. Provide actionable, educational feedback.

ANALYSIS DIMENSIONS:
1. Security: OWASP Top 10, CWE Top 25, hardcoded secrets
2. Quality: Complexity, code smells, dead code, naming
3. Performance: Algorithm efficiency, N+1 queries, memory leaks
4. Architecture: SOLID principles, coupling, design patterns
5. Testing: Coverage gaps, test quality

GUIDELINES:
- Be helpful and educational, not judgmental
- Explain WHY issues matter, not just WHAT is wrong
- Provide code examples for fixes
- Acknowledge when suggestions are opinionated
- Celebrate good patterns when found

CONSTRAINTS:
- NEVER execute code (static analysis only)
- NEVER store code beyond session
- NEVER expose credentials if detected
"""

# Analysis prompts by type
SECURITY_PROMPT = """
Analyze for security vulnerabilities:
- SQL/Command/XSS injection
- Authentication/Authorization flaws
- Cryptography weaknesses
- Hardcoded secrets
- Dependency vulnerabilities

Code:
```{language}
{code}
```

Return JSON with findings including severity, CWE ID, location, and fix.
"""

QUALITY_PROMPT = """
Analyze code quality:
- Cyclomatic/cognitive complexity
- Code smells and anti-patterns
- Dead code and unused imports
- Naming conventions
- Error handling

Code:
```{language}
{code}
```

Return JSON with findings, metrics, and quality score.
"""

PERFORMANCE_PROMPT = """
Analyze for performance issues:
- Algorithm inefficiencies
- Database N+1 problems
- Memory leaks
- Resource management
- Blocking operations

Code:
```{language}
{code}
```

Return JSON with findings, current/improved complexity, and optimized code.
"""

ARCHITECTURE_PROMPT = """
Analyze architecture and design:
- SOLID principle violations
- Circular dependencies
- Coupling and cohesion
- Design pattern opportunities
- API design issues

Code:
```{language}
{code}
```

Return JSON with findings, dependency graph, and refactoring suggestions.
"""


def build_review_prompt(
    code: str,
    language: str,
    analysis_types: List[str],
    file_path: str = "code.py",
) -> str:
    """Build comprehensive review prompt."""
    sections = []
    
    for analysis_type in analysis_types:
        if analysis_type == "security":
            sections.append(SECURITY_PROMPT.format(language=language, code=code))
        elif analysis_type == "quality":
            sections.append(QUALITY_PROMPT.format(language=language, code=code))
        elif analysis_type == "performance":
            sections.append(PERFORMANCE_PROMPT.format(language=language, code=code))
        elif analysis_type == "architecture":
            sections.append(ARCHITECTURE_PROMPT.format(language=language, code=code))
    
    return "\n\n---\n\n".join(sections)


def get_model_routing(analysis_type: str) -> str:
    """Get recommended model for analysis type."""
    routing = {
        "security": "claude-sonnet-4",  # Best at threat detection
        "quality": "gpt-4-turbo",       # Good at patterns
        "performance": "gpt-4-turbo",   # Algorithm analysis
        "architecture": "claude-opus-4", # System design
    }
    return routing.get(analysis_type, "gpt-4-turbo")
