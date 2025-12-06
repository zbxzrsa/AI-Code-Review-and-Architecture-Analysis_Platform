"""
Self-Evolving Bug Fixer AI

Automatically detects, analyzes, and fixes code vulnerabilities and bugs
using the Version Control AI cycle.

Features:
- Static analysis integration
- Pattern-based vulnerability detection
- Automated fix generation
- Self-verification cycle
- Learning from past fixes
"""

import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class BugCategory(str, Enum):
    """Bug category types."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    COMPATIBILITY = "compatibility"


class FixStatus(str, Enum):
    """Fix status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPLIED = "applied"
    VERIFIED = "verified"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class VulnerabilityPattern:
    """Pattern for detecting vulnerabilities."""
    pattern_id: str
    name: str
    description: str
    regex_pattern: str
    category: BugCategory
    severity: Severity
    fix_template: str
    file_extensions: List[str] = field(default_factory=list)
    context_required: int = 5  # Lines of context needed


@dataclass
class DetectedVulnerability:
    """Detected vulnerability instance."""
    vuln_id: str
    pattern_id: str
    file_path: str
    line_number: int
    column: int
    code_snippet: str
    surrounding_context: str
    severity: Severity
    category: BugCategory
    description: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fix_suggestion: Optional[str] = None
    confidence: float = 0.0


@dataclass
class CodeFix:
    """Generated code fix."""
    fix_id: str
    vuln_id: str
    file_path: str
    original_code: str
    fixed_code: str
    fix_description: str
    status: FixStatus = FixStatus.PENDING
    applied_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    verification_result: Optional[Dict[str, Any]] = None
    rollback_reason: Optional[str] = None


@dataclass
class FixResult:
    """Result of applying a fix."""
    fix_id: str
    success: bool
    error: Optional[str] = None
    tests_passed: int = 0
    tests_failed: int = 0
    verification_score: float = 0.0


# =============================================================================
# Vulnerability Patterns Database
# =============================================================================

VULNERABILITY_PATTERNS: List[VulnerabilityPattern] = [
    # Security Patterns
    VulnerabilityPattern(
        pattern_id="SEC-001",
        name="Hardcoded Secret",
        description="Hardcoded secret key or password in code",
        regex_pattern=r'(?:SECRET|PASSWORD|API_KEY|TOKEN)\s*=\s*["\'][^"\']+["\'](?:\s*#.*)?$',
        category=BugCategory.SECURITY,
        severity=Severity.CRITICAL,
        fix_template='''
{var_name} = os.getenv("{env_var}")
if not {var_name}:
    raise ValueError("{env_var} environment variable must be set")
''',
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="SEC-002",
        name="Default Fallback Secret",
        description="Default fallback value for secrets in getenv",
        regex_pattern=r'os\.getenv\(["\'](?:SECRET|KEY|PASSWORD|TOKEN)["\'],\s*["\'][^"\']+["\']\)',
        category=BugCategory.SECURITY,
        severity=Severity.CRITICAL,
        fix_template='''
{var_name} = os.getenv("{env_var}")
if not {var_name}:
    raise ValueError("{env_var} environment variable must be set")
''',
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="SEC-003",
        name="Weak JWT Validation",
        description="JWT decode without proper validation options",
        regex_pattern=r'jwt\.decode\([^)]+\)(?!.*options)',
        category=BugCategory.SECURITY,
        severity=Severity.HIGH,
        fix_template='''
jwt.decode(
    token,
    secret_key,
    algorithms=[algorithm],
    options={{
        "verify_aud": True,
        "verify_iss": True,
        "require": ["exp", "sub", "type", "iat"]
    }},
    audience="code-review-platform",
    issuer="auth-service"
)
''',
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="SEC-004",
        name="Insecure Default Role",
        description="Default user role should be guest, not user",
        regex_pattern=r'\.get\(["\']X-User-Role["\']\s*,\s*["\']user["\']\)',
        category=BugCategory.SECURITY,
        severity=Severity.CRITICAL,
        fix_template='.get("X-User-Role", "guest")',
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="SEC-005",
        name="Weak API Key Validation",
        description="API key validation using only prefix check",
        regex_pattern=r'api_key\.startswith\(["\']',
        category=BugCategory.SECURITY,
        severity=Severity.HIGH,
        fix_template='''
valid_keys = os.getenv("VALID_API_KEYS", "").split(",")
return api_key in [k.strip() for k in valid_keys if k.strip()]
''',
        file_extensions=[".py"],
    ),
    
    # Reliability Patterns
    VulnerabilityPattern(
        pattern_id="REL-001",
        name="Deprecated datetime.utcnow",
        description="Using deprecated datetime.now(timezone.utc) instead of timezone-aware datetime",
        regex_pattern=r'datetime\.utcnow\(\)',
        category=BugCategory.RELIABILITY,
        severity=Severity.MEDIUM,
        fix_template='datetime.now(timezone.utc)',
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="REL-002",
        name="Deprecated get_event_loop",
        description="Using deprecated asyncio.get_event_loop() in async context",
        regex_pattern=r'asyncio\.get_event_loop\(\)',
        category=BugCategory.RELIABILITY,
        severity=Severity.MEDIUM,
        fix_template='asyncio.get_running_loop()',
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="REL-003",
        name="Missing None Check Before Raise",
        description="Raising exception variable that could be None",
        regex_pattern=r'raise\s+last_exception\s*$',
        category=BugCategory.RELIABILITY,
        severity=Severity.MEDIUM,
        fix_template='''
if last_exception:
    raise last_exception
raise RuntimeError("Operation failed with no exception captured")
''',
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="REL-004",
        name="Shallow Copy of Mutable Default",
        description="Using shallow copy for mutable class default",
        regex_pattern=r'self\._\w+\s*=\s*self\.\w+\.copy\(\)',
        category=BugCategory.RELIABILITY,
        severity=Severity.MEDIUM,
        fix_template='self._{attr} = copy.deepcopy(self.{CLASS_ATTR})',
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="REL-005",
        name="Missing Async Lock",
        description="Shared state modification without async lock",
        regex_pattern=r'self\._\w+\[[\w\."\']+\]\s*=',
        category=BugCategory.RELIABILITY,
        severity=Severity.LOW,
        fix_template='''
async with self._lock:
    {original_code}
''',
        file_extensions=[".py"],
    ),
    
    # Performance Patterns
    VulnerabilityPattern(
        pattern_id="PERF-001",
        name="Import Inside Function",
        description="Module import inside function (inefficient)",
        regex_pattern=r'^\s+import\s+\w+',
        category=BugCategory.PERFORMANCE,
        severity=Severity.LOW,
        fix_template="# Move import to module level",
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="PERF-002",
        name="Missing Timeout on Async Call",
        description="Async call without timeout could hang forever",
        regex_pattern=r'await\s+self\._\w+\([^)]*\)(?!.*timeout)',
        category=BugCategory.PERFORMANCE,
        severity=Severity.MEDIUM,
        fix_template='''
await asyncio.wait_for(
    {original_call},
    timeout={timeout}
)
''',
        file_extensions=[".py"],
    ),
    VulnerabilityPattern(
        pattern_id="PERF-003",
        name="Weak Session Key",
        description="Session key using truncated token (not unique enough)",
        regex_pattern=r'session_key\s*=.*\[:20\]',
        category=BugCategory.SECURITY,
        severity=Severity.MEDIUM,
        fix_template='''
token_hash = hashlib.sha256(token.encode()).hexdigest()[:32]
session_key = f"session:{{user_id}}:{{token_hash}}"
''',
        file_extensions=[".py"],
    ),
]


# =============================================================================
# Bug Fixer Engine
# =============================================================================

class BugFixerEngine:
    """
    Core engine for automated bug detection and fixing.
    
    Implements the self-evolving cycle:
    1. Detect vulnerabilities using patterns
    2. Generate fixes using AI or templates
    3. Apply fixes to code
    4. Verify fixes through testing
    5. Learn from results to improve
    """
    
    def __init__(
        self,
        workspace_path: str,
        patterns: Optional[List[VulnerabilityPattern]] = None,
        ai_client: Optional[Any] = None,
        test_runner: Optional[Callable] = None,
    ):
        self.workspace_path = Path(workspace_path)
        self.patterns = patterns or VULNERABILITY_PATTERNS
        self.ai_client = ai_client
        self.test_runner = test_runner
        
        # State
        self._vulnerabilities: Dict[str, DetectedVulnerability] = {}
        self._fixes: Dict[str, CodeFix] = {}
        self._fix_history: List[FixResult] = []
        self._learned_patterns: Dict[str, float] = {}  # pattern_id -> confidence boost
        
        # Locks for thread safety
        self._scan_lock = asyncio.Lock()
        self._fix_lock = asyncio.Lock()
    
    async def scan_codebase(
        self,
        file_paths: Optional[List[str]] = None,
        categories: Optional[List[BugCategory]] = None,
        min_severity: Severity = Severity.LOW,
    ) -> List[DetectedVulnerability]:
        """
        Scan codebase for vulnerabilities.
        
        Args:
            file_paths: Specific files to scan (None = all)
            categories: Filter by category
            min_severity: Minimum severity to report
            
        Returns:
            List of detected vulnerabilities
        """
        async with self._scan_lock:
            vulnerabilities = []
            
            # Get files to scan
            if file_paths:
                files = [Path(p) for p in file_paths]
            else:
                files = self._get_scannable_files()
            
            # Scan each file
            for file_path in files:
                try:
                    file_vulns = await self._scan_file(file_path, categories, min_severity)
                    vulnerabilities.extend(file_vulns)
                except Exception as e:
                    logger.error(f"Error scanning {file_path}: {e}")
            
            # Store results
            for vuln in vulnerabilities:
                self._vulnerabilities[vuln.vuln_id] = vuln
            
            logger.info(f"Scan complete: {len(vulnerabilities)} vulnerabilities found")
            return vulnerabilities
    
    def _get_scannable_files(self) -> List[Path]:
        """Get all scannable files in workspace."""
        files = []
        extensions = {ext for p in self.patterns for ext in p.file_extensions}
        
        for ext in extensions:
            files.extend(self.workspace_path.rglob(f"*{ext}"))
        
        # Filter out common exclusions
        exclusions = {".venv", "node_modules", "__pycache__", ".git", "dist", "build"}
        files = [
            f for f in files
            if not any(exc in str(f) for exc in exclusions)
        ]
        
        return files
    
    def _scan_file(
        self,
        file_path: Path,
        categories: Optional[List[BugCategory]],
        min_severity: Severity,
    ) -> List[DetectedVulnerability]:
        """Scan single file for vulnerabilities."""
        vulnerabilities = []
        
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
        except Exception as e:
            # Avoid logging user-controlled data directly - sanitize file path
            safe_path = str(file_path.name) if hasattr(file_path, 'name') else "[file]"
            logger.warning("Could not read file %s: %s", safe_path, str(e))
            return []
        
        # Get applicable patterns
        ext = file_path.suffix
        applicable_patterns = [
            p for p in self.patterns
            if (not p.file_extensions or ext in p.file_extensions)
            and (not categories or p.category in categories)
            and self._severity_value(p.severity) >= self._severity_value(min_severity)
        ]
        
        # Check each pattern
        for pattern in applicable_patterns:
            try:
                regex = re.compile(pattern.regex_pattern, re.MULTILINE | re.IGNORECASE)
                
                for i, line in enumerate(lines, 1):
                    matches = list(regex.finditer(line))
                    for match in matches:
                        # Get surrounding context
                        start = max(0, i - pattern.context_required - 1)
                        end = min(len(lines), i + pattern.context_required)
                        context = "\n".join(lines[start:end])
                        
                        # Calculate confidence with learned adjustments
                        base_confidence = 0.85
                        learned_boost = self._learned_patterns.get(pattern.pattern_id, 0)
                        confidence = min(1.0, base_confidence + learned_boost)
                        
                        vuln = DetectedVulnerability(
                            vuln_id=str(uuid.uuid4()),
                            pattern_id=pattern.pattern_id,
                            file_path=str(file_path),
                            line_number=i,
                            column=match.start() + 1,
                            code_snippet=line.strip(),
                            surrounding_context=context,
                            severity=pattern.severity,
                            category=pattern.category,
                            description=pattern.description,
                            fix_suggestion=pattern.fix_template,
                            confidence=confidence,
                        )
                        vulnerabilities.append(vuln)
                        
            except re.error as e:
                logger.error(f"Invalid regex in pattern {pattern.pattern_id}: {e}")
        
        return vulnerabilities
    
    def _severity_value(self, severity: Severity) -> int:
        """Get numeric value for severity comparison."""
        return {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
            Severity.INFO: 0,
        }.get(severity, 0)
    
    async def generate_fixes(
        self,
        vuln_ids: Optional[List[str]] = None,
        auto_apply: bool = False,
    ) -> List[CodeFix]:
        """
        Generate fixes for detected vulnerabilities.
        
        Args:
            vuln_ids: Specific vulnerabilities to fix (None = all)
            auto_apply: Whether to automatically apply fixes
            
        Returns:
            List of generated fixes
        """
        fixes = []
        
        vulns = [
            self._vulnerabilities[vid]
            for vid in (vuln_ids or self._vulnerabilities.keys())
            if vid in self._vulnerabilities
        ]
        
        for vuln in vulns:
            try:
                fix = await self._generate_fix(vuln)
                if fix:
                    fixes.append(fix)
                    self._fixes[fix.fix_id] = fix
                    
                    if auto_apply:
                        await self.apply_fix(fix.fix_id)
                        
            except Exception as e:
                logger.error(f"Error generating fix for {vuln.vuln_id}: {e}")
        
        return fixes
    
    async def _generate_fix(self, vuln: DetectedVulnerability) -> Optional[CodeFix]:
        """Generate fix for single vulnerability."""
        pattern = next(
            (p for p in self.patterns if p.pattern_id == vuln.pattern_id),
            None
        )
        
        if not pattern:
            return None
        
        # Read current file content
        try:
            file_path = Path(vuln.file_path)
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
        except Exception as e:
            logger.error(f"Could not read {vuln.file_path}: {e}")
            return None
        
        # Get original code block
        original_code = lines[vuln.line_number - 1]
        
        # Generate fixed code based on pattern
        fixed_code = self._apply_fix_template(
            pattern,
            original_code,
            vuln.surrounding_context,
        )
        
        if not fixed_code or fixed_code == original_code:
            # Try AI-based fix generation if available
            if self.ai_client:
                fixed_code = await self._generate_ai_fix(vuln, pattern)
            
            if not fixed_code:
                return None
        
        return CodeFix(
            fix_id=str(uuid.uuid4()),
            vuln_id=vuln.vuln_id,
            file_path=vuln.file_path,
            original_code=original_code,
            fixed_code=fixed_code,
            fix_description=f"Fix {pattern.name}: {pattern.description}",
        )
    
    def _apply_fix_template(
        self,
        pattern: VulnerabilityPattern,
        original_code: str,
        context: str,
    ) -> str:
        """Apply fix template to generate fixed code."""
        template = pattern.fix_template
        
        # Simple replacements for common patterns
        if pattern.pattern_id == "REL-001":
            # datetime.now(timezone.utc) -> datetime.now(timezone.utc)
            return original_code.replace("datetime.now(timezone.utc)", "datetime.now(timezone.utc)")
        
        elif pattern.pattern_id == "REL-002":
            # asyncio.get_event_loop() -> asyncio.get_running_loop()
            return original_code.replace(
                "asyncio.get_event_loop()",
                "asyncio.get_running_loop()"
            )
        
        elif pattern.pattern_id == "SEC-004":
            # Default role user -> guest
            return original_code.replace(
                '.get("X-User-Role", "user")',
                '.get("X-User-Role", "guest")'
            ).replace(
                ".get('X-User-Role', 'user')",
                ".get('X-User-Role', 'guest')"
            )
        
        # For complex patterns, return template for review
        return template
    
    async def _generate_ai_fix(
        self,
        vuln: DetectedVulnerability,
        pattern: VulnerabilityPattern,
    ) -> Optional[str]:
        """Use AI to generate fix."""
        if not self.ai_client:
            return None
        
        prompt = f"""
        Fix the following vulnerability:
        
        Pattern: {pattern.name}
        Description: {pattern.description}
        Severity: {vuln.severity.value}
        
        Code snippet:
        ```
        {vuln.surrounding_context}
        ```
        
        Line to fix:
        {vuln.code_snippet}
        
        Fix template hint:
        {pattern.fix_template}
        
        Provide ONLY the fixed line of code, no explanation.
        """
        
        try:
            response = await self.ai_client.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"AI fix generation failed: {e}")
            return None
    
    async def apply_fix(
        self,
        fix_id: str,
        backup: bool = True,
    ) -> FixResult:
        """
        Apply a generated fix to the codebase.
        
        Args:
            fix_id: ID of fix to apply
            backup: Whether to create backup before applying
            
        Returns:
            Result of applying the fix
        """
        async with self._fix_lock:
            fix = self._fixes.get(fix_id)
            if not fix:
                return FixResult(fix_id=fix_id, success=False, error="Fix not found")
            
            try:
                file_path = Path(fix.file_path)
                content = file_path.read_text(encoding="utf-8")
                
                # Create backup
                if backup:
                    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
                    backup_path.write_text(content, encoding="utf-8")
                
                # Apply fix
                new_content = content.replace(fix.original_code, fix.fixed_code, 1)
                
                if new_content == content:
                    return FixResult(
                        fix_id=fix_id,
                        success=False,
                        error="Original code not found in file"
                    )
                
                file_path.write_text(new_content, encoding="utf-8")
                
                fix.status = FixStatus.APPLIED
                fix.applied_at = datetime.now(timezone.utc)
                
                logger.info(f"Applied fix {fix_id} to {fix.file_path}")
                
                # Verify if test runner available
                if self.test_runner:
                    result = await self.verify_fix(fix_id)
                    return result
                
                return FixResult(fix_id=fix_id, success=True)
                
            except Exception as e:
                fix.status = FixStatus.FAILED
                logger.error(f"Failed to apply fix {fix_id}: {e}")
                return FixResult(fix_id=fix_id, success=False, error=str(e))
    
    async def verify_fix(self, fix_id: str) -> FixResult:
        """
        Verify that a fix doesn't break anything.
        
        Runs tests and checks for regressions.
        """
        fix = self._fixes.get(fix_id)
        if not fix:
            return FixResult(fix_id=fix_id, success=False, error="Fix not found")
        
        if not self.test_runner:
            return FixResult(fix_id=fix_id, success=True, verification_score=0.5)
        
        try:
            # Run tests
            test_result = await self.test_runner(fix.file_path)
            
            tests_passed = test_result.get("passed", 0)
            tests_failed = test_result.get("failed", 0)
            
            if tests_failed == 0:
                fix.status = FixStatus.VERIFIED
                fix.verified_at = datetime.now(timezone.utc)
                fix.verification_result = test_result
                
                # Learn from successful fix
                self._learn_from_fix(fix, success=True)
                
                return FixResult(
                    fix_id=fix_id,
                    success=True,
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                    verification_score=1.0,
                )
            else:
                # Rollback
                await self.rollback_fix(fix_id, "Tests failed after applying fix")
                
                return FixResult(
                    fix_id=fix_id,
                    success=False,
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                    verification_score=tests_passed / (tests_passed + tests_failed),
                    error=f"{tests_failed} tests failed",
                )
                
        except Exception as e:
            logger.error(f"Verification failed for fix {fix_id}: {e}")
            return FixResult(fix_id=fix_id, success=False, error=str(e))
    
    def rollback_fix(self, fix_id: str, reason: str) -> bool:
        """Rollback an applied fix."""
        fix = self._fixes.get(fix_id)
        if not fix or fix.status != FixStatus.APPLIED:
            return False
        
        try:
            file_path = Path(fix.file_path)
            backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
            
            if backup_path.exists():
                content = backup_path.read_text(encoding="utf-8")
                file_path.write_text(content, encoding="utf-8")
                backup_path.unlink()
                
                fix.status = FixStatus.ROLLED_BACK
                fix.rollback_reason = reason
                
                # Learn from failed fix
                self._learn_from_fix(fix, success=False)
                
                logger.info(f"Rolled back fix {fix_id}: {reason}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Rollback failed for fix {fix_id}: {e}")
            return False
    
    def _learn_from_fix(self, fix: CodeFix, success: bool):
        """Learn from fix result to improve future detection."""
        vuln = self._vulnerabilities.get(fix.vuln_id)
        if not vuln:
            return
        
        pattern_id = vuln.pattern_id
        
        # Adjust confidence for pattern
        current = self._learned_patterns.get(pattern_id, 0)
        adjustment = 0.02 if success else -0.05
        self._learned_patterns[pattern_id] = max(-0.2, min(0.15, current + adjustment))
        
        # Record in history
        self._fix_history.append(FixResult(
            fix_id=fix.fix_id,
            success=success,
            verification_score=1.0 if success else 0.0,
        ))
    
    def get_vulnerability_summary(self) -> Dict[str, Any]:
        """Get summary of detected vulnerabilities."""
        by_severity = {}
        by_category = {}
        by_file = {}
        
        for vuln in self._vulnerabilities.values():
            # By severity
            sev = vuln.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
            
            # By category
            cat = vuln.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            
            # By file
            file = vuln.file_path
            by_file[file] = by_file.get(file, 0) + 1
        
        return {
            "total": len(self._vulnerabilities),
            "by_severity": by_severity,
            "by_category": by_category,
            "by_file": by_file,
            "fixes_generated": len(self._fixes),
            "fixes_applied": sum(1 for f in self._fixes.values() if f.status == FixStatus.APPLIED),
            "fixes_verified": sum(1 for f in self._fixes.values() if f.status == FixStatus.VERIFIED),
        }
    
    def get_fix_report(self) -> List[Dict[str, Any]]:
        """Get detailed report of all fixes."""
        return [
            {
                "fix_id": fix.fix_id,
                "vuln_id": fix.vuln_id,
                "file": fix.file_path,
                "status": fix.status.value,
                "description": fix.fix_description,
                "applied_at": fix.applied_at.isoformat() if fix.applied_at else None,
                "verified_at": fix.verified_at.isoformat() if fix.verified_at else None,
                "original": fix.original_code[:100],
                "fixed": fix.fixed_code[:100],
            }
            for fix in self._fixes.values()
        ]


# =============================================================================
# Automated Bug Fix Cycle
# =============================================================================

class AutoFixCycle:
    """
    Automated bug fixing cycle that runs continuously.
    
    Integrates with Version Control AI for self-evolution.
    """
    
    def __init__(
        self,
        bug_fixer: BugFixerEngine,
        scan_interval: int = 3600,  # 1 hour
        auto_fix_severity: Severity = Severity.CRITICAL,
        max_auto_fixes_per_cycle: int = 10,
    ):
        self.bug_fixer = bug_fixer
        self.scan_interval = scan_interval
        self.auto_fix_severity = auto_fix_severity
        self.max_auto_fixes_per_cycle = max_auto_fixes_per_cycle
        
        self._running = False
        self._cycle_task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._cycle_results: List[Dict[str, Any]] = []
    
    async def start(self):
        """Start the automated fix cycle."""
        if self._running:
            return
        
        self._running = True
        self._cycle_task = asyncio.create_task(self._run_cycle())
        logger.info("Automated bug fix cycle started")
    
    async def stop(self):
        """Stop the automated fix cycle."""
        self._running = False
        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                raise  # Re-raise CancelledError after cleanup
        logger.info("Automated bug fix cycle stopped")
    
    async def _run_cycle(self):
        """Main cycle loop."""
        while self._running:
            try:
                self._cycle_count += 1
                cycle_result = await self._execute_cycle()
                self._cycle_results.append(cycle_result)
                
                # Keep only last 100 results
                if len(self._cycle_results) > 100:
                    self._cycle_results = self._cycle_results[-100:]
                
            except Exception as e:
                logger.error(f"Cycle {self._cycle_count} failed: {e}")
            
            await asyncio.sleep(self.scan_interval)
    
    async def _execute_cycle(self) -> Dict[str, Any]:
        """Execute single fix cycle."""
        start_time = datetime.now(timezone.utc)
        
        # Step 1: Scan for vulnerabilities
        vulns = await self.bug_fixer.scan_codebase(
            min_severity=Severity.LOW,
        )
        
        # Step 2: Filter by auto-fix severity
        auto_fix_vulns = [
            v for v in vulns
            if self.bug_fixer._severity_value(v.severity) >= 
               self.bug_fixer._severity_value(self.auto_fix_severity)
        ][:self.max_auto_fixes_per_cycle]
        
        # Step 3: Generate and apply fixes
        fixes_applied = 0
        fixes_verified = 0
        
        for vuln in auto_fix_vulns:
            fixes = await self.bug_fixer.generate_fixes(
                vuln_ids=[vuln.vuln_id],
                auto_apply=True,
            )
            
            for fix in fixes:
                if fix.status == FixStatus.APPLIED:
                    fixes_applied += 1
                if fix.status == FixStatus.VERIFIED:
                    fixes_verified += 1
        
        end_time = datetime.now(timezone.utc)
        
        result = {
            "cycle": self._cycle_count,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "vulnerabilities_found": len(vulns),
            "auto_fix_candidates": len(auto_fix_vulns),
            "fixes_applied": fixes_applied,
            "fixes_verified": fixes_verified,
        }
        
        logger.info(f"Cycle {self._cycle_count} complete: {result}")
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get cycle status."""
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "scan_interval": self.scan_interval,
            "auto_fix_severity": self.auto_fix_severity.value,
            "recent_cycles": self._cycle_results[-10:],
            "summary": self.bug_fixer.get_vulnerability_summary(),
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_bug_fixer(workspace_path: str, **kwargs) -> BugFixerEngine:
    """Create a bug fixer engine instance."""
    return BugFixerEngine(workspace_path=workspace_path, **kwargs)


def create_auto_fix_cycle(
    workspace_path: str,
    scan_interval: int = 3600,
    **kwargs
) -> AutoFixCycle:
    """Create an automated fix cycle."""
    bug_fixer = create_bug_fixer(workspace_path, **kwargs)
    return AutoFixCycle(
        bug_fixer=bug_fixer,
        scan_interval=scan_interval,
    )
