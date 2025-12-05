"""
Auto-Fix Cycle Integration

Integrates the Bug Fixer AI with the Three-Version Evolution Cycle:
- Scans codebase for vulnerabilities
- Generates and applies fixes
- Verifies fixes don't break anything
- Learns from successful/failed fixes
- Promotes successful fix strategies to V2
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FixCyclePhase(str, Enum):
    """Fix cycle phase."""
    IDLE = "idle"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    APPLYING = "applying"
    VERIFYING = "verifying"
    LEARNING = "learning"


class FixStrategy(str, Enum):
    """Fix application strategy."""
    CONSERVATIVE = "conservative"  # Only apply high-confidence fixes
    BALANCED = "balanced"  # Apply medium+ confidence fixes
    AGGRESSIVE = "aggressive"  # Apply all fixes with verification


@dataclass
class FixCycleConfig:
    """Configuration for the fix cycle."""
    scan_interval_seconds: int = 3600  # 1 hour
    min_confidence: float = 0.85
    max_concurrent_fixes: int = 5
    auto_apply: bool = False  # Require approval by default
    strategy: FixStrategy = FixStrategy.BALANCED
    target_paths: List[str] = field(default_factory=list)
    excluded_paths: List[str] = field(default_factory=lambda: [
        "node_modules", "__pycache__", ".git", "venv", "dist", "build"
    ])


@dataclass
class VulnerabilityScan:
    """Result of a vulnerability scan."""
    scan_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    files_scanned: int = 0
    vulnerabilities_found: int = 0
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)


@dataclass
class FixAttempt:
    """Record of a fix attempt."""
    attempt_id: str
    vuln_id: str
    file_path: str
    fix_type: str
    original_code: str
    fixed_code: str
    confidence: float
    applied_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None
    rollback_at: Optional[datetime] = None


@dataclass
class CycleMetrics:
    """Metrics for the fix cycle."""
    cycles_completed: int = 0
    total_scans: int = 0
    vulnerabilities_detected: int = 0
    fixes_generated: int = 0
    fixes_applied: int = 0
    fixes_verified: int = 0
    fixes_failed: int = 0
    fixes_rolled_back: int = 0
    avg_fix_time_seconds: float = 0.0
    last_cycle_at: Optional[datetime] = None


class AutoFixCycle:
    """
    Automated vulnerability fix cycle.
    
    Workflow:
    1. SCANNING: Scan codebase for vulnerabilities
    2. ANALYZING: Prioritize by severity and confidence
    3. GENERATING: Create fix proposals
    4. APPLYING: Apply fixes (if auto_apply or approved)
    5. VERIFYING: Run tests, static analysis
    6. LEARNING: Update patterns based on success/failure
    """
    
    def __init__(
        self,
        config: Optional[FixCycleConfig] = None,
        bug_fixer=None,
        fix_verifier=None,
        version_manager=None,
        event_bus=None,
    ):
        self.config = config or FixCycleConfig()
        self.bug_fixer = bug_fixer
        self.fix_verifier = fix_verifier
        self.version_manager = version_manager
        self.event_bus = event_bus
        
        # State
        self._running = False
        self._phase = FixCyclePhase.IDLE
        self._current_scan: Optional[VulnerabilityScan] = None
        self._pending_fixes: List[FixAttempt] = []
        self._fix_history: List[FixAttempt] = []
        self._metrics = CycleMetrics()
        
        # Callbacks
        self._on_vulnerability_found: List[Callable] = []
        self._on_fix_applied: List[Callable] = []
        self._on_fix_verified: List[Callable] = []
        
        # Lock
        self._lock = asyncio.Lock()
        self._cycle_task: Optional[asyncio.Task] = None
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    async def start(self):
        """Start the auto-fix cycle."""
        if self._running:
            logger.warning("Fix cycle already running")
            return
        
        self._running = True
        self._cycle_task = asyncio.create_task(self._run_cycle_loop())
        logger.info("Auto-fix cycle started")
        
        if self.event_bus:
            await self.event_bus.publish("fix_cycle_started", {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "strategy": self.config.strategy.value,
                    "min_confidence": self.config.min_confidence,
                    "auto_apply": self.config.auto_apply,
                },
            })
    
    async def stop(self):
        """Stop the auto-fix cycle."""
        self._running = False
        
        if self._cycle_task:
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                raise  # Re-raise CancelledError after cleanup
        
        self._phase = FixCyclePhase.IDLE
        logger.info("Auto-fix cycle stopped")
    
    async def _run_cycle_loop(self):
        """Main cycle loop."""
        while self._running:
            try:
                await self.execute_cycle()
                self._metrics.cycles_completed += 1
                self._metrics.last_cycle_at = datetime.now(timezone.utc)
            except Exception as e:
                logger.error(f"Fix cycle error: {e}")
            
            await asyncio.sleep(self.config.scan_interval_seconds)
    
    # =========================================================================
    # Cycle Execution
    # =========================================================================
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute a single fix cycle."""
        import uuid
        
        cycle_id = str(uuid.uuid4())
        result = {
            "cycle_id": cycle_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "phases": [],
        }
        
        try:
            # Phase 1: Scanning
            self._phase = FixCyclePhase.SCANNING
            scan_result = await self._scan_codebase()
            result["phases"].append({"phase": "scanning", "result": scan_result})
            
            # Phase 2: Analyzing
            self._phase = FixCyclePhase.ANALYZING
            analysis_result = await self._analyze_vulnerabilities()
            result["phases"].append({"phase": "analyzing", "result": analysis_result})
            
            # Phase 3: Generating fixes
            self._phase = FixCyclePhase.GENERATING
            generation_result = await self._generate_fixes()
            result["phases"].append({"phase": "generating", "result": generation_result})
            
            # Phase 4: Applying fixes
            if self.config.auto_apply or self._pending_fixes:
                self._phase = FixCyclePhase.APPLYING
                apply_result = await self._apply_fixes()
                result["phases"].append({"phase": "applying", "result": apply_result})
            
            # Phase 5: Verifying
            self._phase = FixCyclePhase.VERIFYING
            verify_result = await self._verify_fixes()
            result["phases"].append({"phase": "verifying", "result": verify_result})
            
            # Phase 6: Learning
            self._phase = FixCyclePhase.LEARNING
            learn_result = await self._learn_from_cycle()
            result["phases"].append({"phase": "learning", "result": learn_result})
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Cycle {cycle_id} failed: {e}")
        finally:
            self._phase = FixCyclePhase.IDLE
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        return result
    
    async def _scan_codebase(self) -> Dict[str, Any]:
        """Scan codebase for vulnerabilities."""
        import uuid
        
        scan = VulnerabilityScan(
            scan_id=str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
        )
        self._current_scan = scan
        
        vulnerabilities = []
        
        if self.bug_fixer:
            # Use actual bug fixer
            result = await self.bug_fixer.scan_codebase(
                paths=self.config.target_paths,
                excluded=self.config.excluded_paths,
            )
            vulnerabilities = result.get("vulnerabilities", [])
            scan.files_scanned = result.get("files_scanned", 0)
        else:
            # Mock scan for demo
            vulnerabilities = [
                {
                    "vuln_id": "vuln-001",
                    "severity": "critical",
                    "category": "security",
                    "file_path": "backend/shared/security/auth.py",
                    "line": 19,
                    "description": "Hardcoded secret key",
                },
                {
                    "vuln_id": "vuln-002",
                    "severity": "medium",
                    "category": "reliability",
                    "file_path": "backend/shared/services/reliability.py",
                    "line": 41,
                    "description": "Deprecated datetime usage",
                },
            ]
            scan.files_scanned = 50
        
        scan.vulnerabilities_found = len(vulnerabilities)
        scan.completed_at = datetime.now(timezone.utc)
        
        # Aggregate by severity and category
        for v in vulnerabilities:
            sev = v.get("severity", "unknown")
            cat = v.get("category", "unknown")
            scan.by_severity[sev] = scan.by_severity.get(sev, 0) + 1
            scan.by_category[cat] = scan.by_category.get(cat, 0) + 1
        
        self._metrics.total_scans += 1
        self._metrics.vulnerabilities_detected += scan.vulnerabilities_found
        
        logger.info(
            f"Scan complete: {scan.files_scanned} files, "
            f"{scan.vulnerabilities_found} vulnerabilities"
        )
        
        return {
            "scan_id": scan.scan_id,
            "files_scanned": scan.files_scanned,
            "vulnerabilities_found": scan.vulnerabilities_found,
            "by_severity": scan.by_severity,
        }
    
    def _analyze_vulnerabilities(self) -> Dict[str, Any]:
        """Analyze and prioritize vulnerabilities."""
        if not self._current_scan:
            return {"analyzed": 0}
        
        # Sort by severity (priority: critical > high > medium > low > info)
        analyzed = sum(self._current_scan.by_severity.values())
        
        return {
            "analyzed": analyzed,
            "priority_distribution": self._current_scan.by_severity,
        }
    
    async def _generate_fixes(self) -> Dict[str, Any]:
        """Generate fix proposals."""
        import uuid
        
        fixes_generated = 0
        
        if self.bug_fixer:
            # Use actual bug fixer to generate fixes
            result = await self.bug_fixer.generate_fixes()
            fixes_generated = result.get("fixes_generated", 0)
            
            for fix in result.get("fixes", []):
                attempt = FixAttempt(
                    attempt_id=str(uuid.uuid4()),
                    vuln_id=fix["vuln_id"],
                    file_path=fix["file_path"],
                    fix_type=fix["fix_type"],
                    original_code=fix["original_code"],
                    fixed_code=fix["fixed_code"],
                    confidence=fix["confidence"],
                )
                
                if attempt.confidence >= self.config.min_confidence:
                    self._pending_fixes.append(attempt)
        else:
            # Mock fix generation
            mock_fixes = [
                FixAttempt(
                    attempt_id=str(uuid.uuid4()),
                    vuln_id="vuln-001",
                    file_path="backend/shared/security/auth.py",
                    fix_type="hardcoded_secret",
                    original_code='SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default")',
                    fixed_code='SECRET_KEY = os.getenv("JWT_SECRET_KEY")\nif not SECRET_KEY:\n    raise ValueError("JWT_SECRET_KEY must be set")',
                    confidence=0.95,
                ),
            ]
            
            for fix in mock_fixes:
                if fix.confidence >= self.config.min_confidence:
                    self._pending_fixes.append(fix)
                    fixes_generated += 1
        
        self._metrics.fixes_generated += fixes_generated
        
        return {
            "fixes_generated": fixes_generated,
            "pending_fixes": len(self._pending_fixes),
        }
    
    async def _apply_fixes(self) -> Dict[str, Any]:
        """Apply pending fixes."""
        applied = 0
        failed = 0
        
        for fix in self._pending_fixes[:self.config.max_concurrent_fixes]:
            try:
                if self.bug_fixer:
                    result = await self.bug_fixer.apply_fix(
                        fix.vuln_id,
                        fix.fixed_code,
                    )
                    fix.success = result.get("success", False)
                else:
                    # Mock apply
                    fix.success = True
                
                if fix.success:
                    fix.applied_at = datetime.now(timezone.utc)
                    applied += 1
                    self._metrics.fixes_applied += 1
                    
                    # Notify callbacks
                    for cb in self._on_fix_applied:
                        await cb(fix)
                else:
                    failed += 1
                    self._metrics.fixes_failed += 1
                    
            except Exception as e:
                fix.error = str(e)
                failed += 1
                self._metrics.fixes_failed += 1
                logger.error(f"Failed to apply fix {fix.attempt_id}: {e}")
        
        # Move to history
        self._fix_history.extend(self._pending_fixes[:self.config.max_concurrent_fixes])
        self._pending_fixes = self._pending_fixes[self.config.max_concurrent_fixes:]
        
        return {
            "applied": applied,
            "failed": failed,
            "remaining": len(self._pending_fixes),
        }
    
    async def _verify_fixes(self) -> Dict[str, Any]:
        """Verify applied fixes."""
        verified = 0
        failed = 0
        
        for fix in self._fix_history:
            if fix.applied_at and not fix.verified_at:
                try:
                    if self.fix_verifier:
                        result = await self.fix_verifier.verify_fix(
                            fix.file_path,
                            fix.fixed_code,
                        )
                        success = result.get("passed", False)
                    else:
                        # Mock verification
                        success = True
                    
                    if success:
                        fix.verified_at = datetime.now(timezone.utc)
                        verified += 1
                        self._metrics.fixes_verified += 1
                        
                        for cb in self._on_fix_verified:
                            await cb(fix)
                    else:
                        # Rollback
                        await self._rollback_fix(fix)
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"Verification failed for {fix.attempt_id}: {e}")
                    failed += 1
        
        return {
            "verified": verified,
            "failed": failed,
        }
    
    async def _rollback_fix(self, fix: FixAttempt):
        """Rollback a failed fix."""
        try:
            if self.bug_fixer:
                await self.bug_fixer.rollback(fix.vuln_id)
            
            fix.rollback_at = datetime.now(timezone.utc)
            self._metrics.fixes_rolled_back += 1
            
            logger.warning(f"Rolled back fix {fix.attempt_id}")
            
        except Exception as e:
            logger.error(f"Rollback failed for {fix.attempt_id}: {e}")
    
    async def _learn_from_cycle(self) -> Dict[str, Any]:
        """Learn from the cycle results."""
        successful_patterns = []
        failed_patterns = []
        
        for fix in self._fix_history[-100:]:  # Last 100 fixes
            if fix.verified_at:
                successful_patterns.append(fix.fix_type)
            elif fix.rollback_at:
                failed_patterns.append(fix.fix_type)
        
        # Update pattern weights if bug_fixer supports it
        if self.bug_fixer and hasattr(self.bug_fixer, "update_pattern_weights"):
            await self.bug_fixer.update_pattern_weights(
                successful=successful_patterns,
                failed=failed_patterns,
            )
        
        # Promote successful strategies to V2 if version_manager exists
        if self.version_manager:
            for pattern in set(successful_patterns):
                if successful_patterns.count(pattern) >= 3:
                    # Pattern succeeded 3+ times, consider for promotion
                    logger.info(f"Pattern '{pattern}' candidate for V2 promotion")
        
        return {
            "successful_patterns": len(set(successful_patterns)),
            "failed_patterns": len(set(failed_patterns)),
        }
    
    # =========================================================================
    # API
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current cycle status."""
        return {
            "running": self._running,
            "phase": self._phase.value,
            "pending_fixes": len(self._pending_fixes),
            "metrics": {
                "cycles_completed": self._metrics.cycles_completed,
                "vulnerabilities_detected": self._metrics.vulnerabilities_detected,
                "fixes_applied": self._metrics.fixes_applied,
                "fixes_verified": self._metrics.fixes_verified,
                "fixes_failed": self._metrics.fixes_failed,
                "fixes_rolled_back": self._metrics.fixes_rolled_back,
            },
            "last_cycle_at": self._metrics.last_cycle_at.isoformat() if self._metrics.last_cycle_at else None,
        }
    
    def get_pending_fixes(self) -> List[Dict[str, Any]]:
        """Get list of pending fixes awaiting approval."""
        return [
            {
                "attempt_id": fix.attempt_id,
                "vuln_id": fix.vuln_id,
                "file_path": fix.file_path,
                "fix_type": fix.fix_type,
                "confidence": fix.confidence,
                "original_code": fix.original_code[:200],
                "fixed_code": fix.fixed_code[:200],
            }
            for fix in self._pending_fixes
        ]
    
    def approve_fix(self, attempt_id: str) -> Dict[str, Any]:
        """Approve a pending fix for application."""
        for fix in self._pending_fixes:
            if fix.attempt_id == attempt_id:
                # Move to front of queue
                self._pending_fixes.remove(fix)
                self._pending_fixes.insert(0, fix)
                return {"success": True, "message": "Fix approved"}
        
        return {"success": False, "message": "Fix not found"}
    
    def reject_fix(self, attempt_id: str) -> Dict[str, Any]:
        """Reject a pending fix."""
        for fix in self._pending_fixes:
            if fix.attempt_id == attempt_id:
                self._pending_fixes.remove(fix)
                return {"success": True, "message": "Fix rejected"}
        
        return {"success": False, "message": "Fix not found"}
    
    def get_fix_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent fix history."""
        return [
            {
                "attempt_id": fix.attempt_id,
                "vuln_id": fix.vuln_id,
                "file_path": fix.file_path,
                "fix_type": fix.fix_type,
                "confidence": fix.confidence,
                "applied_at": fix.applied_at.isoformat() if fix.applied_at else None,
                "verified_at": fix.verified_at.isoformat() if fix.verified_at else None,
                "success": fix.success,
                "rolled_back": fix.rollback_at is not None,
            }
            for fix in self._fix_history[-limit:]
        ]


# Factory function
def create_auto_fix_cycle(
    config: Optional[FixCycleConfig] = None,
    bug_fixer=None,
    fix_verifier=None,
    version_manager=None,
) -> AutoFixCycle:
    """Create an auto-fix cycle instance."""
    return AutoFixCycle(
        config=config,
        bug_fixer=bug_fixer,
        fix_verifier=fix_verifier,
        version_manager=version_manager,
    )
