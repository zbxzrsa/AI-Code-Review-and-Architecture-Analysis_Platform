"""
Dual-Loop Update Mechanism

Two interconnected update loops:
1. Project Update Loop - Updates to the codebase and project
2. AI Self-Iteration Loop - AI model improvements and evolution

Target: Version iteration cycle ≤ 24 hours
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib

logger = logging.getLogger(__name__)


class LoopStatus(Enum):
    """Loop execution status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class UpdateType(Enum):
    """Types of updates"""
    CODE_CHANGE = "code_change"
    DEPENDENCY_UPDATE = "dependency_update"
    CONFIG_CHANGE = "config_change"
    MODEL_UPDATE = "model_update"
    LEARNING_INTEGRATION = "learning_integration"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BUG_FIX = "bug_fix"
    FEATURE_ADDITION = "feature_addition"


@dataclass
class UpdateCandidate:
    """A candidate update to be processed"""
    update_id: str
    update_type: UpdateType
    title: str
    description: str
    source: str  # 'project', 'ai', 'learning'
    priority: int = 1  # 1-5, higher is more urgent
    created_at: str = ""
    status: str = "pending"
    
    # Impact assessment
    estimated_impact: float = 0.0  # 0-1
    risk_level: str = "low"  # low, medium, high
    test_coverage: float = 0.0
    
    # Related data
    affected_files: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationResult:
    """Result of an iteration cycle"""
    iteration_id: str
    loop_type: str  # 'project' or 'ai'
    started_at: str
    completed_at: str
    duration_seconds: float
    
    updates_processed: int
    updates_succeeded: int
    updates_failed: int
    
    version_before: str
    version_after: str
    
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    
    def is_successful(self) -> bool:
        return self.updates_failed == 0


class ProjectLoop:
    """
    Project Update Loop
    
    Handles:
    - Code changes and improvements
    - Dependency updates
    - Configuration changes
    - Bug fixes and features
    """
    
    def __init__(
        self,
        iteration_interval_hours: float = 24.0,
        max_updates_per_cycle: int = 50
    ):
        self.iteration_interval = timedelta(hours=iteration_interval_hours)
        self.max_updates = max_updates_per_cycle
        
        self.status = LoopStatus.IDLE
        self.pending_updates: List[UpdateCandidate] = []
        self.completed_iterations: List[IterationResult] = []
        
        self.current_version = "1.0.0"
        self.last_iteration: Optional[datetime] = None
        
        # Callbacks
        self.on_update_start: Optional[Callable] = None
        self.on_update_complete: Optional[Callable] = None
        self.on_iteration_complete: Optional[Callable] = None
    
    def add_update(self, update: UpdateCandidate) -> None:
        """Add an update candidate"""
        update.created_at = datetime.now().isoformat()
        self.pending_updates.append(update)
        
        # Sort by priority
        self.pending_updates.sort(key=lambda u: u.priority, reverse=True)
        
        logger.info(f"Added project update: {update.title} (priority: {update.priority})")
    
    async def run_iteration(self) -> IterationResult:
        """Run a single iteration cycle"""
        iteration_id = f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        started_at = datetime.now()
        
        self.status = LoopStatus.RUNNING
        logger.info(f"Starting project iteration: {iteration_id}")
        
        updates_succeeded = 0
        updates_failed = 0
        improvements = []
        issues = []
        version_before = self.current_version
        
        # Process updates
        updates_to_process = self.pending_updates[:self.max_updates]
        
        for update in updates_to_process:
            try:
                if self.on_update_start:
                    await self.on_update_start(update)
                
                # Process the update
                success = await self._process_update(update)
                
                if success:
                    updates_succeeded += 1
                    update.status = "completed"
                    improvements.append({
                        "update_id": update.update_id,
                        "type": update.update_type.value,
                        "title": update.title
                    })
                else:
                    updates_failed += 1
                    update.status = "failed"
                    issues.append(f"Failed: {update.title}")
                
                if self.on_update_complete:
                    await self.on_update_complete(update, success)
                
            except Exception as e:
                updates_failed += 1
                update.status = "failed"
                issues.append(f"Error in {update.title}: {str(e)}")
                logger.error(f"Update error: {e}")
        
        # Remove processed updates
        self.pending_updates = [
            u for u in self.pending_updates
            if u.status not in ["completed", "failed"]
        ]
        
        # Update version if successful
        if updates_succeeded > 0:
            self.current_version = self._increment_version(
                self.current_version,
                minor=updates_succeeded > 5
            )
        
        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()
        
        result = IterationResult(
            iteration_id=iteration_id,
            loop_type="project",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=duration,
            updates_processed=len(updates_to_process),
            updates_succeeded=updates_succeeded,
            updates_failed=updates_failed,
            version_before=version_before,
            version_after=self.current_version,
            improvements=improvements,
            issues=issues
        )
        
        self.completed_iterations.append(result)
        self.last_iteration = completed_at
        self.status = LoopStatus.IDLE
        
        if self.on_iteration_complete:
            await self.on_iteration_complete(result)
        
        logger.info(
            f"Project iteration completed: {updates_succeeded} succeeded, "
            f"{updates_failed} failed, version: {self.current_version}"
        )
        
        return result
    
    async def _process_update(self, update: UpdateCandidate) -> bool:
        """Process a single update"""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # In production, this would:
        # 1. Run automated tests
        # 2. Apply code changes
        # 3. Update dependencies
        # 4. Validate configuration
        
        # Success based on risk and test coverage
        success_probability = 0.95 - (0.1 if update.risk_level == "high" else 0)
        success_probability += update.test_coverage * 0.1
        
        return update.risk_level != "high" or update.test_coverage > 0.8
    
    def _increment_version(self, version: str, minor: bool = False) -> str:
        """Increment version number"""
        parts = version.split(".")
        major, minor_v, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        if minor:
            minor_v += 1
            patch = 0
        else:
            patch += 1
        
        return f"{major}.{minor_v}.{patch}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get loop status"""
        return {
            "status": self.status.value,
            "current_version": self.current_version,
            "pending_updates": len(self.pending_updates),
            "completed_iterations": len(self.completed_iterations),
            "last_iteration": self.last_iteration.isoformat() if self.last_iteration else None,
            "next_iteration_in": str(
                self.iteration_interval - (datetime.now() - self.last_iteration)
                if self.last_iteration else "N/A"
            )
        }


class AIIterationLoop:
    """
    AI Self-Iteration Loop
    
    Handles:
    - Model updates and improvements
    - Learning integration
    - Performance optimization
    - Automatic evaluation and versioning
    """
    
    def __init__(
        self,
        iteration_interval_hours: float = 24.0,
        min_improvement_threshold: float = 0.02
    ):
        self.iteration_interval = timedelta(hours=iteration_interval_hours)
        self.improvement_threshold = min_improvement_threshold
        
        self.status = LoopStatus.IDLE
        self.pending_improvements: List[UpdateCandidate] = []
        self.completed_iterations: List[IterationResult] = []
        
        self.current_model_version = "1.0.0"
        self.last_iteration: Optional[datetime] = None
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.current_performance: Dict[str, float] = {
            "accuracy": 0.85,
            "latency_ms": 200,
            "throughput_rps": 100,
            "error_rate": 0.02
        }
        
        # Callbacks
        self.on_model_update: Optional[Callable] = None
        self.on_performance_change: Optional[Callable] = None
    
    def add_improvement(self, improvement: UpdateCandidate) -> None:
        """Add an AI improvement candidate"""
        improvement.created_at = datetime.now().isoformat()
        improvement.source = "ai"
        self.pending_improvements.append(improvement)
        
        logger.info(f"Added AI improvement: {improvement.title}")
    
    async def run_iteration(self) -> IterationResult:
        """Run a single AI iteration cycle"""
        iteration_id = f"ai_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        started_at = datetime.now()
        
        self.status = LoopStatus.RUNNING
        logger.info(f"Starting AI iteration: {iteration_id}")
        
        # Store current performance
        self.performance_history.append({
            **self.current_performance,
            "timestamp": started_at.isoformat()
        })
        
        updates_succeeded = 0
        updates_failed = 0
        improvements = []
        issues = []
        version_before = self.current_model_version
        
        # Evaluate and process improvements
        for improvement in self.pending_improvements:
            try:
                # Evaluate improvement impact
                impact = await self._evaluate_improvement(improvement)
                
                if impact >= self.improvement_threshold:
                    # Apply improvement
                    success = await self._apply_improvement(improvement)
                    
                    if success:
                        updates_succeeded += 1
                        improvement.status = "completed"
                        improvements.append({
                            "update_id": improvement.update_id,
                            "type": improvement.update_type.value,
                            "impact": impact
                        })
                        
                        # Update performance
                        await self._update_performance(improvement, impact)
                    else:
                        updates_failed += 1
                        improvement.status = "failed"
                        issues.append(f"Failed to apply: {improvement.title}")
                else:
                    improvement.status = "rejected"
                    issues.append(f"Insufficient impact: {improvement.title}")
                
            except Exception as e:
                updates_failed += 1
                improvement.status = "failed"
                issues.append(f"Error: {str(e)}")
                logger.error(f"AI improvement error: {e}")
        
        # Clear processed improvements
        self.pending_improvements = [
            i for i in self.pending_improvements
            if i.status not in ["completed", "failed", "rejected"]
        ]
        
        # Update model version
        if updates_succeeded > 0:
            self.current_model_version = self._increment_version(
                self.current_model_version
            )
        
        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()
        
        result = IterationResult(
            iteration_id=iteration_id,
            loop_type="ai",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=duration,
            updates_processed=len(self.pending_improvements) + updates_succeeded + updates_failed,
            updates_succeeded=updates_succeeded,
            updates_failed=updates_failed,
            version_before=version_before,
            version_after=self.current_model_version,
            improvements=improvements,
            issues=issues
        )
        
        self.completed_iterations.append(result)
        self.last_iteration = completed_at
        self.status = LoopStatus.IDLE
        
        logger.info(
            f"AI iteration completed: {updates_succeeded} improvements applied, "
            f"model version: {self.current_model_version}"
        )
        
        return result
    
    async def _evaluate_improvement(self, improvement: UpdateCandidate) -> float:
        """Evaluate potential impact of an improvement"""
        # Simulate evaluation
        await asyncio.sleep(0.05)
        
        # Base impact from metadata
        base_impact = improvement.estimated_impact
        
        # Adjust based on type
        type_multipliers = {
            UpdateType.PERFORMANCE_OPTIMIZATION: 1.2,
            UpdateType.LEARNING_INTEGRATION: 1.1,
            UpdateType.BUG_FIX: 1.0,
            UpdateType.MODEL_UPDATE: 0.9,
        }
        
        multiplier = type_multipliers.get(improvement.update_type, 1.0)
        return base_impact * multiplier
    
    async def _apply_improvement(self, improvement: UpdateCandidate) -> bool:
        """Apply an improvement to the AI model"""
        # Simulate applying improvement
        await asyncio.sleep(0.1)
        
        # In production, this would:
        # 1. Update model weights
        # 2. Apply configuration changes
        # 3. Run validation tests
        # 4. Update model artifacts
        
        return True
    
    async def _update_performance(
        self,
        improvement: UpdateCandidate,
        impact: float
    ) -> None:
        """Update performance metrics after improvement"""
        if improvement.update_type == UpdateType.PERFORMANCE_OPTIMIZATION:
            self.current_performance["latency_ms"] *= (1 - impact * 0.1)
            self.current_performance["throughput_rps"] *= (1 + impact * 0.05)
        
        elif improvement.update_type == UpdateType.LEARNING_INTEGRATION:
            self.current_performance["accuracy"] = min(
                0.99,
                self.current_performance["accuracy"] + impact * 0.02
            )
        
        elif improvement.update_type == UpdateType.BUG_FIX:
            self.current_performance["error_rate"] *= (1 - impact * 0.2)
        
        if self.on_performance_change:
            await self.on_performance_change(self.current_performance)
    
    def _increment_version(self, version: str) -> str:
        """Increment model version"""
        parts = version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        patch += 1
        return f"{major}.{minor}.{patch}"
    
    def get_performance_trend(self) -> Dict[str, Any]:
        """Get performance trend over iterations"""
        if len(self.performance_history) < 2:
            return {"status": "insufficient_data"}
        
        latest = self.performance_history[-1]
        previous = self.performance_history[-2]
        
        trends = {}
        for metric in ["accuracy", "latency_ms", "throughput_rps", "error_rate"]:
            if metric in latest and metric in previous:
                change = (latest[metric] - previous[metric]) / previous[metric]
                trends[metric] = {
                    "current": latest[metric],
                    "previous": previous[metric],
                    "change_percent": change * 100,
                    "improving": (
                        change > 0 if metric in ["accuracy", "throughput_rps"]
                        else change < 0
                    )
                }
        
        return trends
    
    def get_status(self) -> Dict[str, Any]:
        """Get loop status"""
        return {
            "status": self.status.value,
            "current_model_version": self.current_model_version,
            "pending_improvements": len(self.pending_improvements),
            "completed_iterations": len(self.completed_iterations),
            "current_performance": self.current_performance,
            "last_iteration": self.last_iteration.isoformat() if self.last_iteration else None
        }


class DualLoopUpdater:
    """
    Dual-Loop Update Coordinator
    
    Coordinates the Project Loop and AI Iteration Loop
    with bidirectional communication.
    
    Target: Version iteration cycle ≤ 24 hours
    """
    
    def __init__(
        self,
        iteration_cycle_hours: float = 24.0
    ):
        self.project_loop = ProjectLoop(iteration_cycle_hours / 2)
        self.ai_loop = AIIterationLoop(iteration_cycle_hours / 2)
        
        self.iteration_cycle = timedelta(hours=iteration_cycle_hours)
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        self._loop_task: Optional[asyncio.Task] = None
        
        # Communication queue between loops
        self.cross_loop_updates: asyncio.Queue = asyncio.Queue()
        
        # Setup cross-loop callbacks
        self.project_loop.on_iteration_complete = self._on_project_iteration
        self.ai_loop.on_performance_change = self._on_ai_performance_change
    
    async def _on_project_iteration(self, result: IterationResult) -> None:
        """Handle project iteration completion"""
        # Generate AI improvement candidates based on project changes
        if result.is_successful():
            for improvement in result.improvements:
                # Create AI learning opportunity
                ai_update = UpdateCandidate(
                    update_id=f"ai_from_{improvement['update_id']}",
                    update_type=UpdateType.LEARNING_INTEGRATION,
                    title=f"Learn from: {improvement['title']}",
                    description="Integrate project improvement into AI knowledge",
                    source="project_loop",
                    priority=2,
                    estimated_impact=0.05
                )
                self.ai_loop.add_improvement(ai_update)
    
    async def _on_ai_performance_change(
        self,
        performance: Dict[str, float]
    ) -> None:
        """Handle AI performance changes"""
        # Create project updates based on AI improvements
        if performance.get("accuracy", 0) > 0.9:
            project_update = UpdateCandidate(
                update_id=f"proj_from_ai_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                update_type=UpdateType.FEATURE_ADDITION,
                title="Enable advanced AI features",
                description="AI performance improved, enabling new capabilities",
                source="ai_loop",
                priority=3,
                estimated_impact=0.1
            )
            self.project_loop.add_update(project_update)
    
    async def start(self) -> None:
        """Start the dual-loop system"""
        logger.info("Starting Dual-Loop Update System...")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        self._loop_task = asyncio.create_task(self._run_loops())
        
        logger.info(f"Dual-loop started, iteration cycle: {self.iteration_cycle}")
    
    async def stop(self) -> None:
        """Stop the dual-loop system"""
        logger.info("Stopping Dual-Loop Update System...")
        
        self.is_running = False
        
        if self._loop_task:
            self._loop_task.cancel()
        
        logger.info("Dual-loop stopped")
    
    async def _run_loops(self) -> None:
        """Run both loops continuously"""
        while self.is_running:
            try:
                # Run project loop
                if (
                    self.project_loop.pending_updates or
                    not self.project_loop.last_iteration or
                    datetime.now() - self.project_loop.last_iteration >= self.project_loop.iteration_interval
                ):
                    await self.project_loop.run_iteration()
                
                # Run AI loop
                if (
                    self.ai_loop.pending_improvements or
                    not self.ai_loop.last_iteration or
                    datetime.now() - self.ai_loop.last_iteration >= self.ai_loop.iteration_interval
                ):
                    await self.ai_loop.run_iteration()
                
                # Process cross-loop updates
                await self._process_cross_loop_updates()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Dual-loop error: {e}")
                await asyncio.sleep(5)
    
    async def _process_cross_loop_updates(self) -> None:
        """Process updates between loops"""
        while not self.cross_loop_updates.empty():
            try:
                update = await asyncio.wait_for(
                    self.cross_loop_updates.get(),
                    timeout=1.0
                )
                
                if update.get("target") == "project":
                    self.project_loop.add_update(
                        UpdateCandidate(**update["data"])
                    )
                elif update.get("target") == "ai":
                    self.ai_loop.add_improvement(
                        UpdateCandidate(**update["data"])
                    )
                    
            except asyncio.TimeoutError:
                break
    
    def get_iteration_cycle_status(self) -> Dict[str, Any]:
        """Get status of iteration cycle"""
        now = datetime.now()
        
        # Calculate time since last full cycle
        last_project = self.project_loop.last_iteration
        last_ai = self.ai_loop.last_iteration
        
        if last_project and last_ai:
            last_full_cycle = min(last_project, last_ai)
            time_since_cycle = now - last_full_cycle
            meets_target = time_since_cycle <= self.iteration_cycle
        else:
            time_since_cycle = None
            meets_target = None
        
        return {
            "iteration_cycle_target_hours": self.iteration_cycle.total_seconds() / 3600,
            "time_since_last_cycle_hours": (
                time_since_cycle.total_seconds() / 3600
                if time_since_cycle else None
            ),
            "meets_24h_target": meets_target,
            "project_loop": self.project_loop.get_status(),
            "ai_loop": self.ai_loop.get_status()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        uptime = (
            (datetime.now() - self.start_time).total_seconds()
            if self.start_time else 0
        )
        
        return {
            "is_running": self.is_running,
            "uptime_hours": uptime / 3600,
            "iteration_cycle": self.get_iteration_cycle_status(),
            "project_version": self.project_loop.current_version,
            "model_version": self.ai_loop.current_model_version,
            "ai_performance": self.ai_loop.current_performance
        }
