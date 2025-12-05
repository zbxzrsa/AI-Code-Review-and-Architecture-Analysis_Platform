"""
Version-Specific AI Engines

Each version (V1, V2, V3) has its own:
- Code Review AI (CR-AI): User-facing code analysis
- Version Control AI (VC-AI): Self-evolution management

V1 Experimental: Tests new technologies with relaxed thresholds
V2 Production: Stable, proven technologies for users
V3 Quarantine: Analyzes failed experiments for learning
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AICapability(str, Enum):
    """AI capabilities."""
    CODE_REVIEW = "code_review"
    BUG_DETECTION = "bug_detection"
    SECURITY_SCAN = "security_scan"
    OPTIMIZATION = "optimization"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


@dataclass
class AIConfig:
    """AI engine configuration."""
    version: str
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    capabilities: List[AICapability] = field(default_factory=list)
    enabled_technologies: List[str] = field(default_factory=list)
    fallback_enabled: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3


@dataclass
class ReviewRequest:
    """Code review request."""
    request_id: str
    code: str
    language: str
    context: Optional[str] = None
    review_type: str = "full"
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReviewResult:
    """Code review result."""
    request_id: str
    version: str
    issues: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    tokens_used: int = 0
    model_used: str = ""
    technologies_used: List[str] = field(default_factory=list)
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VersionAIEngine(ABC):
    """
    Abstract base class for version-specific AI engines.
    
    Each version implements its own:
    - Code Review AI for user-facing analysis
    - Version Control AI for self-management
    """
    
    def __init__(self, config: AIConfig, version_manager=None):
        self.config = config
        self.version_manager = version_manager
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def review_code(self, request: ReviewRequest) -> ReviewResult:
        """Perform code review."""
        pass
    
    @abstractmethod
    async def analyze_for_evolution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data for version control AI decisions."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AICapability]:
        """Get available capabilities."""
        pass
    
    async def record_request(self, latency_ms: float, success: bool):
        """Record request metrics."""
        async with self._lock:
            self._request_count += 1
            if not success:
                self._error_count += 1
            self._total_latency += latency_ms
    
    def get_metrics(self) -> Dict[str, float]:
        """Get engine metrics."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / self._request_count if self._request_count > 0 else 0,
            "avg_latency_ms": self._total_latency / self._request_count if self._request_count > 0 else 0,
        }


class V1ExperimentalAI(VersionAIEngine):
    """
    V1 Experimentation AI Engine
    
    Purpose:
    - Test new technologies (attention mechanisms, architectures)
    - Trial-and-error with relaxed thresholds
    - Collect data for evaluation
    
    Features:
    - Multi-technology routing
    - A/B testing support
    - Detailed metrics collection
    """
    
    def __init__(self, config: AIConfig, version_manager=None, experiment_framework=None):
        super().__init__(config, version_manager)
        self.experiment_framework = experiment_framework
        
        # Experimental features enabled
        self.enabled_experiments: List[str] = []
        self.technology_weights: Dict[str, float] = {}
        
        # Default capabilities for V1
        if not config.capabilities:
            config.capabilities = [
                AICapability.CODE_REVIEW,
                AICapability.BUG_DETECTION,
                AICapability.SECURITY_SCAN,
            ]
    
    async def review_code(self, request: ReviewRequest) -> ReviewResult:
        """
        Perform code review with experimental technologies.
        
        Routes request through active experiments for comparison.
        """
        start_time = datetime.now(timezone.utc)
        
        result = ReviewResult(
            request_id=request.request_id,
            version="v1",
        )
        
        try:
            # Select technology for this request
            technology = await self._select_technology()
            result.technologies_used.append(technology)
            
            # Perform analysis (simulated - in production, call actual model)
            issues, suggestions = await self._analyze_code(
                request.code,
                request.language,
                technology,
            )
            
            result.issues = issues
            result.suggestions = suggestions
            result.model_used = technology
            
            # Record for experiment tracking
            if self.experiment_framework:
                for exp_id in self.enabled_experiments:
                    await self.experiment_framework.record_result(
                        exp_id,
                        success=True,
                        latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                        accuracy=0.85,  # Would be calculated from feedback
                    )
            
        except Exception as e:
            logger.error(f"V1 review error: {e}")
            result.issues = [{"type": "error", "message": str(e)}]
            await self.record_request(0, False)
            return result
        
        result.latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        await self.record_request(result.latency_ms, True)
        
        return result
    
    async def _select_technology(self) -> str:
        """Select technology for request based on experiment weights."""
        import random
        
        if not self.technology_weights:
            return "multi_head_attention"  # Default
        
        techs = list(self.technology_weights.keys())
        weights = list(self.technology_weights.values())
        return random.choices(techs, weights=weights)[0]
    
    def _analyze_code(
        self,
        code: str,
        language: str,  # noqa: ARG002 - reserved for language-specific analysis
        technology: str,  # noqa: ARG002 - reserved for tech-specific analysis
    ) -> tuple:
        """Analyze code using specified technology."""
        # Simulated analysis - in production, this would use actual models
        issues = []
        suggestions = []
        
        # Basic pattern matching for demo
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if "password" in line.lower() and "=" in line:
                issues.append({
                    "type": "security",
                    "severity": "high",
                    "line": i,
                    "message": "Potential hardcoded password detected",
                    "technology": technology,
                })
            if "TODO" in line or "FIXME" in line:
                suggestions.append({
                    "type": "maintainability",
                    "line": i,
                    "message": "Address TODO/FIXME comment",
                })
        
        return issues, suggestions
    
    async def analyze_for_evolution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        V1 Version Control AI: Analyze experiments for promotion decisions.
        """
        analysis = {
            "version": "v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations": [],
        }
        
        if self.experiment_framework:
            experiments = await self.experiment_framework.list_experiments()
            
            for exp in experiments:
                if exp.result and exp.result.should_promote:
                    analysis["recommendations"].append({
                        "action": "promote",
                        "experiment_id": exp.experiment_id,
                        "name": exp.config.name,
                        "reason": exp.result.recommendation,
                    })
                elif exp.result and exp.result.should_quarantine:
                    analysis["recommendations"].append({
                        "action": "quarantine",
                        "experiment_id": exp.experiment_id,
                        "name": exp.config.name,
                        "reason": exp.result.recommendation,
                    })
        
        return analysis
    
    def get_capabilities(self) -> List[AICapability]:
        return self.config.capabilities
    
    async def enable_experiment(self, experiment_id: str, weight: float = 0.1):
        """Enable an experiment for traffic routing."""
        self.enabled_experiments.append(experiment_id)
        if self.experiment_framework:
            exp = await self.experiment_framework.get_experiment(experiment_id)
            if exp:
                self.technology_weights[exp.config.technology_type] = weight


class V2ProductionAI(VersionAIEngine):
    """
    V2 Production AI Engine
    
    Purpose:
    - Stable, user-facing code review
    - Only proven technologies from V1
    - Strict SLO enforcement
    
    Features:
    - High reliability with fallbacks
    - Optimized for latency
    - Comprehensive code analysis
    """
    
    def __init__(self, config: AIConfig, version_manager=None):
        super().__init__(config, version_manager)
        
        # Production-grade technologies only
        self.active_technologies = config.enabled_technologies or [
            "multi_head_attention",
            "kv_cache_optimization",
        ]
        
        # SLO thresholds
        self.latency_slo_ms = 3000
        self.error_rate_slo = 0.02
        
        # Full capabilities for production
        if not config.capabilities:
            config.capabilities = list(AICapability)
    
    async def review_code(self, request: ReviewRequest) -> ReviewResult:
        """
        Perform production-grade code review.
        
        Uses proven technologies with fallback support.
        """
        start_time = datetime.now(timezone.utc)
        
        result = ReviewResult(
            request_id=request.request_id,
            version="v2",
        )
        
        try:
            # Use primary technology with fallback
            for tech in self.active_technologies:
                try:
                    issues, suggestions = await self._analyze_code(
                        request.code,
                        request.language,
                        tech,
                    )
                    result.technologies_used.append(tech)
                    result.issues.extend(issues)
                    result.suggestions.extend(suggestions)
                    break  # Success, no need for fallback
                except Exception as e:
                    logger.warning(f"Technology {tech} failed, trying fallback: {e}")
                    continue
            
            # Calculate quality metrics
            result.metrics = await self._calculate_metrics(request.code, result.issues)
            result.model_used = self.active_technologies[0]
            
        except Exception as e:
            logger.error(f"V2 review error: {e}")
            result.issues = [{"type": "error", "message": str(e)}]
            await self.record_request(0, False)
            return result
        
        result.latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        await self.record_request(result.latency_ms, True)
        
        # Check SLO
        if result.latency_ms > self.latency_slo_ms:
            logger.warning(f"V2 latency SLO violation: {result.latency_ms}ms > {self.latency_slo_ms}ms")
        
        return result
    
    def _analyze_code(
        self,
        code: str,
        language: str,  # noqa: ARG002 - reserved for language-specific analysis
        technology: str,  # noqa: ARG002 - reserved for tech-specific analysis
    ) -> tuple:
        """Production-grade code analysis."""
        issues = []
        suggestions = []
        
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            # Security checks
            if any(kw in line.lower() for kw in ["password", "secret", "api_key"]):
                if "=" in line and ("\"" in line or "'" in line):
                    issues.append({
                        "type": "security",
                        "severity": "critical",
                        "line": i,
                        "message": "Potential hardcoded secret",
                        "rule": "SEC-001",
                    })
            
            # Performance checks
            if "select *" in line.lower():
                issues.append({
                    "type": "performance",
                    "severity": "medium",
                    "line": i,
                    "message": "Avoid SELECT *, specify columns",
                    "rule": "PERF-001",
                })
            
            # Code quality
            if len(line) > 120:
                suggestions.append({
                    "type": "style",
                    "line": i,
                    "message": "Line exceeds 120 characters",
                })
        
        return issues, suggestions
    
    async def _calculate_metrics(
        self,
        code: str,
        issues: List[Dict],
    ) -> Dict[str, float]:
        """Calculate code quality metrics."""
        lines = len(code.split("\n"))
        
        return {
            "lines_of_code": lines,
            "issues_per_100_lines": (len(issues) / lines) * 100 if lines > 0 else 0,
            "critical_issues": sum(1 for i in issues if i.get("severity") == "critical"),
            "high_issues": sum(1 for i in issues if i.get("severity") == "high"),
            "medium_issues": sum(1 for i in issues if i.get("severity") == "medium"),
        }
    
    async def analyze_for_evolution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        V2 Version Control AI: Monitor production health, trigger degradation.
        """
        metrics = self.get_metrics()
        
        analysis = {
            "version": "v2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": "healthy",
            "recommendations": [],
        }
        
        # Check for degradation triggers
        if metrics["error_rate"] > self.error_rate_slo:
            analysis["health"] = "degraded"
            analysis["recommendations"].append({
                "action": "investigate",
                "reason": f"Error rate {metrics['error_rate']:.2%} exceeds SLO {self.error_rate_slo:.2%}",
            })
        
        if metrics["avg_latency_ms"] > self.latency_slo_ms:
            analysis["health"] = "degraded"
            analysis["recommendations"].append({
                "action": "optimize",
                "reason": f"Latency {metrics['avg_latency_ms']:.0f}ms exceeds SLO {self.latency_slo_ms}ms",
            })
        
        return analysis
    
    def get_capabilities(self) -> List[AICapability]:
        return self.config.capabilities
    
    async def add_promoted_technology(self, technology: str):
        """Add a technology promoted from V1."""
        if technology not in self.active_technologies:
            self.active_technologies.append(technology)
            logger.info(f"V2: Added promoted technology {technology}")


class V3QuarantineAI(VersionAIEngine):
    """
    V3 Quarantine AI Engine
    
    Purpose:
    - Archive and analyze failed experiments
    - Learn from failures
    - Support re-evaluation requests
    
    Features:
    - Read-only analysis mode
    - Failure pattern detection
    - Re-evaluation support
    """
    
    def __init__(self, config: AIConfig, version_manager=None):
        super().__init__(config, version_manager)
        
        # Quarantined technologies
        self.quarantined_technologies: Dict[str, Dict[str, Any]] = {}
        
        # Failure analysis data
        self.failure_patterns: List[Dict[str, Any]] = []
        
        # Limited capabilities for quarantine
        if not config.capabilities:
            config.capabilities = [
                AICapability.BUG_DETECTION,
                AICapability.TESTING,
            ]
    
    async def review_code(self, request: ReviewRequest) -> ReviewResult:
        """
        Read-only code review for quarantine analysis.
        """
        start_time = datetime.now(timezone.utc)
        
        result = ReviewResult(
            request_id=request.request_id,
            version="v3",
        )
        
        try:
            # Analyze using basic patterns only (no experimental tech)
            issues = await self._basic_analysis(request.code)
            result.issues = issues
            result.model_used = "quarantine_analyzer"
            
        except Exception as e:
            logger.error(f"V3 review error: {e}")
            result.issues = [{"type": "error", "message": str(e)}]
        
        result.latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        await self.record_request(result.latency_ms, True)
        
        return result
    
    async def _basic_analysis(self, code: str) -> List[Dict[str, Any]]:
        """Basic code analysis without experimental technologies."""
        issues = []
        lines = code.split("\n")
        
        for i, line in enumerate(lines, 1):
            if "eval(" in line or "exec(" in line:
                issues.append({
                    "type": "security",
                    "severity": "critical",
                    "line": i,
                    "message": "Dangerous eval/exec usage",
                })
        
        return issues
    
    async def analyze_for_evolution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        V3 Version Control AI: Analyze failure patterns, support re-evaluation.
        """
        analysis = {
            "version": "v3",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quarantined_count": len(self.quarantined_technologies),
            "failure_patterns": [],
            "re_evaluation_candidates": [],
        }
        
        # Identify patterns in failures
        for tech_id, tech_data in self.quarantined_technologies.items():
            if tech_data.get("days_quarantined", 0) >= 30:
                analysis["re_evaluation_candidates"].append({
                    "tech_id": tech_id,
                    "name": tech_data.get("name"),
                    "original_failure": tech_data.get("failure_reason"),
                })
        
        return analysis
    
    def get_capabilities(self) -> List[AICapability]:
        return self.config.capabilities
    
    async def quarantine_technology(
        self,
        tech_id: str,
        name: str,
        failure_reason: str,
        metrics: Dict[str, float],
    ):
        """Add a technology to quarantine."""
        self.quarantined_technologies[tech_id] = {
            "name": name,
            "failure_reason": failure_reason,
            "metrics": metrics,
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "days_quarantined": 0,
        }
        
        # Record failure pattern
        self.failure_patterns.append({
            "tech_id": tech_id,
            "pattern": failure_reason,
            "metrics": metrics,
        })
        
        logger.warning(f"V3: Quarantined technology {name}: {failure_reason}")
    
    async def request_re_evaluation(self, tech_id: str) -> Dict[str, Any]:
        """Request re-evaluation of quarantined technology."""
        tech_data = self.quarantined_technologies.get(tech_id)
        if not tech_data:
            return {"success": False, "reason": "Technology not found"}
        
        if tech_data.get("days_quarantined", 0) < 30:
            return {
                "success": False,
                "reason": "Minimum quarantine period not met (30 days)",
            }
        
        return {
            "success": True,
            "tech_id": tech_id,
            "name": tech_data["name"],
            "action": "move_to_v1",
        }


# Factory function
def create_version_ai(
    version: str,
    version_manager=None,
    experiment_framework=None,
) -> VersionAIEngine:
    """Create AI engine for specified version."""
    if version == "v1":
        config = AIConfig(version="v1", model_name="experimental")
        return V1ExperimentalAI(config, version_manager, experiment_framework)
    elif version == "v2":
        config = AIConfig(version="v2", model_name="production")
        return V2ProductionAI(config, version_manager)
    elif version == "v3":
        config = AIConfig(version="v3", model_name="quarantine")
        return V3QuarantineAI(config, version_manager)
    else:
        raise ValueError(f"Unknown version: {version}")
