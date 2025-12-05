"""
V3 Comparison and Exclusion Engine

V3 (Old/Quarantine) serves as a comparison baseline and exclusion zone for:
- Technologies with poor reviews
- Technologies with poor performance
- Failed experiments that should not be retried

This engine analyzes quarantined technologies and provides:
1. Comparison data for V1 experiments
2. Performance baselines
3. Definitive exclusion decisions
4. Learning from failures to prevent repetition
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid

logger = logging.getLogger(__name__)


class ExclusionReason(str, Enum):
    """Reasons for excluding a technology."""
    POOR_PERFORMANCE = "poor_performance"
    POOR_ACCURACY = "poor_accuracy"
    HIGH_ERROR_RATE = "high_error_rate"
    SECURITY_VULNERABILITY = "security_vulnerability"
    INCOMPATIBLE = "incompatible"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    UNFIXABLE = "unfixable"


class ExclusionDecision(str, Enum):
    """Exclusion decision status."""
    PENDING = "pending"
    TEMPORARY = "temporary"      # Can be retried after cooldown
    PERMANENT = "permanent"       # Never retry
    UNDER_REVIEW = "under_review" # Being re-evaluated


@dataclass
class TechnologyProfile:
    """Profile of a quarantined technology."""
    tech_id: str
    name: str
    category: str
    source: str
    
    # Performance metrics at quarantine
    accuracy: float = 0.0
    error_rate: float = 0.0
    latency_p95_ms: float = 0.0
    cost_per_request: float = 0.0
    
    # Exclusion details
    exclusion_reason: ExclusionReason = ExclusionReason.POOR_PERFORMANCE
    exclusion_decision: ExclusionDecision = ExclusionDecision.PENDING
    exclusion_evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timeline
    quarantined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exclusion_decided_at: Optional[datetime] = None
    retry_after: Optional[datetime] = None
    
    # Comparison data
    comparison_baseline: Dict[str, float] = field(default_factory=dict)
    v2_comparison: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing a technology against V3 baseline."""
    comparison_id: str
    technology_id: str
    baseline_tech_id: str
    
    # Metrics comparison
    metrics_delta: Dict[str, float] = field(default_factory=dict)
    
    # Verdict
    better_than_baseline: bool = False
    recommendation: str = ""
    confidence: float = 0.0
    
    compared_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class V3ComparisonEngine:
    """
    Engine for managing V3 quarantine zone:
    - Track quarantined technologies
    - Provide comparison baselines
    - Make exclusion decisions
    - Learn from failures
    """
    
    def __init__(
        self,
        version_manager=None,
        event_bus=None,
    ):
        self.version_manager = version_manager
        self.event_bus = event_bus
        
        # Quarantined technologies
        self._profiles: Dict[str, TechnologyProfile] = {}
        
        # Exclusion list
        self._permanent_exclusions: Set[str] = set()
        self._temporary_exclusions: Dict[str, datetime] = {}  # tech_id -> retry_after
        
        # Comparison baselines
        self._baselines: Dict[str, Dict[str, float]] = {}
        
        # Learning from failures
        self._failure_patterns: List[Dict[str, Any]] = []
        self._exclusion_rules: List[Dict[str, Any]] = []
        
        # V2 reference metrics for comparison
        self._v2_reference: Dict[str, float] = {
            "accuracy": 0.85,
            "error_rate": 0.02,
            "latency_p95_ms": 3000,
            "cost_per_request": 0.01,
        }
        
        self._lock = asyncio.Lock()
    
    # =========================================================================
    # Quarantine Management
    # =========================================================================
    
    async def quarantine_technology(
        self,
        tech_id: str,
        name: str,
        category: str,
        source: str,
        metrics: Dict[str, float],
        reason: ExclusionReason,
        evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> TechnologyProfile:
        """Quarantine a technology from V1 or V2."""
        profile = TechnologyProfile(
            tech_id=tech_id,
            name=name,
            category=category,
            source=source,
            accuracy=metrics.get("accuracy", 0),
            error_rate=metrics.get("error_rate", 1),
            latency_p95_ms=metrics.get("latency_p95_ms", float("inf")),
            cost_per_request=metrics.get("cost_per_request", 0),
            exclusion_reason=reason,
            exclusion_evidence=evidence or [],
        )
        
        # Calculate comparison baseline
        profile.comparison_baseline = metrics.copy()
        profile.v2_comparison = await self._compare_with_v2(metrics)
        
        self._profiles[tech_id] = profile
        
        # Make initial exclusion decision
        await self._make_exclusion_decision(profile)
        
        # Record failure pattern
        await self._record_failure_pattern(profile)
        
        logger.info(f"Quarantined technology: {name} ({reason.value})")
        
        if self.event_bus:
            await self.event_bus.publish("technology_quarantined", {
                "tech_id": tech_id,
                "name": name,
                "reason": reason.value,
                "decision": profile.exclusion_decision.value,
                "timestamp": profile.quarantined_at.isoformat(),
            })
        
        return profile
    
    def _compare_with_v2(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Compare metrics with V2 reference."""
        comparison = {}
        
        for key, v2_value in self._v2_reference.items():
            if key in metrics:
                tech_value = metrics[key]
                if key in ["accuracy"]:
                    # Higher is better
                    comparison[f"{key}_delta"] = tech_value - v2_value
                    comparison[f"{key}_ratio"] = tech_value / v2_value if v2_value > 0 else 0
                else:
                    # Lower is better
                    comparison[f"{key}_delta"] = v2_value - tech_value
                    comparison[f"{key}_ratio"] = v2_value / tech_value if tech_value > 0 else 0
        
        return comparison
    
    def _make_exclusion_decision(self, profile: TechnologyProfile):
        """Make exclusion decision based on metrics and reason."""
        reason = profile.exclusion_reason
        
        # Check for permanent exclusion criteria
        permanent_reasons = [
            ExclusionReason.SECURITY_VULNERABILITY,
            ExclusionReason.UNFIXABLE,
        ]
        
        if reason in permanent_reasons:
            profile.exclusion_decision = ExclusionDecision.PERMANENT
            self._permanent_exclusions.add(profile.tech_id)
            logger.warning(f"Permanent exclusion: {profile.name} ({reason.value})")
        
        elif profile.error_rate > 0.50 or profile.accuracy < 0.50:
            # Very poor performance = permanent
            profile.exclusion_decision = ExclusionDecision.PERMANENT
            self._permanent_exclusions.add(profile.tech_id)
            logger.warning(f"Permanent exclusion due to critical metrics: {profile.name}")
        
        elif profile.error_rate > 0.20 or profile.accuracy < 0.70:
            # Poor performance = temporary with long cooldown
            profile.exclusion_decision = ExclusionDecision.TEMPORARY
            profile.retry_after = datetime.now(timezone.utc) + timedelta(days=90)
            self._temporary_exclusions[profile.tech_id] = profile.retry_after
            logger.info(f"Temporary exclusion (90 days): {profile.name}")
        
        else:
            # Marginal failure = temporary with short cooldown
            profile.exclusion_decision = ExclusionDecision.TEMPORARY
            profile.retry_after = datetime.now(timezone.utc) + timedelta(days=30)
            self._temporary_exclusions[profile.tech_id] = profile.retry_after
            logger.info(f"Temporary exclusion (30 days): {profile.name}")
        
        profile.exclusion_decided_at = datetime.now(timezone.utc)
    
    async def _record_failure_pattern(self, profile: TechnologyProfile):
        """Record failure pattern for learning."""
        pattern = {
            "tech_id": profile.tech_id,
            "name": profile.name,
            "category": profile.category,
            "reason": profile.exclusion_reason.value,
            "metrics": {
                "accuracy": profile.accuracy,
                "error_rate": profile.error_rate,
                "latency_p95_ms": profile.latency_p95_ms,
            },
            "timestamp": profile.quarantined_at.isoformat(),
        }
        
        self._failure_patterns.append(pattern)
        
        # Generate exclusion rule if pattern repeated
        similar_failures = [
            p for p in self._failure_patterns
            if p["category"] == profile.category 
            and p["reason"] == profile.exclusion_reason.value
        ]
        
        if len(similar_failures) >= 3:
            rule = {
                "rule_id": str(uuid.uuid4()),
                "category": profile.category,
                "reason": profile.exclusion_reason.value,
                "min_occurrences": 3,
                "action": "warn_before_experiment",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._exclusion_rules.append(rule)
            logger.info(f"New exclusion rule created for {profile.category}")
    
    # =========================================================================
    # Comparison Functions
    # =========================================================================
    
    async def compare_technology(
        self,
        tech_id: str,
        tech_metrics: Dict[str, float],
    ) -> ComparisonResult:
        """Compare a V1 technology against V3 quarantine baseline."""
        
        # Find similar quarantined technologies for comparison
        similar = await self._find_similar_quarantined(tech_id)
        
        if not similar:
            # No baseline, compare with V2
            return await self._compare_with_v2_baseline(tech_id, tech_metrics)
        
        baseline = similar[0]  # Most similar
        
        # Calculate deltas
        metrics_delta = {}
        for key in ["accuracy", "error_rate", "latency_p95_ms", "cost_per_request"]:
            tech_val = tech_metrics.get(key, 0)
            base_val = baseline.comparison_baseline.get(key, 0)
            metrics_delta[key] = tech_val - base_val
        
        # Determine if better than baseline
        better = (
            metrics_delta.get("accuracy", 0) > 0.05 and
            metrics_delta.get("error_rate", 0) < -0.02
        )
        
        # Generate recommendation
        if better:
            recommendation = "Technology shows improvement over quarantined baseline. Proceed with caution."
            confidence = 0.75
        else:
            recommendation = "Technology does not show significant improvement. Consider alternative approaches."
            confidence = 0.85
        
        result = ComparisonResult(
            comparison_id=str(uuid.uuid4()),
            technology_id=tech_id,
            baseline_tech_id=baseline.tech_id,
            metrics_delta=metrics_delta,
            better_than_baseline=better,
            recommendation=recommendation,
            confidence=confidence,
        )
        
        return result
    
    def _find_similar_quarantined(
        self,
        tech_id: str,  # noqa: ARG002 - reserved for similarity matching
    ) -> List[TechnologyProfile]:
        """Find similar quarantined technologies for comparison."""
        # In production, would use embedding similarity
        return list(self._profiles.values())[:3]
    
    def _compare_with_v2_baseline(
        self,
        tech_id: str,
        tech_metrics: Dict[str, float],
    ) -> ComparisonResult:
        """Compare with V2 baseline when no V3 comparison exists."""
        metrics_delta = {}
        
        for key in ["accuracy", "error_rate", "latency_p95_ms", "cost_per_request"]:
            tech_val = tech_metrics.get(key, 0)
            v2_val = self._v2_reference.get(key, 0)
            metrics_delta[key] = tech_val - v2_val
        
        # Better if exceeds V2 thresholds
        better = (
            tech_metrics.get("accuracy", 0) >= self._v2_reference["accuracy"] and
            tech_metrics.get("error_rate", 1) <= self._v2_reference["error_rate"]
        )
        
        return ComparisonResult(
            comparison_id=str(uuid.uuid4()),
            technology_id=tech_id,
            baseline_tech_id="v2_reference",
            metrics_delta=metrics_delta,
            better_than_baseline=better,
            recommendation="Compared against V2 production baseline" if better else "Does not meet V2 standards",
            confidence=0.90,
        )
    
    # =========================================================================
    # Exclusion Checking
    # =========================================================================
    
    def is_excluded(self, tech_id: str) -> bool:
        """Check if a technology is excluded from experimentation."""
        if tech_id in self._permanent_exclusions:
            return True
        
        if tech_id in self._temporary_exclusions:
            retry_after = self._temporary_exclusions[tech_id]
            if datetime.now(timezone.utc) < retry_after:
                return True
            else:
                # Cooldown passed, remove from temporary exclusion
                del self._temporary_exclusions[tech_id]
                return False
        
        return False
    
    async def get_exclusion_status(self, tech_id: str) -> Dict[str, Any]:
        """Get detailed exclusion status for a technology."""
        profile = self._profiles.get(tech_id)
        
        if not profile:
            return {"excluded": False, "reason": None}
        
        if tech_id in self._permanent_exclusions:
            return {
                "excluded": True,
                "decision": "permanent",
                "reason": profile.exclusion_reason.value,
                "decided_at": profile.exclusion_decided_at.isoformat() if profile.exclusion_decided_at else None,
            }
        
        if tech_id in self._temporary_exclusions:
            retry_after = self._temporary_exclusions[tech_id]
            now = datetime.now(timezone.utc)
            
            return {
                "excluded": now < retry_after,
                "decision": "temporary",
                "reason": profile.exclusion_reason.value,
                "retry_after": retry_after.isoformat(),
                "days_remaining": (retry_after - now).days if now < retry_after else 0,
            }
        
        return {"excluded": False, "reason": None, "was_quarantined": True}
    
    async def check_before_experiment(
        self,
        tech_name: str,
        tech_category: str,
    ) -> Dict[str, Any]:
        """Check if a new experiment should be warned about similar failures."""
        warnings = []
        
        # Check failure patterns
        similar_failures = [
            p for p in self._failure_patterns
            if p["category"] == tech_category
        ]
        
        if similar_failures:
            warnings.append({
                "type": "similar_failures",
                "message": f"{len(similar_failures)} similar technologies in {tech_category} have failed",
                "details": [
                    {"name": p["name"], "reason": p["reason"]} 
                    for p in similar_failures[:3]
                ],
            })
        
        # Check exclusion rules
        matching_rules = [
            r for r in self._exclusion_rules
            if r["category"] == tech_category
        ]
        
        for rule in matching_rules:
            warnings.append({
                "type": "exclusion_rule",
                "message": f"Category {tech_category} has exclusion rule for {rule['reason']}",
                "action": rule["action"],
            })
        
        return {
            "proceed": len(warnings) == 0,
            "warnings": warnings,
            "recommendation": "Proceed with experiment" if not warnings else "Review warnings before proceeding",
        }
    
    # =========================================================================
    # Re-evaluation
    # =========================================================================
    
    def request_re_evaluation(
        self,
        tech_id: str,
        reason: str,  # noqa: ARG002 - reserved for audit logging
        new_evidence: Optional[List[Dict[str, Any]]] = None,  # noqa: ARG002 - reserved for evidence tracking
    ) -> Dict[str, Any]:
        """Request re-evaluation of a quarantined technology."""
        profile = self._profiles.get(tech_id)
        
        if not profile:
            return {"success": False, "reason": "Technology not found"}
        
        if tech_id in self._permanent_exclusions:
            return {
                "success": False,
                "reason": "Technology has permanent exclusion",
                "original_reason": profile.exclusion_reason.value,
            }
        
        # Check cooldown
        if tech_id in self._temporary_exclusions:
            retry_after = self._temporary_exclusions[tech_id]
            if datetime.now(timezone.utc) < retry_after:
                return {
                    "success": False,
                    "reason": "Cooldown period not complete",
                    "retry_after": retry_after.isoformat(),
                }
        
        # Mark for re-evaluation
        profile.exclusion_decision = ExclusionDecision.UNDER_REVIEW
        
        logger.info(f"Re-evaluation requested for {profile.name}: {reason}")
        
        return {
            "success": True,
            "tech_id": tech_id,
            "name": profile.name,
            "status": "under_review",
            "next_step": "Return to V1 for fresh experimentation",
        }
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def get_quarantine_statistics(self) -> Dict[str, Any]:
        """Get statistics about quarantined technologies."""
        total = len(self._profiles)
        
        by_reason = {}
        for profile in self._profiles.values():
            reason = profile.exclusion_reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        by_decision = {}
        for profile in self._profiles.values():
            decision = profile.exclusion_decision.value
            by_decision[decision] = by_decision.get(decision, 0) + 1
        
        return {
            "total_quarantined": total,
            "permanent_exclusions": len(self._permanent_exclusions),
            "temporary_exclusions": len(self._temporary_exclusions),
            "by_reason": by_reason,
            "by_decision": by_decision,
            "failure_patterns_recorded": len(self._failure_patterns),
            "exclusion_rules": len(self._exclusion_rules),
        }
    
    def get_exclusion_list(self) -> Dict[str, List[str]]:
        """Get list of excluded technologies."""
        permanent = list(self._permanent_exclusions)
        temporary = [
            tech_id for tech_id, retry_after in self._temporary_exclusions.items()
            if datetime.now(timezone.utc) < retry_after
        ]
        
        return {
            "permanent": permanent,
            "temporary": temporary,
        }
    
    def get_failure_insights(self) -> List[Dict[str, Any]]:
        """Get insights from failure patterns."""
        insights = []
        
        # Group by category
        categories = {}
        for pattern in self._failure_patterns:
            cat = pattern["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(pattern)
        
        for cat, patterns in categories.items():
            if len(patterns) >= 2:
                insights.append({
                    "category": cat,
                    "failure_count": len(patterns),
                    "common_reasons": list({p["reason"] for p in patterns}),
                    "avg_accuracy": sum(p["metrics"]["accuracy"] for p in patterns) / len(patterns),
                    "insight": f"Category {cat} has recurring failures - review approach",
                })
        
        return insights
