"""
Quarantine Manager

Handles V1 → V3 quarantine process:
- Failure evidence capture
- Root cause analysis
- V3 archival
- Blacklist management
- Quarterly review scheduling
"""

import asyncio
import logging
import uuid  # FIXED: Moved from inside function
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from .event_types import (
    EventType,
    Version,
    VersionEvent,
    QuarantineRecord,
)

logger = logging.getLogger(__name__)


class FailureCategory:
    """Failure category constants."""
    TECHNICAL = "technical"
    QUALITY = "quality"
    OPERATIONAL = "operational"


@dataclass
class BlacklistEntry:
    """Entry in technique blacklist."""
    entry_id: str
    technique_name: str
    reason: str
    experiment_id: str
    category: str
    severity: str
    created_at: datetime
    expires_at: Optional[datetime] = None  # None = permanent


class QuarantineManager:
    """
    Manages the V1 → V3 quarantine process.
    
    Implements:
    - Failure evidence capture
    - Automated root cause analysis
    - V3 schema archival
    - Blacklist management
    - Quarterly review scheduling
    """
    
    def __init__(
        self,
        event_bus = None,
        db_connection = None,
        review_interval_days: int = 90,  # Quarterly
    ):
        self.event_bus = event_bus
        self.db = db_connection
        self.review_interval_days = review_interval_days
        
        # In-memory storage
        self._quarantine_records: Dict[str, QuarantineRecord] = {}
        self._blacklist: Dict[str, BlacklistEntry] = {}
    
    async def quarantine_experiment(
        self,
        experiment_id: str,
        failure_type: str,
        failure_evidence: Dict[str, Any],
        error_logs: str,
        metrics_at_failure: Dict[str, Any],
    ) -> QuarantineRecord:
        """
        Quarantine a failed V1 experiment to V3.
        
        Steps:
        1. Capture failure evidence
        2. Perform root cause analysis
        3. Store in V3 quarantine schema
        4. Update blacklist if needed
        """
        logger.info(f"Quarantining experiment {experiment_id}")
        
        # Step 1: Capture evidence
        evidence = await self._capture_evidence(
            experiment_id,
            failure_evidence,
            error_logs,
            metrics_at_failure,
        )
        
        # Step 2: Root cause analysis
        rca_result = await self._analyze_root_cause(
            failure_type,
            evidence,
        )
        
        # Step 3: Create quarantine record
        record = QuarantineRecord(
            experiment_id=experiment_id,
            failure_category=rca_result["category"],
            root_cause=rca_result["root_cause"],
            contributing_factors=rca_result["factors"],
            evidence=evidence,
            impact_assessment=rca_result["impact"],
            remediation_steps=rca_result["remediation"],
            review_scheduled_at=datetime.utcnow() + timedelta(days=self.review_interval_days),
        )
        
        # Step 4: Update blacklist if recommended
        if rca_result.get("blacklist_recommended"):
            blacklist_entry = await self._add_to_blacklist(
                technique_name=rca_result["technique_name"],
                reason=rca_result["blacklist_reason"],
                experiment_id=experiment_id,
                category=rca_result["category"],
            )
            record.blacklist_entry = blacklist_entry.entry_id
        
        # Store record
        self._quarantine_records[record.record_id] = record
        
        # Emit event
        await self._emit_event(
            EventType.QUARANTINE_COMPLETED,
            Version.V3_QUARANTINE,
            {
                "record_id": record.record_id,
                "experiment_id": experiment_id,
                "failure_category": record.failure_category,
                "root_cause": record.root_cause,
                "review_scheduled_at": record.review_scheduled_at.isoformat(),
            },
        )
        
        logger.info(f"Experiment {experiment_id} quarantined as {record.record_id}")
        return record
    
    async def _capture_evidence(
        self,
        experiment_id: str,
        failure_evidence: Dict[str, Any],
        error_logs: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Capture comprehensive failure evidence."""
        return {
            "experiment_id": experiment_id,
            "captured_at": datetime.utcnow().isoformat(),
            "failure_evidence": failure_evidence,
            "error_logs": error_logs,
            "metrics_snapshot": metrics,
            # In production, also capture:
            # - Code snapshot
            # - Configuration
            # - Prompts used
            # - Model versions
            # - API responses
        }
    
    async def _analyze_root_cause(
        self,
        failure_type: str,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform automated root cause analysis.
        
        Categories:
        - Technical: Model issues, API failures, resource limits
        - Quality: Accuracy problems, false positives/negatives
        - Operational: Cost overruns, scalability issues
        """
        # Analyze failure patterns
        category = self._categorize_failure(failure_type, evidence)
        
        # Identify root cause
        root_cause = self._identify_root_cause(category, evidence)
        
        # Find contributing factors
        factors = self._find_contributing_factors(evidence)
        
        # Assess impact
        impact = self._assess_impact(evidence)
        
        # Generate remediation steps
        remediation = self._generate_remediation(category, root_cause)
        
        # Determine if blacklisting is needed
        blacklist_info = self._evaluate_blacklist(category, evidence)
        
        return {
            "category": category,
            "root_cause": root_cause,
            "factors": factors,
            "impact": impact,
            "remediation": remediation,
            "blacklist_recommended": blacklist_info.get("recommended", False),
            "technique_name": blacklist_info.get("technique_name", ""),
            "blacklist_reason": blacklist_info.get("reason", ""),
        }
    
    def _categorize_failure(
        self,
        failure_type: str,
        evidence: Dict[str, Any],
    ) -> str:
        """Categorize failure type."""
        technical_indicators = [
            "timeout", "oom", "api_error", "model_error",
            "connection", "throttling", "rate_limit",
        ]
        
        quality_indicators = [
            "accuracy", "precision", "recall", "false_positive",
            "false_negative", "hallucination", "incorrect",
        ]
        
        operational_indicators = [
            "cost", "budget", "scale", "capacity", "throughput",
        ]
        
        failure_lower = failure_type.lower()
        
        if any(ind in failure_lower for ind in technical_indicators):
            return FailureCategory.TECHNICAL
        elif any(ind in failure_lower for ind in quality_indicators):
            return FailureCategory.QUALITY
        elif any(ind in failure_lower for ind in operational_indicators):
            return FailureCategory.OPERATIONAL
        
        return FailureCategory.TECHNICAL  # Default
    
    def _identify_root_cause(
        self,
        category: str,
        evidence: Dict[str, Any],
    ) -> str:
        """Identify primary root cause."""
        error_logs = evidence.get("error_logs", "")
        metrics = evidence.get("metrics_snapshot", {})
        
        if category == FailureCategory.TECHNICAL:
            if "timeout" in error_logs.lower():
                return "AI model API timeout exceeded threshold"
            elif "oom" in error_logs.lower():
                return "Memory exhaustion during analysis"
            elif "rate" in error_logs.lower():
                return "API rate limit exceeded"
            return "Technical failure in AI model integration"
        
        elif category == FailureCategory.QUALITY:
            accuracy = metrics.get("accuracy", 1.0)
            if accuracy < 0.85:
                return f"Accuracy degraded to {accuracy:.1%}, below 85% threshold"
            return "Quality metrics below acceptable thresholds"
        
        elif category == FailureCategory.OPERATIONAL:
            cost = metrics.get("cost_per_review", 0)
            if cost > 0.5:
                return f"Cost per review ${cost:.2f} exceeds budget"
            return "Operational constraints exceeded"
        
        return "Unknown failure cause"
    
    def _find_contributing_factors(
        self,
        evidence: Dict[str, Any],
    ) -> List[str]:
        """Identify contributing factors."""
        factors = []
        metrics = evidence.get("metrics_snapshot", {})
        
        if metrics.get("latency_p95", 0) > 5000:
            factors.append("High latency may have affected model performance")
        
        if metrics.get("error_rate", 0) > 0.05:
            factors.append("Elevated error rate during evaluation")
        
        if metrics.get("sample_size", 0) < 1000:
            factors.append("Insufficient sample size for reliable evaluation")
        
        return factors or ["No additional contributing factors identified"]
    
    def _assess_impact(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Assess failure impact."""
        return {
            "users_affected": 0,  # V1 doesn't affect real users
            "reviews_impacted": evidence.get("metrics_snapshot", {}).get("total_reviews", 0),
            "data_integrity": "intact",  # V1 is isolated
            "security_impact": "none",
        }
    
    def _generate_remediation(
        self,
        category: str,
        root_cause: str,
    ) -> List[str]:
        """Generate remediation steps."""
        if category == FailureCategory.TECHNICAL:
            return [
                "Review API integration code for error handling",
                "Implement retry logic with exponential backoff",
                "Add circuit breaker for external API calls",
                "Consider fallback models in routing chain",
            ]
        elif category == FailureCategory.QUALITY:
            return [
                "Review and refine prompts for better accuracy",
                "Consider using more capable model",
                "Add validation layer for AI outputs",
                "Implement human-in-the-loop for edge cases",
            ]
        elif category == FailureCategory.OPERATIONAL:
            return [
                "Optimize prompts to reduce token usage",
                "Implement request batching",
                "Add caching layer for repeated analyses",
                "Review resource allocation and scaling policies",
            ]
        
        return ["Investigate further before retry"]
    
    def _evaluate_blacklist(
        self,
        category: str,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine if technique should be blacklisted."""
        metrics = evidence.get("metrics_snapshot", {})
        
        # Blacklist if:
        # - Security vulnerability missed
        # - Accuracy below 70%
        # - Cost 3x budget
        
        if metrics.get("security_misses", 0) > 0:
            return {
                "recommended": True,
                "technique_name": evidence.get("experiment_id", "unknown"),
                "reason": "Security vulnerability detection failure",
            }
        
        if metrics.get("accuracy", 1.0) < 0.70:
            return {
                "recommended": True,
                "technique_name": evidence.get("experiment_id", "unknown"),
                "reason": f"Accuracy {metrics.get('accuracy', 0):.1%} unacceptably low",
            }
        
        return {"recommended": False}
    
    async def _add_to_blacklist(
        self,
        technique_name: str,
        reason: str,
        experiment_id: str,
        category: str,
    ) -> BlacklistEntry:
        """Add technique to blacklist."""
        entry = BlacklistEntry(
            entry_id=str(uuid.uuid4()),
            technique_name=technique_name,
            reason=reason,
            experiment_id=experiment_id,
            category=category,
            severity="high",
            created_at=datetime.utcnow(),
        )
        
        self._blacklist[entry.entry_id] = entry
        
        logger.warning(f"Added to blacklist: {technique_name} - {reason}")
        return entry
    
    async def review_quarantined(
        self,
        record_id: str,
        reviewer: str,
        retry_approved: bool,
        notes: str = "",
    ) -> bool:
        """Review quarantined experiment for potential retry."""
        record = self._quarantine_records.get(record_id)
        if not record:
            return False
        
        record.reviewed_at = datetime.utcnow()
        record.retry_approved = retry_approved
        
        await self._emit_event(
            EventType.QUARANTINE_REVIEWED,
            Version.V3_QUARANTINE,
            {
                "record_id": record_id,
                "experiment_id": record.experiment_id,
                "reviewer": reviewer,
                "retry_approved": retry_approved,
                "notes": notes,
            },
        )
        
        if retry_approved:
            # Remove from blacklist if exists
            if record.blacklist_entry:
                del self._blacklist[record.blacklist_entry]
                record.blacklist_entry = None
            
            await self._emit_event(
                EventType.QUARANTINE_RETRY_APPROVED,
                Version.V1_EXPERIMENTATION,
                {
                    "experiment_id": record.experiment_id,
                    "original_record_id": record_id,
                },
            )
        
        return True
    
    def get_quarantine_records(
        self,
        pending_review: bool = False,
    ) -> List[QuarantineRecord]:
        """Get quarantine records."""
        records = list(self._quarantine_records.values())
        
        if pending_review:
            now = datetime.utcnow()
            records = [
                r for r in records
                if r.review_scheduled_at and r.review_scheduled_at <= now
                and r.reviewed_at is None
            ]
        
        return records
    
    def get_blacklist(self) -> List[BlacklistEntry]:
        """Get current blacklist."""
        return list(self._blacklist.values())
    
    def is_blacklisted(self, technique_name: str) -> bool:
        """Check if technique is blacklisted."""
        return any(
            e.technique_name == technique_name
            for e in self._blacklist.values()
        )
    
    async def _emit_event(
        self,
        event_type: EventType,
        version: Version,
        payload: Dict[str, Any],
    ):
        """Emit event to event bus."""
        event = VersionEvent(
            event_type=event_type,
            version=version,
            payload=payload,
            source="quarantine-manager",
        )
        
        if self.event_bus:
            await self.event_bus.publish(event_type.value, event.to_dict())
