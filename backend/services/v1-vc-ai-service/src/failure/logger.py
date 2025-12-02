"""
Failure Logging Protocol for V1 VC-AI

Comprehensive failure documentation with:
- Automatic trigger detection
- V3 quarantine integration
- Root cause analysis
- Blacklist management
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import uuid
import json
import logging
import asyncio

import httpx


logger = logging.getLogger(__name__)


class FailureType(str, Enum):
    """Types of failures"""
    ACCURACY_DROP = "accuracy_drop"
    LATENCY_INCREASE = "latency_increase"
    INFERENCE_ERROR = "inference_error"
    HALLUCINATION = "hallucination"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    TRAINING_DIVERGENCE = "training_divergence"
    OOM_ERROR = "oom_error"
    TIMEOUT = "timeout"
    VALIDATION_FAILURE = "validation_failure"
    SECURITY_VIOLATION = "security_violation"
    UNKNOWN = "unknown"


class BlockingLevel(str, Enum):
    """How severely this failure blocks progress"""
    CRITICAL = "critical"      # Stop all experiments
    HIGH = "high"              # Block similar experiments
    MEDIUM = "medium"          # Warn but allow retry
    LOW = "low"                # Log and continue


class FixComplexity(str, Enum):
    """Estimated complexity to fix the issue"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMPOSSIBLE = "impossible"


@dataclass
class FailureTrigger:
    """Trigger condition for failure detection"""
    name: str
    description: str
    threshold: float
    comparison: str  # gt, lt, gte, lte, eq
    current_value: float
    baseline_value: Optional[float] = None
    
    def is_triggered(self, value: float) -> bool:
        """Check if the trigger condition is met"""
        if self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        elif self.comparison == "eq":
            return value == self.threshold
        return False


@dataclass
class MetricsAtFailure:
    """Metrics captured at the time of failure"""
    accuracy: float
    latency_ms: int
    memory_gb: float
    cost_per_request: float
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    gpu_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "latency_ms": self.latency_ms,
            "memory_gb": self.memory_gb,
            "cost_per_request": self.cost_per_request,
            "throughput_rps": self.throughput_rps,
            "error_rate": self.error_rate,
            "gpu_utilization": self.gpu_utilization,
        }


@dataclass
class FailureRecord:
    """Complete failure record"""
    failure_id: str
    timestamp: datetime
    v1_version: str
    experiment_id: str
    technique_attempted: str
    failure_type: FailureType
    blocking_level: BlockingLevel
    
    # Trigger that caused the failure
    trigger: FailureTrigger
    
    # Metrics at failure
    metrics_at_failure: MetricsAtFailure
    
    # Analysis
    root_cause_analysis: str
    reproduction_steps: List[str]
    estimated_fix_complexity: FixComplexity
    
    # Additional context
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # V3 integration
    pushed_to_v3: bool = False
    v3_quarantine_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "timestamp": self.timestamp.isoformat(),
            "v1_version": self.v1_version,
            "experiment_id": self.experiment_id,
            "technique_attempted": self.technique_attempted,
            "failure_type": self.failure_type.value,
            "blocking_level": self.blocking_level.value,
            "trigger": {
                "name": self.trigger.name,
                "description": self.trigger.description,
                "threshold": self.trigger.threshold,
                "comparison": self.trigger.comparison,
                "current_value": self.trigger.current_value,
                "baseline_value": self.trigger.baseline_value,
            },
            "metrics_at_failure": self.metrics_at_failure.to_dict(),
            "root_cause_analysis": self.root_cause_analysis,
            "reproduction_steps": self.reproduction_steps,
            "estimated_fix_complexity": self.estimated_fix_complexity.value,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "config_snapshot": self.config_snapshot,
            "pushed_to_v3": self.pushed_to_v3,
            "v3_quarantine_id": self.v3_quarantine_id,
        }


class FailureLogger:
    """
    Main failure logging service.
    
    Handles:
    1. Detecting failures based on trigger conditions
    2. Creating detailed failure records
    3. Pushing failures to V3 quarantine
    4. Managing blacklisted techniques
    """
    
    # Default trigger conditions
    DEFAULT_TRIGGERS = [
        FailureTrigger(
            name="accuracy_drop",
            description="Accuracy dropped more than 5% from baseline",
            threshold=0.05,
            comparison="gt",
            current_value=0.0,
        ),
        FailureTrigger(
            name="latency_increase",
            description="Latency increased more than 50% from baseline",
            threshold=0.50,
            comparison="gt",
            current_value=0.0,
        ),
        FailureTrigger(
            name="inference_error_rate",
            description="Inference error rate exceeds 2%",
            threshold=0.02,
            comparison="gt",
            current_value=0.0,
        ),
        FailureTrigger(
            name="memory_usage",
            description="Memory usage exceeds 90% of available",
            threshold=0.90,
            comparison="gt",
            current_value=0.0,
        ),
        FailureTrigger(
            name="training_loss",
            description="Training loss is NaN or Inf (divergence)",
            threshold=float('inf'),
            comparison="eq",
            current_value=0.0,
        ),
    ]
    
    def __init__(
        self,
        v3_api_endpoint: str = "http://v3-quarantine-service:8000/api/v3/quarantine/failures",
        v3_webhook_url: Optional[str] = None,
        response_time_sla_seconds: float = 5.0,
    ):
        self.v3_api_endpoint = v3_api_endpoint
        self.v3_webhook_url = v3_webhook_url
        self.response_time_sla_seconds = response_time_sla_seconds
        
        # Failure storage (in production, use database)
        self.failures: Dict[str, FailureRecord] = {}
        
        # Blacklisted techniques
        self.blacklist: Dict[str, List[str]] = {}  # technique -> failure_ids
        
        # Active triggers
        self.triggers = self.DEFAULT_TRIGGERS.copy()
        
        # HTTP client
        self.http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=self.response_time_sla_seconds)
        return self.http_client
    
    async def close(self):
        """Close HTTP client"""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    def check_triggers(
        self,
        current_metrics: MetricsAtFailure,
        baseline_metrics: Optional[MetricsAtFailure] = None,
    ) -> List[FailureTrigger]:
        """Check all triggers against current metrics"""
        triggered = []
        
        for trigger in self.triggers:
            current_value = 0.0
            baseline_value = None
            
            if trigger.name == "accuracy_drop" and baseline_metrics:
                current_value = baseline_metrics.accuracy - current_metrics.accuracy
                baseline_value = baseline_metrics.accuracy
            elif trigger.name == "latency_increase" and baseline_metrics:
                if baseline_metrics.latency_ms > 0:
                    current_value = (current_metrics.latency_ms - baseline_metrics.latency_ms) / baseline_metrics.latency_ms
                baseline_value = baseline_metrics.latency_ms
            elif trigger.name == "inference_error_rate":
                current_value = current_metrics.error_rate
            elif trigger.name == "memory_usage":
                current_value = current_metrics.memory_gb / 24.0  # Assuming 24GB GPU
            elif trigger.name == "training_loss":
                # This would be passed externally
                pass
            
            trigger.current_value = current_value
            trigger.baseline_value = baseline_value
            
            if trigger.is_triggered(current_value):
                triggered.append(trigger)
        
        return triggered
    
    def detect_failure_type(
        self,
        triggered: List[FailureTrigger],
        error_message: Optional[str] = None,
    ) -> FailureType:
        """Determine the failure type from triggers and error"""
        if error_message:
            error_lower = error_message.lower()
            if "out of memory" in error_lower or "oom" in error_lower:
                return FailureType.OOM_ERROR
            if "timeout" in error_lower:
                return FailureType.TIMEOUT
            if "nan" in error_lower or "inf" in error_lower:
                return FailureType.TRAINING_DIVERGENCE
            if "hallucination" in error_lower:
                return FailureType.HALLUCINATION
            if "security" in error_lower or "injection" in error_lower:
                return FailureType.SECURITY_VIOLATION
        
        if not triggered:
            return FailureType.UNKNOWN
        
        # Map triggers to failure types
        trigger_names = {t.name for t in triggered}
        
        if "accuracy_drop" in trigger_names:
            return FailureType.ACCURACY_DROP
        if "latency_increase" in trigger_names:
            return FailureType.LATENCY_INCREASE
        if "inference_error_rate" in trigger_names:
            return FailureType.INFERENCE_ERROR
        if "memory_usage" in trigger_names:
            return FailureType.MEMORY_EXHAUSTION
        if "training_loss" in trigger_names:
            return FailureType.TRAINING_DIVERGENCE
        
        return FailureType.UNKNOWN
    
    def determine_blocking_level(
        self,
        failure_type: FailureType,
        triggered: List[FailureTrigger],
    ) -> BlockingLevel:
        """Determine how severely this failure should block progress"""
        # Critical failures
        if failure_type in [FailureType.SECURITY_VIOLATION, FailureType.OOM_ERROR]:
            return BlockingLevel.CRITICAL
        
        if failure_type in [FailureType.TRAINING_DIVERGENCE, FailureType.HALLUCINATION]:
            return BlockingLevel.HIGH
        
        # Check severity of triggers
        for trigger in triggered:
            if trigger.name == "accuracy_drop" and trigger.current_value > 0.10:
                return BlockingLevel.HIGH
            if trigger.name == "latency_increase" and trigger.current_value > 1.0:
                return BlockingLevel.HIGH
        
        if failure_type in [FailureType.ACCURACY_DROP, FailureType.LATENCY_INCREASE]:
            return BlockingLevel.MEDIUM
        
        return BlockingLevel.LOW
    
    def estimate_fix_complexity(
        self,
        failure_type: FailureType,
        root_cause: str,
    ) -> FixComplexity:
        """Estimate how complex it would be to fix this issue"""
        # Simple heuristics
        if failure_type == FailureType.SECURITY_VIOLATION:
            return FixComplexity.HIGH
        
        if failure_type in [FailureType.TRAINING_DIVERGENCE, FailureType.HALLUCINATION]:
            if "architecture" in root_cause.lower():
                return FixComplexity.HIGH
            return FixComplexity.MEDIUM
        
        if failure_type == FailureType.OOM_ERROR:
            if "batch_size" in root_cause.lower():
                return FixComplexity.LOW
            return FixComplexity.MEDIUM
        
        if failure_type in [FailureType.ACCURACY_DROP, FailureType.LATENCY_INCREASE]:
            return FixComplexity.MEDIUM
        
        return FixComplexity.MEDIUM
    
    async def log_failure(
        self,
        experiment_id: str,
        v1_version: str,
        technique_attempted: str,
        current_metrics: MetricsAtFailure,
        baseline_metrics: Optional[MetricsAtFailure] = None,
        root_cause_analysis: str = "",
        reproduction_steps: Optional[List[str]] = None,
        error_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        auto_push_to_v3: bool = True,
    ) -> FailureRecord:
        """
        Log a failure with full documentation.
        
        Args:
            experiment_id: ID of the failed experiment
            v1_version: Version of V1 VC-AI
            technique_attempted: The technique/architecture being tested
            current_metrics: Metrics at the time of failure
            baseline_metrics: Baseline metrics for comparison
            root_cause_analysis: Analysis of what caused the failure
            reproduction_steps: Steps to reproduce the failure
            error_message: Error message if any
            stack_trace: Stack trace if available
            config_snapshot: Configuration at time of failure
            auto_push_to_v3: Automatically push to V3 quarantine
            
        Returns:
            Created FailureRecord
        """
        # Check triggers
        triggered = self.check_triggers(current_metrics, baseline_metrics)
        
        if not triggered and not error_message:
            # Force a trigger for explicit failure logging
            triggered = [FailureTrigger(
                name="explicit_failure",
                description="Explicitly logged failure",
                threshold=1.0,
                comparison="eq",
                current_value=1.0,
            )]
        
        # Determine failure type and blocking level
        failure_type = self.detect_failure_type(triggered, error_message)
        blocking_level = self.determine_blocking_level(failure_type, triggered)
        fix_complexity = self.estimate_fix_complexity(failure_type, root_cause_analysis)
        
        # Create failure record
        failure = FailureRecord(
            failure_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            v1_version=v1_version,
            experiment_id=experiment_id,
            technique_attempted=technique_attempted,
            failure_type=failure_type,
            blocking_level=blocking_level,
            trigger=triggered[0] if triggered else FailureTrigger(
                name="unknown", description="", threshold=0, comparison="gt", current_value=0
            ),
            metrics_at_failure=current_metrics,
            root_cause_analysis=root_cause_analysis,
            reproduction_steps=reproduction_steps or [],
            estimated_fix_complexity=fix_complexity,
            error_message=error_message,
            stack_trace=stack_trace,
            config_snapshot=config_snapshot or {},
        )
        
        # Store failure
        self.failures[failure.failure_id] = failure
        
        # Add to blacklist if blocking is high or critical
        if blocking_level in [BlockingLevel.CRITICAL, BlockingLevel.HIGH]:
            self._add_to_blacklist(technique_attempted, failure.failure_id)
        
        # Push to V3
        if auto_push_to_v3:
            await self._push_to_v3(failure)
        
        logger.warning(
            f"Failure logged: {failure.failure_id} - {failure_type.value} "
            f"(blocking: {blocking_level.value})"
        )
        
        return failure
    
    def _add_to_blacklist(self, technique: str, failure_id: str):
        """Add a technique to the blacklist"""
        if technique not in self.blacklist:
            self.blacklist[technique] = []
        self.blacklist[technique].append(failure_id)
        logger.warning(f"Technique '{technique}' added to blacklist")
    
    def is_blacklisted(self, technique: str) -> bool:
        """Check if a technique is blacklisted"""
        return technique in self.blacklist
    
    def get_blacklist_reasons(self, technique: str) -> List[FailureRecord]:
        """Get the failures that caused a technique to be blacklisted"""
        if technique not in self.blacklist:
            return []
        
        return [
            self.failures[fid] for fid in self.blacklist[technique]
            if fid in self.failures
        ]
    
    async def _push_to_v3(self, failure: FailureRecord) -> bool:
        """Push failure to V3 quarantine service"""
        try:
            client = await self._get_client()
            
            response = await client.post(
                self.v3_api_endpoint,
                json=failure.to_dict(),
            )
            
            if response.status_code == 200 or response.status_code == 201:
                result = response.json()
                failure.pushed_to_v3 = True
                failure.v3_quarantine_id = result.get("quarantine_id")
                logger.info(f"Failure {failure.failure_id} pushed to V3: {failure.v3_quarantine_id}")
                return True
            else:
                logger.error(f"Failed to push to V3: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error pushing to V3: {e}")
            return False
    
    async def send_webhook(self, failure: FailureRecord):
        """Send webhook notification for new failure"""
        if not self.v3_webhook_url:
            return
        
        try:
            client = await self._get_client()
            
            await client.post(
                self.v3_webhook_url,
                json={
                    "event": "v1_failure",
                    "failure_id": failure.failure_id,
                    "failure_type": failure.failure_type.value,
                    "blocking_level": failure.blocking_level.value,
                    "experiment_id": failure.experiment_id,
                    "timestamp": failure.timestamp.isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
    
    def get_failure(self, failure_id: str) -> Optional[FailureRecord]:
        """Get a specific failure record"""
        return self.failures.get(failure_id)
    
    def get_failures_by_experiment(self, experiment_id: str) -> List[FailureRecord]:
        """Get all failures for an experiment"""
        return [
            f for f in self.failures.values()
            if f.experiment_id == experiment_id
        ]
    
    def get_failures_by_type(self, failure_type: FailureType) -> List[FailureRecord]:
        """Get all failures of a specific type"""
        return [
            f for f in self.failures.values()
            if f.failure_type == failure_type
        ]
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged failures"""
        total = len(self.failures)
        by_type = {}
        by_blocking = {}
        by_complexity = {}
        
        for f in self.failures.values():
            by_type[f.failure_type.value] = by_type.get(f.failure_type.value, 0) + 1
            by_blocking[f.blocking_level.value] = by_blocking.get(f.blocking_level.value, 0) + 1
            by_complexity[f.estimated_fix_complexity.value] = by_complexity.get(f.estimated_fix_complexity.value, 0) + 1
        
        return {
            "total_failures": total,
            "by_type": by_type,
            "by_blocking_level": by_blocking,
            "by_fix_complexity": by_complexity,
            "blacklisted_techniques": len(self.blacklist),
            "pushed_to_v3": sum(1 for f in self.failures.values() if f.pushed_to_v3),
        }
