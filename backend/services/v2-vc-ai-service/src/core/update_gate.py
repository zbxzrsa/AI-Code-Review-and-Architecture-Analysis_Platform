"""
V2 VC-AI Update Gate

Multi-stage validation pipeline for accepting innovations from V1.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..config.update_gate_config import (
    UPDATE_VALIDATION_PIPELINE,
    VALIDATION_THRESHOLDS,
    AUTOMATED_GATE_CHECKS,
    GateDecision,
    ValidationStage,
)


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StageResult:
    """Result of a validation stage"""
    stage: ValidationStage
    decision: GateDecision
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_minutes: float = 0
    validations: List[ValidationResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class GateResult:
    """Overall gate result"""
    experiment_id: str
    final_decision: GateDecision
    current_stage: ValidationStage
    stages: List[StageResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    ready_for_production: bool = False


class UpdateGate:
    """
    Multi-stage update gate for validating V1 experiments before V2 promotion.
    
    Stages:
    1. V1 Qualification - Entry requirements check
    2. Staging Deployment - Full test suite
    3. Canary Deployment - Limited production traffic
    4. Progressive Rollout - Gradual traffic increase
    5. Full Production - Ongoing monitoring
    """
    
    def __init__(
        self,
        v2_baseline_metrics: Optional[Dict[str, float]] = None,
    ):
        self.v2_baseline = v2_baseline_metrics or {
            "accuracy": 0.95,
            "latency_p99_ms": 500,
            "error_rate": 0.001,
            "throughput_rps": 100,
            "cost_per_request": 0.01,
        }
        
        self._active_gates: Dict[str, GateResult] = {}
        self._pipeline_config = UPDATE_VALIDATION_PIPELINE
        self._thresholds = VALIDATION_THRESHOLDS
    
    async def start_gate(self, experiment_id: str, v1_metrics: Dict[str, Any]) -> GateResult:
        """Start the update gate process for an experiment"""
        logger.info(f"Starting update gate for experiment {experiment_id}")
        
        gate = GateResult(
            experiment_id=experiment_id,
            final_decision=GateDecision.CONDITIONAL,
            current_stage=ValidationStage.V1_QUALIFICATION,
        )
        
        self._active_gates[experiment_id] = gate
        
        # Start Stage 1: V1 Qualification
        stage_result = self._run_v1_qualification(experiment_id, v1_metrics)
        gate.stages.append(stage_result)
        
        if stage_result.decision == GateDecision.REJECT:
            gate.final_decision = GateDecision.REJECT
            gate.completed_at = datetime.now(timezone.utc)
            return gate
        
        gate.current_stage = ValidationStage.STAGING_DEPLOYMENT
        return gate
    
    def _run_v1_qualification(
        self,
        experiment_id: str,  # noqa: ARG002 - Reserved for experiment tracking
        v1_metrics: Dict[str, Any],
    ) -> StageResult:
        """Run Stage 1: V1 Qualification"""
        stage = StageResult(
            stage=ValidationStage.V1_QUALIFICATION,
            decision=GateDecision.CONDITIONAL,
            started_at=datetime.now(timezone.utc),
        )
        
        _ = self._pipeline_config["stage_1_v1_qualification"]  # noqa: F841 - config reserved
        
        # Check entry requirements
        validations = []
        
        # 1. Data duration check
        data_days = v1_metrics.get("experiment_duration_days", 0)
        validations.append(ValidationResult(
            check_name="data_duration",
            passed=data_days >= 7,
            message=f"Experiment has {data_days} days of data (min: 7)",
            details={"days": data_days, "required": 7},
        ))
        
        # 2. Accuracy improvement check
        v1_accuracy = v1_metrics.get("accuracy", 0)
        baseline_accuracy = self.v2_baseline["accuracy"]
        required_accuracy = baseline_accuracy * 1.05  # 5% improvement
        validations.append(ValidationResult(
            check_name="accuracy_improvement",
            passed=v1_accuracy >= required_accuracy,
            message=f"Accuracy {v1_accuracy:.4f} vs required {required_accuracy:.4f}",
            details={
                "v1_accuracy": v1_accuracy,
                "baseline": baseline_accuracy,
                "required": required_accuracy,
            },
        ))
        
        # 3. Latency check
        v1_latency = v1_metrics.get("latency_p99_ms", float("inf"))
        baseline_latency = self.v2_baseline["latency_p99_ms"]
        validations.append(ValidationResult(
            check_name="latency_check",
            passed=v1_latency <= baseline_latency,
            message=f"P99 latency {v1_latency}ms vs baseline {baseline_latency}ms",
            details={
                "v1_latency": v1_latency,
                "baseline": baseline_latency,
            },
        ))
        
        # 4. Error rate check
        v1_error_rate = v1_metrics.get("error_rate", 1.0)
        baseline_error_rate = self.v2_baseline["error_rate"]
        validations.append(ValidationResult(
            check_name="error_rate_check",
            passed=v1_error_rate <= baseline_error_rate,
            message=f"Error rate {v1_error_rate:.4%} vs baseline {baseline_error_rate:.4%}",
            details={
                "v1_error_rate": v1_error_rate,
                "baseline": baseline_error_rate,
            },
        ))
        
        # 5. Cost check
        v1_cost = v1_metrics.get("cost_per_request", float("inf"))
        baseline_cost = self.v2_baseline["cost_per_request"]
        max_cost = baseline_cost * 1.05  # Max 5% increase
        validations.append(ValidationResult(
            check_name="cost_check",
            passed=v1_cost <= max_cost,
            message=f"Cost ${v1_cost:.4f} vs max ${max_cost:.4f}",
            details={
                "v1_cost": v1_cost,
                "baseline": baseline_cost,
                "max_allowed": max_cost,
            },
        ))
        
        # 6. Security audit check
        security_score = v1_metrics.get("security_score", 0)
        validations.append(ValidationResult(
            check_name="security_audit",
            passed=security_score >= 0.95,
            message=f"Security score {security_score:.2%} vs required 95%",
            details={"score": security_score, "required": 0.95},
        ))
        
        # 7. Zero regressions check
        regressions = v1_metrics.get("regression_count", 0)
        validations.append(ValidationResult(
            check_name="zero_regressions",
            passed=regressions == 0,
            message=f"Regressions found: {regressions}",
            details={"regressions": regressions},
        ))
        
        stage.validations = validations
        
        # Determine decision
        critical_checks = ["accuracy_improvement", "security_audit", "zero_regressions"]
        critical_failures = [v for v in validations if not v.passed and v.check_name in critical_checks]
        all_failures = [v for v in validations if not v.passed]
        
        if critical_failures:
            stage.decision = GateDecision.REJECT
            stage.notes.append(f"Critical failures: {[f.check_name for f in critical_failures]}")
        elif all_failures:
            stage.decision = GateDecision.CONDITIONAL
            stage.notes.append(f"Minor issues to address: {[f.check_name for f in all_failures]}")
        else:
            stage.decision = GateDecision.PASS
        
        stage.completed_at = datetime.now(timezone.utc)
        stage.duration_minutes = (stage.completed_at - stage.started_at).total_seconds() / 60
        
        logger.info(f"Stage 1 completed for {experiment_id}: {stage.decision}")
        return stage
    
    async def run_staging_deployment(
        self,
        experiment_id: str,
        test_results: Dict[str, Any],
    ) -> StageResult:
        """Run Stage 2: Staging Deployment"""
        if experiment_id not in self._active_gates:
            raise ValueError(f"No active gate for experiment {experiment_id}")
        
        gate = self._active_gates[experiment_id]
        
        stage = StageResult(
            stage=ValidationStage.STAGING_DEPLOYMENT,
            decision=GateDecision.CONDITIONAL,
            started_at=datetime.now(timezone.utc),
        )
        
        validations = []
        
        # 1. Regression test suite
        regression_passed = test_results.get("regression_tests_passed", 0)
        regression_total = test_results.get("regression_tests_total", 2000)
        validations.append(ValidationResult(
            check_name="regression_suite",
            passed=regression_passed == regression_total,
            message=f"Regression tests: {regression_passed}/{regression_total}",
            details={"passed": regression_passed, "total": regression_total},
        ))
        
        # 2. Load testing
        load_test_passed = test_results.get("load_test_passed", False)
        validations.append(ValidationResult(
            check_name="load_testing",
            passed=load_test_passed,
            message=f"Load test (5x peak): {'PASSED' if load_test_passed else 'FAILED'}",
            details=test_results.get("load_test_details", {}),
        ))
        
        # 3. Stress testing
        stress_test_passed = test_results.get("stress_test_passed", False)
        validations.append(ValidationResult(
            check_name="stress_testing",
            passed=stress_test_passed,
            message=f"Stress test: {'PASSED' if stress_test_passed else 'FAILED'}",
            details=test_results.get("stress_test_details", {}),
        ))
        
        # 4. Chaos engineering
        chaos_passed = test_results.get("chaos_test_passed", False)
        validations.append(ValidationResult(
            check_name="chaos_engineering",
            passed=chaos_passed,
            message=f"Chaos engineering: {'PASSED' if chaos_passed else 'FAILED'}",
            details=test_results.get("chaos_test_details", {}),
        ))
        
        # 5. Disaster recovery
        dr_passed = test_results.get("disaster_recovery_passed", False)
        validations.append(ValidationResult(
            check_name="disaster_recovery",
            passed=dr_passed,
            message=f"Disaster recovery: {'PASSED' if dr_passed else 'FAILED'}",
            details=test_results.get("dr_test_details", {}),
        ))
        
        stage.validations = validations
        
        # Determine decision
        all_passed = all(v.passed for v in validations)
        critical_failures = [v for v in validations if not v.passed and v.check_name in ["regression_suite", "load_testing"]]
        
        if all_passed:
            stage.decision = GateDecision.APPROVED
        elif critical_failures:
            stage.decision = GateDecision.REJECT
        else:
            stage.decision = GateDecision.NEEDS_FIX
        
        stage.completed_at = datetime.now(timezone.utc)
        stage.duration_minutes = (stage.completed_at - stage.started_at).total_seconds() / 60
        
        gate.stages.append(stage)
        
        if stage.decision == GateDecision.APPROVED:
            gate.current_stage = ValidationStage.CANARY_DEPLOYMENT
        elif stage.decision == GateDecision.REJECT:
            gate.final_decision = GateDecision.REJECT
            gate.completed_at = datetime.now(timezone.utc)
        
        logger.info(f"Stage 2 completed for {experiment_id}: {stage.decision}")
        return stage
    
    async def run_canary_deployment(
        self,
        experiment_id: str,
        canary_metrics: Dict[str, Any],
        duration_hours: float = 4.0,
    ) -> StageResult:
        """Run Stage 3: Canary Deployment"""
        if experiment_id not in self._active_gates:
            raise ValueError(f"No active gate for experiment {experiment_id}")
        
        gate = self._active_gates[experiment_id]
        
        stage = StageResult(
            stage=ValidationStage.CANARY_DEPLOYMENT,
            decision=GateDecision.CONDITIONAL,
            started_at=datetime.now(timezone.utc),
        )
        
        config = self._pipeline_config["stage_3_canary_deployment"]
        validations = []
        
        # Monitor metrics against thresholds
        error_rate = canary_metrics.get("error_rate", 1.0)
        error_threshold = config["monitoring"]["metrics"]["error_rate"]["threshold"]
        validations.append(ValidationResult(
            check_name="canary_error_rate",
            passed=error_rate < error_threshold,
            message=f"Error rate {error_rate:.4%} vs threshold {error_threshold:.4%}",
            details={"actual": error_rate, "threshold": error_threshold},
        ))
        
        latency_p99 = canary_metrics.get("latency_p99_ms", float("inf"))
        latency_threshold = config["monitoring"]["metrics"]["latency_p99_ms"]["threshold"]
        validations.append(ValidationResult(
            check_name="canary_latency",
            passed=latency_p99 < latency_threshold,
            message=f"P99 latency {latency_p99}ms vs threshold {latency_threshold}ms",
            details={"actual": latency_p99, "threshold": latency_threshold},
        ))
        
        # Agreement rate with V2
        agreement_rate = canary_metrics.get("agreement_rate", 0)
        agreement_threshold = config["comparison"]["agreement_rate_threshold"]
        validations.append(ValidationResult(
            check_name="agreement_rate",
            passed=agreement_rate >= agreement_threshold,
            message=f"Agreement rate {agreement_rate:.2%} vs threshold {agreement_threshold:.2%}",
            details={"actual": agreement_rate, "threshold": agreement_threshold},
        ))
        
        stage.validations = validations
        stage.metrics = canary_metrics
        
        # Determine decision
        all_passed = all(v.passed for v in validations)
        stage.decision = GateDecision.PROCEED if all_passed else GateDecision.ROLLBACK
        
        stage.completed_at = datetime.now(timezone.utc)
        stage.duration_minutes = (stage.completed_at - stage.started_at).total_seconds() / 60
        
        gate.stages.append(stage)
        
        if stage.decision == GateDecision.PROCEED:
            gate.current_stage = ValidationStage.PROGRESSIVE_ROLLOUT
        else:
            gate.final_decision = GateDecision.ROLLBACK
            gate.completed_at = datetime.now(timezone.utc)
        
        logger.info(f"Stage 3 completed for {experiment_id}: {stage.decision}")
        return stage
    
    async def run_progressive_rollout_step(
        self,
        experiment_id: str,
        current_percentage: int,
        step_metrics: Dict[str, Any],
    ) -> Tuple[GateDecision, Optional[str]]:
        """Run a single step of Stage 4: Progressive Rollout"""
        if experiment_id not in self._active_gates:
            raise ValueError(f"No active gate for experiment {experiment_id}")
        
        gate = self._active_gates[experiment_id]
        config = self._pipeline_config["stage_4_progressive_rollout"]
        
        # Check rollback triggers
        for trigger in config["rollback_triggers"]:
            metric_value = step_metrics.get(trigger["metric"])
            if metric_value is not None:
                if trigger["threshold"] == "any":
                    if metric_value:
                        return GateDecision.ROLLBACK, f"Rollback triggered: {trigger['metric']}"
                elif metric_value > trigger["threshold"]:
                    return GateDecision.ROLLBACK, f"Rollback triggered: {trigger['metric']} = {metric_value}"
        
        # Find next step in schedule
        schedule = config["schedule"]
        next_step = None
        for step in schedule:
            if step["traffic_percentage"] > current_percentage:
                next_step = step
                break
        
        if next_step is None:
            # Rollout complete
            gate.current_stage = ValidationStage.FULL_PRODUCTION
            gate.final_decision = GateDecision.APPROVED
            gate.ready_for_production = True
            gate.completed_at = datetime.now(timezone.utc)
            return GateDecision.CONTINUE, f"Rollout complete, now at 100%"
        
        return GateDecision.CONTINUE, f"Proceeding to {next_step['traffic_percentage']}%"
    
    def get_gate_status(self, experiment_id: str) -> Optional[GateResult]:
        """Get current gate status for an experiment"""
        return self._active_gates.get(experiment_id)
    
    def get_all_active_gates(self) -> Dict[str, GateResult]:
        """Get all active gates"""
        return self._active_gates.copy()
    
    async def abort_gate(self, experiment_id: str, reason: str) -> bool:
        """Abort an active gate"""
        if experiment_id not in self._active_gates:
            return False
        
        gate = self._active_gates[experiment_id]
        gate.final_decision = GateDecision.REJECT
        gate.completed_at = datetime.now(timezone.utc)
        
        # Add abort note to current stage
        if gate.stages:
            gate.stages[-1].notes.append(f"Gate aborted: {reason}")
        
        logger.warning(f"Gate aborted for {experiment_id}: {reason}")
        return True
