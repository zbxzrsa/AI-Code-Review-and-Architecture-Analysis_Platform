"""
Lifecycle Controller Service

Manages the promotion/downgrade lifecycle across V1, V2, and V3:
- V1 → V2 promotion (after shadow evaluation passes)
- V1 → V3 downgrade (on failure)
- V3 → V1 re-evaluation cycle
- Integrates with OPA for policy-as-code decisions
- Triggers Argo Rollouts for gray-scale promotion
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VersionState(str, Enum):
    """Version states in the lifecycle"""
    EXPERIMENT = "experiment"      # V1: Active experimentation
    SHADOW = "shadow"              # V1: Shadow traffic evaluation
    GRAY_1 = "gray_1_percent"      # V2: 1% traffic
    GRAY_5 = "gray_5_percent"      # V2: 5% traffic
    GRAY_25 = "gray_25_percent"    # V2: 25% traffic
    GRAY_50 = "gray_50_percent"    # V2: 50% traffic
    STABLE = "stable"              # V2: 100% production
    QUARANTINE = "quarantine"      # V3: Failed/archived
    RE_EVALUATION = "re_evaluation" # V3 → V1: Recovery attempt


class EvaluationResult(str, Enum):
    """Results from evaluation pipeline"""
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


@dataclass
class PromotionThresholds:
    """Configurable thresholds for promotion decisions"""
    p95_latency_ms: float = 3000.0
    error_rate: float = 0.02
    accuracy_delta: float = 0.02  # Must be >= 2% better
    security_pass_rate: float = 0.99
    cost_increase_max: float = 0.10  # Max 10% cost increase
    min_shadow_requests: int = 1000
    min_shadow_duration_hours: int = 24
    statistical_significance_p: float = 0.05
    consecutive_failures_for_downgrade: int = 3


@dataclass 
class EvaluationMetrics:
    """Metrics collected during evaluation"""
    total_requests: int = 0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_rate: float = 0.0
    accuracy: float = 0.0
    accuracy_delta: float = 0.0
    security_pass_rate: float = 0.0
    cost_per_request: float = 0.0
    cost_delta: float = 0.0
    
    # Statistical test results
    accuracy_p_value: float = 1.0
    latency_p_value: float = 1.0
    cost_p_value: float = 1.0
    
    evaluation_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evaluation_end: Optional[datetime] = None


class VersionConfig(BaseModel):
    """Configuration for a version being managed"""
    version_id: str
    model_version: str
    prompt_version: str
    routing_policy_version: str
    current_state: VersionState
    created_at: datetime
    last_evaluation: Optional[datetime] = None
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = {}


class LifecycleController:
    """
    Main controller for version lifecycle management.
    
    Responsibilities:
    - Monitor shadow traffic evaluation results
    - Make promotion/downgrade decisions using OPA
    - Trigger Argo Rollouts for gray-scale
    - Manage V3 recovery cycles
    """
    
    def __init__(
        self,
        opa_url: str = "http://opa.platform-control-plane.svc:8181",
        argo_url: str = "http://argo-rollouts.platform-control-plane.svc:8080",
        prometheus_url: str = "http://prometheus.platform-monitoring.svc:9090",
        evaluation_url: str = "http://evaluation-pipeline.platform-control-plane.svc:8080",
        gateway_url: str = "http://traffic-controller.platform-control-plane.svc:8080",
    ):
        self.opa_url = opa_url
        self.argo_url = argo_url
        self.prometheus_url = prometheus_url
        self.evaluation_url = evaluation_url
        self.gateway_url = gateway_url
        
        self.thresholds = PromotionThresholds()
        self.active_versions: Dict[str, VersionConfig] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        
        self._http_client: Optional[httpx.AsyncClient] = None
        self._running = False
    
    async def start(self):
        """Start the lifecycle controller"""
        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._running = True
        
        logger.info("Lifecycle Controller started")
        
        # Start background tasks
        asyncio.create_task(self._evaluation_loop())
        asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """Stop the lifecycle controller"""
        self._running = False
        if self._http_client:
            await self._http_client.aclose()
        logger.info("Lifecycle Controller stopped")
    
    # ==================== Evaluation Loop ====================
    
    async def _evaluation_loop(self):
        """Main loop for continuous evaluation"""
        while self._running:
            try:
                # Get all V1 versions in shadow mode
                shadow_versions = [
                    v for v in self.active_versions.values()
                    if v.current_state == VersionState.SHADOW
                ]
                
                for version in shadow_versions:
                    await self._evaluate_shadow_version(version)
                
                # Check gray-scale versions
                gray_versions = [
                    v for v in self.active_versions.values()
                    if v.current_state.value.startswith("gray_")
                ]
                
                for version in gray_versions:
                    await self._evaluate_gray_version(version)
                
                # Check V3 re-evaluation candidates
                quarantine_versions = [
                    v for v in self.active_versions.values()
                    if v.current_state == VersionState.RE_EVALUATION
                ]
                
                for version in quarantine_versions:
                    await self._evaluate_recovery(version)
                
            except Exception as e:
                logger.error(f"Evaluation loop error: {e}")
            
            await asyncio.sleep(60)  # Evaluate every minute
    
    async def _evaluate_shadow_version(self, version: VersionConfig):
        """Evaluate a version in shadow mode"""
        metrics = await self._collect_metrics(version.version_id, "v1")
        
        if metrics.total_requests < self.thresholds.min_shadow_requests:
            logger.info(f"Version {version.version_id}: Insufficient requests ({metrics.total_requests})")
            return
        
        # Check if shadow duration requirement met
        shadow_duration = datetime.now(timezone.utc) - version.created_at
        if shadow_duration < timedelta(hours=self.thresholds.min_shadow_duration_hours):
            logger.info(f"Version {version.version_id}: Shadow duration not met")
            return
        
        # Evaluate using OPA
        decision = await self._opa_evaluate_promotion(version, metrics)
        
        if decision["allow"]:
            logger.info(f"Version {version.version_id}: Promotion approved, starting gray-scale")
            await self._start_gray_scale(version, metrics)
        elif decision.get("downgrade"):
            logger.warning(f"Version {version.version_id}: Downgrade triggered - {decision.get('reason')}")
            await self._downgrade_to_v3(version, decision.get("reason", "Evaluation failed"))
        else:
            logger.info(f"Version {version.version_id}: Continuing shadow evaluation")
            version.last_evaluation = datetime.now(timezone.utc)
    
    async def _evaluate_gray_version(self, version: VersionConfig):
        """Evaluate a version in gray-scale rollout"""
        metrics = await self._collect_metrics(version.version_id, "v2")
        
        # Check SLO compliance
        slo_passed = await self._check_slo_compliance(metrics)
        
        if not slo_passed:
            version.consecutive_failures += 1
            
            if version.consecutive_failures >= self.thresholds.consecutive_failures_for_downgrade:
                logger.error(f"Version {version.version_id}: SLO violations exceeded, rolling back")
                await self._rollback_gray_scale(version)
                return
        else:
            version.consecutive_failures = 0
        
        # OPA decision for next phase
        decision = await self._opa_evaluate_gray_progress(version, metrics)
        
        if decision["advance"]:
            await self._advance_gray_scale(version)
        elif decision.get("rollback"):
            await self._rollback_gray_scale(version)
    
    async def _evaluate_recovery(self, version: VersionConfig):
        """Evaluate a version attempting recovery from V3"""
        metrics = await self._collect_metrics(version.version_id, "v3")
        
        # Run gold-set evaluation
        gold_set_results = await self._run_gold_set_evaluation(version)
        
        if gold_set_results["passed"]:
            logger.info(f"Version {version.version_id}: Recovery successful, moving to V1 shadow")
            await self._promote_to_v1_shadow(version)
        else:
            logger.info(f"Version {version.version_id}: Recovery not ready - {gold_set_results.get('reason')}")
    
    # ==================== OPA Integration ====================
    
    async def _opa_evaluate_promotion(
        self, 
        version: VersionConfig, 
        metrics: EvaluationMetrics
    ) -> Dict[str, Any]:
        """Query OPA for promotion decision"""
        input_data = {
            "version": {
                "id": version.version_id,
                "model": version.model_version,
                "prompt": version.prompt_version,
                "state": version.current_state.value,
            },
            "metrics": {
                "p95_latency_ms": metrics.p95_latency_ms,
                "error_rate": metrics.error_rate,
                "accuracy_delta": metrics.accuracy_delta,
                "security_pass_rate": metrics.security_pass_rate,
                "cost_delta": metrics.cost_delta,
                "total_requests": metrics.total_requests,
            },
            "statistical_tests": {
                "accuracy_p_value": metrics.accuracy_p_value,
                "latency_p_value": metrics.latency_p_value,
                "cost_p_value": metrics.cost_p_value,
            },
            "thresholds": {
                "p95_latency_ms": self.thresholds.p95_latency_ms,
                "error_rate": self.thresholds.error_rate,
                "accuracy_delta": self.thresholds.accuracy_delta,
                "security_pass_rate": self.thresholds.security_pass_rate,
                "cost_increase_max": self.thresholds.cost_increase_max,
                "statistical_significance_p": self.thresholds.statistical_significance_p,
            }
        }
        
        try:
            response = await self._http_client.post(
                f"{self.opa_url}/v1/data/lifecycle/promotion",
                json={"input": input_data}
            )
            result = response.json().get("result", {})
            return result
        except Exception as e:
            logger.error(f"OPA promotion query failed: {e}")
            return {"allow": False, "reason": f"OPA error: {e}"}
    
    async def _opa_evaluate_gray_progress(
        self,
        version: VersionConfig,
        metrics: EvaluationMetrics
    ) -> Dict[str, Any]:
        """Query OPA for gray-scale progress decision"""
        input_data = {
            "version": {
                "id": version.version_id,
                "current_phase": version.current_state.value,
                "consecutive_failures": version.consecutive_failures,
            },
            "metrics": {
                "p95_latency_ms": metrics.p95_latency_ms,
                "error_rate": metrics.error_rate,
                "accuracy": metrics.accuracy,
            },
            "thresholds": {
                "p95_latency_ms": self.thresholds.p95_latency_ms,
                "error_rate": self.thresholds.error_rate,
            }
        }
        
        try:
            response = await self._http_client.post(
                f"{self.opa_url}/v1/data/lifecycle/gray_progress",
                json={"input": input_data}
            )
            return response.json().get("result", {})
        except Exception as e:
            logger.error(f"OPA gray progress query failed: {e}")
            return {"advance": False, "rollback": True, "reason": f"OPA error: {e}"}
    
    # ==================== Metrics Collection ====================
    
    async def _collect_metrics(
        self, 
        version_id: str, 
        namespace: str
    ) -> EvaluationMetrics:
        """Collect metrics from Prometheus"""
        metrics = EvaluationMetrics()
        
        try:
            # Total requests
            query = f'sum(http_requests_total{{version="{version_id}"}})'
            metrics.total_requests = await self._prometheus_query(query)
            
            # Latency percentiles
            query = f'histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket{{version="{version_id}"}}[5m])) by (le)) * 1000'
            metrics.p50_latency_ms = await self._prometheus_query(query)
            
            query = f'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{{version="{version_id}"}}[5m])) by (le)) * 1000'
            metrics.p95_latency_ms = await self._prometheus_query(query)
            
            query = f'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{{version="{version_id}"}}[5m])) by (le)) * 1000'
            metrics.p99_latency_ms = await self._prometheus_query(query)
            
            # Error rate
            query = f'sum(rate(http_requests_total{{version="{version_id}",status=~"5.."}}[5m])) / sum(rate(http_requests_total{{version="{version_id}"}}[5m]))'
            metrics.error_rate = await self._prometheus_query(query)
            
            # Accuracy (from evaluation pipeline)
            query = f'avg(analysis_accuracy{{version="{version_id}"}})'
            metrics.accuracy = await self._prometheus_query(query)
            
            # Accuracy delta vs baseline
            query = f'avg(analysis_accuracy{{version="{version_id}"}}) - avg(analysis_accuracy{{version="baseline"}})'
            metrics.accuracy_delta = await self._prometheus_query(query)
            
            # Security pass rate
            query = f'sum(rate(security_checks_passed{{version="{version_id}"}}[10m])) / sum(rate(security_checks_total{{version="{version_id}"}}[10m]))'
            metrics.security_pass_rate = await self._prometheus_query(query)
            
            # Cost metrics
            query = f'avg(request_cost{{version="{version_id}"}})'
            metrics.cost_per_request = await self._prometheus_query(query)
            
            query = f'(avg(request_cost{{version="{version_id}"}}) - avg(request_cost{{version="baseline"}})) / avg(request_cost{{version="baseline"}})'
            metrics.cost_delta = await self._prometheus_query(query)
            
        except Exception as e:
            logger.error(f"Metrics collection failed for {version_id}: {e}")
        
        return metrics
    
    async def _prometheus_query(self, query: str) -> float:
        """Execute a Prometheus query"""
        try:
            response = await self._http_client.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query}
            )
            result = response.json()
            if result["status"] == "success" and result["data"]["result"]:
                return float(result["data"]["result"][0]["value"][1])
        except Exception as e:
            logger.warning(f"Prometheus query failed: {query} - {e}")
        return 0.0
    
    # ==================== Rollout Control ====================
    
    async def _start_gray_scale(self, version: VersionConfig, metrics: EvaluationMetrics):
        """Initiate gray-scale rollout to V2"""
        # Update version state
        version.current_state = VersionState.GRAY_1
        version.last_evaluation = datetime.now(timezone.utc)
        
        # Configure gateway for 1% traffic
        await self._update_traffic_split(version.version_id, 1)
        
        # Trigger Argo Rollout
        await self._trigger_argo_rollout(version.version_id, "gray-1-percent")
        
        # Log promotion event
        self._log_lifecycle_event(version, "gray_scale_started", metrics)
    
    async def _advance_gray_scale(self, version: VersionConfig):
        """Advance to next gray-scale phase"""
        phase_progression = {
            VersionState.GRAY_1: (VersionState.GRAY_5, 5),
            VersionState.GRAY_5: (VersionState.GRAY_25, 25),
            VersionState.GRAY_25: (VersionState.GRAY_50, 50),
            VersionState.GRAY_50: (VersionState.STABLE, 100),
        }
        
        if version.current_state in phase_progression:
            next_state, percentage = phase_progression[version.current_state]
            version.current_state = next_state
            version.last_evaluation = datetime.now(timezone.utc)
            
            await self._update_traffic_split(version.version_id, percentage)
            await self._trigger_argo_rollout(version.version_id, next_state.value)
            
            if next_state == VersionState.STABLE:
                logger.info(f"Version {version.version_id}: Full production deployment complete!")
                self._log_lifecycle_event(version, "promoted_to_stable", {})
    
    async def _rollback_gray_scale(self, version: VersionConfig):
        """Rollback gray-scale deployment"""
        logger.warning(f"Rolling back version {version.version_id}")
        
        # Reset traffic to stable version
        await self._update_traffic_split(version.version_id, 0)
        
        # Trigger Argo rollback
        await self._trigger_argo_rollback(version.version_id)
        
        # Move to quarantine
        await self._downgrade_to_v3(version, "Gray-scale rollback triggered")
    
    async def _downgrade_to_v3(self, version: VersionConfig, reason: str):
        """Downgrade version to V3 quarantine"""
        version.current_state = VersionState.QUARANTINE
        version.last_evaluation = datetime.now(timezone.utc)
        version.metadata["quarantine_reason"] = reason
        version.metadata["quarantine_time"] = datetime.now(timezone.utc).isoformat()
        
        # Remove from traffic
        await self._update_traffic_split(version.version_id, 0)
        
        self._log_lifecycle_event(version, "downgraded_to_v3", {"reason": reason})
    
    async def _promote_to_v1_shadow(self, version: VersionConfig):
        """Promote recovered version back to V1 shadow"""
        version.current_state = VersionState.SHADOW
        version.consecutive_failures = 0
        version.last_evaluation = datetime.now(timezone.utc)
        version.metadata["recovery_time"] = datetime.now(timezone.utc).isoformat()
        
        self._log_lifecycle_event(version, "recovered_to_v1", {})
    
    # ==================== External Integrations ====================
    
    async def _update_traffic_split(self, version_id: str, percentage: int):
        """Update gateway traffic split"""
        try:
            await self._http_client.put(
                f"{self.gateway_url}/traffic-split",
                json={
                    "version_id": version_id,
                    "percentage": percentage
                }
            )
        except Exception as e:
            logger.error(f"Failed to update traffic split: {e}")
    
    async def _trigger_argo_rollout(self, version_id: str, phase: str):
        """Trigger Argo Rollout progression"""
        try:
            await self._http_client.post(
                f"{self.argo_url}/api/v1/rollouts/platform-v2-stable/promote",
                json={
                    "version_id": version_id,
                    "phase": phase
                }
            )
        except Exception as e:
            logger.error(f"Failed to trigger Argo rollout: {e}")
    
    async def _trigger_argo_rollback(self, version_id: str):
        """Trigger Argo Rollout abort/rollback"""
        try:
            await self._http_client.post(
                f"{self.argo_url}/api/v1/rollouts/platform-v2-stable/abort"
            )
        except Exception as e:
            logger.error(f"Failed to trigger Argo rollback: {e}")
    
    async def _run_gold_set_evaluation(self, version: VersionConfig) -> Dict[str, Any]:
        """Run gold-set evaluation for recovery"""
        try:
            response = await self._http_client.post(
                f"{self.evaluation_url}/evaluate/gold-set",
                json={
                    "version_id": version.version_id,
                    "model_version": version.model_version,
                    "prompt_version": version.prompt_version,
                }
            )
            return response.json()
        except Exception as e:
            logger.error(f"Gold-set evaluation failed: {e}")
            return {"passed": False, "reason": f"Evaluation error: {e}"}
    
    async def _check_slo_compliance(self, metrics: EvaluationMetrics) -> bool:
        """Check if metrics meet SLO requirements"""
        return (
            metrics.p95_latency_ms <= self.thresholds.p95_latency_ms and
            metrics.error_rate <= self.thresholds.error_rate and
            metrics.security_pass_rate >= self.thresholds.security_pass_rate
        )
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self._running:
            try:
                # Check dependencies health
                await self._check_dependencies_health()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            await asyncio.sleep(30)
    
    async def _check_dependencies_health(self):
        """Check health of external dependencies"""
        dependencies = [
            ("OPA", self.opa_url + "/health"),
            ("Prometheus", self.prometheus_url + "/-/healthy"),
            ("Evaluation", self.evaluation_url + "/health"),
        ]
        
        for name, url in dependencies:
            try:
                response = await self._http_client.get(url, timeout=5.0)
                if response.status_code != 200:
                    logger.warning(f"{name} health check failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"{name} health check error: {e}")
    
    def _log_lifecycle_event(
        self, 
        version: VersionConfig, 
        event_type: str, 
        details: Any
    ):
        """Log lifecycle event for audit"""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version_id": version.version_id,
            "event_type": event_type,
            "current_state": version.current_state.value,
            "details": details,
        }
        self.evaluation_history.append(event)
        logger.info(f"Lifecycle event: {event}")


# API Endpoints
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Lifecycle Controller", version="1.0.0")
controller: Optional[LifecycleController] = None


@app.on_event("startup")
async def startup():
    global controller
    controller = LifecycleController()
    await controller.start()


@app.on_event("shutdown")
async def shutdown():
    if controller:
        await controller.stop()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/versions")
async def list_versions():
    return {"versions": [v.dict() for v in controller.active_versions.values()]}


@app.post("/versions/{version_id}/register")
async def register_version(version_id: str, config: VersionConfig):
    controller.active_versions[version_id] = config
    return {"status": "registered", "version_id": version_id}


@app.post("/versions/{version_id}/evaluate")
async def trigger_evaluation(version_id: str):
    if version_id not in controller.active_versions:
        raise HTTPException(404, "Version not found")
    
    version = controller.active_versions[version_id]
    metrics = await controller._collect_metrics(version_id, "v1")
    decision = await controller._opa_evaluate_promotion(version, metrics)
    
    return {
        "version_id": version_id,
        "metrics": metrics.__dict__,
        "decision": decision
    }


@app.get("/history")
async def get_history(limit: int = 100):
    return {"events": controller.evaluation_history[-limit:]}


@app.put("/thresholds")
async def update_thresholds(thresholds: PromotionThresholds):
    controller.thresholds = thresholds
    return {"status": "updated", "thresholds": thresholds}


# Include comparison API routers
from comparison_api import router as comparison_router
from comparison_api import rollback_router, audit_router, stats_router

app.include_router(comparison_router)
app.include_router(rollback_router)
app.include_router(audit_router)
app.include_router(stats_router)
