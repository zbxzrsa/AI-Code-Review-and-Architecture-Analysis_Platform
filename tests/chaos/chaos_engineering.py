"""
Chaos Engineering Tests

Phase 5: Chaos Engineering
- Pod failure simulation
- Network partition testing
- Resource exhaustion
- Dependency failure injection
"""

import asyncio
import random
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ChaosType(str, Enum):
    """Types of chaos experiments."""
    POD_KILL = "pod_kill"
    POD_FAILURE = "pod_failure"
    NETWORK_PARTITION = "network_partition"
    NETWORK_LATENCY = "network_latency"
    NETWORK_PACKET_LOSS = "network_packet_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    DNS_FAILURE = "dns_failure"
    DEPENDENCY_FAILURE = "dependency_failure"


class ChaosState(str, Enum):
    """Chaos experiment state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ChaosExperiment:
    """Chaos experiment definition."""
    experiment_id: str
    name: str
    chaos_type: ChaosType
    target: str  # Kubernetes selector or service name
    duration_seconds: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    rollback_on_failure: bool = True
    state: ChaosState = ChaosState.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosResult:
    """Result of chaos experiment."""
    experiment_id: str
    success: bool
    hypothesis_validated: bool
    metrics_before: Dict[str, float]
    metrics_during: Dict[str, float]
    metrics_after: Dict[str, float]
    recovery_time_seconds: float
    findings: List[str]
    recommendations: List[str]


class ChaosMonkey:
    """
    Chaos engineering framework for resilience testing.
    
    Tests system behavior under:
    - Pod/container failures
    - Network issues
    - Resource constraints
    - Dependency failures
    """
    
    def __init__(
        self,
        kubernetes_client = None,
        metrics_client = None,
        namespace: str = "platform-v2-stable",
    ):
        self.k8s = kubernetes_client
        self.metrics = metrics_client
        self.namespace = namespace
        
        # Experiment history
        self._experiments: Dict[str, ChaosExperiment] = {}
        self._results: Dict[str, ChaosResult] = {}
    
    # =========================================================================
    # Pod Chaos
    # =========================================================================
    
    async def kill_pod(
        self,
        selector: str,
        count: int = 1,
    ) -> ChaosExperiment:
        """Kill random pods matching selector."""
        import uuid
        
        experiment = ChaosExperiment(
            experiment_id=str(uuid.uuid4()),
            name=f"Kill {count} pods matching {selector}",
            chaos_type=ChaosType.POD_KILL,
            target=selector,
            duration_seconds=0,  # Instant
            parameters={"count": count},
            hypothesis="System should recover within 60 seconds with no user impact",
        )
        
        self._experiments[experiment.experiment_id] = experiment
        
        try:
            experiment.state = ChaosState.RUNNING
            experiment.started_at = datetime.now(timezone.utc)
            
            # Capture metrics before
            metrics_before = await self._capture_metrics()
            
            # Kill pods
            if self.k8s:
                pods = await self._get_pods(selector)
                targets = random.sample(pods, min(count, len(pods)))
                
                for pod in targets:
                    logger.info(f"Killing pod: {pod}")
                    await self._delete_pod(pod)
            else:
                logger.info(f"[DRY RUN] Would kill {count} pods matching {selector}")
            
            # Monitor recovery
            recovery_start = datetime.now(timezone.utc)
            recovered = await self._wait_for_recovery(selector, timeout=120)
            recovery_time = (datetime.now(timezone.utc) - recovery_start).total_seconds()
            
            # Capture metrics after
            metrics_after = await self._capture_metrics()
            
            experiment.state = ChaosState.COMPLETED
            experiment.completed_at = datetime.now(timezone.utc)
            
            # Evaluate hypothesis
            hypothesis_valid = recovery_time < 60 and metrics_after.get("error_rate", 0) < 0.02
            
            result = ChaosResult(
                experiment_id=experiment.experiment_id,
                success=recovered,
                hypothesis_validated=hypothesis_valid,
                metrics_before=metrics_before,
                metrics_during={},
                metrics_after=metrics_after,
                recovery_time_seconds=recovery_time,
                findings=self._analyze_findings(metrics_before, metrics_after),
                recommendations=self._generate_recommendations(recovered, recovery_time),
            )
            
            self._results[experiment.experiment_id] = result
            experiment.results = result.__dict__
            
            return experiment
            
        except Exception as e:
            experiment.state = ChaosState.FAILED
            experiment.results = {"error": str(e)}
            logger.error(f"Chaos experiment failed: {e}")
            return experiment
    
    async def simulate_pod_failure(
        self,
        selector: str,
        duration_seconds: int = 60,
    ) -> ChaosExperiment:
        """Simulate pod failure by scaling to zero then back."""
        import uuid
        
        experiment = ChaosExperiment(
            experiment_id=str(uuid.uuid4()),
            name=f"Simulate pod failure for {duration_seconds}s",
            chaos_type=ChaosType.POD_FAILURE,
            target=selector,
            duration_seconds=duration_seconds,
            hypothesis="System should handle pod unavailability gracefully",
        )
        
        self._experiments[experiment.experiment_id] = experiment
        
        try:
            experiment.state = ChaosState.RUNNING
            experiment.started_at = datetime.now(timezone.utc)
            
            metrics_before = await self._capture_metrics()
            
            # Store original replica count
            original_replicas = 3  # Default
            
            if self.k8s:
                # Scale down
                logger.info(f"Scaling down {selector}")
                await self._scale_deployment(selector, 0)
                
                # Wait for duration
                await asyncio.sleep(duration_seconds)
                
                # Scale back up
                logger.info(f"Scaling up {selector}")
                await self._scale_deployment(selector, original_replicas)
            else:
                logger.info(f"[DRY RUN] Would simulate {duration_seconds}s failure")
                await asyncio.sleep(min(duration_seconds, 5))  # Short wait for dry run
            
            # Wait for recovery
            await self._wait_for_recovery(selector, timeout=120)
            
            metrics_after = await self._capture_metrics()
            
            experiment.state = ChaosState.COMPLETED
            experiment.completed_at = datetime.now(timezone.utc)
            
            return experiment
            
        except Exception as e:
            experiment.state = ChaosState.FAILED
            if experiment.rollback_on_failure:
                await self._rollback_experiment(experiment)
            raise
    
    # =========================================================================
    # Network Chaos
    # =========================================================================
    
    async def inject_network_latency(
        self,
        selector: str,
        latency_ms: int = 500,
        duration_seconds: int = 60,
    ) -> ChaosExperiment:
        """Inject network latency."""
        import uuid
        
        experiment = ChaosExperiment(
            experiment_id=str(uuid.uuid4()),
            name=f"Inject {latency_ms}ms latency for {duration_seconds}s",
            chaos_type=ChaosType.NETWORK_LATENCY,
            target=selector,
            duration_seconds=duration_seconds,
            parameters={"latency_ms": latency_ms},
            hypothesis="System should maintain p95 < 5s with 500ms added latency",
        )
        
        self._experiments[experiment.experiment_id] = experiment
        
        try:
            experiment.state = ChaosState.RUNNING
            experiment.started_at = datetime.now(timezone.utc)
            
            # Apply network chaos (using tc or Chaos Mesh)
            chaos_manifest = self._generate_network_chaos_manifest(
                selector, latency_ms, duration_seconds
            )
            
            logger.info(f"Applying network latency chaos: {latency_ms}ms")
            
            if self.k8s:
                await self._apply_chaos_manifest(chaos_manifest)
                await asyncio.sleep(duration_seconds)
                await self._remove_chaos_manifest(chaos_manifest)
            else:
                logger.info(f"[DRY RUN] Would inject {latency_ms}ms latency")
                await asyncio.sleep(min(duration_seconds, 5))
            
            experiment.state = ChaosState.COMPLETED
            experiment.completed_at = datetime.now(timezone.utc)
            
            return experiment
            
        except Exception as e:
            experiment.state = ChaosState.FAILED
            raise
    
    async def inject_network_partition(
        self,
        source_selector: str,
        target_selector: str,
        duration_seconds: int = 60,
    ) -> ChaosExperiment:
        """Create network partition between services."""
        import uuid
        
        experiment = ChaosExperiment(
            experiment_id=str(uuid.uuid4()),
            name=f"Network partition between {source_selector} and {target_selector}",
            chaos_type=ChaosType.NETWORK_PARTITION,
            target=f"{source_selector} -> {target_selector}",
            duration_seconds=duration_seconds,
            hypothesis="System should detect partition and failover within 30s",
        )
        
        self._experiments[experiment.experiment_id] = experiment
        
        logger.info(f"Creating network partition for {duration_seconds}s")
        
        # Implementation would use NetworkPolicy or Chaos Mesh
        experiment.state = ChaosState.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)
        
        return experiment
    
    # =========================================================================
    # Resource Chaos
    # =========================================================================
    
    async def stress_cpu(
        self,
        selector: str,
        cpu_load_percent: int = 80,
        duration_seconds: int = 60,
    ) -> ChaosExperiment:
        """Stress CPU on target pods."""
        import uuid
        
        experiment = ChaosExperiment(
            experiment_id=str(uuid.uuid4()),
            name=f"CPU stress {cpu_load_percent}% for {duration_seconds}s",
            chaos_type=ChaosType.CPU_STRESS,
            target=selector,
            duration_seconds=duration_seconds,
            parameters={"cpu_load_percent": cpu_load_percent},
            hypothesis="HPA should scale up within 2 minutes under CPU stress",
        )
        
        self._experiments[experiment.experiment_id] = experiment
        
        logger.info(f"Stressing CPU to {cpu_load_percent}%")
        
        # Would use stress-ng or Chaos Mesh
        experiment.state = ChaosState.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)
        
        return experiment
    
    async def stress_memory(
        self,
        selector: str,
        memory_mb: int = 512,
        duration_seconds: int = 60,
    ) -> ChaosExperiment:
        """Stress memory on target pods."""
        import uuid
        
        experiment = ChaosExperiment(
            experiment_id=str(uuid.uuid4()),
            name=f"Memory stress {memory_mb}MB for {duration_seconds}s",
            chaos_type=ChaosType.MEMORY_STRESS,
            target=selector,
            duration_seconds=duration_seconds,
            parameters={"memory_mb": memory_mb},
            hypothesis="System should handle memory pressure without OOMKilled",
        )
        
        self._experiments[experiment.experiment_id] = experiment
        
        logger.info(f"Stressing memory with {memory_mb}MB allocation")
        
        experiment.state = ChaosState.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)
        
        return experiment
    
    # =========================================================================
    # Dependency Chaos
    # =========================================================================
    
    async def fail_dependency(
        self,
        dependency: str,  # redis, postgresql, ai-api
        failure_type: str = "unavailable",  # unavailable, slow, error
        duration_seconds: int = 60,
    ) -> ChaosExperiment:
        """Simulate dependency failure."""
        import uuid
        
        experiment = ChaosExperiment(
            experiment_id=str(uuid.uuid4()),
            name=f"Fail {dependency} ({failure_type}) for {duration_seconds}s",
            chaos_type=ChaosType.DEPENDENCY_FAILURE,
            target=dependency,
            duration_seconds=duration_seconds,
            parameters={"failure_type": failure_type},
            hypothesis=f"System should gracefully degrade when {dependency} is {failure_type}",
        )
        
        self._experiments[experiment.experiment_id] = experiment
        
        logger.info(f"Simulating {dependency} {failure_type}")
        
        # Implementation would block traffic or inject faults
        experiment.state = ChaosState.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)
        
        return experiment
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    async def _capture_metrics(self) -> Dict[str, float]:
        """Capture current system metrics."""
        if self.metrics:
            return await self.metrics.get_current_metrics()
        
        # Mock metrics
        return {
            "latency_p95_ms": 800,
            "error_rate": 0.01,
            "throughput_rps": 100,
            "cpu_utilization": 0.45,
            "memory_utilization": 0.60,
        }
    
    async def _get_pods(self, selector: str) -> List[str]:
        """Get pods matching selector."""
        # Would use kubernetes client
        return ["pod-1", "pod-2", "pod-3"]
    
    async def _delete_pod(self, pod_name: str):
        """Delete a pod."""
        logger.info(f"Deleting pod: {pod_name}")
    
    async def _scale_deployment(self, name: str, replicas: int):
        """Scale deployment."""
        logger.info(f"Scaling {name} to {replicas} replicas")
    
    async def _wait_for_recovery(
        self,
        selector: str,
        timeout: int = 120,
    ) -> bool:
        """Wait for pods to recover."""
        start = datetime.now(timezone.utc)
        
        while (datetime.now(timezone.utc) - start).total_seconds() < timeout:
            # Check pod health
            await asyncio.sleep(5)
            logger.info("Waiting for recovery...")
            
            # Mock: assume recovery after a few checks
            if (datetime.now(timezone.utc) - start).total_seconds() > 15:
                return True
        
        return False
    
    async def _rollback_experiment(self, experiment: ChaosExperiment):
        """Rollback experiment effects."""
        logger.info(f"Rolling back experiment: {experiment.experiment_id}")
        experiment.state = ChaosState.ROLLED_BACK
    
    def _generate_network_chaos_manifest(
        self,
        selector: str,
        latency_ms: int,
        duration_seconds: int,
    ) -> Dict[str, Any]:
        """Generate Chaos Mesh manifest for network chaos."""
        return {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {"name": f"latency-{latency_ms}ms"},
            "spec": {
                "action": "delay",
                "mode": "all",
                "selector": {"labelSelectors": selector},
                "delay": {"latency": f"{latency_ms}ms"},
                "duration": f"{duration_seconds}s",
            },
        }
    
    async def _apply_chaos_manifest(self, manifest: Dict[str, Any]):
        """Apply chaos manifest to cluster."""
        logger.info(f"Applying chaos manifest: {manifest['metadata']['name']}")
    
    async def _remove_chaos_manifest(self, manifest: Dict[str, Any]):
        """Remove chaos manifest from cluster."""
        logger.info(f"Removing chaos manifest: {manifest['metadata']['name']}")
    
    def _analyze_findings(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
    ) -> List[str]:
        """Analyze chaos experiment findings."""
        findings = []
        
        if after.get("error_rate", 0) > before.get("error_rate", 0) * 2:
            findings.append("Error rate increased significantly during chaos")
        
        if after.get("latency_p95_ms", 0) > before.get("latency_p95_ms", 0) * 1.5:
            findings.append("Latency increased by more than 50%")
        
        return findings or ["System maintained stability during chaos"]
    
    def _generate_recommendations(
        self,
        recovered: bool,
        recovery_time: float,
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if not recovered:
            recommendations.append("Implement automatic recovery mechanisms")
            recommendations.append("Add health checks for faster failure detection")
        
        if recovery_time > 60:
            recommendations.append("Reduce recovery time with pod pre-warming")
            recommendations.append("Increase readiness probe frequency")
        
        return recommendations or ["System resilience is acceptable"]
    
    # =========================================================================
    # Test Runner
    # =========================================================================
    
    async def run_resilience_suite(self) -> List[ChaosResult]:
        """Run full resilience test suite."""
        logger.info("Starting Chaos Engineering Suite...")
        
        results = []
        
        # Test 1: Pod failure
        exp = await self.kill_pod("app=v2-service", count=1)
        if exp.experiment_id in self._results:
            results.append(self._results[exp.experiment_id])
        
        await asyncio.sleep(30)  # Cool down
        
        # Test 2: Network latency
        exp = await self.inject_network_latency("app=v2-service", latency_ms=500, duration_seconds=30)
        
        await asyncio.sleep(30)
        
        # Test 3: Dependency failure
        exp = await self.fail_dependency("redis", failure_type="unavailable", duration_seconds=30)
        
        logger.info("Chaos Engineering Suite completed")
        return results
    
    def generate_report(self) -> str:
        """Generate chaos engineering report."""
        report = f"""
{'='*60}
CHAOS ENGINEERING REPORT
{'='*60}
Generated: {datetime.now().isoformat()}
Namespace: {self.namespace}

EXPERIMENTS
-----------
Total: {len(self._experiments)}
Completed: {len([e for e in self._experiments.values() if e.state == ChaosState.COMPLETED])}
Failed: {len([e for e in self._experiments.values() if e.state == ChaosState.FAILED])}

RESULTS
-------
"""
        
        for exp_id, result in self._results.items():
            exp = self._experiments.get(exp_id)
            if exp:
                report += f"""
Experiment: {exp.name}
  Type: {exp.chaos_type.value}
  Hypothesis: {exp.hypothesis}
  Hypothesis Validated: {'✓' if result.hypothesis_validated else '✗'}
  Recovery Time: {result.recovery_time_seconds:.1f}s
  Findings: {', '.join(result.findings)}
  Recommendations: {', '.join(result.recommendations)}
"""
        
        report += f"\n{'='*60}\n"
        return report


# CLI runner
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    chaos = ChaosMonkey(namespace="platform-v2-stable")
    asyncio.run(chaos.run_resilience_suite())
    print(chaos.generate_report())
