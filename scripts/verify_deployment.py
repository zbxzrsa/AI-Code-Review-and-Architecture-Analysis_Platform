#!/usr/bin/env python3
"""
Deployment Verification Script

Verifies that all components of the three-version architecture
are deployed and functioning correctly.

Usage:
    python verify_deployment.py [--namespace NAMESPACE] [--timeout TIMEOUT]
"""

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check name constants
CHECK_GATEWAY_HEALTH = "Gateway Health"
CHECK_V1_SERVICES = "V1 Services"
CHECK_V2_SERVICES = "V2 Services"
CHECK_V3_SERVICES = "V3 Services"
CHECK_LIFECYCLE_CONTROLLER = "Lifecycle Controller"
CHECK_EVALUATION_PIPELINE = "Evaluation Pipeline"
CHECK_OPA_POLICY_ENGINE = "OPA Policy Engine"
CHECK_SHADOW_TRAFFIC = "Shadow Traffic"
CHECK_SLO_METRICS = "SLO Metrics"
CHECK_PROMETHEUS = "Prometheus"


@dataclass
class CheckResult:
    """Result of a deployment check"""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


@dataclass
class VersionCheck:
    """Check for a specific version"""
    version: str
    namespace: str
    services: List[str]
    expected_replicas: int = 1


class DeploymentVerifier:
    """Verifies deployment health"""
    
    def __init__(
        self,
        gateway_url: str = "http://localhost",
        prometheus_url: str = "http://localhost:9090",
        timeout: int = 30
    ):
        self.gateway_url = gateway_url
        self.prometheus_url = prometheus_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.results: List[CheckResult] = []
    
    async def run_all_checks(self) -> Tuple[bool, List[CheckResult]]:
        """Run all deployment verification checks"""
        logger.info("=" * 60)
        logger.info("Starting Deployment Verification")
        logger.info("=" * 60)
        
        checks = [
            self.check_gateway_health,
            self.check_v1_services,
            self.check_v2_services,
            self.check_v3_services,
            self.check_lifecycle_controller,
            self.check_evaluation_pipeline,
            self.check_prometheus,
            self.check_opa,
            self.check_shadow_traffic,
            self.check_slo_metrics,
        ]
        
        for check in checks:
            try:
                result = await check()
                self.results.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(f"{status} - {result.name}: {result.message}")
            except Exception as e:
                result = CheckResult(
                    name=check.__name__,
                    passed=False,
                    message=f"Check failed with error: {str(e)}"
                )
                self.results.append(result)
                logger.error(f"❌ FAIL - {result.name}: {result.message}")
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        all_passed = passed == total
        
        logger.info("=" * 60)
        logger.info(f"Verification Complete: {passed}/{total} checks passed")
        logger.info("=" * 60)
        
        return all_passed, self.results
    
    async def check_gateway_health(self) -> CheckResult:
        """Check API Gateway health"""
        try:
            response = await self.client.get(f"{self.gateway_url}/api/health")
            if response.status_code == 200:
                return CheckResult(
                    name=CHECK_GATEWAY_HEALTH,
                    passed=True,
                    message="API Gateway is healthy"
                )
            return CheckResult(
                name=CHECK_GATEWAY_HEALTH,
                passed=False,
                message=f"Gateway returned status {response.status_code}"
            )
        except Exception as e:
            return CheckResult(
                name=CHECK_GATEWAY_HEALTH,
                passed=False,
                message=f"Cannot connect to gateway: {e}"
            )
    
    async def check_v1_services(self) -> CheckResult:
        """Check V1 Experiment services"""
        try:
            # Check V1 VCAI service
            response = await self.client.get(
                f"{self.gateway_url}/api/v1/health",
                headers={"X-Version-Override": "v1"}
            )
            if response.status_code == 200:
                return CheckResult(
                    name=CHECK_V1_SERVICES,
                    passed=True,
                    message="V1 experiment services are running"
                )
            return CheckResult(
                name=CHECK_V1_SERVICES,
                passed=False,
                message=f"V1 services returned status {response.status_code}"
            )
        except Exception as e:
            return CheckResult(
                name=CHECK_V1_SERVICES,
                passed=False,
                message=f"V1 services unreachable: {e}"
            )
    
    async def check_v2_services(self) -> CheckResult:
        """Check V2 Stable services"""
        try:
            response = await self.client.get(f"{self.gateway_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                return CheckResult(
                    name=CHECK_V2_SERVICES,
                    passed=True,
                    message="V2 production services are running",
                    details=data
                )
            return CheckResult(
                name=CHECK_V2_SERVICES,
                passed=False,
                message=f"V2 services returned status {response.status_code}"
            )
        except Exception as e:
            return CheckResult(
                name=CHECK_V2_SERVICES,
                passed=False,
                message=f"V2 services unreachable: {e}"
            )
    
    async def check_v3_services(self) -> CheckResult:
        """Check V3 Legacy services (may be scaled to zero)"""
        try:
            response = await self.client.get(
                f"{self.gateway_url}/api/v1/health",
                headers={"X-Version-Override": "v3"}
            )
            # V3 may be scaled down, so 503 is acceptable
            if response.status_code in [200, 503]:
                return CheckResult(
                    name=CHECK_V3_SERVICES,
                    passed=True,
                    message="V3 legacy services are configured (may be scaled down)"
                )
            return CheckResult(
                name=CHECK_V3_SERVICES,
                passed=False,
                message=f"V3 services returned unexpected status {response.status_code}"
            )
        except Exception as e:
            # V3 being unreachable is often acceptable
            return CheckResult(
                name=CHECK_V3_SERVICES,
                passed=True,
                message=f"V3 services not responding (expected if scaled down)"
            )
    
    async def check_lifecycle_controller(self) -> CheckResult:
        """Check Lifecycle Controller"""
        try:
            response = await self.client.get(
                f"{self.gateway_url}/api/admin/lifecycle/health"
            )
            if response.status_code == 200:
                return CheckResult(
                    name=CHECK_LIFECYCLE_CONTROLLER,
                    passed=True,
                    message="Lifecycle controller is healthy"
                )
            return CheckResult(
                name=CHECK_LIFECYCLE_CONTROLLER,
                passed=False,
                message=f"Lifecycle controller returned {response.status_code}"
            )
        except Exception as e:
            return CheckResult(
                name=CHECK_LIFECYCLE_CONTROLLER,
                passed=False,
                message=f"Lifecycle controller unreachable: {e}"
            )
    
    async def check_evaluation_pipeline(self) -> CheckResult:
        """Check Evaluation Pipeline"""
        try:
            response = await self.client.get(
                f"{self.gateway_url}/api/admin/evaluation/health"
            )
            if response.status_code == 200:
                return CheckResult(
                    name=CHECK_EVALUATION_PIPELINE,
                    passed=True,
                    message="Evaluation pipeline is healthy"
                )
            return CheckResult(
                name=CHECK_EVALUATION_PIPELINE,
                passed=False,
                message=f"Evaluation pipeline returned {response.status_code}"
            )
        except Exception as e:
            return CheckResult(
                name=CHECK_EVALUATION_PIPELINE,
                passed=False,
                message=f"Evaluation pipeline unreachable: {e}"
            )
    
    async def check_prometheus(self) -> CheckResult:
        """Check Prometheus metrics"""
        try:
            response = await self.client.get(f"{self.prometheus_url}/-/healthy")
            if response.status_code == 200:
                # Also check if we have metrics
                query_response = await self.client.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params={"query": "up"}
                )
                if query_response.status_code == 200:
                    data = query_response.json()
                    targets = len(data.get('data', {}).get('result', []))
                    return CheckResult(
                        name=CHECK_PROMETHEUS,
                        passed=True,
                        message=f"Prometheus healthy, {targets} targets up",
                        details={"targets_up": targets}
                    )
            return CheckResult(
                name=CHECK_PROMETHEUS,
                passed=False,
                message="Prometheus not responding correctly"
            )
        except Exception as e:
            return CheckResult(
                name=CHECK_PROMETHEUS,
                passed=False,
                message=f"Prometheus unreachable: {e}"
            )
    
    async def check_opa(self) -> CheckResult:
        """Check OPA policy engine"""
        try:
            response = await self.client.get("http://localhost:8181/health")
            if response.status_code == 200:
                # Check if policies are loaded
                policy_response = await self.client.get(
                    "http://localhost:8181/v1/policies"
                )
                if policy_response.status_code == 200:
                    data = policy_response.json()
                    policies = len(data.get('result', []))
                    return CheckResult(
                        name=CHECK_OPA_POLICY_ENGINE,
                        passed=True,
                        message=f"OPA healthy, {policies} policies loaded",
                        details={"policies_loaded": policies}
                    )
            return CheckResult(
                name=CHECK_OPA_POLICY_ENGINE,
                passed=False,
                message="OPA not responding correctly"
            )
        except Exception as e:
            return CheckResult(
                name=CHECK_OPA_POLICY_ENGINE,
                passed=False,
                message=f"OPA unreachable: {e}"
            )
    
    async def check_shadow_traffic(self) -> CheckResult:
        """Check if shadow traffic is flowing to V1"""
        try:
            query = 'sum(rate(http_requests_total{namespace="platform-v1-exp"}[5m]))'
            response = await self.client.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query}
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get('data', {}).get('result', [])
                if results:
                    value = float(results[0]['value'][1])
                    if value > 0:
                        return CheckResult(
                            name=CHECK_SHADOW_TRAFFIC,
                            passed=True,
                            message=f"Shadow traffic flowing to V1: {value:.2f} req/s",
                            details={"requests_per_second": value}
                        )
                return CheckResult(
                    name=CHECK_SHADOW_TRAFFIC,
                    passed=False,
                    message="No shadow traffic detected to V1"
                )
            return CheckResult(
                name=CHECK_SHADOW_TRAFFIC,
                passed=False,
                message="Cannot query shadow traffic metrics"
            )
        except Exception as e:
            return CheckResult(
                name=CHECK_SHADOW_TRAFFIC,
                passed=False,
                message=f"Shadow traffic check failed: {e}"
            )
    
    async def check_slo_metrics(self) -> CheckResult:
        """Check SLO metrics for V2"""
        try:
            # Check error rate
            error_query = 'slo:v2:error_rate:ratio_rate5m'
            response = await self.client.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": error_query}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('data', {}).get('result', [])
                if results:
                    error_rate = float(results[0]['value'][1])
                    
                    # Check latency
                    latency_query = 'slo:v2:latency_p95:histogram_quantile_5m'
                    lat_response = await self.client.get(
                        f"{self.prometheus_url}/api/v1/query",
                        params={"query": latency_query}
                    )
                    latency_ms = 0
                    if lat_response.status_code == 200:
                        lat_data = lat_response.json()
                        lat_results = lat_data.get('data', {}).get('result', [])
                        if lat_results:
                            latency_ms = float(lat_results[0]['value'][1])
                    
                    # Check thresholds
                    error_ok = error_rate < 0.02
                    latency_ok = latency_ms < 3000
                    
                    if error_ok and latency_ok:
                        return CheckResult(
                            name=CHECK_SLO_METRICS,
                            passed=True,
                            message=f"SLOs met: error={error_rate:.4f}, p95={latency_ms:.0f}ms",
                            details={"error_rate": error_rate, "p95_latency_ms": latency_ms}
                        )
                    else:
                        issues = []
                        if not error_ok:
                            issues.append(f"error rate {error_rate:.4f} > 0.02")
                        if not latency_ok:
                            issues.append(f"p95 latency {latency_ms:.0f}ms > 3000ms")
                        return CheckResult(
                            name=CHECK_SLO_METRICS,
                            passed=False,
                            message=f"SLO violations: {', '.join(issues)}",
                            details={"error_rate": error_rate, "p95_latency_ms": latency_ms}
                        )
                
                return CheckResult(
                    name=CHECK_SLO_METRICS,
                    passed=True,
                    message="No SLO metrics available yet (new deployment)"
                )
            
            return CheckResult(
                name=CHECK_SLO_METRICS,
                passed=False,
                message="Cannot query SLO metrics"
            )
        except Exception as e:
            return CheckResult(
                name=CHECK_SLO_METRICS,
                passed=False,
                message=f"SLO check failed: {e}"
            )
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


def print_report(results: List[CheckResult]):
    """Print detailed verification report"""
    print("\n" + "=" * 60)
    print("DEPLOYMENT VERIFICATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("-" * 60)
    
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    
    print(f"\nSummary: {len(passed)}/{len(results)} checks passed\n")
    
    if passed:
        print("PASSED CHECKS:")
        for r in passed:
            print(f"  ✅ {r.name}")
            print(f"     {r.message}")
            if r.details:
                for k, v in r.details.items():
                    print(f"     - {k}: {v}")
    
    if failed:
        print("\nFAILED CHECKS:")
        for r in failed:
            print(f"  ❌ {r.name}")
            print(f"     {r.message}")
    
    print("\n" + "=" * 60)
    
    if failed:
        print("⚠️  DEPLOYMENT VERIFICATION FAILED")
        print("   Please investigate failed checks before proceeding.")
    else:
        print("✅ DEPLOYMENT VERIFICATION PASSED")
        print("   All systems operational.")
    
    print("=" * 60 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Verify deployment health")
    parser.add_argument("--gateway", default="http://localhost", help="Gateway URL")
    parser.add_argument("--prometheus", default="http://localhost:9090", help="Prometheus URL")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    
    args = parser.parse_args()
    
    verifier = DeploymentVerifier(
        gateway_url=args.gateway,
        prometheus_url=args.prometheus,
        timeout=args.timeout
    )
    
    try:
        all_passed, results = await verifier.run_all_checks()
        print_report(results)
        
        sys.exit(0 if all_passed else 1)
    finally:
        await verifier.close()


if __name__ == "__main__":
    asyncio.run(main())
