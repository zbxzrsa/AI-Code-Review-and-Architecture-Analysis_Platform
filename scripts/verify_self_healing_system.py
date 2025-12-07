#!/usr/bin/env python3
"""
Self-Healing System Verification Script

Comprehensive verification of the self-healing system components,
configuration, and integration. Run before production deployment.

Usage:
    python scripts/verify_self_healing_system.py [--verbose] [--fix]
"""

import sys
import os
import asyncio
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import argparse
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "shared"))


# =============================================================================
# Check Name Constants (avoid duplicate literals)
# =============================================================================

CHECK_PACKAGE_EXPORTS = "Package exports"
CHECK_INTEGRATION_WORKFLOW = "Integration workflow"
CHECK_CALLBACK_WIRING = "Callback wiring"
CHECK_HEALTH_THRESHOLDS = "Health thresholds"
CHECK_REPAIR_STRATEGIES = "Repair strategies"
CHECK_ALERT_CHANNELS = "Alert channels"
CHECK_UNIT_TESTS = "Unit tests"


class CheckStatus(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️ WARN"
    SKIP = "⏭️ SKIP"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = None


class SelfHealingVerifier:
    """Verifies self-healing system components and configuration."""

    def __init__(self, verbose: bool = False, fix: bool = False):
        self.verbose = verbose
        self.fix = fix
        self.results: List[CheckResult] = []

    def log(self, msg: str):
        """Log message if verbose mode."""
        if self.verbose:
            print(f"  [DEBUG] {msg}")

    def add_result(self, result: CheckResult):
        """Add check result."""
        self.results.append(result)
        status_str = result.status.value
        print(f"  {status_str}: {result.name}")
        if result.message and (self.verbose or result.status != CheckStatus.PASS):
            print(f"           {result.message}")

    async def run_all_checks(self) -> bool:
        """Run all verification checks."""
        print("\n" + "=" * 60)
        print("Self-Healing System Verification")
        print("=" * 60)

        # Module checks
        print("\n📦 Module Verification")
        print("-" * 40)
        await self.check_module_imports()
        await self.check_module_exports()

        # Component checks
        print("\n🔧 Component Verification")
        print("-" * 40)
        await self.check_health_monitor()
        await self.check_auto_repair()
        await self.check_alert_manager()
        await self.check_metrics_collector()
        await self.check_orchestrator()

        # Integration checks
        print("\n🔗 Integration Verification")
        print("-" * 40)
        await self.check_orchestrator_integration()
        await self.check_callback_wiring()

        # Configuration checks
        print("\n⚙️ Configuration Verification")
        print("-" * 40)
        await self.check_thresholds()
        await self.check_repair_strategies()
        await self.check_alert_channels()

        # Test suite checks
        print("\n🧪 Test Suite Verification")
        print("-" * 40)
        await self.check_unit_tests()
        await self.check_integration_tests()

        # Documentation checks
        print("\n📚 Documentation Verification")
        print("-" * 40)
        await self.check_runbook()
        await self.check_api_docs()

        # Summary
        return self.print_summary()

    async def check_module_imports(self):
        """Verify all modules can be imported."""
        modules = [
            ("self_healing", "Main package"),
            ("self_healing.health_monitor", "Health Monitor"),
            ("self_healing.auto_repair", "Auto Repair"),
            ("self_healing.alert_manager", "Alert Manager"),
            ("self_healing.metrics_collector", "Metrics Collector"),
            ("self_healing.orchestrator", "Orchestrator"),
        ]

        for module_name, description in modules:
            try:
                importlib.import_module(module_name)
                self.add_result(CheckResult(
                    name=f"Import {description}",
                    status=CheckStatus.PASS,
                    message=""
                ))
            except ImportError as e:
                self.add_result(CheckResult(
                    name=f"Import {description}",
                    status=CheckStatus.FAIL,
                    message=str(e)
                ))

    async def check_module_exports(self):
        """Verify package exports all expected symbols."""
        try:
            import self_healing

            expected_exports = [
                "HealthMonitor", "HealthStatus", "HealthMetrics",
                "AutoRepair", "RepairAction", "RepairResult",
                "AlertManager", "Alert", "AlertSeverity",
                "MetricsCollector", "Metric", "MetricSource",
                "SelfHealingOrchestrator",
            ]

            missing = [e for e in expected_exports if not hasattr(self_healing, e)]

            if not missing:
                self.add_result(CheckResult(
                    name=CHECK_PACKAGE_EXPORTS,
                    status=CheckStatus.PASS,
                    message=f"All {len(expected_exports)} exports available"
                ))
            else:
                self.add_result(CheckResult(
                    name=CHECK_PACKAGE_EXPORTS,
                    status=CheckStatus.FAIL,
                    message=f"Missing: {', '.join(missing)}"
                ))

        except ImportError as e:
            self.add_result(CheckResult(
                name=CHECK_PACKAGE_EXPORTS,
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_health_monitor(self):
        """Verify HealthMonitor component."""
        try:
            from self_healing import HealthMonitor, HealthStatus, HealthMetrics

            monitor = HealthMonitor()

            # Check attributes
            required_attrs = [
                "add_health_check", "run_health_checks",
                "detect_anomalies", "start", "stop"
            ]

            missing = [a for a in required_attrs if not hasattr(monitor, a)]

            if not missing:
                self.add_result(CheckResult(
                    name="HealthMonitor API",
                    status=CheckStatus.PASS,
                    message="All required methods available"
                ))
            else:
                self.add_result(CheckResult(
                    name="HealthMonitor API",
                    status=CheckStatus.FAIL,
                    message=f"Missing: {', '.join(missing)}"
                ))

            # Check HealthMetrics
            metrics = HealthMetrics(
                timestamp="2024-01-01T00:00:00",
                error_rate=0.01
            )
            score = metrics.calculate_health_score()

            if 0 <= score <= 100:
                self.add_result(CheckResult(
                    name="HealthMetrics scoring",
                    status=CheckStatus.PASS,
                    message=f"Health score calculation works (score={score})"
                ))
            else:
                self.add_result(CheckResult(
                    name="HealthMetrics scoring",
                    status=CheckStatus.FAIL,
                    message=f"Invalid score: {score}"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name="HealthMonitor",
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_auto_repair(self):
        """Verify AutoRepair component."""
        try:
            from self_healing import AutoRepair, RepairAction, RepairResult

            repair = AutoRepair(dry_run=True)

            # Check repair actions
            expected_actions = [
                RepairAction.RESTART_SERVICE,
                RepairAction.SCALE_UP,
                RepairAction.CLEAR_CACHE,
                RepairAction.ROLLBACK_VERSION,
                RepairAction.DRAIN_QUEUE,
            ]

            for action in expected_actions:
                if action in repair.repair_strategies:
                    self.log(f"Repair action {action.value} configured")

            # Test repair execution
            record = await repair.execute_repair(
                RepairAction.CLEAR_CACHE,
                {"triggered_by": "verification", "reason": "test"}
            )

            if record.success:
                self.add_result(CheckResult(
                    name="AutoRepair execution",
                    status=CheckStatus.PASS,
                    message="Dry-run repair successful"
                ))
            else:
                self.add_result(CheckResult(
                    name="AutoRepair execution",
                    status=CheckStatus.WARN,
                    message=f"Repair failed: {record.error_message}"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name="AutoRepair",
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_alert_manager(self):
        """Verify AlertManager component."""
        try:
            from self_healing import AlertManager, AlertSeverity

            manager = AlertManager()

            # Test alert generation
            alert = await manager.generate_alert(
                severity=AlertSeverity.INFO,
                title="Verification Test",
                message="Testing alert generation",
                source="verification"
            )

            if alert is not None:
                self.add_result(CheckResult(
                    name="AlertManager generation",
                    status=CheckStatus.PASS,
                    message="Alert generated successfully"
                ))
            else:
                self.add_result(CheckResult(
                    name="AlertManager generation",
                    status=CheckStatus.WARN,
                    message="Alert was deduplicated or failed"
                ))

            # Check deduplication
            alert2 = await manager.generate_alert(
                severity=AlertSeverity.INFO,
                title="Verification Test",
                message="Testing alert generation",
                source="verification"
            )

            if alert2 is None:
                self.add_result(CheckResult(
                    name="AlertManager deduplication",
                    status=CheckStatus.PASS,
                    message="Duplicate alert correctly suppressed"
                ))
            else:
                self.add_result(CheckResult(
                    name="AlertManager deduplication",
                    status=CheckStatus.WARN,
                    message="Deduplication may not be working"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name="AlertManager",
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_metrics_collector(self):
        """Verify MetricsCollector component."""
        try:
            from self_healing import MetricsCollector, MetricSource, AggregationType

            collector = MetricsCollector()

            # Test collection
            metrics = await collector.collect_all()

            if MetricSource.SYSTEM.value in metrics:
                system_metrics = metrics[MetricSource.SYSTEM.value]
                self.add_result(CheckResult(
                    name="MetricsCollector system metrics",
                    status=CheckStatus.PASS,
                    message=f"Collected {len(system_metrics)} system metrics"
                ))
            else:
                self.add_result(CheckResult(
                    name="MetricsCollector system metrics",
                    status=CheckStatus.FAIL,
                    message="No system metrics collected"
                ))

            # Check CPU metric specifically
            cpu_metric = collector.get_metric(MetricSource.SYSTEM, "cpu_usage_percent")

            if cpu_metric and 0 <= cpu_metric.value <= 100:
                self.add_result(CheckResult(
                    name="MetricsCollector CPU metric",
                    status=CheckStatus.PASS,
                    message=f"CPU usage: {cpu_metric.value:.1f}%"
                ))
            else:
                self.add_result(CheckResult(
                    name="MetricsCollector CPU metric",
                    status=CheckStatus.WARN,
                    message="CPU metric not available or invalid"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name="MetricsCollector",
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_orchestrator(self):
        """Verify SelfHealingOrchestrator component."""
        try:
            from self_healing import SelfHealingOrchestrator

            orchestrator = SelfHealingOrchestrator(dry_run=True)

            # Check component initialization
            if (orchestrator.health_monitor and
                orchestrator.auto_repair and
                orchestrator.alert_manager and
                orchestrator.metrics_collector):
                self.add_result(CheckResult(
                    name="Orchestrator components",
                    status=CheckStatus.PASS,
                    message="All components initialized"
                ))
            else:
                self.add_result(CheckResult(
                    name="Orchestrator components",
                    status=CheckStatus.FAIL,
                    message="Some components not initialized"
                ))

            # Test start/stop
            await orchestrator.start()
            is_running = orchestrator.is_running
            await orchestrator.stop()

            if is_running:
                self.add_result(CheckResult(
                    name="Orchestrator lifecycle",
                    status=CheckStatus.PASS,
                    message="Start/stop cycle successful"
                ))
            else:
                self.add_result(CheckResult(
                    name="Orchestrator lifecycle",
                    status=CheckStatus.FAIL,
                    message="Failed to start orchestrator"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name="Orchestrator",
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_orchestrator_integration(self):
        """Verify orchestrator integrates all components."""
        try:
            from self_healing import SelfHealingOrchestrator, HealthMetrics

            orchestrator = SelfHealingOrchestrator(
                enable_auto_repair=True,
                dry_run=True
            )

            await orchestrator.start()

            # Inject bad metrics to trigger workflow
            bad_metrics = HealthMetrics(
                timestamp="2024-01-01T00:00:00",
                error_rate=0.10  # 10% - should trigger alert
            )

            await orchestrator.health_monitor.run_health_checks(bad_metrics)
            await asyncio.sleep(0.3)  # Allow async processing

            # Check stats
            stats = orchestrator.get_status()

            await orchestrator.stop()

            if stats.get("issues_detected", 0) > 0:
                self.add_result(CheckResult(
                    name=CHECK_INTEGRATION_WORKFLOW,
                    status=CheckStatus.PASS,
                    message=f"Issues detected: {stats.get('issues_detected')}, Alerts: {stats.get('alerts_sent', 0)}"
                ))
            else:
                self.add_result(CheckResult(
                    name=CHECK_INTEGRATION_WORKFLOW,
                    status=CheckStatus.WARN,
                    message="No issues detected from bad metrics"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name=CHECK_INTEGRATION_WORKFLOW,
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_callback_wiring(self):
        """Verify callbacks are properly wired."""
        try:
            from self_healing import SelfHealingOrchestrator

            orchestrator = SelfHealingOrchestrator(dry_run=True)

            # Check callbacks exist
            monitor = orchestrator.health_monitor

            callbacks_wired = (
                monitor.on_threshold_exceeded is not None and
                monitor.on_anomaly_detected is not None and
                monitor.on_status_change is not None
            )

            if callbacks_wired:
                self.add_result(CheckResult(
                    name=CHECK_CALLBACK_WIRING,
                    status=CheckStatus.PASS,
                    message="All callbacks connected"
                ))
            else:
                self.add_result(CheckResult(
                    name=CHECK_CALLBACK_WIRING,
                    status=CheckStatus.FAIL,
                    message="Some callbacks not connected"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name=CHECK_CALLBACK_WIRING,
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_thresholds(self):
        """Verify health thresholds are configured."""
        try:
            from self_healing import HealthMonitor

            monitor = HealthMonitor()

            if len(monitor.health_checks) > 0:
                self.add_result(CheckResult(
                    name=CHECK_HEALTH_THRESHOLDS,
                    status=CheckStatus.PASS,
                    message=f"{len(monitor.health_checks)} checks configured"
                ))
            else:
                self.add_result(CheckResult(
                    name=CHECK_HEALTH_THRESHOLDS,
                    status=CheckStatus.WARN,
                    message="No default health checks configured"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name=CHECK_HEALTH_THRESHOLDS,
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_repair_strategies(self):
        """Verify repair strategies are configured."""
        try:
            from self_healing import AutoRepair, RepairAction

            repair = AutoRepair()

            expected = len(RepairAction)
            actual = len(repair.repair_strategies)

            if actual >= 5:  # At least the main 5 actions
                self.add_result(CheckResult(
                    name=CHECK_REPAIR_STRATEGIES,
                    status=CheckStatus.PASS,
                    message=f"{actual} strategies configured"
                ))
            else:
                self.add_result(CheckResult(
                    name=CHECK_REPAIR_STRATEGIES,
                    status=CheckStatus.WARN,
                    message=f"Only {actual}/{expected} strategies configured"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name=CHECK_REPAIR_STRATEGIES,
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_alert_channels(self):
        """Verify alert channels are configured."""
        try:
            from self_healing import AlertManager, AlertSeverity

            manager = AlertManager()

            # Check routing rules
            if hasattr(manager, 'routing_rules') and len(manager.routing_rules) > 0:
                self.add_result(CheckResult(
                    name=CHECK_ALERT_CHANNELS,
                    status=CheckStatus.PASS,
                    message=f"{len(manager.routing_rules)} routing rules configured"
                ))
            else:
                self.add_result(CheckResult(
                    name=CHECK_ALERT_CHANNELS,
                    status=CheckStatus.WARN,
                    message="No routing rules found"
                ))

        except Exception as e:
            self.add_result(CheckResult(
                name=CHECK_ALERT_CHANNELS,
                status=CheckStatus.FAIL,
                message=str(e)
            ))

    async def check_unit_tests(self):
        """Verify unit tests exist."""
        test_files = [
            PROJECT_ROOT / "tests" / "unit" / "test_critical_fixes.py",
            PROJECT_ROOT / "tests" / "unit" / "test_metrics_collector.py",
        ]

        existing = [f for f in test_files if f.exists()]

        if len(existing) >= 2:
            self.add_result(CheckResult(
                name=CHECK_UNIT_TESTS,
                status=CheckStatus.PASS,
                message=f"{len(existing)} test files found"
            ))
        elif len(existing) > 0:
            self.add_result(CheckResult(
                name=CHECK_UNIT_TESTS,
                status=CheckStatus.WARN,
                message=f"Only {len(existing)} test files found"
            ))
        else:
            self.add_result(CheckResult(
                name=CHECK_UNIT_TESTS,
                status=CheckStatus.FAIL,
                message="No unit test files found"
            ))

    async def check_integration_tests(self):
        """Verify integration tests exist."""
        test_file = PROJECT_ROOT / "tests" / "integration" / "test_self_healing_system.py"

        if test_file.exists():
            # Count test classes
            content = test_file.read_text()
            test_count = content.count("def test_")

            self.add_result(CheckResult(
                name="Integration tests",
                status=CheckStatus.PASS,
                message=f"{test_count} integration tests found"
            ))
        else:
            self.add_result(CheckResult(
                name="Integration tests",
                status=CheckStatus.FAIL,
                message="Integration test file not found"
            ))

    async def check_runbook(self):
        """Verify operations runbook exists."""
        runbook = PROJECT_ROOT / "docs" / "OPERATIONS_RUNBOOK.md"

        if runbook.exists():
            content = runbook.read_text()
            sections = content.count("##")

            self.add_result(CheckResult(
                name="Operations runbook",
                status=CheckStatus.PASS,
                message=f"Runbook found with {sections} sections"
            ))
        else:
            self.add_result(CheckResult(
                name="Operations runbook",
                status=CheckStatus.FAIL,
                message="Operations runbook not found"
            ))

    async def check_api_docs(self):
        """Verify API documentation."""
        docs = [
            PROJECT_ROOT / "SELF_HEALING_SYSTEM.md",
            PROJECT_ROOT / "docs" / "OPERATIONS_RUNBOOK.md",
        ]

        existing = [d for d in docs if d.exists()]

        if len(existing) == len(docs):
            self.add_result(CheckResult(
                name="API documentation",
                status=CheckStatus.PASS,
                message=f"All {len(docs)} documentation files present"
            ))
        else:
            self.add_result(CheckResult(
                name="API documentation",
                status=CheckStatus.WARN,
                message=f"Only {len(existing)}/{len(docs)} docs found"
            ))

    def print_summary(self) -> bool:
        """Print verification summary."""
        print("\n" + "=" * 60)
        print("Verification Summary")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.status == CheckStatus.PASS)
        failed = sum(1 for r in self.results if r.status == CheckStatus.FAIL)
        warned = sum(1 for r in self.results if r.status == CheckStatus.WARN)
        skipped = sum(1 for r in self.results if r.status == CheckStatus.SKIP)
        total = len(self.results)

        print(f"\n  Total Checks: {total}")
        print(f"  ✅ Passed:    {passed}")
        print(f"  ❌ Failed:    {failed}")
        print(f"  ⚠️ Warnings:  {warned}")
        print(f"  ⏭️ Skipped:   {skipped}")

        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\n  Success Rate: {success_rate:.1f}%")

        if failed == 0:
            print("\n🎉 All critical checks passed!")
            print("   Self-healing system is ready for deployment.")
            return True
        else:
            print("\n⚠️ Some checks failed!")
            print("   Please fix the issues before deployment.")

            print("\n  Failed checks:")
            for r in self.results:
                if r.status == CheckStatus.FAIL:
                    print(f"    - {r.name}: {r.message}")

            return False


async def main():
    parser = argparse.ArgumentParser(
        description="Verify self-healing system components"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix issues automatically"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    verifier = SelfHealingVerifier(verbose=args.verbose, fix=args.fix)
    success = await verifier.run_all_checks()

    if args.json:
        results = [
            {
                "name": r.name,
                "status": r.status.name,
                "message": r.message
            }
            for r in verifier.results
        ]
        print(json.dumps(results, indent=2))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
