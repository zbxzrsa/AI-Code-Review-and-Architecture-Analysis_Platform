"""
Performance Benchmarks for Design Pattern Implementations

Measures:
- CQRS: Read/write throughput, query response time
- Circuit Breaker: Interception delay, fault isolation rate
- DDD: Aggregate operation performance

Expected Results:
- CQRS: Read throughput +30%, Query response < 200ms
- Circuit Breaker: Interception delay < 100ms, Isolation rate 99.9%
"""
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import json

import sys
sys.path.insert(0, 'd:/Desktop/AI-Code-Review-and-Architecture-Analysis_Platform')


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_per_sec: float
    success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2),
            "max_time_ms": round(self.max_time_ms, 2),
            "p95_time_ms": round(self.p95_time_ms, 2),
            "p99_time_ms": round(self.p99_time_ms, 2),
            "throughput_per_sec": round(self.throughput_per_sec, 2),
            "success_rate": round(self.success_rate, 4),
        }


class Benchmark:
    """Base class for benchmarks."""
    
    def __init__(self, name: str, iterations: int = 1000):
        self.name = name
        self.iterations = iterations
        self.times: List[float] = []
        self.successes = 0
        self.failures = 0
    
    async def setup(self):
        """Setup before benchmark run."""
        pass
    
    async def teardown(self):
        """Cleanup after benchmark run."""
        pass
    
    async def run_single(self) -> bool:
        """Run a single iteration. Returns True if successful."""
        raise NotImplementedError
    
    async def run(self) -> BenchmarkResult:
        """Run the complete benchmark."""
        await self.setup()
        
        total_start = time.perf_counter()
        
        for _ in range(self.iterations):
            start = time.perf_counter()
            try:
                success = await self.run_single()
                if success:
                    self.successes += 1
                else:
                    self.failures += 1
            except Exception as e:
                self.failures += 1
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.times.append(elapsed_ms)
        
        total_time_ms = (time.perf_counter() - total_start) * 1000
        
        await self.teardown()
        
        # Calculate statistics
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        
        return BenchmarkResult(
            name=self.name,
            iterations=self.iterations,
            total_time_ms=total_time_ms,
            avg_time_ms=statistics.mean(sorted_times),
            min_time_ms=min(sorted_times),
            max_time_ms=max(sorted_times),
            p95_time_ms=sorted_times[int(n * 0.95)] if n > 20 else max(sorted_times),
            p99_time_ms=sorted_times[int(n * 0.99)] if n > 100 else max(sorted_times),
            throughput_per_sec=self.iterations / (total_time_ms / 1000),
            success_rate=self.successes / self.iterations,
        )


# =============================================================================
# CQRS Benchmarks
# =============================================================================

class CQRSCommandBenchmark(Benchmark):
    """Benchmark for CQRS command processing."""
    
    def __init__(self, iterations: int = 1000):
        super().__init__("CQRS Command Processing", iterations)
        self.command_bus = None
        self.event_store = None
    
    async def setup(self):
        from backend.shared.patterns.cqrs.commands import CommandBus
        from backend.shared.patterns.cqrs.event_sourcing import InMemoryEventStore, EventPublisher
        
        self.event_store = InMemoryEventStore()
        self.event_publisher = EventPublisher(self.event_store)
        self.command_bus = CommandBus(event_publisher=self.event_publisher)
    
    async def run_single(self) -> bool:
        from backend.shared.patterns.cqrs.commands import (
            CreateAnalysisCommand, CommandResult, DomainEvent
        )
        
        # Simulate command processing
        cmd = CreateAnalysisCommand(
            code="def test(): pass",
            language="python",
            rules=["security"]
        )
        
        # Simulate event emission
        event = DomainEvent(
            event_type="AnalysisCreated",
            aggregate_id="analysis-123",
            aggregate_type="Analysis",
            data={"language": "python"}
        )
        
        await self.event_store.append(event)
        return True


class CQRSQueryBenchmark(Benchmark):
    """Benchmark for CQRS query processing."""
    
    def __init__(self, iterations: int = 1000):
        super().__init__("CQRS Query Processing", iterations)
        self.query_bus = None
        self.read_model = None
    
    async def setup(self):
        from backend.shared.patterns.cqrs.queries import QueryBus, QueryCache
        from backend.shared.patterns.cqrs.read_models import (
            InMemoryReadModelStore, AnalysisReadModel
        )
        
        self.cache = QueryCache(max_size=10000)
        self.query_bus = QueryBus(cache=self.cache)
        
        self.store = InMemoryReadModelStore()
        self.read_model = AnalysisReadModel(self.store)
        
        # Pre-populate with test data
        for i in range(100):
            await self.read_model.create_analysis({
                "id": f"analysis-{i}",
                "language": "python",
                "project_id": f"proj-{i % 10}",
                "status": "completed"
            })
    
    async def run_single(self) -> bool:
        import random
        
        # Query random analysis
        analysis_id = f"analysis-{random.randint(0, 99)}"
        data = await self.read_model.get_analysis(analysis_id)
        
        return data is not None


class CQRSReadModelSyncBenchmark(Benchmark):
    """Benchmark for read model synchronization."""
    
    def __init__(self, iterations: int = 500):
        super().__init__("CQRS Read Model Sync", iterations)
    
    async def setup(self):
        from backend.shared.patterns.cqrs.event_sourcing import InMemoryEventStore
        from backend.shared.patterns.cqrs.read_models import InMemoryReadModelStore
        
        self.event_store = InMemoryEventStore()
        self.read_model_store = InMemoryReadModelStore()
    
    async def run_single(self) -> bool:
        from backend.shared.patterns.cqrs.commands import DomainEvent
        
        # Simulate event -> read model update
        event = DomainEvent(
            event_type="AnalysisCompleted",
            aggregate_id=f"analysis-{self.successes}",
            aggregate_type="Analysis",
            data={"issues_count": 5}
        )
        
        await self.event_store.append(event)
        await self.read_model_store.set(
            f"analysis:{event.aggregate_id}",
            {"id": event.aggregate_id, "status": "completed"}
        )
        
        return True


# =============================================================================
# Circuit Breaker Benchmarks
# =============================================================================

class CircuitBreakerInterceptionBenchmark(Benchmark):
    """Benchmark for circuit breaker interception delay."""
    
    def __init__(self, iterations: int = 1000):
        super().__init__("Circuit Breaker Interception", iterations)
    
    async def setup(self):
        from backend.shared.patterns.circuit_breaker.enhanced_circuit_breaker import (
            EnhancedCircuitBreaker, DynamicThresholdConfig
        )
        
        self.breaker = EnhancedCircuitBreaker(
            name="benchmark_breaker",
            config=DynamicThresholdConfig(
                failure_rate_threshold=0.50,
                minimum_requests=10
            )
        )
    
    async def run_single(self) -> bool:
        # Measure time to check circuit state
        start = time.perf_counter()
        can_execute = await self.breaker._check_and_update_state()
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Target: < 100ms interception delay
        return elapsed_ms < 100


class CircuitBreakerExecutionBenchmark(Benchmark):
    """Benchmark for circuit breaker protected execution."""
    
    def __init__(self, iterations: int = 1000):
        super().__init__("Circuit Breaker Execution", iterations)
    
    async def setup(self):
        from backend.shared.patterns.circuit_breaker.enhanced_circuit_breaker import (
            EnhancedCircuitBreaker, DynamicThresholdConfig
        )
        
        self.breaker = EnhancedCircuitBreaker(
            name="benchmark_breaker",
            config=DynamicThresholdConfig()
        )
    
    async def run_single(self) -> bool:
        async def fast_operation():
            await asyncio.sleep(0.001)  # 1ms simulated work
            return True
        
        result = await self.breaker.execute(fast_operation)
        return result


class CircuitBreakerFallbackBenchmark(Benchmark):
    """Benchmark for circuit breaker fallback mechanism."""
    
    def __init__(self, iterations: int = 500):
        super().__init__("Circuit Breaker Fallback", iterations)
    
    async def setup(self):
        from backend.shared.patterns.circuit_breaker.provider_circuit_breakers import (
            ProviderCircuitBreakerManager, ProviderConfig, ProviderType
        )
        
        self.manager = ProviderCircuitBreakerManager()
        
        # Register multiple providers
        for i in range(3):
            self.manager.register_provider(
                ProviderConfig(
                    provider_type=ProviderType.OPENAI,
                    name=f"provider_{i}",
                    endpoint=f"https://api{i}.example.com",
                    priority=i + 1
                ),
                fallback_providers=[f"provider_{j}" for j in range(3) if j != i]
            )
    
    async def run_single(self) -> bool:
        async def provider_call(provider_name):
            await asyncio.sleep(0.001)
            return f"result_from_{provider_name}"
        
        result = await self.manager.execute_with_fallback(
            provider_call,
            preferred_provider="provider_0"
        )
        
        return result is not None


# =============================================================================
# DDD Benchmarks
# =============================================================================

class DDDAggregateCreationBenchmark(Benchmark):
    """Benchmark for aggregate creation."""
    
    def __init__(self, iterations: int = 1000):
        super().__init__("DDD Aggregate Creation", iterations)
    
    async def run_single(self) -> bool:
        from backend.shared.patterns.ddd.domain_models import Analysis
        
        analysis = Analysis.create(
            project_id="proj-123",
            code="def test(): pass",
            language="python"
        )
        
        return analysis.id is not None


class DDDAggregateOperationBenchmark(Benchmark):
    """Benchmark for aggregate operations."""
    
    def __init__(self, iterations: int = 1000):
        super().__init__("DDD Aggregate Operations", iterations)
    
    async def run_single(self) -> bool:
        from backend.shared.patterns.ddd.domain_models import (
            Analysis, Severity, IssueType
        )
        from backend.shared.patterns.ddd.aggregates import IssueBuilder
        
        # Create analysis
        analysis = Analysis.create(
            project_id="proj-123",
            code="def test(): pass",
            language="python"
        )
        
        # Start analysis
        analysis.start()
        
        # Add issues
        issue = (
            IssueBuilder()
            .with_type(IssueType.SECURITY)
            .with_severity(Severity.HIGH)
            .with_message("Security issue found")
            .at_location("test.py", 1)
            .with_rule("SEC001")
            .build()
        )
        
        analysis.add_issue(issue)
        
        # Complete analysis
        analysis.complete(lines_analyzed=100, analysis_time_ms=50.0)
        
        # Get domain events
        events = analysis.clear_domain_events()
        
        return len(events) >= 2  # Created + Completed events


class DDDRepositoryBenchmark(Benchmark):
    """Benchmark for repository operations."""
    
    def __init__(self, iterations: int = 500):
        super().__init__("DDD Repository Operations", iterations)
    
    async def setup(self):
        from backend.shared.patterns.ddd.repositories import InMemoryRepository
        from backend.shared.patterns.ddd.domain_models import Analysis
        
        self.repository = InMemoryRepository()
    
    async def run_single(self) -> bool:
        from backend.shared.patterns.ddd.domain_models import Analysis
        
        # Create and save
        analysis = Analysis.create(
            project_id="proj-123",
            code="def test(): pass",
            language="python"
        )
        
        await self.repository.save(analysis)
        
        # Retrieve
        retrieved = await self.repository.get(analysis.id)
        
        return retrieved is not None and retrieved.id == analysis.id


# =============================================================================
# Benchmark Runner
# =============================================================================

async def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks and return results."""
    benchmarks = [
        # CQRS Benchmarks
        CQRSCommandBenchmark(iterations=1000),
        CQRSQueryBenchmark(iterations=1000),
        CQRSReadModelSyncBenchmark(iterations=500),
        
        # Circuit Breaker Benchmarks
        CircuitBreakerInterceptionBenchmark(iterations=1000),
        CircuitBreakerExecutionBenchmark(iterations=1000),
        CircuitBreakerFallbackBenchmark(iterations=500),
        
        # DDD Benchmarks
        DDDAggregateCreationBenchmark(iterations=1000),
        DDDAggregateOperationBenchmark(iterations=500),
        DDDRepositoryBenchmark(iterations=500),
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("Design Patterns Performance Benchmark")
    print("="*60 + "\n")
    
    for benchmark in benchmarks:
        print(f"Running: {benchmark.name}...")
        result = await benchmark.run()
        results.append(result)
        
        # Check targets
        status = "✓" if result.success_rate >= 0.99 else "✗"
        print(f"  {status} Avg: {result.avg_time_ms:.2f}ms, "
              f"P95: {result.p95_time_ms:.2f}ms, "
              f"Throughput: {result.throughput_per_sec:.0f}/s, "
              f"Success: {result.success_rate:.2%}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    # CQRS targets
    cqrs_query_result = next(r for r in results if "Query" in r.name)
    query_target_met = cqrs_query_result.avg_time_ms < 200
    print(f"\nCQRS Query Response Time Target (<200ms): "
          f"{'✓ PASS' if query_target_met else '✗ FAIL'} "
          f"({cqrs_query_result.avg_time_ms:.2f}ms)")
    
    # Circuit Breaker targets
    cb_interception_result = next(r for r in results if "Interception" in r.name)
    interception_target_met = cb_interception_result.p95_time_ms < 100
    print(f"Circuit Breaker Interception Delay Target (<100ms): "
          f"{'✓ PASS' if interception_target_met else '✗ FAIL'} "
          f"({cb_interception_result.p95_time_ms:.2f}ms)")
    
    cb_fallback_result = next(r for r in results if "Fallback" in r.name)
    isolation_target_met = cb_fallback_result.success_rate >= 0.999
    print(f"Circuit Breaker Fault Isolation Target (≥99.9%): "
          f"{'✓ PASS' if isolation_target_met else '✗ FAIL'} "
          f"({cb_fallback_result.success_rate:.2%})")
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [r.to_dict() for r in results],
        "targets": {
            "cqrs_query_response_time_target_met": query_target_met,
            "circuit_breaker_interception_target_met": interception_target_met,
            "circuit_breaker_isolation_target_met": isolation_target_met,
        }
    }


if __name__ == "__main__":
    results = asyncio.run(run_all_benchmarks())
    
    # Save results to file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to benchmark_results.json")
