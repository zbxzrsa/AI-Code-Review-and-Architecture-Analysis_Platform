"""
Performance Benchmark Module

Provides comprehensive performance benchmarking for:
- API response time
- Resource utilization
- Concurrent processing capacity
"""

import time
import asyncio
import statistics
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Awaitable
from datetime import datetime, timezone
from functools import wraps
from contextlib import asynccontextmanager
import psutil

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # Timing
    warmup_iterations: int = 5
    benchmark_iterations: int = 100
    timeout_seconds: float = 30.0
    
    # Concurrency testing
    concurrent_users: List[int] = field(default_factory=lambda: [1, 10, 50, 100])
    
    # Thresholds
    max_response_time_ms: float = 500.0
    max_p95_latency_ms: float = 1000.0
    min_requests_per_second: float = 100.0
    
    # Resource limits
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    std_dev_ms: float
    requests_per_second: float
    success_count: int
    error_count: int
    cpu_percent: float
    memory_percent: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "timing": {
                "total_ms": round(self.total_time_ms, 2),
                "avg_ms": round(self.avg_time_ms, 2),
                "min_ms": round(self.min_time_ms, 2),
                "max_ms": round(self.max_time_ms, 2),
                "p50_ms": round(self.p50_time_ms, 2),
                "p95_ms": round(self.p95_time_ms, 2),
                "p99_ms": round(self.p99_time_ms, 2),
                "std_dev_ms": round(self.std_dev_ms, 2),
            },
            "throughput": {
                "requests_per_second": round(self.requests_per_second, 2),
                "success_count": self.success_count,
                "error_count": self.error_count,
            },
            "resources": {
                "cpu_percent": round(self.cpu_percent, 2),
                "memory_percent": round(self.memory_percent, 2),
            },
            "timestamp": self.timestamp.isoformat(),
        }
    
    def passes_thresholds(self, config: BenchmarkConfig) -> bool:
        """Check if result passes configured thresholds."""
        return (
            self.avg_time_ms <= config.max_response_time_ms and
            self.p95_time_ms <= config.max_p95_latency_ms and
            self.requests_per_second >= config.min_requests_per_second and
            self.cpu_percent <= config.max_cpu_percent and
            self.memory_percent <= config.max_memory_percent
        )


class PerformanceBenchmark:
    """
    Performance benchmark runner.
    
    Usage:
        benchmark = PerformanceBenchmark()
        
        # Benchmark a function
        result = await benchmark.run(
            "api_endpoint",
            lambda: client.get("/api/endpoint")
        )
        
        # Benchmark with concurrency
        results = await benchmark.run_concurrent(
            "api_endpoint",
            lambda: client.get("/api/endpoint"),
            concurrent_users=[1, 10, 50]
        )
        
        # Generate report
        report = benchmark.generate_report()
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._results: List[BenchmarkResult] = []
    
    async def run(
        self,
        name: str,
        func: Callable[[], Awaitable[Any]],
        iterations: Optional[int] = None,
        warmup: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Run benchmark on an async function.
        
        Args:
            name: Benchmark name
            func: Async function to benchmark
            iterations: Number of iterations
            warmup: Warmup iterations
        
        Returns:
            BenchmarkResult with timing statistics
        """
        iterations = iterations or self.config.benchmark_iterations
        warmup = warmup or self.config.warmup_iterations
        
        # Warmup phase
        logger.info(f"Warming up {name} ({warmup} iterations)")
        for _ in range(warmup):
            try:
                await asyncio.wait_for(func(), timeout=self.config.timeout_seconds)
            except Exception:
                pass
        
        # Benchmark phase
        logger.info(f"Running benchmark {name} ({iterations} iterations)")
        times: List[float] = []
        success_count = 0
        error_count = 0
        
        # Get initial resource usage
        process = psutil.Process()
        cpu_start = process.cpu_percent()
        mem_start = process.memory_percent()
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            iter_start = time.perf_counter()
            try:
                await asyncio.wait_for(func(), timeout=self.config.timeout_seconds)
                success_count += 1
            except Exception as e:
                error_count += 1
                logger.debug(f"Benchmark iteration {i} error: {e}")
            
            iter_time = (time.perf_counter() - iter_start) * 1000
            times.append(iter_time)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Get final resource usage
        cpu_end = process.cpu_percent()
        mem_end = process.memory_percent()
        
        # Calculate statistics
        sorted_times = sorted(times)
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time,
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            p50_time_ms=sorted_times[int(len(sorted_times) * 0.50)],
            p95_time_ms=sorted_times[int(len(sorted_times) * 0.95)],
            p99_time_ms=sorted_times[int(len(sorted_times) * 0.99)],
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
            requests_per_second=(iterations / total_time) * 1000,
            success_count=success_count,
            error_count=error_count,
            cpu_percent=(cpu_start + cpu_end) / 2,
            memory_percent=(mem_start + mem_end) / 2,
        )
        
        self._results.append(result)
        logger.info(f"Benchmark {name} complete: avg={result.avg_time_ms:.2f}ms, p95={result.p95_time_ms:.2f}ms")
        
        return result
    
    async def run_concurrent(
        self,
        name: str,
        func: Callable[[], Awaitable[Any]],
        concurrent_users: Optional[List[int]] = None
    ) -> List[BenchmarkResult]:
        """
        Run benchmark with varying concurrency levels.
        
        Args:
            name: Benchmark name
            func: Async function to benchmark
            concurrent_users: List of concurrency levels
        
        Returns:
            List of BenchmarkResults for each concurrency level
        """
        concurrent_users = concurrent_users or self.config.concurrent_users
        results = []
        
        for users in concurrent_users:
            logger.info(f"Running {name} with {users} concurrent users")
            
            async def concurrent_run():
                await asyncio.gather(*[func() for _ in range(users)])
            
            result = await self.run(
                f"{name}_concurrent_{users}",
                concurrent_run,
                iterations=self.config.benchmark_iterations // users
            )
            results.append(result)
        
        return results
    
    @asynccontextmanager
    async def measure(self, name: str):
        """
        Context manager for measuring code block performance.
        
        Usage:
            async with benchmark.measure("operation"):
                await do_something()
        """
        start_time = time.perf_counter()
        process = psutil.Process()
        cpu_start = process.cpu_percent()
        mem_start = process.memory_percent()
        
        try:
            yield
            success = True
        except Exception:
            success = False
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            cpu_end = process.cpu_percent()
            mem_end = process.memory_percent()
            
            result = BenchmarkResult(
                name=name,
                iterations=1,
                total_time_ms=elapsed_ms,
                avg_time_ms=elapsed_ms,
                min_time_ms=elapsed_ms,
                max_time_ms=elapsed_ms,
                p50_time_ms=elapsed_ms,
                p95_time_ms=elapsed_ms,
                p99_time_ms=elapsed_ms,
                std_dev_ms=0,
                requests_per_second=1000 / elapsed_ms if elapsed_ms > 0 else 0,
                success_count=1 if success else 0,
                error_count=0 if success else 1,
                cpu_percent=(cpu_start + cpu_end) / 2,
                memory_percent=(mem_start + mem_end) / 2,
            )
            self._results.append(result)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self._results:
            return {"error": "No benchmark results"}
        
        return {
            "summary": {
                "total_benchmarks": len(self._results),
                "total_iterations": sum(r.iterations for r in self._results),
                "avg_response_time_ms": round(statistics.mean(r.avg_time_ms for r in self._results), 2),
                "avg_throughput_rps": round(statistics.mean(r.requests_per_second for r in self._results), 2),
            },
            "results": [r.to_dict() for r in self._results],
            "thresholds": {
                "max_response_time_ms": self.config.max_response_time_ms,
                "max_p95_latency_ms": self.config.max_p95_latency_ms,
                "min_requests_per_second": self.config.min_requests_per_second,
            },
            "pass_rate": sum(1 for r in self._results if r.passes_thresholds(self.config)) / len(self._results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def clear_results(self) -> None:
        """Clear all benchmark results."""
        self._results.clear()


# Global benchmark instance
_benchmark: Optional[PerformanceBenchmark] = None


def get_benchmark() -> PerformanceBenchmark:
    """Get global benchmark instance."""
    global _benchmark
    if _benchmark is None:
        _benchmark = PerformanceBenchmark()
    return _benchmark


def benchmark(name: Optional[str] = None):
    """
    Decorator for benchmarking async functions.
    
    Usage:
        @benchmark("api_call")
        async def my_function():
            ...
    """
    def decorator(func: Callable):
        bench_name = name or func.__name__
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            bench = get_benchmark()
            async with bench.measure(bench_name):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def get_benchmark_results() -> Dict[str, Any]:
    """Get all benchmark results."""
    return get_benchmark().generate_report()
