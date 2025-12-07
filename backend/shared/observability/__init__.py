"""
可观测性模块 (Observability Module)

模块功能描述:
    提供性能基准测试、结构化日志和监控功能。

主要功能:
    - 性能基准测试
    - 结构化日志记录
    - 指标收集和跟踪

主要组件:
    - PerformanceBenchmark: 性能基准测试
    - StructuredLogger: 结构化日志器
    - MetricsCollector: 指标收集器

辅助函数:
    - benchmark(): 基准测试装饰器
    - get_logger(): 获取日志器
    - track_request(): 跟踪请求
    - track_latency(): 跟踪延迟

最后修改日期: 2024-12-07
"""

from .benchmarks import (
    PerformanceBenchmark,
    BenchmarkConfig,
    benchmark,
    get_benchmark_results,
)
from .structured_logging import (
    StructuredLogger,
    LogConfig,
    get_logger,
    setup_logging,
)
from .metrics import (
    MetricsCollector,
    track_request,
    track_latency,
)

__all__ = [
    # Benchmarks
    "PerformanceBenchmark",
    "BenchmarkConfig",
    "benchmark",
    "get_benchmark_results",
    # Logging
    "StructuredLogger",
    "LogConfig",
    "get_logger",
    "setup_logging",
    # Metrics
    "MetricsCollector",
    "track_request",
    "track_latency",
]
