"""
监控模块 (Monitoring Module)

模块功能描述:
    提供 Prometheus 指标收集和系统监控能力。

主要组件:
    - MetricsCollector: 中央指标注册表
    - PrometheusMiddleware: FastAPI HTTP 指标中间件
    - 装饰器: @track_time, @count_calls
    - 辅助函数: track_in_flight, get_metrics

使用示例:
    from backend.shared.monitoring import MetricsCollector, PrometheusMiddleware
    
    # 添加中间件到 FastAPI
    app.add_middleware(PrometheusMiddleware)
    
    # 记录指标
    MetricsCollector.record_vulnerability("critical", "sql_injection")
    MetricsCollector.record_ai_request("openai", "gpt-4", 1.5, "success")
    
    # 使用装饰器
    @track_time("scan_duration", scan_type="full")
    async def scan_code():
        ...

最后修改日期: 2024-12-07
"""

from .metrics import (
    # Core collector
    MetricsCollector,
    
    # Middleware
    PrometheusMiddleware,
    
    # Decorators
    track_time,
    count_calls,
    track_in_flight,
    
    # Endpoint helpers
    get_metrics,
    get_metrics_content_type,
    
    # Constants
    Severity,
    VulnerabilityCategory,
    AIProvider,
    AnalysisType,
    
    # Availability flag
    PROMETHEUS_AVAILABLE,
)

__all__ = [
    "MetricsCollector",
    "PrometheusMiddleware",
    "track_time",
    "count_calls",
    "track_in_flight",
    "get_metrics",
    "get_metrics_content_type",
    "Severity",
    "VulnerabilityCategory",
    "AIProvider",
    "AnalysisType",
    "PROMETHEUS_AVAILABLE",
]
