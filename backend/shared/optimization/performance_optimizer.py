"""
Performance Optimizer

分析系统瓶颈，优化关键路径执行效率。
提供性能分析和优化建议。
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import asyncio
import logging
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerformanceMetric(str, Enum):
    """性能指标类型"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DATABASE_QUERY_TIME = "database_query_time"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class PerformanceMeasurement:
    """性能测量结果"""
    metric_type: PerformanceMetric
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bottleneck:
    """性能瓶颈"""
    bottleneck_id: str
    location: str
    metric_type: PerformanceMetric
    current_value: float
    threshold: float
    impact: str  # high, medium, low
    recommendation: str
    estimated_improvement: str


@dataclass
class PerformanceReport:
    """性能报告"""
    report_id: str
    timestamp: datetime
    measurements: List[PerformanceMeasurement]
    bottlenecks: List[Bottleneck]
    recommendations: List[str]
    summary: Dict[str, Any]


class PerformanceProfiler:
    """
    性能分析器
    
    功能：
    1. 测量函数执行时间
    2. 分析数据库查询性能
    3. 检测内存使用
    4. 识别性能瓶颈
    """
    
    def __init__(self):
        self.measurements: List[PerformanceMeasurement] = []
        self.function_timings: Dict[str, List[float]] = defaultdict(list)
        self.query_timings: Dict[str, List[float]] = defaultdict(list)
    
    def profile_function(self, func: Callable) -> Callable:
        """
        性能分析装饰器
        
        Args:
            func: 要分析的函数
        
        Returns:
            包装后的函数
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = (time.perf_counter() - start_time) * 1000  # ms
                self.function_timings[func.__name__].append(duration)
                
                self.measurements.append(PerformanceMeasurement(
                    metric_type=PerformanceMetric.RESPONSE_TIME,
                    value=duration,
                    unit="ms",
                    timestamp=datetime.now(),
                    context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                ))
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = (time.perf_counter() - start_time) * 1000  # ms
                self.function_timings[func.__name__].append(duration)
                
                self.measurements.append(PerformanceMeasurement(
                    metric_type=PerformanceMetric.RESPONSE_TIME,
                    value=duration,
                    unit="ms",
                    timestamp=datetime.now(),
                    context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                ))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    def record_query_time(self, query: str, duration_ms: float) -> None:
        """记录数据库查询时间"""
        # 简化查询用于分组
        normalized_query = self._normalize_query(query)
        
        self.query_timings[normalized_query].append(duration_ms)
        
        self.measurements.append(PerformanceMeasurement(
            metric_type=PerformanceMetric.DATABASE_QUERY_TIME,
            value=duration_ms,
            unit="ms",
            timestamp=datetime.now(),
            context={"query": normalized_query[:100]}  # 截断长查询
        ))
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询（用于分组）"""
        # 移除参数值，保留结构
        import re
        normalized = re.sub(r'\$\d+', '?', query)
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r'"[^"]*"', '"?"', normalized)
        return normalized.strip()
    
    def analyze_bottlenecks(
        self,
        response_time_threshold_ms: float = 1000,
        query_time_threshold_ms: float = 100
    ) -> List[Bottleneck]:
        """分析性能瓶颈"""
        bottlenecks = []
        
        # 1. 分析函数执行时间
        for func_name, timings in self.function_timings.items():
            if not timings:
                continue
            
            avg_time = sum(timings) / len(timings)
            max_time = max(timings)
            
            if avg_time > response_time_threshold_ms:
                bottlenecks.append(Bottleneck(
                    bottleneck_id=f"func_{func_name}",
                    location=func_name,
                    metric_type=PerformanceMetric.RESPONSE_TIME,
                    current_value=avg_time,
                    threshold=response_time_threshold_ms,
                    impact="high" if avg_time > response_time_threshold_ms * 2 else "medium",
                    recommendation=f"优化函数 {func_name}，考虑缓存、异步处理或算法优化",
                    estimated_improvement="30-50%"
                ))
        
        # 2. 分析数据库查询
        for query, timings in self.query_timings.items():
            if not timings:
                continue
            
            avg_time = sum(timings) / len(timings)
            
            if avg_time > query_time_threshold_ms:
                bottlenecks.append(Bottleneck(
                    bottleneck_id=f"query_{hash(query) % 10000}",
                    location=query[:50],
                    metric_type=PerformanceMetric.DATABASE_QUERY_TIME,
                    current_value=avg_time,
                    threshold=query_time_threshold_ms,
                    impact="high" if avg_time > query_time_threshold_ms * 5 else "medium",
                    recommendation=f"优化查询性能，考虑添加索引、使用缓存或查询重构",
                    estimated_improvement="40-60%"
                ))
        
        return bottlenecks
    
    def generate_report(self) -> PerformanceReport:
        """生成性能报告"""
        bottlenecks = self.analyze_bottlenecks()
        
        # 计算统计信息
        if self.measurements:
            response_times = [
                m.value for m in self.measurements
                if m.metric_type == PerformanceMetric.RESPONSE_TIME
            ]
            
            query_times = [
                m.value for m in self.measurements
                if m.metric_type == PerformanceMetric.DATABASE_QUERY_TIME
            ]
            
            summary = {
                "total_measurements": len(self.measurements),
                "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
                "max_response_time_ms": max(response_times) if response_times else 0,
                "avg_query_time_ms": sum(query_times) / len(query_times) if query_times else 0,
                "bottlenecks_count": len(bottlenecks)
            }
        else:
            summary = {}
        
        # 生成建议
        recommendations = []
        if bottlenecks:
            recommendations.append(f"发现 {len(bottlenecks)} 个性能瓶颈，建议优先处理高影响项")
        
        high_impact = [b for b in bottlenecks if b.impact == "high"]
        if high_impact:
            recommendations.append(f"有 {len(high_impact)} 个高影响瓶颈需要立即处理")
        
        return PerformanceReport(
            report_id=f"perf_{datetime.now().isoformat()}",
            timestamp=datetime.now(),
            measurements=self.measurements[-1000:],  # 保留最近1000条
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            summary=summary
        )


class QueryOptimizer:
    """
    查询优化器
    
    专门优化数据库查询性能
    """
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self.query_cache: Dict[str, tuple] = {}  # query -> (result, timestamp, ttl)
        self.slow_queries: List[Dict[str, Any]] = []
    
    async def execute_optimized_query(
        self,
        query: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_ttl: int = 300
    ) -> tuple:
        """
        执行优化查询
        
        Args:
            query: SQL查询
            params: 查询参数
            use_cache: 是否使用缓存
            cache_ttl: 缓存TTL（秒）
        
        Returns:
            (结果, 执行时间ms)
        """
        # 1. 检查缓存
        if use_cache and self.cache_enabled:
            cache_key = self._generate_cache_key(query, params)
            cached = self.query_cache.get(cache_key)
            
            if cached:
                result, timestamp, ttl = cached
                if (datetime.now() - timestamp).total_seconds() < ttl:
                    logger.debug(f"Query cache hit: {query[:50]}")
                    return result, 0.0
        
        # 2. 执行查询
        start_time = time.perf_counter()
        # TODO: 实际执行查询
        result = None  # 占位符
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # 3. 记录慢查询
        if duration_ms > 100:  # 100ms阈值
            self.slow_queries.append({
                "query": query,
                "params": params,
                "duration_ms": duration_ms,
                "timestamp": datetime.now()
            })
        
        # 4. 缓存结果
        if use_cache and self.cache_enabled:
            cache_key = self._generate_cache_key(query, params)
            self.query_cache[cache_key] = (result, datetime.now(), cache_ttl)
        
        return result, duration_ms
    
    def _generate_cache_key(self, query: str, params: Optional[Dict]) -> str:
        """生成缓存键"""
        import hashlib
        key_str = f"{query}:{sorted(params.items()) if params else ''}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取慢查询列表"""
        return sorted(
            self.slow_queries,
            key=lambda x: x["duration_ms"],
            reverse=True
        )[:limit]
    
    def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """使缓存失效"""
        if pattern:
            # 按模式清除
            keys_to_remove = [
                k for k in self.query_cache.keys()
                if pattern in k
            ]
            for key in keys_to_remove:
                del self.query_cache[key]
            return len(keys_to_remove)
        else:
            # 清除所有
            count = len(self.query_cache)
            self.query_cache.clear()
            return count

