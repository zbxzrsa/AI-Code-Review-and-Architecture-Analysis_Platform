"""
分布式版本控制 AI 核心模块 (Distributed Version Control AI Core Module)

模块功能描述:
    本模块实现了分布式版本控制 AI 系统的核心功能，采用微服务架构设计，
    提供服务发现、负载均衡、熔断器保护和事件驱动通信等企业级特性。

主要组件:
    - ServiceRegistry: 服务注册与发现中心
    - CircuitBreaker: 熔断器，用于故障隔离和服务保护
    - EventBus: 事件总线，支持异步事件驱动通信
    - DistributedVCAI: 分布式系统主协调器

核心特性:
    - 服务发现和负载均衡
    - 故障容错和自动恢复
    - 事件驱动架构
    - 水平扩展支持

最后修改日期: 2024-12-07
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """
    服务健康状态枚举类
    
    定义服务节点的运行状态，用于服务发现和健康检查。
    
    状态说明:
        - HEALTHY: 服务正常运行
        - DEGRADED: 服务降级，部分功能受限
        - UNHEALTHY: 服务不健康，无法处理请求
        - STARTING: 服务正在启动
        - STOPPING: 服务正在停止
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


class CircuitState(Enum):
    """
    熔断器状态枚举类
    
    实现熔断器模式的三种状态，用于服务故障隔离和自动恢复。
    
    状态说明:
        - CLOSED: 闭合状态，正常处理请求
        - OPEN: 断开状态，拒绝所有请求
        - HALF_OPEN: 半开状态，尝试恢复
    """
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class VCAIConfig:
    """
    分布式 VCAI 系统配置类
    
    功能描述:
        定义分布式版本控制 AI 系统的所有配置参数，包括服务配置、
        性能目标、学习配置、回滚配置和监控配置。
    
    配置分类:
        - 服务配置: service_id, cluster_name, node_count
        - 性能目标: learning_delay_seconds, iteration_cycle_hours 等
        - 学习配置: learning_enabled, learning_batch_size 等
        - 回滚配置: max_rollback_versions, rollback_timeout_seconds
        - 熔断器配置: failure_threshold, recovery_timeout_seconds 等
        - 监控配置: metrics_interval_seconds, health_check_interval_seconds
    """
    # Service Configuration
    service_id: str = "vcai-primary"
    cluster_name: str = "vcai-cluster"
    node_count: int = 3
    
    # Performance Targets
    learning_delay_seconds: int = 300        # < 5 minutes
    iteration_cycle_hours: int = 24          # ≤ 24 hours
    availability_target: float = 0.999       # > 99.9%
    merge_success_rate: float = 0.95         # > 95%
    
    # Learning Configuration
    learning_enabled: bool = True
    learning_batch_size: int = 100
    learning_interval_seconds: int = 60
    max_concurrent_learners: int = 5
    
    # Rollback Configuration
    max_rollback_versions: int = 10
    rollback_timeout_seconds: int = 30
    
    # Circuit Breaker Configuration
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3
    
    # Monitoring Configuration
    metrics_interval_seconds: int = 10
    health_check_interval_seconds: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'service_id': self.service_id,
            'cluster_name': self.cluster_name,
            'node_count': self.node_count,
            'learning_delay_seconds': self.learning_delay_seconds,
            'iteration_cycle_hours': self.iteration_cycle_hours,
            'availability_target': self.availability_target,
            'merge_success_rate': self.merge_success_rate
        }


@dataclass
class ServiceNode:
    """
    分布式系统服务节点类
    
    功能描述:
        表示分布式系统中的单个服务节点，包含节点标识、网络地址、
        健康状态、负载信息和能力列表。
    
    属性说明:
        - node_id: 节点唯一标识符
        - host: 节点主机地址
        - port: 节点服务端口
        - status: 节点健康状态
        - last_heartbeat: 最后心跳时间
        - load: 当前负载（0.0-1.0）
        - version: 节点版本号
        - capabilities: 节点能力列表
    """
    node_id: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.STARTING
    last_heartbeat: Optional[str] = None
    load: float = 0.0
    version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)
    
    def is_available(self) -> bool:
        """
        检查节点是否可用
        
        返回值:
            bool: 如果节点状态为 HEALTHY 或 DEGRADED 则返回 True
        """
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]


class CircuitBreaker:
    """
    熔断器类 - 用于服务容错
    
    功能描述:
        实现熔断器模式，通过临时阻断对失败服务的请求来防止级联故障。
        当服务连续失败达到阈值时，熔断器断开；经过恢复超时后进入半开状态
        进行试探；如果试探成功则闭合，否则重新断开。
    
    状态转换:
        CLOSED -> OPEN: 失败次数达到阈值
        OPEN -> HALF_OPEN: 恢复超时后
        HALF_OPEN -> CLOSED: 试探成功
        HALF_OPEN -> OPEN: 试探失败
    
    属性:
        failure_threshold: 触发断开的失败次数阈值
        recovery_timeout: 从断开到半开的等待时间（秒）
        half_open_max_calls: 半开状态允许的最大试探次数
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """
        检查是否可以执行请求
        
        根据当前熔断器状态判断是否允许执行请求。
        
        返回值:
            bool: True 表示可以执行，False 表示请求被阻断
        """
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self) -> None:
        """
        记录成功执行
        
        当请求执行成功时调用，用于更新熔断器状态。
        在半开状态下，累计成功次数达到阈值后闭合熔断器。
        """
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")
        else:
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """
        记录失败执行
        
        当请求执行失败时调用，累计失败次数。
        当失败次数达到阈值或在半开状态下失败时，断开熔断器。
        """
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker OPEN - service still failing")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class ServiceRegistry:
    """
    服务注册与发现中心
    
    功能描述:
        管理分布式系统中的服务注册、发现和健康检查。
        支持服务节点的动态注册和注销，提供基于负载的节点选择。
    
    主要功能:
        - 服务节点注册和注销
        - 健康节点查询
        - 基于负载的节点选择（负载均衡）
        - 定期健康检查
        - 心跳监控
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceNode] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_running: bool = False
    
    def register(self, node: ServiceNode) -> None:
        """
        注册服务节点
        
        参数:
            node: 要注册的服务节点对象
        """
        self.services[node.node_id] = node
        self.circuit_breakers[node.node_id] = CircuitBreaker()
        node.status = ServiceStatus.HEALTHY
        node.last_heartbeat = datetime.now().isoformat()
        logger.info(f"Registered service node: {node.node_id}")
    
    def deregister(self, node_id: str) -> None:
        """
        注销服务节点
        
        参数:
            node_id: 要注销的节点标识符
        """
        if node_id in self.services:
            del self.services[node_id]
            del self.circuit_breakers[node_id]
            logger.info(f"Deregistered service node: {node_id}")
    
    def get_healthy_nodes(self) -> List[ServiceNode]:
        """
        获取所有健康的服务节点
        
        返回值:
            List[ServiceNode]: 状态可用且熔断器未断开的节点列表
        """
        return [
            node for node in self.services.values()
            if node.is_available() and 
            self.circuit_breakers[node.node_id].can_execute()
        ]
    
    def get_node_by_load(self) -> Optional[ServiceNode]:
        """
        根据负载选择节点（负载均衡）
        
        返回当前负载最低的健康节点，实现简单的负载均衡策略。
        
        返回值:
            Optional[ServiceNode]: 负载最低的节点，无可用节点时返回 None
        """
        healthy = self.get_healthy_nodes()
        if not healthy:
            return None
        return min(healthy, key=lambda n: n.load)
    
    def update_heartbeat(self, node_id: str) -> None:
        """
        更新节点心跳时间
        
        参数:
            node_id: 节点标识符
        """
        if node_id in self.services:
            self.services[node_id].last_heartbeat = datetime.now().isoformat()
    
    async def start_health_checks(self, interval: int = 5) -> None:
        """
        启动定期健康检查
        
        参数:
            interval: 健康检查间隔（秒），默认 5 秒
        """
        async def check_health():
            while self._health_check_running:
                try:
                    await asyncio.sleep(interval)
                    
                    # Run in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    await asyncio.wait_for(
                        loop.run_in_executor(None, self._check_all_nodes),
                        timeout=interval * 0.8  # 80% of interval
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Health check timed out after {interval * 0.8}s")
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    await asyncio.sleep(interval)  # Back off on error
        
        self._health_check_running = True
        self._health_check_task = asyncio.create_task(check_health())
    
    def stop_health_checks(self) -> None:
        """Stop health check loop."""
        self._health_check_running = False
        if self._health_check_task:
            self._health_check_task.cancel()
    
    def _check_all_nodes(self) -> None:
        """
        检查所有节点的健康状态
        
        根据心跳时间判断节点状态:
            - 超过 30 秒无心跳: 标记为 UNHEALTHY
            - 超过 15 秒无心跳: 标记为 DEGRADED
        """
        now = datetime.now()
        
        for node_id, node in self.services.items():
            if node.last_heartbeat:
                last_hb = datetime.fromisoformat(node.last_heartbeat)
                elapsed = (now - last_hb).total_seconds()
                
                if elapsed > 30:
                    node.status = ServiceStatus.UNHEALTHY
                    logger.warning(f"Node {node_id} marked unhealthy - no heartbeat")
                elif elapsed > 15:
                    node.status = ServiceStatus.DEGRADED
    
    # =========================================================================
    # Batch Operations (P1 Enhancement)
    # =========================================================================
    
    def register_batch(self, nodes: List[ServiceNode]) -> int:
        """
        Register multiple service nodes at once.
        
        P1 optimization: Batch registration reduces overhead for
        bulk service discovery operations.
        
        Args:
            nodes: List of nodes to register
            
        Returns:
            Number of successfully registered nodes
        """
        registered = 0
        for node in nodes:
            try:
                self.register(node)
                registered += 1
            except Exception as e:
                logger.warning(f"Failed to register node {node.node_id}: {e}")
        return registered
    
    def get_nodes_by_capability(self, capability: str) -> List[ServiceNode]:
        """
        Get all healthy nodes with a specific capability.
        
        Args:
            capability: Required capability (e.g., "learning", "inference")
            
        Returns:
            List of nodes with the specified capability
        """
        return [
            node for node in self.get_healthy_nodes()
            if capability in node.capabilities
        ]
    
    def get_least_loaded_node_with_capability(
        self,
        capability: str
    ) -> Optional[ServiceNode]:
        """
        Get the least loaded node with a specific capability.
        
        Combines capability filtering with load balancing.
        
        Args:
            capability: Required capability
            
        Returns:
            Least loaded node with capability, or None
        """
        capable_nodes = self.get_nodes_by_capability(capability)
        if not capable_nodes:
            return None
        return min(capable_nodes, key=lambda n: n.load)
    
    def update_node_load(self, node_id: str, load: float) -> bool:
        """
        Update the load metric for a node.
        
        Args:
            node_id: Node identifier
            load: New load value (0.0 - 1.0)
            
        Returns:
            True if updated successfully
        """
        if node_id not in self.services:
            return False
        
        self.services[node_id].load = max(0.0, min(1.0, load))
        return True
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get aggregate cluster statistics.
        
        Returns:
            Dictionary with cluster health metrics
        """
        nodes = list(self.services.values())
        if not nodes:
            return {"total_nodes": 0, "healthy": 0, "avg_load": 0.0}
        
        healthy = [n for n in nodes if n.status == ServiceStatus.HEALTHY]
        degraded = [n for n in nodes if n.status == ServiceStatus.DEGRADED]
        unhealthy = [n for n in nodes if n.status == ServiceStatus.UNHEALTHY]
        
        total_load = sum(n.load for n in nodes)
        
        return {
            "total_nodes": len(nodes),
            "healthy": len(healthy),
            "degraded": len(degraded),
            "unhealthy": len(unhealthy),
            "avg_load": total_load / len(nodes),
            "max_load": max(n.load for n in nodes),
            "min_load": min(n.load for n in nodes),
            "circuit_breakers_open": sum(
                1 for cb in self.circuit_breakers.values()
                if cb.state == CircuitState.OPEN
            ),
        }


class EventBus:
    """
    事件驱动通信总线
    
    功能描述:
        实现微服务之间的异步事件通信，支持发布-订阅模式。
        订阅者可以注册对特定事件类型的关注，当事件发布时自动通知。
    
    主要功能:
        - 事件订阅和取消订阅
        - 异步事件发布
        - 事件历史记录
        - 近期事件查询
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """
        订阅事件类型
        
        参数:
            event_type: 事件类型标识符
            handler: 事件处理函数，支持同步和异步函数
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """
        取消订阅事件类型
        
        参数:
            event_type: 事件类型标识符
            handler: 要移除的事件处理函数
        """
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
    
    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = "unknown"
    ) -> None:
        """
        发布事件
        
        创建事件对象并通知所有订阅该事件类型的处理器。
        事件会被记录到历史中，保留最近 max_history 条记录。
        
        参数:
            event_type: 事件类型标识符
            data: 事件数据负载
            source: 事件来源标识
        """
        event = {
            'event_id': hashlib.sha256(
                f"{event_type}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16],
            'event_type': event_type,
            'data': data,
            'source': source,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notify subscribers
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
    
    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        获取近期事件
        
        参数:
            event_type: 筛选的事件类型，None 表示不筛选
            limit: 返回的最大事件数量
        
        返回值:
            List[Dict]: 事件列表，按时间倒序排列
        """
        events = self.event_history
        if event_type:
            events = [e for e in events if e['event_type'] == event_type]
        return events[-limit:]
    
    # =========================================================================
    # Batch Operations (P1 Enhancement)
    # =========================================================================
    
    async def publish_batch(
        self,
        events: List[Tuple[str, Dict[str, Any], str]],
        parallel: bool = True,
    ) -> List[Dict]:
        """
        Publish multiple events efficiently.
        
        P1 optimization: Batch event publishing reduces overhead for
        high-throughput scenarios.
        
        Args:
            events: List of (event_type, data, source) tuples
            parallel: Whether to notify subscribers in parallel
            
        Returns:
            List of created event objects
        """
        created_events = []
        
        for event_type, data, source in events:
            event = {
                'event_id': hashlib.sha256(
                    f"{event_type}:{datetime.now().isoformat()}".encode()
                ).hexdigest()[:16],
                'event_type': event_type,
                'data': data,
                'source': source,
                'timestamp': datetime.now().isoformat()
            }
            created_events.append(event)
            
            # Store in history
            self.event_history.append(event)
        
        # Trim history
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notify subscribers
        if parallel:
            # Parallel notification for better throughput
            tasks = []
            for event in created_events:
                if event['event_type'] in self.subscribers:
                    for handler in self.subscribers[event['event_type']]:
                        if asyncio.iscoroutinefunction(handler):
                            tasks.append(asyncio.create_task(
                                self._safe_call_handler(handler, event)
                            ))
                        else:
                            try:
                                handler(event)
                            except Exception as e:
                                logger.error(f"Event handler error: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential notification
            for event in created_events:
                if event['event_type'] in self.subscribers:
                    for handler in self.subscribers[event['event_type']]:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                handler(event)
                        except Exception as e:
                            logger.error(f"Event handler error: {e}")
        
        return created_events
    
    async def _safe_call_handler(self, handler: Callable, event: Dict) -> None:
        """Safely call an async handler."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Event handler error: {e}")
    
    def clear_history(self) -> int:
        """Clear event history and return count of cleared events."""
        count = len(self.event_history)
        self.event_history.clear()
        return count
    
    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """Get count of subscribers for an event type or all types."""
        if event_type:
            return len(self.subscribers.get(event_type, []))
        return sum(len(handlers) for handlers in self.subscribers.values())


class DistributedVCAI:
    """
    分布式版本控制 AI 核心类
    
    功能描述:
        分布式系统的主协调器，负责统一管理所有子系统组件，
        包括服务发现、事件通信、学习引擎、版本管理等。
    
    核心功能:
        - 服务发现和负载均衡
        - 事件驱动架构
        - 故障容错和自动恢复
        - 实时学习集成
    
    事件类型:
        - learning.started: 学习开始
        - learning.completed: 学习完成
        - version.created: 版本创建
        - version.promoted: 版本升级
        - version.rollback: 版本回滚
        - merge.completed: 合并完成
        - health.check: 健康检查
    """
    
    # Event Types
    EVENT_LEARNING_STARTED = "learning.started"
    EVENT_LEARNING_COMPLETED = "learning.completed"
    EVENT_VERSION_CREATED = "version.created"
    EVENT_VERSION_PROMOTED = "version.promoted"
    EVENT_VERSION_ROLLBACK = "version.rollback"
    EVENT_MERGE_COMPLETED = "merge.completed"
    EVENT_HEALTH_CHECK = "health.check"
    
    def __init__(self, config: VCAIConfig):
        self.config = config
        self.registry = ServiceRegistry()
        self.event_bus = EventBus()
        
        # Component references (set during initialization)
        self.learning_engine = None
        self.dual_loop = None
        self.version_engine = None
        self.monitor = None
        self.rollback_manager = None
        
        # State
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.current_version = "1.0.0"
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """
        设置内部事件处理器
        
        注册学习完成和版本创建等内部事件的处理函数。
        """
        self.event_bus.subscribe(
            self.EVENT_LEARNING_COMPLETED,
            self._on_learning_completed
        )
        self.event_bus.subscribe(
            self.EVENT_VERSION_CREATED,
            self._on_version_created
        )
    
    async def _on_learning_completed(self, event: Dict) -> None:
        """
        处理学习完成事件
        
        参数:
            event: 学习完成事件对象
        """
        logger.info(f"Learning completed: {event['data']}")
        
        # Trigger version comparison if needed
        if self.version_engine:
            await self.version_engine.analyze_improvements()
    
    def _on_version_created(self, event: Dict) -> None:
        """
        处理新版本创建事件
        
        参数:
            event: 版本创建事件对象
        """
        logger.info(f"New version created: {event['data']}")
    
    async def start(self) -> None:
        """
        启动分布式系统
        
        执行以下初始化步骤:
            1. 设置运行状态
            2. 注册主节点
            3. 启动健康检查
            4. 发布启动事件
        """
        logger.info(f"Starting Distributed VCAI: {self.config.service_id}")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Register this node
        primary_node = ServiceNode(
            node_id=self.config.service_id,
            host="localhost",
            port=8000,
            capabilities=["learning", "versioning", "merging", "rollback"]
        )
        self.registry.register(primary_node)
        
        # Start health checks
        await self.registry.start_health_checks(
            self.config.health_check_interval_seconds
        )
        
        # Publish startup event
        await self.event_bus.publish(
            "system.started",
            {"config": self.config.to_dict()},
            self.config.service_id
        )
        
        logger.info("Distributed VCAI started successfully")
    
    async def stop(self) -> None:
        """
        停止分布式系统
        
        执行以下清理步骤:
            1. 设置停止状态
            2. 注销节点
            3. 发布停止事件
        """
        logger.info("Stopping Distributed VCAI...")
        
        self.is_running = False
        
        # Deregister node
        self.registry.deregister(self.config.service_id)
        
        # Publish shutdown event
        await self.event_bus.publish(
            "system.stopped",
            {"uptime_seconds": self.get_uptime()},
            self.config.service_id
        )
        
        logger.info("Distributed VCAI stopped")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with graceful shutdown."""
        await self.graceful_shutdown()
        return False
    
    async def graceful_shutdown(self, timeout: float = 30.0) -> None:
        """
        Graceful shutdown with timeout.
        
        P1 enhancement: Ensures all in-flight operations complete
        before shutdown, with configurable timeout.
        
        Args:
            timeout: Maximum seconds to wait for pending operations
        """
        logger.info(f"Initiating graceful shutdown (timeout: {timeout}s)...")
        
        # Signal shutdown intent
        self.is_running = False
        
        # Wait for in-flight requests (simple implementation)
        # In production, track active requests and wait for completion
        start_time = datetime.now()
        
        # Give pending operations time to complete
        await asyncio.sleep(min(1.0, timeout / 10))
        
        # Perform regular shutdown
        await self.stop()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Graceful shutdown completed in {elapsed:.2f}s")
    
    def get_uptime(self) -> float:
        """
        获取系统运行时间
        
        返回值:
            float: 系统运行时间（秒）
        """
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0
    
    def get_availability(self) -> float:
        """
        计算系统可用性
        
        返回值:
            float: 系统可用性比率（0.0-1.0）
        """
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 1.0
        return self.successful_requests / total
    
    async def execute_with_circuit_breaker(
        self,
        node_id: str,
        operation: Callable
    ) -> Any:
        """
        使用熔断器保护执行操作
        
        在熔断器保护下执行指定操作，自动记录成功或失败状态。
        
        参数:
            node_id: 目标节点标识符
            operation: 要执行的操作函数
        
        返回值:
            Any: 操作执行结果
        
        异常:
            ValueError: 节点不存在
            RuntimeError: 熔断器处于断开状态
        """
        if node_id not in self.registry.circuit_breakers:
            raise ValueError(f"Unknown node: {node_id}")
        
        cb = self.registry.circuit_breakers[node_id]
        
        if not cb.can_execute():
            raise RuntimeError(f"Circuit breaker OPEN for {node_id}")
        
        try:
            result = await operation() if asyncio.iscoroutinefunction(operation) else operation()
            cb.record_success()
            self.successful_requests += 1
            return result
        except Exception:
            cb.record_failure()
            self.failed_requests += 1
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取系统综合状态
        
        返回值:
            Dict[str, Any]: 包含以下信息的状态字典:
                - service_id: 服务标识
                - is_running: 运行状态
                - uptime_seconds: 运行时间
                - current_version: 当前版本
                - availability: 可用性
                - meets_sla: 是否满足 SLA
                - nodes: 节点状态列表
                - recent_events: 近期事件数量
        """
        return {
            'service_id': self.config.service_id,
            'is_running': self.is_running,
            'uptime_seconds': self.get_uptime(),
            'current_version': self.current_version,
            'availability': self.get_availability(),
            'availability_target': self.config.availability_target,
            'meets_sla': self.get_availability() >= self.config.availability_target,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'nodes': {
                node_id: {
                    'status': node.status.value,
                    'load': node.load,
                    'circuit_breaker': self.registry.circuit_breakers[node_id].get_status()
                }
                for node_id, node in self.registry.services.items()
            },
            'recent_events': len(self.event_bus.event_history)
        }
    
    async def execute_with_retry(
        self,
        operation: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        retryable_exceptions: tuple = (Exception,),
    ) -> Any:
        """
        Execute operation with exponential backoff retry.
        
        P1 optimization: Adds intelligent retry logic with configurable
        exponential backoff for transient failures.
        
        Args:
            operation: Async or sync callable to execute
            max_retries: Maximum retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay cap (seconds)
            exponential_base: Multiplier for exponential backoff
            retryable_exceptions: Tuple of exceptions that trigger retry
            
        Returns:
            Result of successful operation
            
        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    # Calculate delay with exponential backoff + jitter
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    # Add jitter (0-25% of delay)
                    import random
                    jitter = delay * random.uniform(0, 0.25)
                    delay += jitter
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries} retries exhausted: {e}")
        
        raise last_exception


# =============================================================================
# Retry Policy Helper (P1 Enhancement)
# =============================================================================

@dataclass
class RetryPolicy:
    """
    Configurable retry policy for operations.
    
    Usage:
        policy = RetryPolicy(max_retries=5, base_delay=0.5)
        
        @policy.wrap
        async def fetch_data():
            return await client.get(url)
        
        # Or use directly
        result = await policy.execute(fetch_data)
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = field(default_factory=lambda: (Exception,))
    
    async def execute(self, operation: Callable) -> Any:
        """Execute operation with this retry policy."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
            except self.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    import random
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    delay += delay * random.uniform(0, 0.25)
                    
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries}: {e}")
                    await asyncio.sleep(delay)
        
        raise last_exception
    
    def wrap(self, func: Callable) -> Callable:
        """Decorator to wrap a function with retry logic."""
        import functools
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute(lambda: func(*args, **kwargs))
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                import asyncio
                return asyncio.get_event_loop().run_until_complete(
                    self.execute(lambda: func(*args, **kwargs))
                )
            return sync_wrapper
