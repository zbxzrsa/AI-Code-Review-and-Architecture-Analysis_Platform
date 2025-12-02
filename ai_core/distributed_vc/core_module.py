"""
Distributed Version Control AI Core Module

Microservice-based architecture with:
- Service discovery and load balancing
- Fault tolerance and circuit breaker
- Event-driven communication
- Horizontal scaling support
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
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class VCAIConfig:
    """Configuration for Distributed VCAI System"""
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
    """A node in the distributed system"""
    node_id: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.STARTING
    last_heartbeat: Optional[str] = None
    load: float = 0.0
    version: str = "1.0.0"
    capabilities: List[str] = field(default_factory=list)
    
    def is_available(self) -> bool:
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]


class CircuitBreaker:
    """
    Circuit Breaker for Fault Tolerance
    
    Prevents cascading failures by temporarily blocking
    requests to failing services.
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
        """Check if request can be executed"""
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
        """Record successful execution"""
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
        """Record failed execution"""
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
    Service Discovery and Registry
    
    Manages service registration, discovery, and health checking.
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceNode] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._health_check_task: Optional[asyncio.Task] = None
    
    def register(self, node: ServiceNode) -> None:
        """Register a service node"""
        self.services[node.node_id] = node
        self.circuit_breakers[node.node_id] = CircuitBreaker()
        node.status = ServiceStatus.HEALTHY
        node.last_heartbeat = datetime.now().isoformat()
        logger.info(f"Registered service node: {node.node_id}")
    
    def deregister(self, node_id: str) -> None:
        """Deregister a service node"""
        if node_id in self.services:
            del self.services[node_id]
            del self.circuit_breakers[node_id]
            logger.info(f"Deregistered service node: {node_id}")
    
    def get_healthy_nodes(self) -> List[ServiceNode]:
        """Get all healthy nodes"""
        return [
            node for node in self.services.values()
            if node.is_available() and 
            self.circuit_breakers[node.node_id].can_execute()
        ]
    
    def get_node_by_load(self) -> Optional[ServiceNode]:
        """Get node with lowest load (load balancing)"""
        healthy = self.get_healthy_nodes()
        if not healthy:
            return None
        return min(healthy, key=lambda n: n.load)
    
    def update_heartbeat(self, node_id: str) -> None:
        """Update node heartbeat"""
        if node_id in self.services:
            self.services[node_id].last_heartbeat = datetime.now().isoformat()
    
    async def start_health_checks(self, interval: int = 5) -> None:
        """Start periodic health checks"""
        async def check_health():
            while True:
                await asyncio.sleep(interval)
                await self._check_all_nodes()
        
        self._health_check_task = asyncio.create_task(check_health())
    
    async def _check_all_nodes(self) -> None:
        """Check health of all nodes"""
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


class EventBus:
    """
    Event-Driven Communication Bus
    
    Enables async communication between microservices.
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type"""
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(handler)
    
    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str = "unknown"
    ) -> None:
        """Publish an event"""
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
        """Get recent events"""
        events = self.event_history
        if event_type:
            events = [e for e in events if e['event_type'] == event_type]
        return events[-limit:]


class DistributedVCAI:
    """
    Distributed Version Control AI Core
    
    Main orchestrator for the distributed system with:
    - Service discovery and load balancing
    - Event-driven architecture
    - Fault tolerance
    - Real-time learning integration
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
        """Setup internal event handlers"""
        self.event_bus.subscribe(
            self.EVENT_LEARNING_COMPLETED,
            self._on_learning_completed
        )
        self.event_bus.subscribe(
            self.EVENT_VERSION_CREATED,
            self._on_version_created
        )
    
    async def _on_learning_completed(self, event: Dict) -> None:
        """Handle learning completion event"""
        logger.info(f"Learning completed: {event['data']}")
        
        # Trigger version comparison if needed
        if self.version_engine:
            await self.version_engine.analyze_improvements()
    
    async def _on_version_created(self, event: Dict) -> None:
        """Handle new version creation"""
        logger.info(f"New version created: {event['data']}")
    
    async def start(self) -> None:
        """Start the distributed system"""
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
        """Stop the distributed system"""
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
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0
    
    def get_availability(self) -> float:
        """Calculate system availability"""
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 1.0
        return self.successful_requests / total
    
    async def execute_with_circuit_breaker(
        self,
        node_id: str,
        operation: Callable
    ) -> Any:
        """Execute operation with circuit breaker protection"""
        if node_id not in self.registry.circuit_breakers:
            raise ValueError(f"Unknown node: {node_id}")
        
        cb = self.registry.circuit_breakers[node_id]
        
        if not cb.can_execute():
            raise Exception(f"Circuit breaker OPEN for {node_id}")
        
        try:
            result = await operation() if asyncio.iscoroutinefunction(operation) else operation()
            cb.record_success()
            self.successful_requests += 1
            return result
        except Exception as e:
            cb.record_failure()
            self.failed_requests += 1
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
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
