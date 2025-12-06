"""
Bidirectional Communication Protocol

Project-AI communication protocol with:
- Real-time event streaming
- Request-response patterns
- Automated testing pipeline
- Conflict resolution
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json
import hashlib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Protocol message types"""
    # Request-Response
    REQUEST = "request"
    RESPONSE = "response"
    
    # Events
    EVENT = "event"
    NOTIFICATION = "notification"
    
    # Control
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    ERROR = "error"
    
    # Sync
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"


class ProtocolVersion(Enum):
    """Protocol versions"""
    V1 = "1.0"
    V2 = "2.0"
    CURRENT = "2.0"


class ChannelStatus(Enum):
    """Communication channel status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class ProtocolMessage:
    """A protocol message"""
    message_id: str
    message_type: MessageType
    timestamp: str
    
    # Content
    action: str
    payload: Dict[str, Any]
    
    # Routing
    source: str
    target: str
    
    # Metadata
    protocol_version: str = ProtocolVersion.CURRENT.value
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    # Validation
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "action": self.action,
            "payload": self.payload,
            "source": self.source,
            "target": self.target,
            "protocol_version": self.protocol_version,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolMessage":
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            timestamp=data["timestamp"],
            action=data["action"],
            payload=data["payload"],
            source=data["source"],
            target=data["target"],
            protocol_version=data.get("protocol_version", ProtocolVersion.CURRENT.value),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            checksum=data.get("checksum")
        )
    
    def compute_checksum(self) -> str:
        """Compute message checksum"""
        content = f"{self.action}:{json.dumps(self.payload, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def verify_checksum(self) -> bool:
        """Verify message integrity"""
        if not self.checksum:
            return True
        return self.checksum == self.compute_checksum()


@dataclass
class TestResult:
    """Result of an automated test"""
    test_id: str
    test_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration_ms: float
    
    message: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """A test suite for iteration verification"""
    suite_id: str
    name: str
    tests: List[TestResult]
    
    started_at: str
    completed_at: Optional[str] = None
    
    @property
    def total(self) -> int:
        return len(self.tests)
    
    @property
    def passed(self) -> int:
        return len([t for t in self.tests if t.status == "passed"])
    
    @property
    def failed(self) -> int:
        return len([t for t in self.tests if t.status == "failed"])
    
    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0


class MessageHandler(ABC):
    """Abstract message handler"""
    
    @abstractmethod
    async def handle(self, message: ProtocolMessage) -> Optional[ProtocolMessage]:
        """Handle a message and optionally return a response"""
        pass


class BidirectionalProtocol:
    """
    Bidirectional Communication Protocol
    
    Features:
    - Async message passing
    - Request-response patterns
    - Event streaming
    - Message validation
    - Retry logic
    """
    
    def __init__(
        self,
        node_id: str,
        max_message_queue: int = 1000,
        heartbeat_interval_seconds: int = 30,
        message_timeout_seconds: int = 60
    ):
        self.node_id = node_id
        self.max_queue = max_message_queue
        self.heartbeat_interval = heartbeat_interval_seconds
        self.message_timeout = message_timeout_seconds
        
        self.status = ChannelStatus.DISCONNECTED
        self.connected_peers: Dict[str, datetime] = {}
        
        # Message queues
        self.outbound_queue: asyncio.Queue = asyncio.Queue(maxsize=max_message_queue)
        self.inbound_queue: asyncio.Queue = asyncio.Queue(maxsize=max_message_queue)
        
        # Handlers
        self.handlers: Dict[str, MessageHandler] = {}
        self.action_handlers: Dict[str, Callable] = {}
        
        # Pending requests
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Message counter
        self._message_counter = 0
        
        # Tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._processor_task: Optional[asyncio.Task] = None
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        self._message_counter += 1
        return f"{self.node_id}:{self._message_counter}:{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    def start(self) -> None:
        """Start the protocol"""
        self.status = ChannelStatus.CONNECTING
        
        # Start message processor
        self._processor_task = asyncio.create_task(self._process_messages())
        
        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        self.status = ChannelStatus.CONNECTED
        logger.info(f"Protocol started for node: {self.node_id}")
    
    def stop(self) -> None:
        """Stop the protocol"""
        self.status = ChannelStatus.DISCONNECTED
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        if self._processor_task:
            self._processor_task.cancel()
        
        logger.info(f"Protocol stopped for node: {self.node_id}")
    
    def register_handler(self, action: str, handler: Callable) -> None:
        """Register an action handler"""
        self.action_handlers[action] = handler
        logger.info(f"Registered handler for action: {action}")
    
    async def send(
        self,
        target: str,
        action: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.EVENT
    ) -> None:
        """Send a message (fire-and-forget)"""
        message = ProtocolMessage(
            message_id=self._generate_message_id(),
            message_type=message_type,
            timestamp=datetime.now().isoformat(),
            action=action,
            payload=payload,
            source=self.node_id,
            target=target
        )
        message.checksum = message.compute_checksum()
        
        await self.outbound_queue.put(message)
    
    async def request(
        self,
        target: str,
        action: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Send a request and wait for response"""
        timeout = timeout or self.message_timeout
        
        message = ProtocolMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.REQUEST,
            timestamp=datetime.now().isoformat(),
            action=action,
            payload=payload,
            source=self.node_id,
            target=target
        )
        message.checksum = message.compute_checksum()
        
        # Create future for response
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending_requests[message.message_id] = future
        
        # Send message
        await self.outbound_queue.put(message)
        
        try:
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response.payload
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {action}")
            raise
        finally:
            self.pending_requests.pop(message.message_id, None)
    
    async def respond(
        self,
        request: ProtocolMessage,
        payload: Dict[str, Any],
        success: bool = True
    ) -> None:
        """Send a response to a request"""
        response = ProtocolMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.RESPONSE,
            timestamp=datetime.now().isoformat(),
            action=f"{request.action}_response",
            payload={
                "success": success,
                **payload
            },
            source=self.node_id,
            target=request.source,
            correlation_id=request.message_id
        )
        response.checksum = response.compute_checksum()
        
        await self.outbound_queue.put(response)
    
    async def _handle_response_message(self, message: ProtocolMessage) -> bool:
        """Handle response message type. Returns True if handled."""
        if message.message_type == MessageType.RESPONSE:
            if message.correlation_id in self.pending_requests:
                self.pending_requests[message.correlation_id].set_result(message)
            return True
        return False
    
    async def _execute_action_handler(self, message: ProtocolMessage, handler: Callable) -> Any:
        """Execute action handler (async or sync)."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(message)
        return handler(message)
    
    async def _handle_action_message(self, message: ProtocolMessage) -> None:
        """Handle action message type."""
        if message.action not in self.action_handlers:
            return
            
        handler = self.action_handlers[message.action]
        try:
            result = await self._execute_action_handler(message, handler)
            
            # Send response if it was a request
            if message.message_type == MessageType.REQUEST:
                await self.respond(message, result or {})
        except Exception as e:
            logger.error(f"Handler error: {e}")
            if message.message_type == MessageType.REQUEST:
                await self.respond(
                    message,
                    {"error": str(e)},
                    success=False
                )
    
    async def _process_messages(self) -> None:
        """Process inbound messages"""
        while self.status == ChannelStatus.CONNECTED:
            try:
                message = await asyncio.wait_for(
                    self.inbound_queue.get(),
                    timeout=1.0
                )
                
                # Verify message
                if not message.verify_checksum():
                    logger.warning(f"Invalid checksum: {message.message_id}")
                    continue
                
                # Handle response messages
                if await self._handle_response_message(message):
                    continue
                
                # Handle action messages
                await self._handle_action_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Message processing error: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats"""
        while self.status == ChannelStatus.CONNECTED:
            await asyncio.sleep(self.heartbeat_interval)
            
            # Send heartbeat to all peers
            for peer_id in self.connected_peers:
                await self.send(
                    peer_id,
                    "heartbeat",
                    {"timestamp": datetime.now().isoformat()},
                    MessageType.HEARTBEAT
                )
    
    def receive_message(self, message: ProtocolMessage) -> None:
        """Receive an inbound message (called externally)"""
        try:
            self.inbound_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("Inbound queue full, dropping message")
    
    def get_status(self) -> Dict[str, Any]:
        """Get protocol status"""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "connected_peers": len(self.connected_peers),
            "outbound_queue_size": self.outbound_queue.qsize(),
            "inbound_queue_size": self.inbound_queue.qsize(),
            "pending_requests": len(self.pending_requests),
            "registered_handlers": list(self.action_handlers.keys())
        }


class AutomatedTestingPipeline:
    """
    Automated Testing Pipeline
    
    Verifies each iteration through comprehensive testing.
    """
    
    def __init__(self):
        self.test_suites: List[TestSuite] = []
        self.test_definitions: Dict[str, Callable] = {}
        
        self._register_default_tests()
    
    def _register_default_tests(self) -> None:
        """Register default test cases"""
        self.register_test("health_check", self._test_health_check)
        self.register_test("api_response", self._test_api_response)
        self.register_test("model_accuracy", self._test_model_accuracy)
        self.register_test("latency_threshold", self._test_latency_threshold)
        self.register_test("error_rate", self._test_error_rate)
        self.register_test("rollback_capability", self._test_rollback_capability)
        self.register_test("merge_success", self._test_merge_success)
    
    def register_test(self, name: str, test_fn: Callable) -> None:
        """Register a test function"""
        self.test_definitions[name] = test_fn
    
    def _test_health_check(self, context: Dict[str, Any]) -> TestResult:
        """Test system health"""
        start = datetime.now()
        
        is_healthy = context.get("is_healthy", True)
        
        return TestResult(
            test_id=f"test_health_{datetime.now().strftime('%H%M%S')}",
            test_name="health_check",
            status="passed" if is_healthy else "failed",
            duration_ms=(datetime.now() - start).total_seconds() * 1000,
            message="System is healthy" if is_healthy else "Health check failed"
        )
    
    async def _test_api_response(self, context: Dict[str, Any]) -> TestResult:
        """Test API response"""
        start = datetime.now()
        
        # Simulate API test
        await asyncio.sleep(0.1)
        api_available = context.get("api_available", True)
        
        return TestResult(
            test_id=f"test_api_{datetime.now().strftime('%H%M%S')}",
            test_name="api_response",
            status="passed" if api_available else "failed",
            duration_ms=(datetime.now() - start).total_seconds() * 1000
        )
    
    def _test_model_accuracy(self, context: Dict[str, Any]) -> TestResult:
        """Test model accuracy threshold"""
        start = datetime.now()
        
        accuracy = context.get("accuracy", 0.85)
        threshold = context.get("accuracy_threshold", 0.85)
        
        passed = accuracy >= threshold
        
        return TestResult(
            test_id=f"test_accuracy_{datetime.now().strftime('%H%M%S')}",
            test_name="model_accuracy",
            status="passed" if passed else "failed",
            duration_ms=(datetime.now() - start).total_seconds() * 1000,
            message=f"Accuracy: {accuracy:.2%}, Threshold: {threshold:.2%}",
            details={"accuracy": accuracy, "threshold": threshold}
        )
    
    def _test_latency_threshold(self, context: Dict[str, Any]) -> TestResult:
        """Test latency threshold"""
        start = datetime.now()
        
        latency = context.get("latency_p95_ms", 200)
        threshold = context.get("latency_threshold", 200)
        
        passed = latency <= threshold
        
        return TestResult(
            test_id=f"test_latency_{datetime.now().strftime('%H%M%S')}",
            test_name="latency_threshold",
            status="passed" if passed else "failed",
            duration_ms=(datetime.now() - start).total_seconds() * 1000,
            message=f"P95 Latency: {latency}ms, Threshold: {threshold}ms"
        )
    
    def _test_error_rate(self, context: Dict[str, Any]) -> TestResult:
        """Test error rate threshold"""
        start = datetime.now()
        
        error_rate = context.get("error_rate", 0.02)
        threshold = context.get("error_threshold", 0.05)
        
        passed = error_rate <= threshold
        
        return TestResult(
            test_id=f"test_error_{datetime.now().strftime('%H%M%S')}",
            test_name="error_rate",
            status="passed" if passed else "failed",
            duration_ms=(datetime.now() - start).total_seconds() * 1000,
            message=f"Error rate: {error_rate:.2%}, Threshold: {threshold:.2%}"
        )
    
    def _test_rollback_capability(self, context: Dict[str, Any]) -> TestResult:
        """Test rollback capability"""
        start = datetime.now()
        
        snapshots_available = context.get("snapshots_count", 0) > 0
        
        return TestResult(
            test_id=f"test_rollback_{datetime.now().strftime('%H%M%S')}",
            test_name="rollback_capability",
            status="passed" if snapshots_available else "failed",
            duration_ms=(datetime.now() - start).total_seconds() * 1000,
            message="Rollback snapshots available" if snapshots_available else "No snapshots"
        )
    
    def _test_merge_success(self, context: Dict[str, Any]) -> TestResult:
        """Test merge success rate"""
        start = datetime.now()
        
        success_rate = context.get("merge_success_rate", 0.95)
        threshold = 0.95
        
        passed = success_rate >= threshold
        
        return TestResult(
            test_id=f"test_merge_{datetime.now().strftime('%H%M%S')}",
            test_name="merge_success",
            status="passed" if passed else "failed",
            duration_ms=(datetime.now() - start).total_seconds() * 1000,
            message=f"Merge success rate: {success_rate:.2%}, Target: {threshold:.2%}"
        )
    
    async def run_suite(
        self,
        suite_name: str,
        context: Dict[str, Any],
        test_names: Optional[List[str]] = None
    ) -> TestSuite:
        """Run a test suite"""
        suite = TestSuite(
            suite_id=f"suite_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=suite_name,
            tests=[],
            started_at=datetime.now().isoformat()
        )
        
        tests_to_run = test_names or list(self.test_definitions.keys())
        
        for test_name in tests_to_run:
            if test_name in self.test_definitions:
                try:
                    result = await self.test_definitions[test_name](context)
                    suite.tests.append(result)
                except Exception as e:
                    suite.tests.append(TestResult(
                        test_id=f"test_{test_name}_error",
                        test_name=test_name,
                        status="error",
                        duration_ms=0,
                        error=str(e)
                    ))
        
        suite.completed_at = datetime.now().isoformat()
        self.test_suites.append(suite)
        
        logger.info(
            f"Test suite completed: {suite.name} - "
            f"{suite.passed}/{suite.total} passed ({suite.success_rate:.0%})"
        )
        
        return suite
    
    async def verify_iteration(
        self,
        iteration_context: Dict[str, Any]
    ) -> Tuple[bool, TestSuite]:
        """Verify an iteration meets all criteria"""
        suite = await self.run_suite(
            f"iteration_verification_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            iteration_context
        )
        
        # Iteration passes if all critical tests pass
        critical_tests = ["health_check", "model_accuracy", "error_rate"]
        critical_passed = all(
            any(t.test_name == name and t.status == "passed" for t in suite.tests)
            for name in critical_tests
        )
        
        return critical_passed, suite
    
    def get_test_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent test history"""
        recent = sorted(
            self.test_suites,
            key=lambda s: s.started_at,
            reverse=True
        )[:limit]
        
        return [
            {
                "suite_id": s.suite_id,
                "name": s.name,
                "started_at": s.started_at,
                "total": s.total,
                "passed": s.passed,
                "failed": s.failed,
                "success_rate": s.success_rate
            }
            for s in recent
        ]


# Import Tuple for type hints
from typing import Tuple
