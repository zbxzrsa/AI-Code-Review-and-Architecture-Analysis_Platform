"""
Event Publisher for Lifecycle Controller

Publishes lifecycle events to various backends:
- Redis Pub/Sub for real-time notifications
- Kafka for event sourcing
- Webhooks for external integrations

Events enable:
- Real-time dashboards
- Audit logging
- External system integration
- Alerting pipelines
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of lifecycle events"""
    # Version state changes
    EXPERIMENT_REGISTERED = "experiment_registered"
    SHADOW_STARTED = "shadow_started"
    PROMOTION_STARTED = "promotion_started"
    PROMOTION_COMPLETED = "promotion_completed"
    PROMOTION_FAILED = "promotion_failed"
    
    # Gray-scale phases
    GRAY_SCALE_ADVANCED = "gray_scale_advanced"
    GRAY_SCALE_STABLE = "gray_scale_stable"
    
    # Demotions
    DEMOTION_TRIGGERED = "demotion_triggered"
    QUARANTINE_ENTERED = "quarantine_entered"
    
    # Recovery
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_ATTEMPTED = "recovery_attempted"
    RECOVERY_SUCCEEDED = "recovery_succeeded"
    RECOVERY_FAILED = "recovery_failed"
    RECOVERY_ABANDONED = "recovery_abandoned"
    
    # Rollbacks
    ROLLBACK_TRIGGERED = "rollback_triggered"
    ROLLBACK_COMPLETED = "rollback_completed"
    
    # Evaluations
    EVALUATION_COMPLETED = "evaluation_completed"
    GOLD_SET_PASSED = "gold_set_passed"
    GOLD_SET_FAILED = "gold_set_failed"
    
    # Cycle health
    CYCLE_HEALTH_DEGRADED = "cycle_health_degraded"
    CYCLE_HEALTH_RESTORED = "cycle_health_restored"
    
    # Three-Version Spiral Evolution Events
    EVOLUTION_CYCLE_STARTED = "evolution_cycle_started"
    EVOLUTION_CYCLE_STOPPED = "evolution_cycle_stopped"
    EVOLUTION_CYCLE_COMPLETED = "evolution_cycle_completed"
    
    # V1 Experimentation
    V1_EXPERIMENT_STARTED = "v1_experiment_started"
    V1_EXPERIMENT_COMPLETED = "v1_experiment_completed"
    V1_ERROR_REPORTED = "v1_error_reported"
    
    # V2 Error Remediation
    V2_FIX_GENERATED = "v2_fix_generated"
    V2_FIX_APPLIED = "v2_fix_applied"
    V2_FIX_VERIFIED = "v2_fix_verified"
    V2_FIX_FAILED = "v2_fix_failed"
    
    # V3 Quarantine
    V3_TECHNOLOGY_QUARANTINED = "v3_technology_quarantined"
    V3_PERMANENT_EXCLUSION = "v3_permanent_exclusion"
    V3_TEMPORARY_EXCLUSION = "v3_temporary_exclusion"
    V3_REEVAL_REQUESTED = "v3_reeval_requested"
    V3_REEVAL_APPROVED = "v3_reeval_approved"
    
    # Dual-AI Events
    AI_STATUS_CHANGED = "ai_status_changed"
    AI_REQUEST_ROUTED = "ai_request_routed"
    
    # Cross-Version Communication
    CROSS_VERSION_HANDOFF = "cross_version_handoff"
    TECHNOLOGY_KNOWLEDGE_TRANSFERRED = "technology_knowledge_transferred"


@dataclass
class LifecycleEvent:
    """A lifecycle event"""
    event_id: str
    event_type: EventType
    timestamp: str
    version_id: str
    source: str = "lifecycle-controller"
    
    # Event details
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    reason: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    
    # Metadata
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "version_id": self.version_id,
            "source": self.source,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "reason": self.reason,
            "metrics": self.metrics,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class EventBackend(ABC):
    """Abstract base class for event backends"""
    
    @abstractmethod
    async def publish(self, event: LifecycleEvent) -> bool:
        pass
    
    @abstractmethod
    async def connect(self):
        pass
    
    @abstractmethod
    async def disconnect(self):
        pass


class RedisEventBackend(EventBackend):
    """Redis Pub/Sub backend"""
    
    def __init__(
        self,
        redis_url: str = "redis://redis.platform-control-plane.svc:6379",
        channel_prefix: str = "lifecycle:"
    ):
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self._client = None
    
    async def connect(self):
        try:
            import redis.asyncio as aioredis
            self._client = await aioredis.from_url(self.redis_url)
            logger.info("Connected to Redis for event publishing")
        except ImportError:
            logger.warning("Redis not available - events will not be published to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    async def disconnect(self):
        if self._client:
            await self._client.close()
    
    async def publish(self, event: LifecycleEvent) -> bool:
        if not self._client:
            return False
        
        try:
            channel = f"{self.channel_prefix}{event.event_type.value}"
            await self._client.publish(channel, event.to_json())
            
            # Also publish to version-specific channel
            version_channel = f"{self.channel_prefix}version:{event.version_id}"
            await self._client.publish(version_channel, event.to_json())
            
            return True
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")
            return False


class WebhookEventBackend(EventBackend):
    """Webhook backend for external integrations"""
    
    def __init__(
        self,
        webhook_urls: List[str] = None,
        headers: Dict[str, str] = None,
        timeout: float = 10.0
    ):
        self.webhook_urls = webhook_urls or []
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def connect(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        logger.info(f"Webhook backend ready with {len(self.webhook_urls)} endpoints")
    
    async def disconnect(self):
        if self._client:
            await self._client.aclose()
    
    async def publish(self, event: LifecycleEvent) -> bool:
        if not self._client or not self.webhook_urls:
            return False
        
        success = True
        for url in self.webhook_urls:
            try:
                response = await self._client.post(
                    url,
                    json=event.to_dict(),
                    headers=self.headers
                )
                if response.status_code >= 400:
                    logger.warning(f"Webhook {url} returned {response.status_code}")
                    success = False
            except Exception as e:
                logger.error(f"Failed to publish to webhook {url}: {e}")
                success = False
        
        return success
    
    def add_webhook(self, url: str):
        """Add a webhook URL"""
        if url not in self.webhook_urls:
            self.webhook_urls.append(url)
    
    def remove_webhook(self, url: str):
        """Remove a webhook URL"""
        if url in self.webhook_urls:
            self.webhook_urls.remove(url)


class InMemoryEventBackend(EventBackend):
    """In-memory backend for testing and local development"""
    
    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: List[LifecycleEvent] = []
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def connect(self):
        logger.info("In-memory event backend ready")
    
    async def disconnect(self):
        pass
    
    async def publish(self, event: LifecycleEvent) -> bool:
        self.events.append(event)
        
        # Keep only last N events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Notify subscribers
        event_type = event.event_type.value
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")
        
        return True
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        version_id: Optional[str] = None,
        limit: int = 100
    ) -> List[LifecycleEvent]:
        """Get events with optional filtering"""
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type.value == event_type]
        
        if version_id:
            events = [e for e in events if e.version_id == version_id]
        
        return events[-limit:]


class EventPublisher:
    """
    Main event publisher that dispatches to multiple backends.
    """
    
    def __init__(self):
        self.backends: List[EventBackend] = []
        self._event_counter = 0
    
    def add_backend(self, backend: EventBackend):
        """Add an event backend"""
        self.backends.append(backend)
    
    async def start(self):
        """Start all backends"""
        for backend in self.backends:
            await backend.connect()
        logger.info(f"Event publisher started with {len(self.backends)} backends")
    
    async def stop(self):
        """Stop all backends"""
        for backend in self.backends:
            await backend.disconnect()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        self._event_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"evt-{timestamp}-{self._event_counter:06d}"
    
    async def publish(
        self,
        event_type: EventType,
        version_id: str,
        from_state: Optional[str] = None,
        to_state: Optional[str] = None,
        reason: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> LifecycleEvent:
        """Publish an event to all backends"""
        event = LifecycleEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            version_id=version_id,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            metrics=metrics,
            correlation_id=correlation_id,
            user_id=user_id,
        )
        
        # Publish to all backends
        for backend in self.backends:
            try:
                await backend.publish(event)
            except Exception as e:
                logger.error(f"Backend publish error: {e}")
        
        logger.info(f"Published event: {event.event_type.value} for {version_id}")
        return event
    
    # ==================== Convenience Methods ====================
    
    async def publish_promotion_started(
        self,
        version_id: str,
        from_state: str,
        metrics: Dict[str, Any]
    ):
        """Publish promotion started event"""
        await self.publish(
            event_type=EventType.PROMOTION_STARTED,
            version_id=version_id,
            from_state=from_state,
            to_state="gray_1_percent",
            metrics=metrics
        )
    
    async def publish_promotion_completed(
        self,
        version_id: str,
        duration_hours: float
    ):
        """Publish promotion completed event"""
        await self.publish(
            event_type=EventType.PROMOTION_COMPLETED,
            version_id=version_id,
            to_state="stable",
            metrics={"duration_hours": duration_hours}
        )
    
    async def publish_demotion(
        self,
        version_id: str,
        from_state: str,
        reason: str,
        metrics: Dict[str, Any]
    ):
        """Publish demotion event"""
        await self.publish(
            event_type=EventType.DEMOTION_TRIGGERED,
            version_id=version_id,
            from_state=from_state,
            to_state="quarantine",
            reason=reason,
            metrics=metrics
        )
    
    async def publish_recovery_success(
        self,
        version_id: str,
        attempts: int,
        score: float
    ):
        """Publish recovery success event"""
        await self.publish(
            event_type=EventType.RECOVERY_SUCCEEDED,
            version_id=version_id,
            from_state="quarantine",
            to_state="shadow",
            metrics={"attempts": attempts, "score": score}
        )
    
    async def publish_rollback(
        self,
        version_id: str,
        from_phase: str,
        reason: str
    ):
        """Publish rollback event"""
        await self.publish(
            event_type=EventType.ROLLBACK_TRIGGERED,
            version_id=version_id,
            from_state=from_phase,
            reason=reason
        )
