"""
Domain Events Integration

Provides integration between DDD domain events and the CQRS event system.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class IntegrationEvent:
    """
    Integration event for cross-bounded context communication.
    
    Unlike domain events (internal to a bounded context),
    integration events are used for communication between contexts.
    """
    event_id: str
    event_type: str
    source_context: str
    target_context: Optional[str]
    payload: Dict[str, Any]
    correlation_id: Optional[str]
    timestamp: datetime
    
    @classmethod
    def from_domain_event(
        cls,
        domain_event,
        source_context: str,
        target_context: Optional[str] = None
    ) -> 'IntegrationEvent':
        """Create integration event from domain event."""
        return cls(
            event_id=str(uuid4()),
            event_type=domain_event.event_type,
            source_context=source_context,
            target_context=target_context,
            payload=domain_event.__dict__,
            correlation_id=getattr(domain_event, 'correlation_id', None),
            timestamp=datetime.now(timezone.utc),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_context": self.source_context,
            "target_context": self.target_context,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
        }


class IntegrationEventBus:
    """
    Bus for publishing and subscribing to integration events.
    
    Handles cross-context event communication.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._context_subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    def subscribe_context(self, context_name: str, handler: Callable):
        """Subscribe to all events from a specific context."""
        if context_name not in self._context_subscribers:
            self._context_subscribers[context_name] = []
        self._context_subscribers[context_name].append(handler)
    
    async def publish(self, event: IntegrationEvent):
        """Publish an integration event."""
        # Notify type-specific subscribers
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Integration event handler error: {e}")
        
        # Notify context subscribers
        context_handlers = self._context_subscribers.get(event.source_context, [])
        for handler in context_handlers:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Context event handler error: {e}")


# Event type constants for cross-context communication
class IntegrationEventTypes:
    """Standard integration event types."""
    
    # Code Analysis Context
    ANALYSIS_REQUESTED = "code_analysis.analysis_requested"
    ANALYSIS_COMPLETED = "code_analysis.analysis_completed"
    ANALYSIS_FAILED = "code_analysis.analysis_failed"
    
    # Version Management Context
    EXPERIMENT_STARTED = "version_management.experiment_started"
    EXPERIMENT_COMPLETED = "version_management.experiment_completed"
    VERSION_PROMOTED = "version_management.version_promoted"
    VERSION_DEMOTED = "version_management.version_demoted"
    
    # User Auth Context
    USER_AUTHENTICATED = "user_auth.user_authenticated"
    USER_AUTHORIZED = "user_auth.user_authorized"
    
    # Provider Management Context
    PROVIDER_HEALTH_CHANGED = "provider_management.health_changed"
    QUOTA_EXCEEDED = "provider_management.quota_exceeded"
    
    # Audit Context
    AUDIT_ENTRY_CREATED = "audit.entry_created"
    INTEGRITY_VIOLATION = "audit.integrity_violation"


# Global integration event bus
_integration_bus: Optional[IntegrationEventBus] = None


def get_integration_bus() -> IntegrationEventBus:
    """Get or create the global integration event bus."""
    global _integration_bus
    if _integration_bus is None:
        _integration_bus = IntegrationEventBus()
    return _integration_bus
