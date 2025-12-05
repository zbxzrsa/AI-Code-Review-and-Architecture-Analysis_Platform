"""
Event Bus - Event-driven architecture for service communication.

Enables Version Control AI to listen to experiment.completed events
and trigger automatic evaluation and promotion decisions.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Any, Optional
from enum import Enum
from uuid import uuid4
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types in the system."""
    EXPERIMENT_CREATED = "experiment.created"
    EXPERIMENT_STARTED = "experiment.started"
    EXPERIMENT_COMPLETED = "experiment.completed"
    EXPERIMENT_FAILED = "experiment.failed"
    PROMOTION_REQUESTED = "promotion.requested"
    PROMOTION_APPROVED = "promotion.approved"
    PROMOTION_REJECTED = "promotion.rejected"
    QUARANTINE_REQUESTED = "quarantine.requested"
    CODE_REVIEW_REQUESTED = "code_review.requested"
    CODE_REVIEW_COMPLETED = "code_review.completed"


@dataclass
class Event:
    """Base event class."""
    id: str
    type: EventType
    timestamp: datetime
    source: str  # Service that emitted the event
    data: Dict[str, Any]
    correlation_id: Optional[str] = None  # For tracing related events

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "correlation_id": self.correlation_id,
        }


class EventHandler:
    """Handler for specific event types."""

    def __init__(self, event_type: EventType, handler_func: Callable):
        """Initialize event handler."""
        self.event_type = event_type
        self.handler_func = handler_func
        self.id = str(uuid4())

    async def handle(self, event: Event) -> None:
        """Handle event."""
        try:
            if asyncio.iscoroutinefunction(self.handler_func):
                await self.handler_func(event)
            else:
                self.handler_func(event)
        except Exception as e:
            logger.error(
                "Event handler failed",
                handler_id=self.id,
                event_type=event.type.value,
                error=str(e),
            )
            raise


class EventBus:
    """Central event bus for event-driven architecture."""

    def __init__(self):
        """Initialize event bus."""
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.max_history = 10000

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable,
    ) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async function to handle the event

        Returns:
            Handler ID for later unsubscription
        """
        event_handler = EventHandler(event_type, handler)
        self.handlers[event_type].append(event_handler)

        logger.info(
            "Handler subscribed",
            handler_id=event_handler.id,
            event_type=event_type.value,
        )

        return event_handler.id

    def unsubscribe(self, event_type: EventType, handler_id: str) -> bool:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event
            handler_id: ID of handler to remove

        Returns:
            True if handler was removed, False if not found
        """
        handlers = self.handlers[event_type]
        for i, handler in enumerate(handlers):
            if handler.id == handler_id:
                handlers.pop(i)
                logger.info(
                    "Handler unsubscribed",
                    handler_id=handler_id,
                    event_type=event_type.value,
                )
                return True
        return False

    async def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Emit an event.

        Args:
            event_type: Type of event
            data: Event data
            source: Source service
            correlation_id: Optional correlation ID for tracing

        Returns:
            Event ID
        """
        event = Event(
            id=str(uuid4()),
            type=event_type,
            timestamp=datetime.now(timezone.utc),
            source=source,
            data=data,
            correlation_id=correlation_id or str(uuid4()),
        )

        logger.info(
            "Event emitted",
            event_id=event.id,
            event_type=event_type.value,
            source=source,
            correlation_id=event.correlation_id,
        )

        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)

        # Call all handlers for this event type
        handlers = self.handlers.get(event_type, [])
        if handlers:
            tasks = [handler.handle(event) for handler in handlers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "Handler execution failed",
                        handler_id=handlers[i].id,
                        error=str(result),
                    )

        return event.id

    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """
        Get event history with optional filtering.

        Args:
            event_type: Filter by event type
            source: Filter by source
            correlation_id: Filter by correlation ID
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        results = self.event_history

        if event_type:
            results = [e for e in results if e.type == event_type]

        if source:
            results = [e for e in results if e.source == source]

        if correlation_id:
            results = [e for e in results if e.correlation_id == correlation_id]

        # Return most recent events
        return results[-limit:]

    def get_handler_count(self, event_type: Optional[EventType] = None) -> int:
        """Get number of handlers for event type(s)."""
        if event_type:
            return len(self.handlers.get(event_type, []))
        return sum(len(handlers) for handlers in self.handlers.values())

    def clear_history(self) -> int:
        """Clear event history and return count of cleared events."""
        count = len(self.event_history)
        self.event_history.clear()
        return count


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


async def emit_experiment_completed(
    experiment_id: str,
    metrics: Dict[str, Any],
    correlation_id: Optional[str] = None,
) -> str:
    """Emit experiment.completed event."""
    bus = get_event_bus()
    return await bus.emit(
        event_type=EventType.EXPERIMENT_COMPLETED,
        data={
            "experiment_id": experiment_id,
            "metrics": metrics,
        },
        source="v1-experimentation",
        correlation_id=correlation_id,
    )


async def emit_promotion_approved(
    experiment_id: str,
    report_id: str,
    correlation_id: Optional[str] = None,
) -> str:
    """Emit promotion.approved event."""
    bus = get_event_bus()
    return await bus.emit(
        event_type=EventType.PROMOTION_APPROVED,
        data={
            "experiment_id": experiment_id,
            "report_id": report_id,
        },
        source="version-control-ai",
        correlation_id=correlation_id,
    )


async def emit_code_review_completed(
    review_id: str,
    result_id: str,
    correlation_id: Optional[str] = None,
) -> str:
    """Emit code_review.completed event."""
    bus = get_event_bus()
    return await bus.emit(
        event_type=EventType.CODE_REVIEW_COMPLETED,
        data={
            "review_id": review_id,
            "result_id": result_id,
        },
        source="code-review-ai",
        correlation_id=correlation_id,
    )
