"""
Dead Letter Queue Implementation

Implements:
- Failed message handling
- Automatic retry with exponential backoff
- Message inspection and replay
- Alerting on DLQ growth
"""

import asyncio
import logging
import json
import hashlib
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

# Redis key constants
DLQ_MESSAGES_KEY = "dlq:messages"


class MessageState(str, Enum):
    """Message processing state."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"
    RETRYING = "retrying"


@dataclass
class Message:
    """Event bus message."""
    id: str
    topic: str
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempts: int = 0
    max_attempts: int = 3
    state: MessageState = MessageState.PENDING
    last_error: Optional[str] = None
    last_attempt_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "state": self.state.value,
            "last_error": self.last_error,
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "metadata": self.metadata,
        }


@dataclass
class DLQStats:
    """Dead letter queue statistics."""
    total_messages: int = 0
    messages_by_topic: Dict[str, int] = field(default_factory=dict)
    oldest_message_age_hours: float = 0
    average_attempts: float = 0
    error_distribution: Dict[str, int] = field(default_factory=dict)


class DeadLetterQueue:
    """
    Dead letter queue for failed messages.
    
    Features:
    - Message storage with metadata
    - Automatic retry with backoff
    - Manual replay capability
    - Statistics and alerting
    """
    
    def __init__(
        self,
        redis_client = None,
        max_dlq_size: int = 10000,
        base_retry_delay: int = 60,  # seconds
        max_retry_delay: int = 3600,  # 1 hour
        alert_threshold: int = 100,
    ):
        self.redis = redis_client
        self.max_dlq_size = max_dlq_size
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        self.alert_threshold = alert_threshold
        
        # In-memory storage
        self._dlq: Dict[str, Message] = {}
        self._retry_queue: List[Message] = []
        self._alert_callbacks: List[Callable[[DLQStats], Awaitable[None]]] = []
    
    async def add_message(
        self,
        message: Message,
        error: str,
    ):
        """Add failed message to DLQ."""
        message.state = MessageState.DEAD_LETTERED
        message.last_error = error
        message.last_attempt_at = datetime.now(timezone.utc)
        
        # Store message
        self._dlq[message.id] = message
        
        logger.warning(
            f"Message {message.id} added to DLQ: topic={message.topic}, "
            f"attempts={message.attempts}, error={error[:100]}"
        )
        
        # Check if we need to alert
        if len(self._dlq) >= self.alert_threshold:
            await self._trigger_alert()
        
        # Enforce max size
        if len(self._dlq) > self.max_dlq_size:
            self._evict_oldest()
        
        if self.redis:
            await self.redis.hset(
                DLQ_MESSAGES_KEY,
                message.id,
                json.dumps(message.to_dict()),
            )
    
    async def retry_message(
        self,
        message_id: str,
        handler: Callable[[Message], Awaitable[bool]],
    ) -> bool:
        """Retry a dead-lettered message."""
        message = self._dlq.get(message_id)
        if not message:
            logger.warning(f"Message {message_id} not found in DLQ")
            return False
        
        message.attempts += 1
        message.state = MessageState.RETRYING
        message.last_attempt_at = datetime.now(timezone.utc)
        
        try:
            success = await handler(message)
            
            if success:
                message.state = MessageState.COMPLETED
                del self._dlq[message_id]
                logger.info(f"Message {message_id} successfully retried")
                
                if self.redis:
                    await self.redis.hdel(DLQ_MESSAGES_KEY, message_id)
                
                return True
            else:
                message.state = MessageState.DEAD_LETTERED
                message.last_error = "Handler returned False"
                return False
                
        except Exception as e:
            message.state = MessageState.DEAD_LETTERED
            message.last_error = str(e)
            logger.error(f"Retry failed for message {message_id}: {e}")
            return False
    
    async def schedule_retry(
        self,
        message: Message,
    ):
        """Schedule message for retry with exponential backoff."""
        if message.attempts >= message.max_attempts:
            await self.add_message(message, f"Max attempts ({message.max_attempts}) exceeded")
            return
        
        # Calculate backoff delay
        delay = min(
            self.base_retry_delay * (2 ** message.attempts),
            self.max_retry_delay,
        )
        
        message.next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
        message.state = MessageState.RETRYING
        
        self._retry_queue.append(message)
        
        logger.info(
            f"Scheduled retry for message {message.id} in {delay}s "
            f"(attempt {message.attempts + 1}/{message.max_attempts})"
        )
    
    async def process_retry_queue(
        self,
        handler: Callable[[Message], Awaitable[bool]],
    ) -> int:
        """Process messages ready for retry."""
        now = datetime.now(timezone.utc)
        processed = 0
        
        ready_messages = [
            m for m in self._retry_queue
            if m.next_retry_at and m.next_retry_at <= now
        ]
        
        for message in ready_messages:
            self._retry_queue.remove(message)
            message.attempts += 1
            
            try:
                success = await handler(message)
                
                if success:
                    message.state = MessageState.COMPLETED
                    processed += 1
                    logger.info(f"Retry succeeded for message {message.id}")
                else:
                    await self.schedule_retry(message)
                    
            except Exception as e:
                message.last_error = str(e)
                await self.schedule_retry(message)
        
        return processed
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """Get message from DLQ."""
        return self._dlq.get(message_id)
    
    def list_messages(
        self,
        topic: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Message]:
        """List messages in DLQ."""
        messages = list(self._dlq.values())
        
        if topic:
            messages = [m for m in messages if m.topic == topic]
        
        messages.sort(key=lambda m: m.created_at, reverse=True)
        return messages[offset:offset + limit]
    
    def get_stats(self) -> DLQStats:
        """Get DLQ statistics."""
        if not self._dlq:
            return DLQStats()
        
        messages = list(self._dlq.values())
        
        # Count by topic
        by_topic: Dict[str, int] = {}
        for m in messages:
            by_topic[m.topic] = by_topic.get(m.topic, 0) + 1
        
        # Error distribution
        errors: Dict[str, int] = {}
        for m in messages:
            if m.last_error:
                # Normalize error message
                error_key = m.last_error[:50]
                errors[error_key] = errors.get(error_key, 0) + 1
        
        # Oldest message
        oldest = min(m.created_at for m in messages)
        oldest_age = (datetime.now(timezone.utc) - oldest).total_seconds() / 3600
        
        # Average attempts
        avg_attempts = sum(m.attempts for m in messages) / len(messages)
        
        return DLQStats(
            total_messages=len(messages),
            messages_by_topic=by_topic,
            oldest_message_age_hours=oldest_age,
            average_attempts=avg_attempts,
            error_distribution=errors,
        )
    
    async def purge_topic(self, topic: str) -> int:
        """Purge all messages for a topic."""
        to_delete = [mid for mid, m in self._dlq.items() if m.topic == topic]
        
        for mid in to_delete:
            del self._dlq[mid]
        
        if self.redis and to_delete:
            await self.redis.hdel(DLQ_MESSAGES_KEY, *to_delete)
        
        logger.info(f"Purged {len(to_delete)} messages for topic {topic}")
        return len(to_delete)
    
    async def replay_all(
        self,
        topic: Optional[str],
        handler: Callable[[Message], Awaitable[bool]],
    ) -> Dict[str, int]:
        """Replay all messages (optionally filtered by topic)."""
        messages = self.list_messages(topic=topic, limit=self.max_dlq_size)
        
        results = {"success": 0, "failed": 0}
        
        for message in messages:
            success = await self.retry_message(message.id, handler)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    def on_alert(self, callback: Callable[[DLQStats], Awaitable[None]]):
        """Register alert callback."""
        self._alert_callbacks.append(callback)
    
    async def _trigger_alert(self):
        """Trigger alert callbacks."""
        stats = self.get_stats()
        
        for callback in self._alert_callbacks:
            try:
                await callback(stats)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _evict_oldest(self):
        """Evict oldest messages when at capacity."""
        if not self._dlq:
            return
        
        # Find oldest messages
        sorted_messages = sorted(
            self._dlq.items(),
            key=lambda x: x[1].created_at,
        )
        
        # Remove oldest 10%
        to_remove = len(self._dlq) // 10
        for mid, _ in sorted_messages[:to_remove]:
            del self._dlq[mid]
        
        logger.warning(f"Evicted {to_remove} oldest messages from DLQ")


class EnhancedEventBus:
    """
    Event bus with dead letter queue support.
    
    Extends basic event bus with:
    - Automatic DLQ routing
    - Retry logic
    - Message tracking
    """
    
    def __init__(
        self,
        dlq: DeadLetterQueue,
        max_retries: int = 3,
    ):
        self.dlq = dlq
        self.max_retries = max_retries
        
        self._handlers: Dict[str, List[Callable]] = {}
        self._message_counter = 0
    
    def subscribe(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
    ):
        """Subscribe to topic."""
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)
    
    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> str:
        """Publish message to topic."""
        self._message_counter += 1
        
        message = Message(
            id=idempotency_key or f"msg_{self._message_counter}_{datetime.now(timezone.utc).timestamp()}",
            topic=topic,
            payload=payload,
            max_attempts=self.max_retries,
        )
        
        await self._process_message(message)
        return message.id
    
    async def _process_message(self, message: Message):
        """Process message with retry and DLQ support."""
        handlers = self._handlers.get(message.topic, [])
        
        if not handlers:
            logger.warning(f"No handlers for topic {message.topic}")
            return
        
        message.state = MessageState.PROCESSING
        message.attempts += 1
        message.last_attempt_at = datetime.now(timezone.utc)
        
        for handler in handlers:
            try:
                await handler(message.payload)
                message.state = MessageState.COMPLETED
                
            except Exception as e:
                message.last_error = str(e)
                
                if message.attempts < message.max_attempts:
                    await self.dlq.schedule_retry(message)
                else:
                    await self.dlq.add_message(message, str(e))
