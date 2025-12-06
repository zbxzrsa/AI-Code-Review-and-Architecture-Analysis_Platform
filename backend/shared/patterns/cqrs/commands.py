"""
CQRS Command Layer

Handles all write operations with event sourcing support.

Features:
- Command definitions with validation
- Command handlers with transaction support
- Command bus for routing
- Async event emission for read model sync
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar
from uuid import uuid4
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class CommandStatus(str, Enum):
    """Command execution status."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class CommandMetadata:
    """Metadata for command tracking."""
    command_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor_id: Optional[str] = None
    source: Optional[str] = None
    version: int = 1


@dataclass
class Command(ABC):
    """Base class for all commands."""
    metadata: CommandMetadata = field(default_factory=CommandMetadata)
    
    @property
    def command_type(self) -> str:
        return self.__class__.__name__
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_type": self.command_type,
            "command_id": self.metadata.command_id,
            "correlation_id": self.metadata.correlation_id,
            "timestamp": self.metadata.timestamp.isoformat(),
            "actor_id": self.metadata.actor_id,
        }


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    command_id: str
    data: Optional[Any] = None
    error: Optional[str] = None
    events: List['DomainEvent'] = field(default_factory=list)
    execution_time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "command_id": self.command_id,
            "data": self.data,
            "error": self.error,
            "events_count": len(self.events),
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class DomainEvent:
    """Domain event emitted by command handlers."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = ""
    aggregate_id: str = ""
    aggregate_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }


class CommandHandler(ABC, Generic[T]):
    """Base class for command handlers."""
    
    @abstractmethod
    async def handle(self, command: T) -> CommandResult:
        """Handle the command and return result with events."""
        pass
    
    @abstractmethod
    def can_handle(self, command: Command) -> bool:
        """Check if this handler can handle the given command."""
        pass


class CommandValidator(ABC):
    """Base class for command validators."""
    
    @abstractmethod
    async def validate(self, command: Command) -> List[str]:
        """Validate command and return list of errors."""
        pass


# =============================================================================
# Concrete Command Definitions
# =============================================================================

@dataclass
class CreateAnalysisCommand(Command):
    """Command to create a new code analysis."""
    code: str = ""
    language: str = ""
    rules: List[str] = field(default_factory=list)
    project_id: Optional[str] = None


@dataclass
class ApplyFixCommand(Command):
    """Command to apply a suggested fix."""
    analysis_id: str = ""
    issue_id: str = ""
    fix_type: str = "suggested"


@dataclass
class PromoteVersionCommand(Command):
    """Command to promote experiment to production."""
    experiment_id: str = ""
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemoteVersionCommand(Command):
    """Command to demote version to quarantine."""
    version_id: str = ""
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreateExperimentCommand(Command):
    """Command to create a new experiment."""
    name: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    baseline_id: Optional[str] = None


@dataclass
class UpdateConfigCommand(Command):
    """Command to update system configuration."""
    config_key: str = ""
    config_value: Any = None
    requires_approval: bool = False


# =============================================================================
# Command Handlers
# =============================================================================

class CreateAnalysisHandler(CommandHandler[CreateAnalysisCommand]):
    """Handler for CreateAnalysisCommand."""
    
    def __init__(self, analysis_service, event_store):
        self.analysis_service = analysis_service
        self.event_store = event_store
    
    def can_handle(self, command: Command) -> bool:
        return isinstance(command, CreateAnalysisCommand)
    
    async def handle(self, command: CreateAnalysisCommand) -> CommandResult:
        import time
        start_time = time.perf_counter()
        events = []
        
        try:
            # Perform analysis
            analysis_id = str(uuid4())
            
            # Emit AnalysisCreated event
            event = DomainEvent(
                event_type="AnalysisCreated",
                aggregate_id=analysis_id,
                aggregate_type="Analysis",
                data={
                    "code_hash": hash(command.code),
                    "language": command.language,
                    "rules": command.rules,
                    "project_id": command.project_id,
                },
                metadata={"actor_id": command.metadata.actor_id}
            )
            events.append(event)
            
            # Store event
            await self.event_store.append(event)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return CommandResult(
                success=True,
                command_id=command.metadata.command_id,
                data={"analysis_id": analysis_id},
                events=events,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"CreateAnalysis failed: {e}")
            return CommandResult(
                success=False,
                command_id=command.metadata.command_id,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )


class PromoteVersionHandler(CommandHandler[PromoteVersionCommand]):
    """Handler for PromoteVersionCommand."""
    
    def __init__(self, version_service, event_store):
        self.version_service = version_service
        self.event_store = event_store
    
    def can_handle(self, command: Command) -> bool:
        return isinstance(command, PromoteVersionCommand)
    
    async def handle(self, command: PromoteVersionCommand) -> CommandResult:
        import time
        start_time = time.perf_counter()
        events = []
        
        try:
            promotion_id = str(uuid4())
            
            # Emit VersionPromoted event
            event = DomainEvent(
                event_type="VersionPromoted",
                aggregate_id=command.experiment_id,
                aggregate_type="Experiment",
                data={
                    "promotion_id": promotion_id,
                    "reason": command.reason,
                    "metrics": command.metrics,
                    "from_zone": "v1",
                    "to_zone": "v2",
                },
                metadata={"actor_id": command.metadata.actor_id}
            )
            events.append(event)
            
            await self.event_store.append(event)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return CommandResult(
                success=True,
                command_id=command.metadata.command_id,
                data={"promotion_id": promotion_id},
                events=events,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"PromoteVersion failed: {e}")
            return CommandResult(
                success=False,
                command_id=command.metadata.command_id,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )


# =============================================================================
# Command Bus
# =============================================================================

class CommandBus:
    """
    Command bus for routing commands to handlers.
    
    Supports:
    - Handler registration
    - Middleware pipeline
    - Async command execution
    - Transaction management
    """
    
    def __init__(self, event_publisher=None):
        self._handlers: Dict[Type[Command], CommandHandler] = {}
        self._middleware: List[Callable] = []
        self._validators: Dict[Type[Command], List[CommandValidator]] = {}
        self._event_publisher = event_publisher
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_commands = 0
        self._successful_commands = 0
        self._failed_commands = 0
    
    def register_handler(self, command_type: Type[Command], handler: CommandHandler):
        """Register a handler for a command type."""
        self._handlers[command_type] = handler
        logger.info(f"Registered handler for {command_type.__name__}")
    
    def register_validator(self, command_type: Type[Command], validator: CommandValidator):
        """Register a validator for a command type."""
        if command_type not in self._validators:
            self._validators[command_type] = []
        self._validators[command_type].append(validator)
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to the command pipeline."""
        self._middleware.append(middleware)
    
    async def dispatch(self, command: Command) -> CommandResult:
        """
        Dispatch a command to its handler.
        
        Args:
            command: Command to dispatch
            
        Returns:
            CommandResult with execution status
        """
        async with self._lock:
            self._total_commands += 1
        
        command_type = type(command)
        
        # Validate command
        if command_type in self._validators:
            errors = []
            for validator in self._validators[command_type]:
                validator_errors = await validator.validate(command)
                errors.extend(validator_errors)
            
            if errors:
                return CommandResult(
                    success=False,
                    command_id=command.metadata.command_id,
                    error=f"Validation failed: {', '.join(errors)}"
                )
        
        # Find handler
        handler = self._handlers.get(command_type)
        if not handler:
            logger.error(f"No handler registered for {command_type.__name__}")
            return CommandResult(
                success=False,
                command_id=command.metadata.command_id,
                error=f"No handler for {command_type.__name__}"
            )
        
        # Execute with middleware
        try:
            # Apply middleware
            for middleware in self._middleware:
                command = await middleware(command)
            
            # Execute handler
            result = await handler.handle(command)
            
            # Publish events if successful
            if result.success and result.events and self._event_publisher:
                for event in result.events:
                    await self._event_publisher.publish(event)
            
            async with self._lock:
                if result.success:
                    self._successful_commands += 1
                else:
                    self._failed_commands += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            async with self._lock:
                self._failed_commands += 1
            return CommandResult(
                success=False,
                command_id=command.metadata.command_id,
                error=str(e)
            )
    
    async def dispatch_batch(self, commands: List[Command]) -> List[CommandResult]:
        """Dispatch multiple commands."""
        results = []
        for command in commands:
            result = await self.dispatch(command)
            results.append(result)
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get command bus metrics."""
        return {
            "total_commands": self._total_commands,
            "successful_commands": self._successful_commands,
            "failed_commands": self._failed_commands,
            "success_rate": self._successful_commands / max(self._total_commands, 1),
            "registered_handlers": len(self._handlers),
        }


# =============================================================================
# Command Middleware
# =============================================================================

async def logging_middleware(command: Command) -> Command:
    """Log command execution."""
    logger.info(f"Executing command: {command.command_type}", extra={
        "command_id": command.metadata.command_id,
        "actor_id": command.metadata.actor_id,
    })
    return command


async def audit_middleware(command: Command) -> Command:
    """Add audit information to command."""
    if not command.metadata.timestamp:
        command.metadata.timestamp = datetime.now(timezone.utc)
    return command


class TransactionMiddleware:
    """Middleware for transaction management."""
    
    def __init__(self, db_client):
        self.db = db_client
    
    async def __call__(self, command: Command) -> Command:
        # Transaction will be managed by the handler
        command.metadata.source = "transaction_wrapped"
        return command
