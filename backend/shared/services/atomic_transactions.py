"""
Atomic Transaction Manager

Implements:
- Distributed transactions with saga pattern
- Atomic version promotions
- Automatic rollback on failure
- Idempotency key support
"""

import asyncio
import logging
import hashlib
import uuid
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class TransactionState(str, Enum):
    """Transaction state."""
    PENDING = "pending"
    RUNNING = "running"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class StepState(str, Enum):
    """Step execution state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"


@dataclass
class TransactionStep:
    """Individual step in a transaction."""
    name: str
    execute: Callable[..., Awaitable[Any]]
    compensate: Callable[..., Awaitable[None]]
    state: StepState = StepState.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Transaction:
    """Distributed transaction."""
    id: str
    idempotency_key: str
    steps: List[TransactionStep] = field(default_factory=list)
    state: TransactionState = TransactionState.PENDING
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class AtomicTransactionManager:
    """
    Manages distributed transactions with saga pattern.
    
    Features:
    - Step-by-step execution with compensation
    - Automatic rollback on failure
    - Idempotency key support
    - Transaction logging
    """
    
    def __init__(
        self,
        redis_client = None,
        idempotency_ttl: int = 86400,  # 24 hours
    ):
        self.redis = redis_client
        self.idempotency_ttl = idempotency_ttl
        
        # In-memory storage for transactions
        self._transactions: Dict[str, Transaction] = {}
        self._idempotency_cache: Dict[str, str] = {}
    
    async def check_idempotency(self, key: str) -> Optional[str]:
        """Check if operation with idempotency key was already processed."""
        if self.redis:
            result = await self.redis.get(f"idempotency:{key}")
            return result
        return self._idempotency_cache.get(key)
    
    async def save_idempotency(self, key: str, transaction_id: str):
        """Save idempotency key result."""
        if self.redis:
            await self.redis.set(
                f"idempotency:{key}",
                transaction_id,
                ex=self.idempotency_ttl,
            )
        else:
            self._idempotency_cache[key] = transaction_id
    
    def create_transaction(
        self,
        idempotency_key: Optional[str] = None,
    ) -> Transaction:
        """Create new transaction."""
        transaction_id = str(uuid.uuid4())
        
        if not idempotency_key:
            idempotency_key = transaction_id
        
        transaction = Transaction(
            id=transaction_id,
            idempotency_key=idempotency_key,
        )
        
        self._transactions[transaction_id] = transaction
        return transaction
    
    def add_step(
        self,
        transaction: Transaction,
        name: str,
        execute: Callable[..., Awaitable[Any]],
        compensate: Callable[..., Awaitable[None]],
    ):
        """Add step to transaction."""
        step = TransactionStep(
            name=name,
            execute=execute,
            compensate=compensate,
        )
        transaction.steps.append(step)
    
    async def execute_transaction(
        self,
        transaction: Transaction,
        **context,
    ) -> Any:
        """
        Execute transaction with automatic rollback.
        
        Returns result of last step on success.
        Raises exception on failure after rollback.
        """
        # Check idempotency
        existing = await self.check_idempotency(transaction.idempotency_key)
        if existing:
            logger.info(f"Idempotent request, returning existing transaction {existing}")
            existing_tx = self._transactions.get(existing)
            if existing_tx:
                return existing_tx.context.get("result")
            return None
        
        transaction.state = TransactionState.RUNNING
        transaction.context.update(context)
        
        executed_steps: List[TransactionStep] = []
        
        try:
            # Execute steps in order
            for step in transaction.steps:
                step.state = StepState.RUNNING
                step.started_at = datetime.utcnow()
                
                logger.info(f"Transaction {transaction.id}: executing step '{step.name}'")
                
                try:
                    result = await step.execute(transaction.context)
                    step.result = result
                    step.state = StepState.COMPLETED
                    step.completed_at = datetime.utcnow()
                    
                    # Store result in context for next steps
                    transaction.context[f"{step.name}_result"] = result
                    executed_steps.append(step)
                    
                except Exception as e:
                    step.state = StepState.FAILED
                    step.error = str(e)
                    step.completed_at = datetime.utcnow()
                    logger.error(f"Transaction {transaction.id}: step '{step.name}' failed: {e}")
                    raise
            
            # All steps completed
            transaction.state = TransactionState.COMMITTED
            transaction.completed_at = datetime.utcnow()
            transaction.context["result"] = executed_steps[-1].result if executed_steps else None
            
            # Save idempotency key
            await self.save_idempotency(transaction.idempotency_key, transaction.id)
            
            logger.info(f"Transaction {transaction.id} committed successfully")
            return transaction.context["result"]
            
        except Exception as e:
            # Rollback executed steps in reverse order
            logger.info(f"Transaction {transaction.id}: rolling back {len(executed_steps)} steps")
            
            for step in reversed(executed_steps):
                try:
                    step.state = StepState.RUNNING
                    await step.compensate(transaction.context)
                    step.state = StepState.COMPENSATED
                    logger.info(f"Transaction {transaction.id}: compensated step '{step.name}'")
                except Exception as comp_error:
                    logger.error(
                        f"Transaction {transaction.id}: compensation failed for '{step.name}': {comp_error}"
                    )
            
            transaction.state = TransactionState.ROLLED_BACK
            transaction.completed_at = datetime.utcnow()
            transaction.error = str(e)
            
            raise TransactionFailedError(transaction.id, str(e))
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID."""
        return self._transactions.get(transaction_id)


class TransactionFailedError(Exception):
    """Transaction failed and was rolled back."""
    
    def __init__(self, transaction_id: str, error: str):
        self.transaction_id = transaction_id
        self.error = error
        super().__init__(f"Transaction {transaction_id} failed: {error}")


class VersionPromotionTransaction:
    """
    Atomic version promotion from V1 to V2.
    
    Steps:
    1. Validate V1 experiment meets criteria
    2. Create V2 version record
    3. Update routing to new version
    4. Archive V1 experiment
    5. Notify subscribers
    """
    
    def __init__(
        self,
        transaction_manager: AtomicTransactionManager,
        db = None,
        event_bus = None,
    ):
        self.tx_manager = transaction_manager
        self.db = db
        self.event_bus = event_bus
    
    async def promote(
        self,
        experiment_id: str,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Promote V1 experiment to V2 production."""
        if not idempotency_key:
            idempotency_key = f"promote:{experiment_id}:{datetime.utcnow().date()}"
        
        transaction = self.tx_manager.create_transaction(idempotency_key)
        
        # Step 1: Validate experiment
        self.tx_manager.add_step(
            transaction,
            name="validate_experiment",
            execute=self._validate_experiment,
            compensate=self._noop,  # No compensation needed for validation
        )
        
        # Step 2: Create V2 version
        self.tx_manager.add_step(
            transaction,
            name="create_v2_version",
            execute=self._create_v2_version,
            compensate=self._rollback_v2_version,
        )
        
        # Step 3: Update routing
        self.tx_manager.add_step(
            transaction,
            name="update_routing",
            execute=self._update_routing,
            compensate=self._rollback_routing,
        )
        
        # Step 4: Archive V1 experiment
        self.tx_manager.add_step(
            transaction,
            name="archive_experiment",
            execute=self._archive_experiment,
            compensate=self._restore_experiment,
        )
        
        # Step 5: Notify subscribers
        self.tx_manager.add_step(
            transaction,
            name="notify_subscribers",
            execute=self._notify_subscribers,
            compensate=self._noop,  # Notification failure is non-critical
        )
        
        result = await self.tx_manager.execute_transaction(
            transaction,
            experiment_id=experiment_id,
        )
        
        return result
    
    async def _validate_experiment(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment meets promotion criteria."""
        experiment_id = ctx["experiment_id"]
        
        # In production, fetch from database and validate
        # - Minimum runtime (1 week)
        # - Accuracy >= baseline + 5%
        # - Error rate < 5%
        # - No critical issues
        
        return {
            "experiment_id": experiment_id,
            "validated": True,
            "metrics": {"accuracy": 0.95, "error_rate": 0.02},
        }
    
    async def _create_v2_version(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Create new V2 version record."""
        experiment_id = ctx["experiment_id"]
        version_id = str(uuid.uuid4())
        
        # In production, insert into database
        logger.info(f"Created V2 version {version_id} from experiment {experiment_id}")
        
        ctx["v2_version_id"] = version_id
        return {"version_id": version_id, "status": "created"}
    
    async def _rollback_v2_version(self, ctx: Dict[str, Any]):
        """Rollback V2 version creation."""
        version_id = ctx.get("v2_version_id")
        if version_id:
            # In production, delete from database
            logger.info(f"Rolled back V2 version {version_id}")
    
    async def _update_routing(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Update routing to point to new version."""
        version_id = ctx.get("v2_version_id")
        
        # Store previous routing for rollback
        ctx["previous_routing"] = {"version": "v2-old"}
        
        # In production, update routing configuration
        logger.info(f"Updated routing to version {version_id}")
        
        return {"routing_updated": True}
    
    async def _rollback_routing(self, ctx: Dict[str, Any]):
        """Rollback routing changes."""
        previous = ctx.get("previous_routing")
        if previous:
            # Restore previous routing
            logger.info(f"Rolled back routing to {previous}")
    
    async def _archive_experiment(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Archive V1 experiment."""
        experiment_id = ctx["experiment_id"]
        
        # Store experiment state for potential restoration
        ctx["experiment_state"] = {"status": "active"}
        
        # In production, update status in database
        logger.info(f"Archived experiment {experiment_id}")
        
        return {"archived": True}
    
    async def _restore_experiment(self, ctx: Dict[str, Any]):
        """Restore archived experiment."""
        experiment_id = ctx["experiment_id"]
        previous_state = ctx.get("experiment_state")
        
        if previous_state:
            # Restore experiment status
            logger.info(f"Restored experiment {experiment_id} to {previous_state}")
    
    async def _notify_subscribers(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Notify subscribers of promotion."""
        version_id = ctx.get("v2_version_id")
        experiment_id = ctx["experiment_id"]
        
        if self.event_bus:
            await self.event_bus.publish(
                "version_promoted",
                {
                    "experiment_id": experiment_id,
                    "version_id": version_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
        
        return {"notified": True}
    
    async def _noop(self, ctx: Dict[str, Any]):
        """No-op compensation."""
        pass


# Idempotency key decorator
def idempotent(key_func: Callable[..., str]):
    """
    Decorator for idempotent operations.
    
    Usage:
        @idempotent(lambda req: f"create_analysis:{req.project_id}:{req.commit}")
        async def create_analysis(req):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs)
            
            # Check if already processed
            # In production, check Redis/database
            
            result = await func(*args, **kwargs)
            
            # Store idempotency key
            # In production, store in Redis/database
            
            return result
        return wrapper
    return decorator
