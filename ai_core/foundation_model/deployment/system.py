"""
Main orchestrator for Plan A: Lightweight Continuous Learning.

Integrates all deployment components:
- Frozen base model with LoRA adapters
- RAG system for real-time knowledge
- Periodic retraining
- Quantization for efficiency
- Fault tolerance
- Cost control

Async Context Manager Support:
    The PracticalDeploymentSystem supports the async context manager pattern
    for automatic resource management:

    ```python
    async with PracticalDeploymentSystem(model, config) as system:
        result = await system.process("Review this code...")
    # Resources automatically cleaned up here
    ```

    This ensures:
    - Proper initialization of all components
    - Automatic cleanup on exit (even with exceptions)
    - Checkpoint saving before shutdown
    - Thread-safe operation
"""

import asyncio
import logging
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Type
from types import TracebackType

import torch.nn as nn

from .config import PracticalDeploymentConfig
from .cost_control import CostController
from .distillation import ModelDistiller
from .fault_tolerance import FaultToleranceManager, HealthChecker
from .lora import LoRAAdapterManager
from .quantization import ModelQuantizer
from .rag import RAGSystem
from .retraining import RetrainingDataCollector, RetrainingScheduler

logger = logging.getLogger(__name__)


# =============================================================================
# Resource Management Types
# =============================================================================

class SystemState(str, Enum):
    """System lifecycle states."""
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ContextManagerState:
    """Tracks context manager nesting and state."""
    nesting_level: int = 0
    entry_count: int = 0
    exit_count: int = 0
    last_exception: Optional[Exception] = None
    cleanup_performed: bool = False


class StartupError(Exception):
    """Raised when system fails to start."""
    pass


class ShutdownError(Exception):
    """Raised when system fails to stop cleanly."""
    pass


class PracticalDeploymentSystem:
    """
    Main orchestrator for Plan A: Lightweight Continuous Learning.
    
    Integrates:
    - Frozen base model
    - LoRA adapters
    - RAG system
    - Periodic retraining
    - Quantization
    - Fault tolerance
    - Cost control
    
    Supports async context manager for automatic resource management.
    
    Usage (Manual):
        ```python
        config = PracticalDeploymentConfig()
        system = PracticalDeploymentSystem(base_model, config)
        await system.start()
        try:
            result = await system.process("Review this code...")
        finally:
            await system.stop()
        ```
    
    Usage (Context Manager - Recommended):
        ```python
        config = PracticalDeploymentConfig()
        async with PracticalDeploymentSystem(base_model, config) as system:
            result = await system.process("Review this code...")
        # Automatic cleanup, checkpoint saved
        ```
    
    Nested Context Managers:
        ```python
        async with PracticalDeploymentSystem(model, config) as system:
            # Nesting is supported - inner contexts reuse the running system
            async with system:
                result = await system.process("query")
        ```
    
    Exception Handling:
        The context manager ensures cleanup even when exceptions occur:
        ```python
        async with PracticalDeploymentSystem(model, config) as system:
            raise ValueError("Error!")
        # __aexit__ is still called, resources cleaned up
        ```
    
    Cancellation:
        Async cancellation is handled gracefully:
        ```python
        async with PracticalDeploymentSystem(model, config) as system:
            await asyncio.sleep(100)  # If cancelled, cleanup still runs
        ```
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: PracticalDeploymentConfig,
    ):
        self.base_model = base_model
        self.config = config
        
        # Initialize components
        self.adapter_manager = LoRAAdapterManager(base_model, config)
        self.rag_system = RAGSystem(config)
        self.data_collector = RetrainingDataCollector(config)
        self.retraining_scheduler = RetrainingScheduler(
            self.adapter_manager,
            self.data_collector,
            config,
        )
        self.quantizer = ModelQuantizer(config)
        self.health_checker = HealthChecker(config)
        self.fault_tolerance = FaultToleranceManager(config)
        self.cost_controller = CostController(config)
        
        # Optional distiller
        self.distiller: Optional[ModelDistiller] = None
        if config.enable_distillation:
            self.distiller = ModelDistiller(base_model, config)
        
        # System state
        self._state = SystemState.UNINITIALIZED
        self.is_running = False
        self.initialized_at: Optional[datetime] = None
        self.stopped_at: Optional[datetime] = None
        self.request_count = 0
        
        # Context manager state for thread safety and nesting
        self._context_state = ContextManagerState()
        self._context_lock = asyncio.Lock()
    
    # =========================================================================
    # Async Context Manager Protocol
    # =========================================================================
    
    async def __aenter__(self) -> "PracticalDeploymentSystem":
        """
        Enter the async context manager.
        
        Initializes and starts all system components.
        Supports nested context managers - subsequent entries reuse the running system.
        
        Returns:
            The initialized PracticalDeploymentSystem instance.
        
        Raises:
            StartupError: If system fails to start.
        
        Example:
            async with PracticalDeploymentSystem(model, config) as system:
                result = await system.process("input")
        """
        async with self._context_lock:
            self._context_state.entry_count += 1
            self._context_state.nesting_level += 1
            
            logger.debug(
                f"Context enter: nesting_level={self._context_state.nesting_level}, "
                f"entry_count={self._context_state.entry_count}"
            )
            
            # If already running (nested context), just return
            if self._state == SystemState.RUNNING:
                logger.debug("System already running, reusing for nested context")
                return self
            
            # Start the system
            self._state = SystemState.STARTING
            
            try:
                await self.start()
                self._state = SystemState.RUNNING
                self._context_state.cleanup_performed = False
                
                logger.info(
                    f"PracticalDeploymentSystem entered context successfully "
                    f"(nesting_level={self._context_state.nesting_level})"
                )
                
                return self
                
            except asyncio.CancelledError:
                # Handle cancellation during startup
                logger.warning("System startup cancelled")
                self._state = SystemState.ERROR
                self._context_state.last_exception = asyncio.CancelledError()
                raise StartupError("System startup was cancelled") from None
                
            except Exception as e:
                # Handle any startup failures
                logger.error(f"System startup failed: {e}\n{traceback.format_exc()}")
                self._state = SystemState.ERROR
                self._context_state.last_exception = e
                
                # Attempt partial cleanup
                await self._emergency_cleanup()
                
                raise StartupError(f"Failed to start system: {e}") from e
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """
        Exit the async context manager.
        
        Performs comprehensive cleanup regardless of success or failure.
        Handles all exception types and ensures resources are deallocated.
        
        Args:
            exc_type: Exception type if an exception was raised, None otherwise.
            exc_val: Exception instance if raised, None otherwise.
            exc_tb: Exception traceback if raised, None otherwise.
        
        Returns:
            False - exceptions are not suppressed, they propagate to caller.
        
        Note:
            - Cleanup runs even if an exception occurred
            - Nested contexts only cleanup on final exit
            - Cancellation is handled gracefully
        """
        async with self._context_lock:
            self._context_state.exit_count += 1
            self._context_state.nesting_level -= 1
            
            # Log exception info if present
            if exc_type is not None:
                self._context_state.last_exception = exc_val
                logger.warning(
                    f"Context exit with exception: {exc_type.__name__}: {exc_val}"
                )
            
            logger.debug(
                f"Context exit: nesting_level={self._context_state.nesting_level}, "
                f"exit_count={self._context_state.exit_count}"
            )
            
            # Only cleanup on final exit (nesting_level reaches 0)
            if self._context_state.nesting_level > 0:
                logger.debug("Nested context exit, deferring cleanup")
                return False
            
            # Perform cleanup
            if not self._context_state.cleanup_performed:
                await self._perform_cleanup(exc_type, exc_val, exc_tb)
            
            return False  # Don't suppress exceptions
    
    async def _perform_cleanup(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Perform comprehensive resource cleanup.
        
        Called on final context exit. Ensures all resources are properly released.
        """
        logger.info("Performing context manager cleanup...")
        self._state = SystemState.STOPPING
        cleanup_errors: List[str] = []
        
        try:
            # Stop the system (saves checkpoint, stops background services)
            await self.stop()
            
        except asyncio.CancelledError:
            # Handle cancellation during cleanup
            logger.warning("Cleanup was cancelled, performing emergency cleanup")
            cleanup_errors.append("Cleanup cancelled")
            await self._emergency_cleanup()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}\n{traceback.format_exc()}")
            cleanup_errors.append(f"Cleanup error: {e}")
            await self._emergency_cleanup()
        
        finally:
            self._state = SystemState.STOPPED
            self._context_state.cleanup_performed = True
            self.stopped_at = datetime.now(timezone.utc)
            
            if cleanup_errors:
                logger.warning(f"Cleanup completed with errors: {cleanup_errors}")
            else:
                logger.info(
                    f"PracticalDeploymentSystem cleanup complete. "
                    f"Ran for {self._get_uptime_seconds():.1f}s, "
                    f"processed {self.request_count} requests"
                )
    
    async def _emergency_cleanup(self) -> None:
        """
        Emergency cleanup when normal cleanup fails.
        
        Attempts to release critical resources without relying on
        normal shutdown procedures.
        """
        logger.warning("Performing emergency cleanup...")
        
        # Force stop flag
        self.is_running = False
        
        # Cancel background tasks directly
        try:
            if hasattr(self.health_checker, '_check_task') and self.health_checker._check_task:
                self.health_checker._check_task.cancel()
        except Exception as e:
            logger.debug(f"Error cancelling health checker: {e}")
        
        try:
            if hasattr(self.retraining_scheduler, '_scheduler_task') and self.retraining_scheduler._scheduler_task:
                self.retraining_scheduler._scheduler_task.cancel()
        except Exception as e:
            logger.debug(f"Error cancelling retraining scheduler: {e}")
        
        logger.info("Emergency cleanup completed")
    
    def _get_uptime_seconds(self) -> float:
        """Calculate system uptime in seconds."""
        if self.initialized_at is None:
            return 0.0
        
        end_time = self.stopped_at or datetime.now(timezone.utc)
        return (end_time - self.initialized_at).total_seconds()
    
    @property
    def state(self) -> SystemState:
        """Get current system state."""
        return self._state
    
    @property
    def context_info(self) -> Dict[str, Any]:
        """Get context manager state information."""
        return {
            "state": self._state.value,
            "nesting_level": self._context_state.nesting_level,
            "entry_count": self._context_state.entry_count,
            "exit_count": self._context_state.exit_count,
            "cleanup_performed": self._context_state.cleanup_performed,
            "last_exception": str(self._context_state.last_exception) if self._context_state.last_exception else None,
            "uptime_seconds": self._get_uptime_seconds(),
        }
    
    # =========================================================================
    # System Lifecycle Methods
    # =========================================================================
    
    async def start(self):
        """Start the deployment system."""
        if self.is_running:
            logger.warning("System already running")
            return
        
        self.is_running = True
        self.initialized_at = datetime.now(timezone.utc)
        
        # Start background services
        await self.health_checker.start()
        await self.retraining_scheduler.start()
        
        # Create default adapter
        self.adapter_manager.create_adapter("default")
        self.adapter_manager.activate_adapter("default")
        
        logger.info("Practical deployment system started")
    
    async def stop(self):
        """Stop the deployment system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop background services
        await self.health_checker.stop()
        await self.retraining_scheduler.stop()
        
        # Save checkpoint
        try:
            await self.fault_tolerance.save_checkpoint(
                self.adapter_manager,
                self.rag_system,
                additional_state={
                    "request_count": self.request_count,
                    "stopped_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
        
        logger.info("Practical deployment system stopped")
    
    async def process(
        self,
        input_text: str,
        use_rag: bool = True,
        adapter_name: Optional[str] = None,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Process input with the full system.
        
        1. Check cost limits
        2. RAG augmentation
        3. Model inference with adapter
        4. Collect for retraining
        5. Return result
        """
        self.request_count += 1
        
        result = {
            "request_id": f"req_{self.request_count}",
            "input": input_text,
            "output": None,
            "rag_context": [],
            "adapter_used": adapter_name or self.adapter_manager.active_adapter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success",
        }
        
        # Check limits before processing
        within_limits, warning = self.cost_controller.check_limits()
        if not within_limits:
            result["status"] = "rate_limited"
            result["error"] = warning
            return result
        
        if warning:
            result["warning"] = warning
        
        try:
            augmented_prompt = input_text
            
            # RAG augmentation
            if use_rag and self.config.enable_rag:
                augmented_prompt = self.rag_system.augment_prompt(input_text)
                retrieved = self.rag_system.retrieve(input_text)
                result["rag_context"] = [
                    {"content": c[:200] + "..." if len(c) > 200 else c, "score": s}
                    for c, s in retrieved
                ]
            
            # Model inference (placeholder)
            # In production, this would:
            # 1. Tokenize the augmented prompt
            # 2. Run through base model + LoRA adapter
            # 3. Generate response
            output = f"[Response to: {input_text[:100]}...]"
            result["output"] = output
            
            # Estimate and record token usage
            input_tokens = len(input_text.split())
            output_tokens = len(output.split())
            
            within_limits, warning = self.cost_controller.record_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation="process",
            )
            
            result["tokens"] = {
                "input": input_tokens,
                "output": output_tokens,
            }
            
            if warning:
                result["warning"] = warning
            
            # Collect for retraining (if response was good)
            self.data_collector.add_sample(input_text, output)
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            result["status"] = "error"
            result["error"] = str(e)
        
        return result
    
    async def process_batch(
        self,
        inputs: List[str],
        use_rag: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process multiple inputs."""
        results = []
        for input_text in inputs:
            result = await self.process(input_text, use_rag=use_rag)
            results.append(result)
        return results
    
    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add knowledge to RAG system."""
        return self.rag_system.add_knowledge(content, metadata)
    
    def add_knowledge_batch(
        self,
        documents: List[tuple],
    ) -> List[str]:
        """Add multiple documents to RAG system."""
        return self.rag_system.add_knowledge_batch(documents)
    
    async def trigger_retraining(
        self,
        adapter_name: str = "default",
        epochs: int = 1,
    ):
        """Manually trigger adapter retraining."""
        return await self.retraining_scheduler.trigger_retraining(adapter_name, epochs)
    
    async def quantize_model(
        self,
        calibration_data: Optional[List] = None,
    ):
        """Quantize the base model."""
        model, stats = self.quantizer.quantize(
            self.base_model,
            calibration_data=calibration_data,
        )
        self.base_model = model
        return stats
    
    async def save_checkpoint(self):
        """Manually save a checkpoint."""
        return await self.fault_tolerance.save_checkpoint(
            self.adapter_manager,
            self.rag_system,
            additional_state={
                "request_count": self.request_count,
            },
        )
    
    async def load_checkpoint(self, checkpoint_dir: Optional[str] = None):
        """Load a checkpoint."""
        return await self.fault_tolerance.load_checkpoint(
            self.adapter_manager,
            self.rag_system,
            checkpoint_dir=checkpoint_dir,
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "is_running": self.is_running,
            "initialized_at": self.initialized_at.isoformat() if self.initialized_at else None,
            "request_count": self.request_count,
            "health": self.health_checker.get_status(),
            "cost": self.cost_controller.get_usage_summary(),
            "adapters": {
                "available": self.adapter_manager.list_adapters(),
                "active": self.adapter_manager.active_adapter,
            },
            "retraining": self.retraining_scheduler.get_status(),
            "data_collector": self.data_collector.get_stats(),
            "rag": {
                "documents": len(self.rag_system.index),
                "cache_size": len(self.rag_system.embedding_cache),
            },
            "fault_tolerance": self.fault_tolerance.get_stats(),
        }
    
    def get_adapter_info(self, adapter_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about an adapter."""
        return self.adapter_manager.get_adapter_info(adapter_name)
    
    async def create_adapter(
        self,
        adapter_name: str,
        target_modules: Optional[List[str]] = None,
    ):
        """Create a new adapter."""
        return self.adapter_manager.create_adapter(adapter_name, target_modules)
    
    def switch_adapter(self, adapter_name: str):
        """Switch to a different adapter."""
        self.adapter_manager.activate_adapter(adapter_name)
    
    async def merge_adapters(
        self,
        adapter_names: List[str],
        weights: Optional[List[float]] = None,
        new_name: str = "merged",
    ):
        """Merge multiple adapters."""
        return self.adapter_manager.merge_adapters(adapter_names, weights, new_name)
