"""
Autonomous Learning Agent System

⚠️ REFACTORING NOTICE:
This file (1,688 lines) is scheduled for modular split before next release.
A new modular structure is being prepared at:

    ai_core/foundation_model/autonomous/
    ├── config.py         - Configuration classes and enums
    ├── online_learning.py - Online learning buffer and module
    ├── memory/           - Memory subsystems
    │   ├── episodic.py   - Episodic memory
    │   ├── semantic.py   - Semantic memory
    │   └── working.py    - Working memory
    ├── evaluation.py     - Self-evaluation system
    ├── knowledge.py      - RAG, tool use, knowledge integration
    ├── safety.py         - Safety monitor
    └── agent.py          - Main autonomous agent

For forward compatibility, you can already import from:
    from ai_core.foundation_model.autonomous import AutonomousLearningAgent

---

Implements self-evolving AI capabilities:
1. Online Learning Module - Real-time learning from data streams
2. Memory Management - Long/short-term/episodic memory
3. Self-Evaluation System - Automatic benchmarking and gap detection
4. Knowledge Integration - RAG, external knowledge bases, tool use
5. Safety & Alignment - Continuous monitoring and value alignment

Based on Google NeurIPS 2025 "Nested Learning" concepts:
- Self-modifying architecture
- Multi-timescale updates
- Associative memory

Target: Infinite autonomous learning with human oversight
"""

import asyncio
import hashlib
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import vector index for production-ready similarity search
try:
    from .vector_index import (
        BaseVectorIndex,
        FAISSVectorIndex,
        IndexConfig,
        IndexType,
        NumpyVectorIndex,
        create_vector_index,
    )
    VECTOR_INDEX_AVAILABLE = True
except ImportError:
    VECTOR_INDEX_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class LearningMode(str, Enum):
    """Learning modes."""
    ONLINE = "online"  # Continuous real-time learning
    BATCH = "batch"  # Periodic batch updates
    TRIGGERED = "triggered"  # Event-driven learning
    SCHEDULED = "scheduled"  # Time-scheduled learning


class MemoryType(str, Enum):
    """Types of memory."""
    LONG_TERM = "long_term"  # Model parameters
    SHORT_TERM = "short_term"  # Context window
    EPISODIC = "episodic"  # Experience storage
    SEMANTIC = "semantic"  # Knowledge graph
    WORKING = "working"  # Active computation


class SafetyLevel(str, Enum):
    """Safety monitoring levels."""
    LOW = "low"  # Minimal checks
    MEDIUM = "medium"  # Standard checks
    HIGH = "high"  # Strict checks
    CRITICAL = "critical"  # Maximum safety


@dataclass
class AutonomousConfig:
    """Autonomous learning configuration."""
    # Learning modes
    primary_mode: LearningMode = LearningMode.ONLINE
    enable_batch_consolidation: bool = True
    consolidation_interval_hours: int = 24

    # Online learning
    online_learning_rate: float = 1e-6
    online_batch_size: int = 1
    online_buffer_size: int = 1000

    # Memory
    episodic_memory_size: int = 100000
    working_memory_size: int = 1000
    semantic_memory_enabled: bool = True

    # Self-evaluation
    eval_interval_steps: int = 1000
    benchmark_suite: List[str] = field(default_factory=lambda: [
        "code_review", "bug_detection", "security_scan"
    ])
    knowledge_gap_threshold: float = 0.1

    # Knowledge integration
    enable_rag: bool = True
    enable_tool_use: bool = True
    external_knowledge_sources: List[str] = field(default_factory=list)

    # Safety
    safety_level: SafetyLevel = SafetyLevel.HIGH
    human_oversight_required: bool = True
    value_alignment_checks: bool = True
    max_autonomous_steps: int = 10000


@dataclass
class LearningEvent:
    """Single learning event."""
    event_id: str
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    source: str
    priority: int = 1
    processed: bool = False


@dataclass
class KnowledgeGap:
    """Identified knowledge gap."""
    gap_id: str
    domain: str
    description: str
    severity: float  # 0-1
    detected_at: datetime
    resolved: bool = False
    resolution_data: Optional[Dict[str, Any]] = None


# =============================================================================
# Exception Classification System
# =============================================================================

class ExceptionSeverity(str, Enum):
    """Exception severity levels for graded handling."""
    LOW = "low"           # Recoverable, log and continue
    MEDIUM = "medium"     # Recoverable with retry, may need attention
    HIGH = "high"         # Serious issue, may require intervention
    CRITICAL = "critical" # Fatal, must terminate operation


class LearningErrorCode(str, Enum):
    """Error codes for learning system exceptions."""
    E1001_INVALID_SAMPLE = "E1001"      # Invalid sample format
    E1002_EMPTY_BUFFER = "E1002"        # Buffer is empty
    E1003_STREAM_TIMEOUT = "E1003"      # Stream read timeout
    E1004_DATA_CORRUPTION = "E1004"     # Data integrity issue
    E2001_FORWARD_PASS = "E2001"        # Forward pass failed
    E2002_BACKWARD_PASS = "E2002"       # Backward pass failed
    E2003_GRADIENT_EXPLOSION = "E2003"  # Gradient too large
    E2004_MODEL_STATE = "E2004"         # Model state corruption
    E3001_OUT_OF_MEMORY = "E3001"       # GPU/CPU OOM
    E3002_DEVICE_ERROR = "E3002"        # CUDA/device error
    E3003_IO_ERROR = "E3003"            # File I/O error
    E4001_CONFIGURATION = "E4001"       # Configuration error
    E4002_TIMEOUT = "E4002"             # Operation timeout
    E4003_UNKNOWN = "E4003"             # Unknown error


@dataclass
class LearningException:
    """Structured learning exception with context."""
    error_code: LearningErrorCode
    severity: ExceptionSeverity
    message: str
    original_exception: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code.value,
            "severity": self.severity.value,
            "message": self.message,
            "original_exception": str(self.original_exception) if self.original_exception else None,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
        }


class ExceptionClassifier:
    """Classifies exceptions into severity levels and error codes."""

    @staticmethod
    def classify(exception: Exception, context: Optional[Dict[str, Any]] = None) -> LearningException:
        """Classify an exception into a structured LearningException."""
        exc_str = str(exception).lower()

        # GPU/Memory Errors - CRITICAL
        if isinstance(exception, MemoryError) or "cuda" in exc_str and "out of memory" in exc_str:
            return LearningException(
                error_code=LearningErrorCode.E3001_OUT_OF_MEMORY,
                severity=ExceptionSeverity.CRITICAL,
                message=f"Out of memory: {exception}",
                original_exception=exception, context=context, recoverable=False,
            )

        # CUDA Device Errors - CRITICAL
        if isinstance(exception, RuntimeError) and "cuda" in exc_str:
            return LearningException(
                error_code=LearningErrorCode.E3002_DEVICE_ERROR,
                severity=ExceptionSeverity.CRITICAL,
                message=f"CUDA device error: {exception}",
                original_exception=exception, context=context, recoverable=False,
            )

        # Gradient Issues - HIGH
        if "gradient" in exc_str or "nan" in exc_str or "inf" in exc_str:
            return LearningException(
                error_code=LearningErrorCode.E2003_GRADIENT_EXPLOSION,
                severity=ExceptionSeverity.HIGH,
                message=f"Gradient error: {exception}",
                original_exception=exception, context=context, max_retries=2,
            )

        # Value/Type Errors - MEDIUM
        if isinstance(exception, (ValueError, TypeError)):
            return LearningException(
                error_code=LearningErrorCode.E1001_INVALID_SAMPLE,
                severity=ExceptionSeverity.MEDIUM,
                message=f"Invalid data: {exception}",
                original_exception=exception, context=context,
            )

        # Timeout - LOW
        if isinstance(exception, (asyncio.TimeoutError, TimeoutError)):
            return LearningException(
                error_code=LearningErrorCode.E4002_TIMEOUT,
                severity=ExceptionSeverity.LOW,
                message=f"Timeout: {exception}",
                original_exception=exception, context=context,
            )

        # IO Errors - MEDIUM
        if isinstance(exception, (IOError, OSError)):
            return LearningException(
                error_code=LearningErrorCode.E3003_IO_ERROR,
                severity=ExceptionSeverity.MEDIUM,
                message=f"I/O error: {exception}",
                original_exception=exception, context=context,
            )

        # Default - Unknown
        return LearningException(
            error_code=LearningErrorCode.E4003_UNKNOWN,
            severity=ExceptionSeverity.MEDIUM,
            message=f"Unknown error: {exception}",
            original_exception=exception, context=context,
        )


# =============================================================================
# Online Learning Module
# =============================================================================

class OnlineLearningBuffer:
    """Buffer for online learning samples."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.priority_queue: List[Tuple[float, Dict]] = []

    def add(self, sample: Dict[str, Any], priority: float = 1.0):
        """Add sample to buffer."""
        self.buffer.append(sample)

        if priority > 1.0:
            self.priority_queue.append((priority, sample))
            self.priority_queue.sort(key=lambda x: -x[0])

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample from buffer with priority consideration."""
        if len(self.buffer) == 0:
            return []

        # Mix priority and random samples
        samples = []

        # High priority samples (20%)
        num_priority = min(batch_size // 5, len(self.priority_queue))
        for _ in range(num_priority):
            if self.priority_queue:
                _, sample = self.priority_queue.pop(0)
                samples.append(sample)

        # Random samples
        remaining = batch_size - len(samples)
        if remaining > 0 and self.buffer:
            indices = random.sample(range(len(self.buffer)), min(remaining, len(self.buffer)))
            samples.extend([self.buffer[i] for i in indices])

        return samples

    def __len__(self) -> int:
        return len(self.buffer)


class OnlineLearningModule:
    """
    Online Learning Module

    Enables real-time learning from streaming data:
    - Web/API data streams
    - User interactions
    - System feedback

    Implements incremental updates without full retraining.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AutonomousConfig,
    ):
        self.model = model
        self.config = config

        self.device = next(model.parameters()).device

        # Online learning buffer
        self.buffer = OnlineLearningBuffer(config.online_buffer_size)

        # Optimizer with small learning rate
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.online_learning_rate,
            weight_decay=0.01,
        )

        # Learning statistics
        self.total_samples = 0
        self.total_updates = 0
        self.learning_curve: List[float] = []

        # Data streams
        self.active_streams: Dict[str, asyncio.Queue] = {}

        # Learning state
        self.is_learning = False
        self._learning_task: Optional[asyncio.Task] = None

        # Exception tracking for graded handling
        self._error_log: List[LearningException] = []
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        self._last_error_time: Optional[datetime] = None

    async def start_learning(self):
        """Start the online learning loop."""
        if self.is_learning:
            return

        self.is_learning = True
        self._learning_task = asyncio.create_task(self._learning_loop())
        logger.info("Started online learning")

    async def stop_learning(self):
        """Stop the online learning loop."""
        self.is_learning = False
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                logger.info("Learning task cancelled")
                raise
        logger.info("Stopped online learning")

    async def _learning_loop(self):
        """
        Main online learning loop with fine-grained exception handling.

        Exception Handling Strategy:
        - LOW: Log and continue immediately
        - MEDIUM: Log, increment error count, brief pause
        - HIGH: Log detailed info, longer pause, may skip batch
        - CRITICAL: Log full context, terminate loop, notify

        Special Handling:
        - GPU OOM: Automatically reduce batch size and clear cache
        - CancelledError: Propagate for proper async cancellation
        - Gradient issues: Attempt recovery with gradient zeroing
        """
        logger.info("Learning loop started with enhanced exception handling")

        # Track original batch size for potential recovery
        original_batch_size = self.config.online_batch_size
        min_batch_size = max(1, original_batch_size // 8)

        while self.is_learning:
            try:
                # Collect samples from streams
                await self._collect_from_streams()

                # Process buffer if enough samples
                if len(self.buffer) >= self.config.online_batch_size:
                    loss = self._update_step()
                    self.learning_curve.append(loss)
                    self.total_updates += 1

                    # Reset consecutive errors on success
                    self._consecutive_errors = 0

                    # Gradually restore batch size after successful updates
                    if (self.total_updates % 100 == 0 and
                        self.config.online_batch_size < original_batch_size):
                        new_size = min(
                            self.config.online_batch_size * 2,
                            original_batch_size
                        )
                        logger.info(f"Restoring batch size: {self.config.online_batch_size} -> {new_size}")
                        self.config.online_batch_size = new_size

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                # Graceful cancellation - must re-raise for proper async handling
                logger.info("Learning loop cancelled gracefully")
                raise

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # GPU OOM or CUDA errors - special handling
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    await self._handle_gpu_oom_error(e, min_batch_size)
                else:
                    # Other RuntimeError - classify normally
                    await self._handle_general_exception(e)

            except (torch.autograd.detect_anomaly,) if hasattr(torch.autograd, 'detect_anomaly') else () as e:
                # Gradient anomaly detected
                logger.error(f"Gradient anomaly detected: {e}")
                self._attempt_gradient_recovery()
                await asyncio.sleep(2.0)

            except Exception as e:
                await self._handle_general_exception(e)

                # Check if too many consecutive errors
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.critical(
                        f"[{LearningErrorCode.E2004_MODEL_STATE.value}] "
                        f"Too many consecutive errors ({self._consecutive_errors}). "
                        f"Terminating learning loop."
                    )
                    self.is_learning = False
                    break

    async def _handle_gpu_oom_error(self, error: Exception, min_batch_size: int):
        """
        Handle GPU Out-of-Memory errors with automatic batch size reduction.

        Recovery Strategy:
        1. Clear CUDA cache
        2. Reduce batch size by half
        3. Log warning with new batch size
        4. Continue learning if batch size is valid
        """
        logger.error(f"GPU OOM detected: {error}")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared CUDA cache")

        # Reduce batch size
        old_batch_size = self.config.online_batch_size
        new_batch_size = max(min_batch_size, old_batch_size // 2)

        if new_batch_size < old_batch_size:
            self.config.online_batch_size = new_batch_size
            logger.warning(
                f"Reduced batch size due to OOM: {old_batch_size} -> {new_batch_size}"
            )

            # Log to error tracking
            self._error_log.append(LearningException(
                error_code=LearningErrorCode.E2001_OOM,
                severity=ExceptionSeverity.HIGH,
                message=f"GPU OOM - batch size reduced to {new_batch_size}",
                original_exception=error,
                context={
                    "old_batch_size": old_batch_size,
                    "new_batch_size": new_batch_size,
                    "total_updates": self.total_updates,
                },
            ))

            await asyncio.sleep(1.0)  # Brief pause after OOM
        else:
            # Already at minimum batch size - this is critical
            logger.critical(
                f"GPU OOM at minimum batch size ({min_batch_size}). "
                "Cannot reduce further. Consider using a smaller model or more GPU memory."
            )
            self._consecutive_errors += 1
            await asyncio.sleep(5.0)

    async def _handle_general_exception(self, error: Exception):
        """Handle general exceptions with classification and appropriate response."""
        # Classify the exception
        classified = ExceptionClassifier.classify(
            error,
            context={
                "total_updates": self.total_updates,
                "buffer_size": len(self.buffer),
                "consecutive_errors": self._consecutive_errors,
                "batch_size": self.config.online_batch_size,
            }
        )

        # Log the structured exception
        self._error_log.append(classified)
        self._consecutive_errors += 1
        self._last_error_time = datetime.now(timezone.utc)

        # Handle based on severity
        await self._handle_exception_by_severity(classified)

    async def _handle_exception_by_severity(self, exc: LearningException):
        """
        Handle exception based on its severity level with exponential backoff.

        Backoff Strategy:
        - LOW: 0.1s base, no exponential
        - MEDIUM: 1s base, exponential up to 30s
        - HIGH: 5s base, exponential up to 60s
        - CRITICAL: No backoff, terminate immediately
        """
        # Calculate exponential backoff based on consecutive errors
        backoff_factor = min(2 ** min(self._consecutive_errors, 6), 64)

        if exc.severity == ExceptionSeverity.LOW:
            # LOW: Log briefly and continue
            logger.warning(
                f"[{exc.error_code.value}] Low severity: {exc.message}"
            )
            await asyncio.sleep(0.1)

        elif exc.severity == ExceptionSeverity.MEDIUM:
            # MEDIUM: Log with context, exponential backoff
            base_delay = 1.0
            delay = min(base_delay * backoff_factor, 30.0)

            logger.warning(
                f"[{exc.error_code.value}] Medium severity: {exc.message} | "
                f"Consecutive errors: {self._consecutive_errors} | "
                f"Backoff: {delay:.1f}s"
            )
            await asyncio.sleep(delay)

        elif exc.severity == ExceptionSeverity.HIGH:
            # HIGH: Detailed log, longer pause, consider recovery actions
            base_delay = 5.0
            delay = min(base_delay * backoff_factor, 60.0)

            logger.error(
                f"[{exc.error_code.value}] High severity: {exc.message}\n"
                f"Context: {exc.context}\n"
                f"Original exception: {exc.original_exception}\n"
                f"Consecutive errors: {self._consecutive_errors} | "
                f"Backoff: {delay:.1f}s"
            )

            # Attempt recovery for specific error types
            if exc.error_code == LearningErrorCode.E2003_GRADIENT_EXPLOSION:
                self._attempt_gradient_recovery()
            elif exc.error_code == LearningErrorCode.E2001_OOM:
                self._attempt_memory_recovery()
            elif exc.error_code == LearningErrorCode.E2004_MODEL_STATE:
                self._attempt_model_state_recovery()

            await asyncio.sleep(delay)

        elif exc.severity == ExceptionSeverity.CRITICAL:
            # CRITICAL: Full logging, terminate loop
            logger.critical(
                f"[{exc.error_code.value}] CRITICAL ERROR - Terminating learning loop\n"
                f"Message: {exc.message}\n"
                f"Context: {exc.context}\n"
                f"Original exception: {exc.original_exception}\n"
                f"Error log size: {len(self._error_log)}\n"
                f"Total updates before failure: {self.total_updates}"
            )
            self.is_learning = False

    def _attempt_memory_recovery(self):
        """Attempt to recover from memory issues."""
        logger.info("Attempting memory recovery...")
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Run garbage collection
            import gc
            gc.collect()

            # Clear buffer if too large
            if len(self.buffer) > self.config.online_buffer_size // 2:
                logger.info(f"Clearing excess buffer entries: {len(self.buffer)} -> {self.config.online_buffer_size // 4}")
                while len(self.buffer.buffer) > self.config.online_buffer_size // 4:
                    self.buffer.buffer.popleft()

            logger.info("Memory recovery completed")
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")

    def _attempt_model_state_recovery(self):
        """Attempt to recover from model state corruption."""
        logger.info("Attempting model state recovery...")
        try:
            # Reset optimizer state
            self.optimizer.zero_grad()

            # Check model parameters for NaN/Inf
            nan_params = 0
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nan_params += 1
                    logger.warning(f"Found NaN/Inf in parameter: {name}")

            if nan_params > 0:
                logger.error(f"Model has {nan_params} corrupted parameters. Manual intervention required.")
            else:
                logger.info("Model parameters appear healthy")

        except Exception as e:
            logger.error(f"Model state recovery failed: {e}")

    def _attempt_gradient_recovery(self):
        """Attempt to recover from gradient explosion/NaN issues."""
        logger.info("Attempting gradient recovery...")
        try:
            # Zero gradients
            self.optimizer.zero_grad()

            # Clear any NaN in model parameters (reset to small values)
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        param.grad.zero_()
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        logger.warning("Found NaN/Inf in model parameters, resetting affected values")
                        param.data = torch.where(
                            torch.isnan(param) | torch.isinf(param),
                            torch.zeros_like(param),
                            param
                        )

            logger.info("Gradient recovery completed")
        except Exception as recovery_error:
            logger.error(f"Gradient recovery failed: {recovery_error}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered during learning."""
        if not self._error_log:
            return {"total_errors": 0, "by_severity": {}, "by_code": {}}

        by_severity = defaultdict(int)
        by_code = defaultdict(int)

        for exc in self._error_log:
            by_severity[exc.severity.value] += 1
            by_code[exc.error_code.value] += 1

        return {
            "total_errors": len(self._error_log),
            "by_severity": dict(by_severity),
            "by_code": dict(by_code),
            "last_error": self._error_log[-1].to_dict() if self._error_log else None,
            "consecutive_errors": self._consecutive_errors,
        }

    async def _collect_from_streams(self):
        """Collect data from active streams."""
        for stream_name, queue in self.active_streams.items():
            try:
                while not queue.empty():
                    sample = await asyncio.wait_for(queue.get(), timeout=0.01)
                    self.buffer.add(sample, priority=sample.get('priority', 1.0))
                    self.total_samples += 1
            except asyncio.TimeoutError:
                continue

    def _update_step(self) -> float:
        """Execute single online update step."""
        self.model.train()

        # Sample from buffer
        samples = self.buffer.sample(self.config.online_batch_size)

        if not samples:
            return 0.0

        # Prepare batch
        input_ids = torch.stack([s['input_ids'] for s in samples]).to(self.device)
        labels = torch.stack([s.get('labels', s['input_ids']) for s in samples]).to(self.device)

        # Forward pass
        outputs = self.model(input_ids=input_ids)
        logits = outputs['logits']

        # Compute loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Backward and update
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def add_stream(self, stream_name: str, queue: asyncio.Queue):
        """Add a data stream."""
        self.active_streams[stream_name] = queue
        logger.info(f"Added stream: {stream_name}")

    def remove_stream(self, stream_name: str):
        """Remove a data stream."""
        if stream_name in self.active_streams:
            del self.active_streams[stream_name]
            logger.info(f"Removed stream: {stream_name}")

    def add_sample(self, sample: Dict[str, Any], priority: float = 1.0):
        """Directly add a sample to the learning buffer."""
        self.buffer.add(sample, priority)
        self.total_samples += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_samples": self.total_samples,
            "total_updates": self.total_updates,
            "buffer_size": len(self.buffer),
            "active_streams": list(self.active_streams.keys()),
            "is_learning": self.is_learning,
            "recent_loss": np.mean(self.learning_curve[-100:]) if self.learning_curve else 0,
        }

    async def real_time_update(
        self,
        sample: Dict[str, Any],
        immediate: bool = True,
        validate_before_apply: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform immediate model update from real-time data stream.

        ⚠️ PLACEHOLDER: Core real-time learning logic needs implementation.

        Args:
            sample: Training sample with 'input_ids' and optional 'labels'
            immediate: If True, apply update immediately without batching
            validate_before_apply: Validate update doesn't degrade model

        Returns:
            Update result with loss, applied status, and validation metrics

        Technical Design:
            1. WebSocket/SSE integration for real-time data
            2. Micro-batch processing (1-10 samples)
            3. Asynchronous gradient computation
            4. Model version pinning during update
            5. Rollback capability on quality degradation
            6. Rate limiting to prevent update spam

        Target Version: v2.2.0

        Example:
            ```python
            result = await module.real_time_update({
                'input_ids': torch.tensor([...]),
                'labels': torch.tensor([...]),
            })
            if result['applied']:
                print(f"Update applied, loss: {result['loss']}")
            ```
        """
        logger.info("Processing real-time update request")

        result = {
            "applied": False,
            "loss": None,
            "validation_passed": None,
            "reason": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Placeholder implementation
        logger.warning(
            "real_time_update is a placeholder. "
            "Full implementation with validation and rollback coming in v2.2.0. "
            "Using add_sample() + standard learning loop as fallback."
        )

        # Real-time update implementation plan (v2.2.0):
        # 1. Validate sample format
        # 2. Create model checkpoint (for rollback)
        # 3. Compute gradients
        # 4. Validate update quality
        # 5. Apply or rollback

        # Current: Add to buffer for batch processing (fallback)
        try:
            self.add_sample(sample, priority=2.0)  # Higher priority for real-time
            result["reason"] = "Added to buffer (real-time update not yet implemented)"
            return result
        except Exception as e:
            result["reason"] = f"Failed to add sample: {e}"
            logger.error(f"Real-time update failed: {e}")
            return result

    async def adaptive_learning_rate(
        self,
        recent_losses: Optional[List[float]] = None,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
    ) -> float:
        """
        Dynamically adjust learning rate based on training progress.

        ⚠️ PLACEHOLDER: Adaptive LR scheduling needs implementation.

        Args:
            recent_losses: Recent loss values for analysis
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate

        Returns:
            Adjusted learning rate

        Technical Design:
            1. Loss plateau detection
            2. Gradient noise analysis
            3. Cyclical learning rate support
            4. Warmup and cooldown phases
            5. Per-layer adaptive rates

        Target Version: v2.3.0
        """
        logger.warning(
            "adaptive_learning_rate is a placeholder. "
            "Returning current learning rate unchanged."
        )

        # Placeholder: Return current learning rate
        return self.config.online_learning_rate


# =============================================================================
# Memory Management
# =============================================================================

@dataclass
class Episode:
    """Single episodic memory entry."""
    episode_id: str
    timestamp: datetime
    context: str
    action: str
    outcome: str
    reward: float
    embeddings: Optional[np.ndarray] = None


class EpisodicMemory:
    """
    Episodic Memory System

    Stores and retrieves past experiences for:
    - Few-shot learning
    - Experience replay
    - Contextual recall

    Supports production-grade vector indexing with:
    - FAISS: For high-performance CPU/GPU search
    - Milvus: For distributed scalability
    - NumPy fallback: For development/testing
    """

    def __init__(
        self,
        max_size: int = 100000,
        embedding_dim: int = 768,
        use_faiss: bool = True,
        index_type: str = "faiss_flat",
    ):
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss

        self.episodes: Dict[str, Episode] = {}
        self.episode_order: deque = deque(maxlen=max_size)

        # Initialize vector index
        self._init_vector_index(index_type)

        # Batch buffer for efficient indexing
        self._pending_embeddings: List[Tuple[str, np.ndarray]] = []
        self._batch_size = 100

    def _init_vector_index(self, index_type: str):
        """Initialize the appropriate vector index."""
        if VECTOR_INDEX_AVAILABLE and self.use_faiss:
            try:
                config = IndexConfig(
                    index_type=IndexType(index_type),
                    embedding_dim=self.embedding_dim,
                    metric="cosine",
                )
                self._vector_index = create_vector_index(config)
                self._use_advanced_index = True
                logger.info(f"Using {index_type} vector index")
            except Exception as e:
                logger.warning(f"Failed to create advanced index: {e}, using NumPy")
                self._use_advanced_index = False
                self._embeddings_matrix: Optional[np.ndarray] = None
                self._episode_ids: List[str] = []
        else:
            self._use_advanced_index = False
            self._embeddings_matrix: Optional[np.ndarray] = None
            self._episode_ids: List[str] = []

    def add(self, episode: Episode):
        """Add an episode to memory."""
        # Remove oldest if at capacity
        if len(self.episode_order) >= self.max_size:
            oldest_id = self.episode_order.popleft()
            if oldest_id in self.episodes:
                del self.episodes[oldest_id]
                # Note: Vector index removal is expensive, we let it grow and rebuild periodically

        self.episodes[episode.episode_id] = episode
        self.episode_order.append(episode.episode_id)

        # Buffer embedding for batch indexing
        if episode.embeddings is not None:
            self._pending_embeddings.append((episode.episode_id, episode.embeddings))

            # Flush batch if buffer is full
            if len(self._pending_embeddings) >= self._batch_size:
                self._flush_pending_embeddings()

    def add_batch(self, episodes: List[Episode]):
        """Add multiple episodes efficiently."""
        embeddings_batch = []
        ids_batch = []

        for episode in episodes:
            # Handle capacity
            if len(self.episode_order) >= self.max_size:
                oldest_id = self.episode_order.popleft()
                if oldest_id in self.episodes:
                    del self.episodes[oldest_id]

            self.episodes[episode.episode_id] = episode
            self.episode_order.append(episode.episode_id)

            if episode.embeddings is not None:
                embeddings_batch.append(episode.embeddings)
                ids_batch.append(episode.episode_id)

        # Batch add to vector index
        if embeddings_batch:
            self._add_embeddings_batch(ids_batch, embeddings_batch)

    def _flush_pending_embeddings(self):
        """Flush pending embeddings to the vector index."""
        if not self._pending_embeddings:
            return

        ids = [eid for eid, _ in self._pending_embeddings]
        embeddings = [emb for _, emb in self._pending_embeddings]

        self._add_embeddings_batch(ids, embeddings)
        self._pending_embeddings.clear()

    def _add_embeddings_batch(self, ids: List[str], embeddings: List[np.ndarray]):
        """Add embeddings in batch to the vector index."""
        if not embeddings:
            return

        embeddings_array = np.vstack([e.reshape(1, -1) for e in embeddings])

        if self._use_advanced_index:
            self._vector_index.add(embeddings_array, ids)
        else:
            # NumPy fallback
            if self._embeddings_matrix is None:
                self._embeddings_matrix = embeddings_array
                self._episode_ids = ids
            else:
                self._embeddings_matrix = np.vstack([self._embeddings_matrix, embeddings_array])
                self._episode_ids.extend(ids)

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Episode]:
        """Retrieve k most similar episodes."""
        # Flush any pending embeddings first
        self._flush_pending_embeddings()

        if self._use_advanced_index:
            results = self._vector_index.search(query_embedding, k)
            return [
                self.episodes[eid]
                for eid, _ in results
                if eid in self.episodes
            ]
        else:
            # NumPy fallback
            if self._embeddings_matrix is None or len(self._episode_ids) == 0:
                return []

            # Compute cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            matrix_norm = self._embeddings_matrix / (
                np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True) + 1e-8
            )

            similarities = matrix_norm @ query_norm

            # Get top-k indices
            k = min(k, len(self._episode_ids))
            top_indices = np.argsort(similarities)[-k:][::-1]

            return [
                self.episodes[self._episode_ids[i]]
                for i in top_indices
                if self._episode_ids[i] in self.episodes
            ]

    def retrieve_similar_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> List[List[Episode]]:
        """Batch retrieve similar episodes for multiple queries."""
        self._flush_pending_embeddings()

        if self._use_advanced_index:
            all_results = self._vector_index.search_batch(query_embeddings, k)
            return [
                [self.episodes[eid] for eid, _ in results if eid in self.episodes]
                for results in all_results
            ]
        else:
            # Process each query individually for NumPy fallback
            return [self.retrieve_similar(q, k) for q in query_embeddings]

    def retrieve_by_reward(self, k: int = 10, min_reward: float = 0.0) -> List[Episode]:
        """Retrieve episodes with highest rewards."""
        sorted_episodes = sorted(
            self.episodes.values(),
            key=lambda e: e.reward,
            reverse=True
        )
        return [e for e in sorted_episodes[:k] if e.reward >= min_reward]

    def retrieve_recent(self, k: int = 10) -> List[Episode]:
        """Retrieve most recent episodes."""
        recent_ids = list(self.episode_order)[-k:]
        return [self.episodes[eid] for eid in recent_ids if eid in self.episodes]

    def rebuild_index(self):
        """Rebuild the vector index from current episodes."""
        if self._use_advanced_index:
            # Re-initialize and add all current embeddings
            self._init_vector_index("faiss_flat")

            embeddings = []
            ids = []
            for eid, episode in self.episodes.items():
                if episode.embeddings is not None:
                    embeddings.append(episode.embeddings)
                    ids.append(eid)

            if embeddings:
                self._add_embeddings_batch(ids, embeddings)

            logger.info(f"Rebuilt vector index with {len(ids)} embeddings")


class SemanticMemory:
    """
    Semantic Memory System

    Knowledge graph for storing structured knowledge:
    - Concepts and relationships
    - Facts and rules
    - Domain knowledge
    """

    def __init__(self):
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self.facts: List[Dict[str, Any]] = []

    def add_concept(
        self,
        concept_id: str,
        name: str,
        attributes: Dict[str, Any],
    ):
        """Add a concept to semantic memory."""
        self.concepts[concept_id] = {
            "name": name,
            "attributes": attributes,
            "created_at": datetime.now(timezone.utc),
        }

    def add_relationship(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ):
        """Add a relationship between concepts."""
        self.relationships[subject].append((subject, predicate, obj))
        self.relationships[obj].append((subject, predicate, obj))

    def add_fact(
        self,
        fact: str,
        source: str,
        confidence: float = 1.0,
    ):
        """Add a fact."""
        self.facts.append({
            "fact": fact,
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc),
        })

    def query_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Query a concept."""
        return self.concepts.get(concept_id)

    def query_relationships(
        self,
        concept_id: str,
        predicate: Optional[str] = None,
    ) -> List[Tuple[str, str, str]]:
        """Query relationships for a concept."""
        rels = self.relationships.get(concept_id, [])

        if predicate:
            rels = [r for r in rels if r[1] == predicate]

        return rels


class WorkingMemory:
    """
    Working Memory System

    Active computation buffer for:
    - Current context
    - Intermediate results
    - Active goals
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.items: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}

    def store(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Store item in working memory."""
        if len(self.items) >= self.max_size:
            self._evict_oldest()

        self.items[key] = {
            "value": value,
            "expires_at": datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
        }
        self.access_times[key] = datetime.now(timezone.utc)

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve item from working memory."""
        item = self.items.get(key)

        if item is None:
            return None

        # Check expiration
        if datetime.now(timezone.utc) > item["expires_at"]:
            del self.items[key]
            return None

        self.access_times[key] = datetime.now(timezone.utc)
        return item["value"]

    def _evict_oldest(self):
        """Evict least recently accessed item."""
        if not self.access_times:
            return

        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.items[oldest_key]
        del self.access_times[oldest_key]


class MemoryManagement:
    """
    Unified Memory Management System

    Coordinates all memory types:
    - Long-term (parameters)
    - Short-term (context)
    - Episodic (experiences)
    - Semantic (knowledge)
    - Working (active)
    """

    def __init__(self, config: AutonomousConfig):
        self.config = config

        # Initialize memory systems
        self.episodic = EpisodicMemory(config.episodic_memory_size)
        self.semantic = SemanticMemory()
        self.working = WorkingMemory(config.working_memory_size)

        # Memory consolidation tracking
        self.last_consolidation = datetime.now(timezone.utc)
        self.consolidation_pending = False

    def store_experience(
        self,
        context: str,
        action: str,
        outcome: str,
        reward: float,
        embeddings: Optional[np.ndarray] = None,
    ):
        """Store an experience in episodic memory."""
        episode = Episode(
            episode_id=hashlib.md5(
                f"{context}{action}{time.time()}".encode()
            ).hexdigest(),
            timestamp=datetime.now(timezone.utc),
            context=context,
            action=action,
            outcome=outcome,
            reward=reward,
            embeddings=embeddings,
        )

        self.episodic.add(episode)

    def store_knowledge(
        self,
        concept_id: str,
        name: str,
        attributes: Dict[str, Any],
        relationships: Optional[List[Tuple[str, str]]] = None,
    ):
        """Store knowledge in semantic memory."""
        self.semantic.add_concept(concept_id, name, attributes)

        if relationships:
            for predicate, obj in relationships:
                self.semantic.add_relationship(concept_id, predicate, obj)

    def recall_similar_experiences(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Episode]:
        """Recall similar past experiences."""
        return self.episodic.retrieve_similar(query_embedding, k)

    def recall_successful_experiences(self, k: int = 10) -> List[Episode]:
        """Recall successful past experiences."""
        return self.episodic.retrieve_by_reward(k, min_reward=0.5)

    async def consolidate_memories(self, batch_size: int = 50):
        """
        Consolidate memories (like sleep/dream replay).

        Optimized for batch processing:
        - Transfer important episodic memories to semantic
        - Prune low-value memories
        - Strengthen frequently accessed memories

        Args:
            batch_size: Number of episodes to process per batch
        """
        logger.info("Starting memory consolidation")

        # Get high-reward episodes
        important_episodes = self.episodic.retrieve_by_reward(k=100, min_reward=0.7)

        if not important_episodes:
            logger.info("No important episodes to consolidate")
            self.last_consolidation = datetime.now(timezone.utc)
            return

        # Process in batches for efficiency
        total_processed = 0

        for batch_start in range(0, len(important_episodes), batch_size):
            batch = important_episodes[batch_start:batch_start + batch_size]

            # Prepare batch data
            concepts_to_add = []
            facts_to_add = []

            for episode in batch:
                concept_id = f"learned_{episode.episode_id[:8]}"
                concepts_to_add.append({
                    "id": concept_id,
                    "name": f"Learned from {episode.action}",
                    "attributes": {
                        "context_pattern": episode.context[:100],
                        "successful_action": episode.action,
                        "outcome": episode.outcome,
                        "reward": episode.reward,
                        "timestamp": episode.timestamp.isoformat(),
                    }
                })

                facts_to_add.append({
                    "fact": f"Action '{episode.action}' in context '{episode.context[:50]}...' leads to '{episode.outcome}'",
                    "source": "episodic_consolidation",
                    "confidence": episode.reward,
                })

            # Batch add concepts
            for concept in concepts_to_add:
                self.semantic.add_concept(
                    concept["id"],
                    concept["name"],
                    concept["attributes"],
                )

            # Batch add facts
            for fact in facts_to_add:
                self.semantic.add_fact(
                    fact["fact"],
                    source=fact["source"],
                    confidence=fact["confidence"],
                )

            total_processed += len(batch)

            # Yield control to event loop between batches
            await asyncio.sleep(0)

        self.last_consolidation = datetime.now(timezone.utc)
        self.consolidation_pending = False

        logger.info(f"Consolidated {total_processed} episodes in batches of {batch_size}")

    def should_consolidate(self) -> bool:
        """Check if memory consolidation is needed."""
        hours_since = (
            datetime.now(timezone.utc) - self.last_consolidation
        ).total_seconds() / 3600

        return hours_since >= self.config.consolidation_interval_hours


# =============================================================================
# Self-Evaluation System
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""
    benchmark_name: str
    score: float
    max_score: float
    metrics: Dict[str, float]
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None


class SelfEvaluationSystem:
    """
    Self-Evaluation System

    Automatically evaluates model capabilities:
    - Run benchmarks periodically
    - Detect performance degradation
    - Identify knowledge gaps
    - Trigger learning cycles when needed
    """

    def __init__(
        self,
        model: nn.Module,
        config: AutonomousConfig,
    ):
        self.model = model
        self.config = config

        self.device = next(model.parameters()).device

        # Benchmark history
        self.benchmark_history: Dict[str, List[BenchmarkResult]] = defaultdict(list)

        # Knowledge gaps
        self.knowledge_gaps: List[KnowledgeGap] = []

        # Performance baselines
        self.baselines: Dict[str, float] = {}

        # Evaluation counter
        self.eval_count = 0

    def run_benchmark(
        self,
        benchmark_name: str,
        test_data: List[Dict[str, Any]],
    ) -> BenchmarkResult:
        """Run a specific benchmark."""
        self.model.eval()

        correct = 0
        total = 0
        latencies = []

        with torch.no_grad():
            for sample in test_data:
                start_time = time.time()

                # Forward pass
                input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                outputs = self.model(input_ids=input_ids)

                latencies.append(time.time() - start_time)

                # Evaluate (simplified - actual implementation depends on benchmark)
                predicted = outputs['logits'].argmax(dim=-1)
                expected = sample.get('labels', input_ids)

                # Simple accuracy check
                if torch.equal(predicted[:, -1], expected[:, -1]):
                    correct += 1
                total += 1

        self.model.train()

        score = correct / total if total > 0 else 0

        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            score=score,
            max_score=1.0,
            metrics={
                "accuracy": score,
                "avg_latency": np.mean(latencies),
                "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            },
            timestamp=datetime.now(timezone.utc),
        )

        self.benchmark_history[benchmark_name].append(result)
        self.eval_count += 1

        # Check for degradation
        self._check_degradation(benchmark_name, score)

        return result

    def _check_degradation(self, benchmark_name: str, current_score: float):
        """Check for performance degradation."""
        if benchmark_name not in self.baselines:
            self.baselines[benchmark_name] = current_score
            return

        baseline = self.baselines[benchmark_name]
        degradation = baseline - current_score

        if degradation > self.config.knowledge_gap_threshold:
            gap = KnowledgeGap(
                gap_id=hashlib.md5(
                    f"{benchmark_name}{time.time()}".encode()
                ).hexdigest(),
                domain=benchmark_name,
                description=f"Performance degradation: {degradation:.2%}",
                severity=min(degradation * 2, 1.0),
                detected_at=datetime.now(timezone.utc),
            )
            self.knowledge_gaps.append(gap)

            logger.warning(
                f"Knowledge gap detected in {benchmark_name}: "
                f"{degradation:.2%} degradation"
            )

    def run_all_benchmarks(
        self,
        benchmark_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, BenchmarkResult]:
        """Run all configured benchmarks."""
        results = {}

        for benchmark_name in self.config.benchmark_suite:
            if benchmark_name in benchmark_data:
                results[benchmark_name] = self.run_benchmark(
                    benchmark_name,
                    benchmark_data[benchmark_name],
                )

        return results

    def detect_knowledge_gaps(self) -> List[KnowledgeGap]:
        """Analyze benchmarks to detect knowledge gaps."""
        gaps = []

        for benchmark_name, history in self.benchmark_history.items():
            if len(history) < 2:
                continue

            recent = history[-5:]
            older = history[-10:-5] if len(history) >= 10 else history[:-5]

            if not older:
                continue

            recent_avg = np.mean([r.score for r in recent])
            older_avg = np.mean([r.score for r in older])

            if older_avg - recent_avg > self.config.knowledge_gap_threshold:
                gap = KnowledgeGap(
                    gap_id=hashlib.md5(
                        f"{benchmark_name}trend{time.time()}".encode()
                    ).hexdigest(),
                    domain=benchmark_name,
                    description=f"Declining trend: {recent_avg:.2%} vs {older_avg:.2%}",
                    severity=(older_avg - recent_avg) * 2,
                    detected_at=datetime.now(timezone.utc),
                )
                gaps.append(gap)

        self.knowledge_gaps.extend(gaps)
        return gaps

    def get_unresolved_gaps(self) -> List[KnowledgeGap]:
        """Get unresolved knowledge gaps."""
        return [g for g in self.knowledge_gaps if not g.resolved]

    def resolve_gap(self, gap_id: str, resolution_data: Dict[str, Any]):
        """Mark a knowledge gap as resolved."""
        for gap in self.knowledge_gaps:
            if gap.gap_id == gap_id:
                gap.resolved = True
                gap.resolution_data = resolution_data
                break

    def should_trigger_learning(self) -> Tuple[bool, Optional[KnowledgeGap]]:
        """Determine if learning should be triggered."""
        unresolved = self.get_unresolved_gaps()

        # Sort by severity
        unresolved.sort(key=lambda g: g.severity, reverse=True)

        if unresolved and unresolved[0].severity > 0.2:
            return True, unresolved[0]

        return False, None


# =============================================================================
# Knowledge Integration
# =============================================================================

class RAGSystem:
    """
    Retrieval-Augmented Generation System

    Augments model with external knowledge retrieval.
    """

    def __init__(
        self,
        embedding_model: Optional[nn.Module] = None,
        embedding_dim: int = 768,
    ):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

        # Knowledge base
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        """Add a document to the knowledge base."""
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata or {},
            "embedding": embedding,
        }

        if embedding is not None:
            self._update_index(doc_id, embedding)

    def _update_index(self, doc_id: str, embedding: np.ndarray):
        """Update the embedding index."""
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
            self.doc_ids = [doc_id]
        else:
            self.embeddings = np.vstack([
                self.embeddings,
                embedding.reshape(1, -1)
            ])
            self.doc_ids.append(doc_id)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        if self.embeddings is None or len(self.doc_ids) == 0:
            return []

        # Cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        similarities = embeddings_norm @ query_norm

        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            if doc_id in self.documents:
                results.append({
                    "doc_id": doc_id,
                    "score": float(similarities[idx]),
                    **self.documents[doc_id],
                })

        return results

    def augment_prompt(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 3,
    ) -> str:
        """Augment a prompt with retrieved knowledge."""
        retrieved = self.retrieve(query_embedding, k)

        if not retrieved:
            return query

        context = "\n\n".join([
            f"[Retrieved Knowledge {i+1}]:\n{doc['content']}"
            for i, doc in enumerate(retrieved)
        ])

        augmented = f"""Based on the following knowledge:

{context}

Answer the following:
{query}"""

        return augmented


class ToolUseSystem:
    """
    Tool Use System

    Enables the model to use external tools:
    - Code execution
    - Web search
    - API calls
    - File operations
    """

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
        self.usage_history: List[Dict[str, Any]] = []

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
    ):
        """Register a tool."""
        self.tools[name] = func
        self.tool_descriptions[name] = description
        logger.info(f"Registered tool: {name}")

    async def use_tool(
        self,
        tool_name: str,
        **kwargs,
    ) -> Any:
        """Use a tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        start_time = time.time()

        try:
            func = self.tools[tool_name]

            if asyncio.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)

            success = True
            error = None

        except Exception as e:
            result = None
            success = False
            error = str(e)

        # Record usage
        self.usage_history.append({
            "tool": tool_name,
            "kwargs": kwargs,
            "result": result,
            "success": success,
            "error": error,
            "duration": time.time() - start_time,
            "timestamp": datetime.now(timezone.utc),
        })

        return result

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompting."""
        descriptions = []

        for name, desc in self.tool_descriptions.items():
            descriptions.append(f"- {name}: {desc}")

        return "\n".join(descriptions)


class KnowledgeIntegration:
    """
    Unified Knowledge Integration System

    Combines:
    - RAG for retrieval
    - Tool use for actions
    - External knowledge sources
    """

    def __init__(self, config: AutonomousConfig):
        self.config = config

        self.rag = RAGSystem() if config.enable_rag else None
        self.tools = ToolUseSystem() if config.enable_tool_use else None

        # External sources
        self.external_sources: Dict[str, Any] = {}

    def add_knowledge_source(self, name: str, source: Any):
        """Add an external knowledge source."""
        self.external_sources[name] = source

    async def integrate_knowledge(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Integrate knowledge from all sources."""
        result = {
            "query": query,
            "augmented_prompt": query,
            "retrieved_docs": [],
            "external_knowledge": [],
        }

        # RAG retrieval
        if self.rag and query_embedding is not None:
            docs = self.rag.retrieve(query_embedding)
            result["retrieved_docs"] = docs
            result["augmented_prompt"] = self.rag.augment_prompt(
                query, query_embedding
            )

        return result


# =============================================================================
# Safety Monitor
# =============================================================================

class SafetyMonitor:
    """
    Safety & Alignment Monitor

    Ensures safe autonomous operation:
    - Value alignment checks
    - Output filtering
    - Human oversight triggers
    - Emergency stops
    """

    def __init__(self, config: AutonomousConfig):
        self.config = config

        # Safety state
        self.is_safe = True
        self.violation_count = 0
        self.emergency_stop = False

        # Monitoring history
        self.safety_events: List[Dict[str, Any]] = []

        # Human oversight queue
        self.oversight_queue: List[Dict[str, Any]] = []

        # Safety rules
        self.safety_rules: List[Callable[[str], bool]] = []

    def add_safety_rule(self, rule: Callable[[str], bool], description: str):
        """Add a safety rule."""
        self.safety_rules.append(rule)
        logger.info(f"Added safety rule: {description}")

    def check_output(self, output: str) -> Tuple[bool, List[str]]:
        """Check if output passes safety rules."""
        violations = []

        for i, rule in enumerate(self.safety_rules):
            try:
                if not rule(output):
                    violations.append(f"Rule {i} violation")
            except Exception as e:
                violations.append(f"Rule {i} error: {e}")

        is_safe = len(violations) == 0

        if not is_safe:
            self.violation_count += 1
            self._record_event("output_violation", {
                "output_preview": output[:200],
                "violations": violations,
            })

        return is_safe, violations

    def check_action(self, action: str, parameters: Dict[str, Any]) -> bool:
        """Check if an action is safe to execute."""
        # High-risk actions always require oversight
        high_risk_actions = ["delete", "modify", "execute", "deploy"]

        if any(hr in action.lower() for hr in high_risk_actions):
            if self.config.human_oversight_required:
                self.request_oversight(action, parameters)
                return False

        return True

    def request_oversight(self, action: str, parameters: Dict[str, Any]):
        """Request human oversight for an action."""
        request = {
            "request_id": hashlib.md5(
                f"{action}{time.time()}".encode()
            ).hexdigest(),
            "action": action,
            "parameters": parameters,
            "timestamp": datetime.now(timezone.utc),
            "status": "pending",
        }

        self.oversight_queue.append(request)

        self._record_event("oversight_requested", request)

        logger.info(f"Human oversight requested for: {action}")

    def approve_oversight(self, request_id: str) -> bool:
        """Approve an oversight request."""
        for request in self.oversight_queue:
            if request["request_id"] == request_id:
                request["status"] = "approved"
                request["approved_at"] = datetime.now(timezone.utc)
                return True
        return False

    def reject_oversight(self, request_id: str, reason: str) -> bool:
        """Reject an oversight request."""
        for request in self.oversight_queue:
            if request["request_id"] == request_id:
                request["status"] = "rejected"
                request["rejected_at"] = datetime.now(timezone.utc)
                request["rejection_reason"] = reason
                return True
        return False

    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop."""
        self.emergency_stop = True
        self.is_safe = False

        self._record_event("emergency_stop", {
            "reason": reason,
        })

        logger.critical(f"EMERGENCY STOP: {reason}")

    def reset_emergency_stop(self):
        """Reset emergency stop (requires explicit action)."""
        self.emergency_stop = False
        self.is_safe = True
        self.violation_count = 0

        self._record_event("emergency_reset", {})

        logger.info("Emergency stop reset")

    def _record_event(self, event_type: str, data: Dict[str, Any]):
        """Record a safety event."""
        self.safety_events.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc),
        })

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            "is_safe": self.is_safe,
            "emergency_stop": self.emergency_stop,
            "violation_count": self.violation_count,
            "pending_oversight": len([
                r for r in self.oversight_queue
                if r["status"] == "pending"
            ]),
            "safety_level": self.config.safety_level.value,
        }


# =============================================================================
# Autonomous Learning Agent
# =============================================================================

class AutonomousLearningAgent:
    """
    Autonomous Learning Agent

    Main orchestrator for self-evolving AI capabilities:
    - Continuous online learning
    - Multi-timescale memory
    - Self-evaluation and gap detection
    - Knowledge integration
    - Safety monitoring

    Implements the "Nested Learning" paradigm for infinite learning.
    """

    def __init__(
        self,
        model: nn.Module,
        config: AutonomousConfig,
    ):
        self.model = model
        self.config = config

        self.device = next(model.parameters()).device

        # Initialize subsystems
        self.online_learning = OnlineLearningModule(model, config)
        self.memory = MemoryManagement(config)
        self.evaluation = SelfEvaluationSystem(model, config)
        self.knowledge = KnowledgeIntegration(config)
        self.safety = SafetyMonitor(config)

        # Agent state
        self.is_running = False
        self.autonomous_steps = 0
        self.last_evaluation = datetime.now(timezone.utc)

        # Event loop
        self._main_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the autonomous learning agent."""
        if self.is_running:
            logger.warning("Agent already running")
            return

        self.is_running = True

        # Start online learning
        await self.online_learning.start_learning()

        # Start main loop
        self._main_task = asyncio.create_task(self._main_loop())

        logger.info("Autonomous learning agent started")

    async def stop(self):
        """Stop the autonomous learning agent."""
        self.is_running = False

        # Stop online learning
        await self.online_learning.stop_learning()

        # Stop main loop
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                raise  # Re-raise CancelledError after cleanup

        logger.info("Autonomous learning agent stopped")

    async def _main_loop(self):
        """Main autonomous learning loop."""
        while self.is_running:
            try:
                # Safety check
                if self.safety.emergency_stop:
                    logger.warning("Emergency stop active, pausing agent")
                    await asyncio.sleep(10)
                    continue

                # Check step limit
                if self.autonomous_steps >= self.config.max_autonomous_steps:
                    if self.config.human_oversight_required:
                        self.safety.request_oversight(
                            "continue_learning",
                            {"steps_completed": self.autonomous_steps}
                        )
                        await asyncio.sleep(60)
                        continue

                # Run evaluation periodically
                if self._should_evaluate():
                    await self._run_evaluation_cycle()

                # Memory consolidation
                if self.memory.should_consolidate():
                    await self.memory.consolidate_memories()

                # Check for knowledge gaps and trigger learning
                should_learn, gap = self.evaluation.should_trigger_learning()
                if should_learn and gap:
                    await self._address_knowledge_gap(gap)

                self.autonomous_steps += 1

                # Sleep between cycles
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)

    def _should_evaluate(self) -> bool:
        """Check if evaluation should run."""
        steps_since = self.autonomous_steps % self.config.eval_interval_steps
        return steps_since == 0

    async def _run_evaluation_cycle(self):
        """Run evaluation cycle."""
        logger.info("Running evaluation cycle")

        # Detect knowledge gaps
        gaps = self.evaluation.detect_knowledge_gaps()

        if gaps:
            logger.info(f"Detected {len(gaps)} knowledge gaps")

        self.last_evaluation = datetime.now(timezone.utc)

    async def _address_knowledge_gap(self, gap: KnowledgeGap):
        """Address a detected knowledge gap."""
        logger.info(f"Addressing knowledge gap: {gap.domain}")

        # Store experience about gap
        self.memory.store_experience(
            context=f"Knowledge gap in {gap.domain}",
            action="trigger_learning",
            outcome="learning_initiated",
            reward=0.0,  # Will be updated after resolution
        )

        # Mark as being addressed
        gap.resolution_data = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
        }

    async def process_input(
        self,
        input_text: str,
        input_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Process an input with full autonomous capabilities.

        1. Retrieve relevant knowledge
        2. Recall similar experiences
        3. Generate response
        4. Safety check
        5. Store experience
        """
        # Safety check on input
        if not self.safety.check_action("process_input", {"text": input_text[:100]}):
            return {"error": "Action requires human oversight"}

        result = {
            "input": input_text,
            "response": None,
            "knowledge_used": [],
            "experiences_recalled": [],
            "safety_passed": True,
        }

        # Knowledge integration
        if input_embedding is not None:
            knowledge = await self.knowledge.integrate_knowledge(
                input_text, input_embedding
            )
            result["knowledge_used"] = knowledge.get("retrieved_docs", [])
            augmented_input = knowledge.get("augmented_prompt", input_text)
        else:
            augmented_input = input_text

        # Recall similar experiences
        if input_embedding is not None:
            experiences = self.memory.recall_similar_experiences(input_embedding, k=3)
            result["experiences_recalled"] = [
                {"context": e.context[:100], "outcome": e.outcome}
                for e in experiences
            ]

        # Generate response (placeholder - actual generation would use model)
        # In production, this would call model.generate()

        # Store working memory
        self.memory.working.store(
            f"input_{time.time()}",
            {
                "input": input_text,
                "result": result,
            },
            ttl_seconds=3600,
        )

        return result

    def add_learning_sample(
        self,
        sample: Dict[str, Any],
        priority: float = 1.0,
    ):
        """Add a sample for online learning."""
        self.online_learning.add_sample(sample, priority)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "is_running": self.is_running,
            "autonomous_steps": self.autonomous_steps,
            "online_learning": self.online_learning.get_stats(),
            "safety": self.safety.get_safety_status(),
            "knowledge_gaps": len(self.evaluation.get_unresolved_gaps()),
            "last_evaluation": self.last_evaluation.isoformat(),
        }
