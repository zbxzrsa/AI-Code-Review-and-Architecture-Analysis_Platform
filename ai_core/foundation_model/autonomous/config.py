"""
Configuration Module for Autonomous Learning System

Contains all configuration classes, enums, and data models for the
autonomous learning system.

Classes:
- LearningMode: Enum for learning modes (online, batch, triggered, scheduled)
- MemoryType: Enum for memory types (long_term, short_term, episodic, semantic, working)
- SafetyLevel: Enum for safety levels (low, medium, high, critical)
- AutonomousConfig: Main configuration dataclass
- LearningEvent: Single learning event
- KnowledgeGap: Identified knowledge gap
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums
# =============================================================================

class LearningMode(str, Enum):
    """Learning modes for the autonomous agent."""
    ONLINE = "online"        # Continuous real-time learning
    BATCH = "batch"          # Periodic batch updates
    TRIGGERED = "triggered"  # Event-driven learning
    SCHEDULED = "scheduled"  # Time-scheduled learning


class MemoryType(str, Enum):
    """Types of memory in the autonomous system."""
    LONG_TERM = "long_term"    # Model parameters
    SHORT_TERM = "short_term"  # Context window
    EPISODIC = "episodic"      # Experience storage
    SEMANTIC = "semantic"      # Knowledge graph
    WORKING = "working"        # Active computation


class SafetyLevel(str, Enum):
    """Safety monitoring levels."""
    LOW = "low"          # Minimal checks
    MEDIUM = "medium"    # Standard checks
    HIGH = "high"        # Strict checks
    CRITICAL = "critical"  # Maximum safety


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


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class AutonomousConfig:
    """
    Autonomous learning configuration.
    
    Controls all aspects of the autonomous learning agent including
    learning modes, memory settings, evaluation, and safety.
    """
    # Learning modes
    primary_mode: LearningMode = LearningMode.ONLINE
    enable_batch_consolidation: bool = True
    consolidation_interval_hours: int = 24
    
    # Online learning
    online_learning_rate: float = 1e-6
    online_batch_size: int = 1
    online_buffer_size: int = 1000
    gradient_accumulation_steps: int = 4
    max_gradient_norm: float = 1.0
    
    # Memory
    episodic_memory_size: int = 100000
    working_memory_size: int = 1000
    semantic_memory_enabled: bool = True
    memory_consolidation_threshold: float = 0.7
    
    # Self-evaluation
    eval_interval_steps: int = 1000
    benchmark_suite: List[str] = field(default_factory=lambda: [
        "code_review", "bug_detection", "security_scan"
    ])
    knowledge_gap_threshold: float = 0.1
    min_benchmark_score: float = 0.8
    
    # Knowledge integration
    enable_rag: bool = True
    enable_tool_use: bool = True
    external_knowledge_sources: List[str] = field(default_factory=list)
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    
    # Safety
    safety_level: SafetyLevel = SafetyLevel.HIGH
    human_oversight_required: bool = True
    value_alignment_checks: bool = True
    max_autonomous_steps: int = 10000
    dangerous_action_patterns: List[str] = field(default_factory=lambda: [
        "delete", "drop", "remove", "destroy", "format", "shutdown"
    ])


@dataclass
class LearningEvent:
    """
    Single learning event.
    
    Represents an event that triggers learning, such as new data,
    user feedback, or system observations.
    """
    event_id: str
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    source: str
    priority: int = 1
    processed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "source": self.source,
            "priority": self.priority,
            "processed": self.processed,
        }


@dataclass
class KnowledgeGap:
    """
    Identified knowledge gap.
    
    Represents a detected area where the model's knowledge is
    insufficient and learning is needed.
    """
    gap_id: str
    domain: str
    description: str
    severity: float  # 0-1
    detected_at: datetime
    resolved: bool = False
    resolution_data: Optional[Dict[str, Any]] = None
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "domain": self.domain,
            "description": self.description,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
        }


@dataclass
class LearningException:
    """
    Structured learning exception with context.
    
    Provides detailed information about errors during learning
    for proper handling and recovery.
    """
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
    
    def should_retry(self) -> bool:
        """Check if the exception allows retry."""
        return self.recoverable and self.retry_count < self.max_retries
    
    def increment_retry(self) -> "LearningException":
        """Increment retry count and return self."""
        self.retry_count += 1
        return self


@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation."""
    benchmark_name: str
    score: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class Episode:
    """
    Single episode in episodic memory.
    
    Represents a complete experience including context, action,
    outcome, and reward for experience replay.
    """
    episode_id: str
    context: str
    action: str
    outcome: str
    reward: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[Any] = None  # numpy array
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "context": self.context,
            "action": self.action,
            "outcome": self.outcome,
            "reward": self.reward,
            "timestamp": self.timestamp.isoformat(),
        }
