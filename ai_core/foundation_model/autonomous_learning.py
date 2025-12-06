"""
Autonomous Learning Agent System

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
                raise  # Re-raise CancelledError after cleanup
        logger.info("Stopped online learning")
    
    async def _learning_loop(self):
        """Main online learning loop."""
        while self.is_learning:
            try:
                # Collect samples from streams
                await self._collect_from_streams()
                
                # Process buffer if enough samples
                if len(self.buffer) >= self.config.online_batch_size:
                    loss = self._update_step()
                    self.learning_curve.append(loss)
                    self.total_updates += 1
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Online learning error: {e}")
                await asyncio.sleep(1.0)
    
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
    """
    
    def __init__(self, max_size: int = 100000, embedding_dim: int = 768):
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        
        self.episodes: Dict[str, Episode] = {}
        self.episode_order: deque = deque(maxlen=max_size)
        
        # Index for similarity search
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.episode_ids: List[str] = []
    
    def add(self, episode: Episode):
        """Add an episode to memory."""
        # Remove oldest if at capacity
        if len(self.episode_order) >= self.max_size:
            oldest_id = self.episode_order.popleft()
            if oldest_id in self.episodes:
                del self.episodes[oldest_id]
        
        self.episodes[episode.episode_id] = episode
        self.episode_order.append(episode.episode_id)
        
        # Update embedding index
        if episode.embeddings is not None:
            self._update_index(episode)
    
    def _update_index(self, episode: Episode):
        """Update the embedding index."""
        if self.embeddings_matrix is None:
            self.embeddings_matrix = episode.embeddings.reshape(1, -1)
            self.episode_ids = [episode.episode_id]
        else:
            self.embeddings_matrix = np.vstack([
                self.embeddings_matrix,
                episode.embeddings.reshape(1, -1)
            ])
            self.episode_ids.append(episode.episode_id)
    
    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[Episode]:
        """Retrieve k most similar episodes."""
        if self.embeddings_matrix is None or len(self.episode_ids) == 0:
            return []
        
        # Compute cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        matrix_norm = self.embeddings_matrix / np.linalg.norm(
            self.embeddings_matrix, axis=1, keepdims=True
        )
        
        similarities = matrix_norm @ query_norm
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [
            self.episodes[self.episode_ids[i]]
            for i in top_indices
            if self.episode_ids[i] in self.episodes
        ]
    
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
    
    async def consolidate_memories(self):
        """
        Consolidate memories (like sleep/dream replay).
        
        - Transfer important episodic memories to semantic
        - Prune low-value memories
        - Strengthen frequently accessed memories
        """
        logger.info("Starting memory consolidation")
        
        # Get high-reward episodes
        important_episodes = self.episodic.retrieve_by_reward(k=100, min_reward=0.7)
        
        # Extract patterns and store in semantic memory
        for episode in important_episodes:
            # Create concept from episode
            concept_id = f"learned_{episode.episode_id[:8]}"
            self.semantic.add_concept(
                concept_id,
                f"Learned from {episode.action}",
                {
                    "context_pattern": episode.context[:100],
                    "successful_action": episode.action,
                    "outcome": episode.outcome,
                }
            )
            
            # Add fact
            self.semantic.add_fact(
                f"Action '{episode.action}' in context '{episode.context[:50]}...' "
                f"leads to '{episode.outcome}'",
                source="episodic_consolidation",
                confidence=episode.reward,
            )
        
        self.last_consolidation = datetime.now(timezone.utc)
        self.consolidation_pending = False
        
        logger.info(f"Consolidated {len(important_episodes)} episodes")
    
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
