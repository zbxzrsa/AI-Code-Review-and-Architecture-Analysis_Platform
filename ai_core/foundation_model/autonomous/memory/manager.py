"""
Memory Management Module

Unified orchestration of all memory subsystems:
- Episodic Memory: Experience storage
- Semantic Memory: Knowledge storage
- Working Memory: Active computation cache

Provides consolidation, coordination, and cross-memory operations.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .working import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    # Episodic memory
    episodic_memory_size: int = 100000
    episodic_embedding_dim: int = 768
    
    # Semantic memory
    semantic_memory_size: int = 50000
    
    # Working memory
    working_memory_size: int = 1000
    working_memory_ttl: int = 3600
    
    # Consolidation
    consolidation_interval_hours: int = 24
    consolidation_threshold: float = 0.7
    
    # Cross-memory
    enable_cross_memory_links: bool = True


class MemoryManagement:
    """
    Unified memory management for autonomous learning.
    
    Coordinates episodic, semantic, and working memory subsystems.
    Provides consolidation, cross-referencing, and memory operations.
    
    Usage:
        config = MemoryConfig(episodic_memory_size=100000)
        memory = MemoryManagement(config)
        
        # Store experience
        memory.store_experience(
            context="User asked about Python",
            action="Provided code example",
            outcome="User was satisfied",
            reward=1.0,
        )
        
        # Store concept
        memory.store_concept(
            name="Python",
            description="A programming language",
        )
        
        # Recall similar experiences
        experiences = memory.recall_similar_experiences(embedding, k=5)
        
        # Consolidate memories
        await memory.consolidate_memories()
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory management.
        
        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        
        # Initialize memory subsystems
        self.episodic = EpisodicMemory(
            max_size=self.config.episodic_memory_size,
            embedding_dim=self.config.episodic_embedding_dim,
        )
        
        self.semantic = SemanticMemory(
            max_concepts=self.config.semantic_memory_size,
            embedding_dim=self.config.episodic_embedding_dim,
        )
        
        self.working = WorkingMemory(
            max_size=self.config.working_memory_size,
            default_ttl_seconds=self.config.working_memory_ttl,
        )
        
        # Consolidation state
        self._last_consolidation = datetime.now(timezone.utc)
        self._consolidation_count = 0
        
        # Cross-memory links
        self._episode_to_concepts: Dict[str, List[str]] = {}
    
    # =========================================================================
    # Experience Operations (Episodic Memory)
    # =========================================================================
    
    def store_experience(
        self,
        context: str,
        action: str,
        outcome: str,
        reward: float,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store an experience in episodic memory.
        
        Args:
            context: Context/situation description
            action: Action taken
            outcome: Result of the action
            reward: Reward signal
            embedding: Optional embedding vector
            metadata: Additional metadata
            
        Returns:
            Episode ID
        """
        episode_id = self.episodic.store_experience(
            context=context,
            action=action,
            outcome=outcome,
            reward=reward,
            embedding=embedding,
            metadata=metadata,
        )
        
        # Store in working memory for quick access
        self.working.store(
            f"recent_episode_{episode_id}",
            {
                "context": context[:200],
                "action": action[:200],
                "outcome": outcome[:200],
                "reward": reward,
            },
            ttl_seconds=3600,
        )
        
        return episode_id
    
    def recall_similar_experiences(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List:
        """Recall experiences similar to query."""
        return self.episodic.recall_similar_experiences(query_embedding, k=k)
    
    def sample_experiences_for_replay(self, batch_size: int = 32) -> List:
        """Sample experiences for replay learning."""
        return self.episodic.sample_for_replay(batch_size=batch_size)
    
    def get_recent_experiences(self, n: int = 10) -> List:
        """Get recent experiences."""
        return self.episodic.get_recent_episodes(n=n)
    
    # =========================================================================
    # Knowledge Operations (Semantic Memory)
    # =========================================================================
    
    def store_concept(
        self,
        name: str,
        description: str,
        embedding: Optional[np.ndarray] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a concept in semantic memory.
        
        Args:
            name: Concept name
            description: Concept description
            embedding: Optional embedding vector
            attributes: Additional attributes
            
        Returns:
            Concept ID
        """
        return self.semantic.store_concept(
            name=name,
            description=description,
            embedding=embedding,
            attributes=attributes,
        )
    
    def get_concept(self, name: str):
        """Get a concept by name."""
        return self.semantic.get_concept_by_name(name)
    
    def search_concepts(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List:
        """Search concepts by embedding similarity."""
        return self.semantic.search_concepts(query_embedding, k=k)
    
    def add_concept_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        strength: float = 1.0,
    ) -> bool:
        """Add a relationship between concepts."""
        return self.semantic.add_relationship(
            source_name=source,
            target_name=target,
            relation_type=relation_type,
            strength=strength,
        )
    
    # =========================================================================
    # Working Memory Operations
    # =========================================================================
    
    def cache_computation(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ):
        """Cache a computation result."""
        self.working.store(key, value, ttl_seconds=ttl_seconds)
    
    def get_cached(self, key: str, default: Any = None) -> Any:
        """Get a cached computation result."""
        return self.working.get(key, default)
    
    def has_cached(self, key: str) -> bool:
        """Check if a key is cached."""
        return self.working.has(key)
    
    # =========================================================================
    # Consolidation
    # =========================================================================
    
    def should_consolidate(self) -> bool:
        """Check if consolidation should run."""
        hours_since = (
            datetime.now(timezone.utc) - self._last_consolidation
        ).total_seconds() / 3600
        
        return hours_since >= self.config.consolidation_interval_hours
    
    async def consolidate_memories(self):
        """
        Consolidate memories across subsystems.
        
        Transfers important patterns from episodic to semantic memory.
        """
        logger.info("Starting memory consolidation")
        
        # Get high-reward experiences
        high_reward_episodes = self.episodic.get_high_reward_episodes(
            threshold=self.config.consolidation_threshold,
            limit=100,
        )
        
        concepts_created = 0
        
        for episode in high_reward_episodes:
            # Extract concepts from successful episodes
            # In production, this would use NLP to extract entities
            
            # Simple heuristic: create concept from context keywords
            words = episode.context.split()[:5]
            if words:
                concept_name = "_".join(words)
                self.semantic.store_concept(
                    name=concept_name,
                    description=f"Learned from experience: {episode.context[:100]}",
                    embedding=episode.embedding,
                    attributes={
                        "source_episode": episode.episode_id,
                        "reward": episode.reward,
                    },
                )
                concepts_created += 1
        
        # Decay episodic importance
        self.episodic.decay_importance()
        
        # Update consolidation state
        self._last_consolidation = datetime.now(timezone.utc)
        self._consolidation_count += 1
        
        logger.info(f"Consolidation complete: {concepts_created} concepts created")
    
    # =========================================================================
    # Cross-Memory Operations
    # =========================================================================
    
    def link_episode_to_concept(self, episode_id: str, concept_id: str):
        """Link an episode to a concept."""
        if episode_id not in self._episode_to_concepts:
            self._episode_to_concepts[episode_id] = []
        self._episode_to_concepts[episode_id].append(concept_id)
    
    def get_episode_concepts(self, episode_id: str) -> List[str]:
        """Get concepts linked to an episode."""
        return self._episode_to_concepts.get(episode_id, [])
    
    def integrated_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Search across all memory systems.
        
        Args:
            query_embedding: Query embedding
            k: Number of results per system
            
        Returns:
            Dict with results from each memory system
        """
        return {
            "experiences": self.episodic.recall_similar_experiences(query_embedding, k),
            "concepts": self.semantic.search_concepts(query_embedding, k),
        }
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        return {
            "episodic": self.episodic.get_stats(),
            "semantic": self.semantic.get_stats(),
            "working": self.working.get_stats(),
            "consolidation_count": self._consolidation_count,
            "last_consolidation": self._last_consolidation.isoformat(),
            "cross_memory_links": len(self._episode_to_concepts),
        }
    
    def clear_all(self):
        """Clear all memory systems."""
        self.episodic.clear()
        self.semantic.clear()
        self.working.clear()
        self._episode_to_concepts.clear()
