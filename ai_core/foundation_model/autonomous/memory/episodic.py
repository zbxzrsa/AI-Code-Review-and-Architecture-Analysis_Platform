"""
Episodic Memory Module

Stores experiences for replay and learning. Supports:
- Experience storage with embeddings
- Similarity-based retrieval
- Importance-based retention
- Periodic consolidation
"""

import hashlib
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import Episode

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Episodic memory for storing and retrieving experiences.
    
    Features:
    - Stores episodes with context, action, outcome, reward
    - Supports embedding-based similarity search
    - Importance-weighted retention
    - Experience replay sampling
    
    Usage:
        memory = EpisodicMemory(max_size=100000)
        
        # Store experience
        memory.store_experience(
            context="User asked about Python",
            action="Provided code example",
            outcome="User was satisfied",
            reward=1.0,
        )
        
        # Recall similar experiences
        similar = memory.recall_similar_experiences(query_embedding, k=5)
        
        # Sample for replay
        batch = memory.sample_for_replay(batch_size=32)
    """
    
    def __init__(
        self,
        max_size: int = 100000,
        embedding_dim: int = 768,
        importance_decay: float = 0.99,
    ):
        """
        Initialize episodic memory.
        
        Args:
            max_size: Maximum number of episodes to store
            embedding_dim: Dimension of embeddings
            importance_decay: Decay factor for importance over time
        """
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.importance_decay = importance_decay
        
        # Episode storage
        self.episodes: Dict[str, Episode] = {}
        self.episode_order: deque = deque(maxlen=max_size)
        
        # Embedding index (simple numpy-based)
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Importance scores for retention
        self.importance_scores: Dict[str, float] = {}
        
        # Statistics
        self.total_stored = 0
        self.total_recalled = 0
    
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
            reward: Reward signal (-1 to 1)
            embedding: Optional embedding vector
            metadata: Additional metadata
            
        Returns:
            Episode ID
        """
        # Generate episode ID
        episode_id = self._generate_episode_id(context, action)
        
        # Create episode
        episode = Episode(
            episode_id=episode_id,
            context=context,
            action=action,
            outcome=outcome,
            reward=reward,
            timestamp=datetime.now(timezone.utc),
            embedding=embedding,
            metadata=metadata or {},
        )
        
        # Check capacity
        if len(self.episodes) >= self.max_size:
            self._evict_least_important()
        
        # Store
        self.episodes[episode_id] = episode
        self.episode_order.append(episode_id)
        
        # Store embedding if provided
        if embedding is not None:
            self.embeddings[episode_id] = embedding
        
        # Initial importance based on reward
        self.importance_scores[episode_id] = abs(reward) + 0.1
        
        self.total_stored += 1
        
        return episode_id
    
    def recall_similar_experiences(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Episode]:
        """
        Recall experiences similar to query.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of experiences to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar episodes
        """
        if not self.embeddings:
            return []
        
        # Compute similarities
        similarities: List[Tuple[str, float]] = []
        
        for episode_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity >= min_similarity:
                similarities.append((episode_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k episodes
        results = []
        for episode_id, _ in similarities[:k]:
            if episode_id in self.episodes:
                results.append(self.episodes[episode_id])
                # Boost importance for recalled episodes
                self.importance_scores[episode_id] *= 1.1
        
        self.total_recalled += len(results)
        return results
    
    def sample_for_replay(
        self,
        batch_size: int = 32,
        prioritized: bool = True,
    ) -> List[Episode]:
        """
        Sample episodes for experience replay.
        
        Args:
            batch_size: Number of episodes to sample
            prioritized: Whether to use importance-weighted sampling
            
        Returns:
            List of sampled episodes
        """
        if not self.episodes:
            return []
        
        episode_ids = list(self.episodes.keys())
        
        if prioritized and self.importance_scores:
            # Importance-weighted sampling
            weights = [
                self.importance_scores.get(eid, 1.0)
                for eid in episode_ids
            ]
            total = sum(weights)
            probs = [w / total for w in weights]
            
            sample_size = min(batch_size, len(episode_ids))
            sampled_ids = np.random.choice(
                episode_ids,
                size=sample_size,
                replace=False,
                p=probs,
            )
        else:
            # Uniform sampling
            sample_size = min(batch_size, len(episode_ids))
            sampled_ids = random.sample(episode_ids, sample_size)
        
        return [self.episodes[eid] for eid in sampled_ids]
    
    def get_recent_episodes(self, n: int = 10) -> List[Episode]:
        """Get n most recent episodes."""
        recent_ids = list(self.episode_order)[-n:]
        return [
            self.episodes[eid]
            for eid in reversed(recent_ids)
            if eid in self.episodes
        ]
    
    def get_high_reward_episodes(
        self,
        threshold: float = 0.5,
        limit: int = 100,
    ) -> List[Episode]:
        """Get episodes with high rewards."""
        high_reward = [
            ep for ep in self.episodes.values()
            if ep.reward >= threshold
        ]
        high_reward.sort(key=lambda x: x.reward, reverse=True)
        return high_reward[:limit]
    
    def decay_importance(self):
        """Apply decay to all importance scores."""
        for episode_id in self.importance_scores:
            self.importance_scores[episode_id] *= self.importance_decay
    
    def _evict_least_important(self):
        """Evict the least important episode."""
        if not self.importance_scores:
            # Evict oldest if no importance scores
            if self.episode_order:
                oldest_id = self.episode_order.popleft()
                self._remove_episode(oldest_id)
            return
        
        # Find least important
        min_id = min(self.importance_scores, key=self.importance_scores.get)
        self._remove_episode(min_id)
    
    def _remove_episode(self, episode_id: str):
        """Remove an episode from memory."""
        self.episodes.pop(episode_id, None)
        self.embeddings.pop(episode_id, None)
        self.importance_scores.pop(episode_id, None)
    
    def _generate_episode_id(self, context: str, action: str) -> str:
        """Generate unique episode ID."""
        content = f"{context}{action}{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_episodes": len(self.episodes),
            "max_size": self.max_size,
            "total_stored": self.total_stored,
            "total_recalled": self.total_recalled,
            "embeddings_count": len(self.embeddings),
            "avg_importance": (
                sum(self.importance_scores.values()) / len(self.importance_scores)
                if self.importance_scores else 0.0
            ),
        }
    
    def clear(self):
        """Clear all memories."""
        self.episodes.clear()
        self.embeddings.clear()
        self.importance_scores.clear()
        self.episode_order.clear()
