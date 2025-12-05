"""
Long-Term Memory System
Prevents catastrophic forgetting through memory consolidation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Single memory item"""
    input: torch.Tensor
    target: torch.Tensor
    task_id: int
    importance: float
    timestamp: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperienceReplay:
    """
    Experience Replay Buffer
    
    Features:
    - Priority-based sampling
    - Reservoir sampling for streaming data
    - Task-balanced sampling
    - Importance-weighted updates
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        prioritized: bool = True,
        alpha: float = 0.6,
        beta: float = 0.4,
        seed: int = 42
    ):
        """
        Initialize Experience Replay
        
        Args:
            capacity: Maximum buffer size
            prioritized: Whether to use prioritized replay
            alpha: Priority exponent
            beta: Importance sampling exponent
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.buffer: List[MemoryItem] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        self.timestamp = 0
    
    def add(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        task_id: int = 0,
        importance: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add experience to buffer"""
        item = MemoryItem(
            input=input.detach().clone(),  # detach from graph, then clone
            target=target.detach().clone() if isinstance(target, torch.Tensor) else torch.tensor(target),
            task_id=task_id,
            importance=importance,
            timestamp=self.timestamp,
            metadata=metadata or {}
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        
        # Set priority
        if self.prioritized:
            self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.timestamp += 1
    
    def add_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int = 0
    ) -> None:
        """Add a batch of experiences"""
        for i in range(inputs.size(0)):
            self.add(inputs[i], targets[i], task_id)
    
    def sample(
        self,
        batch_size: int,
        task_balanced: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
        """
        Sample from buffer
        
        Args:
            batch_size: Number of samples
            task_balanced: Whether to balance across tasks
            
        Returns:
            inputs, targets, indices, weights
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if task_balanced:
            indices = self._task_balanced_sample(batch_size)
        elif self.prioritized:
            indices, weights = self._prioritized_sample(batch_size)
        else:
            indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
            weights = torch.ones(batch_size)
        
        inputs = torch.stack([self.buffer[i].input for i in indices])
        targets = torch.stack([self.buffer[i].target for i in indices])
        
        if not self.prioritized:
            weights = torch.ones(batch_size)
        
        return inputs, targets, list(indices), weights
    
    def _prioritized_sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Prioritized sampling"""
        buffer_len = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:buffer_len]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = self.rng.choice(buffer_len, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        min_prob = probs.min()
        max_weight = (buffer_len * min_prob) ** (-self.beta)
        weights = (buffer_len * probs[indices]) ** (-self.beta) / max_weight
        
        return indices, torch.tensor(weights, dtype=torch.float32)
    
    def _task_balanced_sample(self, batch_size: int) -> np.ndarray:
        """Sample balanced across tasks"""
        # Group by task
        task_indices: Dict[int, List[int]] = {}
        for i, item in enumerate(self.buffer):
            if item.task_id not in task_indices:
                task_indices[item.task_id] = []
            task_indices[item.task_id].append(i)
        
        # Sample from each task
        samples_per_task = batch_size // len(task_indices)
        indices = []
        
        for task_id, task_idx in task_indices.items():
            n_samples = min(samples_per_task, len(task_idx))
            sampled = self.rng.choice(task_idx, size=n_samples, replace=False)
            indices.extend(sampled.tolist())
        
        # Fill remaining with random samples
        remaining = batch_size - len(indices)
        if remaining > 0:
            available = [i for i in range(len(self.buffer)) if i not in indices]
            if available:
                extra = self.rng.choice(available, size=min(remaining, len(available)), replace=False)
                indices.extend(extra.tolist())
        
        return np.array(indices)
    
    def update_priorities(
        self,
        indices: List[int],
        td_errors: torch.Tensor
    ) -> None:
        """Update priorities based on TD errors"""
        if not self.prioritized:
            return
        
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error.item()) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def reservoir_sample(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        task_id: int = 0
    ) -> bool:
        """
        Reservoir sampling for streaming data
        
        Returns:
            Whether item was added
        """
        if len(self.buffer) < self.capacity:
            self.add(input, target, task_id)
            return True
        
        # Reservoir sampling probability
        prob = self.capacity / self.timestamp
        if self.rng.random() < prob:
            idx = self.rng.integers(0, self.capacity)
            self.buffer[idx] = MemoryItem(
                input=input.detach().clone(),  # detach from graph, then clone
                target=target.detach().clone() if isinstance(target, torch.Tensor) else torch.tensor(target),
                task_id=task_id,
                importance=1.0,
                timestamp=self.timestamp
            )
            self.timestamp += 1
            return True
        
        self.timestamp += 1
        return False
    
    def get_task_distribution(self) -> Dict[int, int]:
        """Get distribution of tasks in buffer"""
        distribution = {}
        for item in self.buffer:
            if item.task_id not in distribution:
                distribution[item.task_id] = 0
            distribution[item.task_id] += 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self, path: str) -> None:
        """Save buffer to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'buffer': self.buffer,
                'priorities': self.priorities,
                'position': self.position,
                'max_priority': self.max_priority,
                'timestamp': self.timestamp
            }, f)
    
    def load(self, path: str) -> None:
        """Load buffer from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.buffer = data['buffer']
            self.priorities = data['priorities']
            self.position = data['position']
            self.max_priority = data['max_priority']
            self.timestamp = data['timestamp']


class LongTermMemory:
    """
    Long-Term Memory System
    
    Features:
    - Hierarchical memory organization
    - Memory consolidation
    - Importance-based retention
    - Semantic clustering
    - Dream-like replay
    """
    
    def __init__(
        self,
        short_term_capacity: int = 1000,
        long_term_capacity: int = 10000,
        consolidation_threshold: float = 0.7,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 42
    ):
        """
        Initialize Long-Term Memory
        
        Args:
            short_term_capacity: Short-term buffer size
            long_term_capacity: Long-term storage size
            consolidation_threshold: Importance threshold for consolidation
            device: Device to use
            seed: Random seed for reproducibility
        """
        self.device = device
        self.consolidation_threshold = consolidation_threshold
        self.rng = np.random.default_rng(seed)
        
        # Short-term memory (recent experiences)
        self.short_term = ExperienceReplay(
            capacity=short_term_capacity,
            prioritized=True
        )
        
        # Long-term memory (consolidated experiences)
        self.long_term = ExperienceReplay(
            capacity=long_term_capacity,
            prioritized=True
        )
        
        # Episodic memory (task-specific clusters)
        self.episodic: Dict[int, List[MemoryItem]] = {}
        
        # Memory statistics
        self.consolidation_count = 0
        self.total_memories = 0
    
    def store(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        task_id: int = 0,
        importance: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store a new memory"""
        self.short_term.add(input, target, task_id, importance, metadata)
        self.total_memories += 1
        
        # Auto-consolidate if short-term is full
        if len(self.short_term) >= self.short_term.capacity * 0.9:
            self.consolidate()
    
    def store_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int = 0
    ) -> None:
        """Store a batch of memories"""
        for i in range(inputs.size(0)):
            self.store(inputs[i], targets[i], task_id)
    
    def consolidate(
        self,
        model: Optional[nn.Module] = None
    ) -> int:
        """
        Consolidate short-term to long-term memory
        
        Args:
            model: Optional model for importance estimation
            
        Returns:
            Number of consolidated memories
        """
        consolidated = 0
        
        for item in self.short_term.buffer:
            # Calculate final importance
            if model is not None:
                importance = self._estimate_importance(model, item)
            else:
                importance = item.importance
            
            # Consolidate if important enough
            if importance >= self.consolidation_threshold:
                self.long_term.add(
                    item.input, item.target, item.task_id,
                    importance, item.metadata
                )
                
                # Add to episodic memory
                if item.task_id not in self.episodic:
                    self.episodic[item.task_id] = []
                self.episodic[item.task_id].append(item)
                
                consolidated += 1
        
        # Clear short-term memory
        self.short_term.buffer.clear()
        self.short_term.position = 0
        
        self.consolidation_count += 1
        logger.info(f"Consolidated {consolidated} memories to long-term storage")
        
        return consolidated
    
    def _estimate_importance(
        self,
        model: nn.Module,
        item: MemoryItem
    ) -> float:
        """Estimate memory importance using model uncertainty"""
        model.eval()
        
        with torch.no_grad():
            input_tensor = item.input.unsqueeze(0).to(self.device)
            output = model(input_tensor)
            
            # Use prediction entropy as importance
            probs = F.softmax(output, dim=1)
            entropy = -(probs * probs.log()).sum().item()
            
            # Normalize
            importance = min(1.0, entropy / np.log(output.size(1)))
        
        return importance
    
    def recall(
        self,
        batch_size: int,
        memory_type: str = 'both',
        task_id: Optional[int] = None  # noqa: ARG002 - reserved for future task filtering
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recall memories
        
        Args:
            batch_size: Number of memories to recall
            memory_type: 'short', 'long', or 'both'
            task_id: Optional task filter (reserved for future use)
            
        Returns:
            inputs, targets
        """
        if memory_type == 'short':
            inputs, targets, _, _ = self.short_term.sample(batch_size)
        elif memory_type == 'long':
            inputs, targets, _, _ = self.long_term.sample(batch_size)
        else:
            # Mix short and long term
            short_size = batch_size // 2
            long_size = batch_size - short_size
            
            short_inputs, short_targets, _, _ = self.short_term.sample(
                min(short_size, len(self.short_term))
            )
            long_inputs, long_targets, _, _ = self.long_term.sample(
                min(long_size, len(self.long_term))
            )
            
            inputs = torch.cat([short_inputs, long_inputs])
            targets = torch.cat([short_targets, long_targets])
        
        return inputs.to(self.device), targets.to(self.device)
    
    def recall_episodic(
        self,
        task_id: int,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recall episodic memories for a specific task"""
        if task_id not in self.episodic:
            return torch.tensor([]), torch.tensor([])
        
        episodes = self.episodic[task_id]
        n_samples = min(batch_size, len(episodes))
        
        indices = self.rng.choice(len(episodes), size=n_samples, replace=False)
        samples = [episodes[i] for i in indices]
        
        inputs = torch.stack([s.input for s in samples])
        targets = torch.stack([s.target for s in samples])
        
        return inputs.to(self.device), targets.to(self.device)
    
    def dream_replay(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        n_iterations: int = 100,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Dream-like replay for memory consolidation
        
        Args:
            model: Model to train
            optimizer: Optimizer
            n_iterations: Number of replay iterations
            batch_size: Batch size
            
        Returns:
            Training metrics
        """
        model.train()
        total_loss = 0.0
        
        for _ in range(n_iterations):
            # Sample from long-term memory
            inputs, targets = self.recall(batch_size, 'long')
            
            if inputs.size(0) == 0:
                break
            
            # Forward pass
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'dream_loss': total_loss / n_iterations,
            'iterations': n_iterations
        }
    
    def generative_replay(
        self,
        generator: nn.Module,
        classifier: nn.Module,
        optimizer: torch.optim.Optimizer,
        n_iterations: int = 100,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Generative replay using a generator model
        
        Args:
            generator: Generator model
            classifier: Classifier model
            optimizer: Optimizer
            n_iterations: Number of iterations
            batch_size: Batch size
            
        Returns:
            Training metrics
        """
        classifier.train()
        generator.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            # Generate pseudo-samples
            z = torch.randn(batch_size * n_iterations, generator.latent_dim).to(self.device)
            generated = generator(z)
        
        # Get pseudo-labels from current classifier
        with torch.no_grad():
            pseudo_labels = classifier(generated).argmax(dim=1)
        
        # Train on generated samples
        for i in range(n_iterations):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = generated[start_idx:end_idx]
            batch_targets = pseudo_labels[start_idx:end_idx]
            
            outputs = classifier(batch_inputs)
            loss = F.cross_entropy(outputs, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return {
            'generative_replay_loss': total_loss / n_iterations
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'short_term_size': len(self.short_term),
            'long_term_size': len(self.long_term),
            'episodic_tasks': len(self.episodic),
            'total_memories': self.total_memories,
            'consolidation_count': self.consolidation_count,
            'task_distribution': self.long_term.get_task_distribution()
        }
    
    def save(self, path: str) -> None:
        """Save memory system to disk"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.short_term.save(str(save_path / 'short_term.pkl'))
        self.long_term.save(str(save_path / 'long_term.pkl'))
        
        with open(save_path / 'episodic.pkl', 'wb') as f:
            pickle.dump(self.episodic, f)
        
        with open(save_path / 'stats.pkl', 'wb') as f:
            pickle.dump({
                'consolidation_count': self.consolidation_count,
                'total_memories': self.total_memories
            }, f)
    
    def load(self, path: str) -> None:
        """Load memory system from disk"""
        load_path = Path(path)
        
        self.short_term.load(str(load_path / 'short_term.pkl'))
        self.long_term.load(str(load_path / 'long_term.pkl'))
        
        with open(load_path / 'episodic.pkl', 'rb') as f:
            self.episodic = pickle.load(f)
        
        with open(load_path / 'stats.pkl', 'rb') as f:
            stats = pickle.load(f)
            self.consolidation_count = stats['consolidation_count']
            self.total_memories = stats['total_memories']
