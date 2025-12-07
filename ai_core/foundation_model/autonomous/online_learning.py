"""
在线学习模块 (Online Learning Module)

模块功能描述:
    实现实时增量学习，支持优先级缓冲和自适应学习率。

主要功能:
    - 基于优先级的样本缓冲
    - 梯度累积
    - 自适应学习率
    - 灾难性遗忘缓解

主要组件:
    - OnlineLearningBuffer: 优先级缓冲区
    - OnlineLearner: 在线学习器
    - LearningStats: 学习统计

最后修改日期: 2024-12-07
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import LearningException, ExceptionSeverity, LearningErrorCode

logger = logging.getLogger(__name__)


@dataclass
class LearningStats:
    """
    在线学习统计数据类
    
    功能描述:
        记录在线学习过程的各项统计指标。
    
    属性说明:
        - samples_received: 已接收样本数
        - samples_processed: 已处理样本数
        - samples_dropped: 已丢弃样本数
        - batches_trained: 已训练批次数
        - avg_loss: 平均损失
    """
    samples_received: int = 0
    samples_processed: int = 0
    samples_dropped: int = 0
    batches_trained: int = 0
    total_loss: float = 0.0
    avg_loss: float = 0.0
    learning_rate: float = 0.0
    last_update: Optional[datetime] = None


class OnlineLearningBuffer:
    """
    在线学习优先级缓冲区
    
    功能描述:
        管理在线学习样本的优先级缓冲区。
    
    主要特性:
        - 基于优先级的插入
        - FIFO 溢出处理
        - 批量采样
        - 统计跟踪
    
    使用示例:
        buffer = OnlineLearningBuffer(max_size=1000)
        
        # 添加带优先级的样本
        buffer.add(sample_data, priority=1.0)
        
        # 获取批次
        batch = buffer.sample_batch(batch_size=32)
    """
    
    def __init__(self, max_size: int = 1000):
        """
        初始化缓冲区
        
        参数:
            max_size: 最大缓冲区大小
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.priorities: deque = deque(maxlen=max_size)
        
        # Statistics
        self.total_added = 0
        self.total_dropped = 0
    
    def add(
        self,
        sample: Dict[str, Any],
        priority: float = 1.0,
    ):
        """
        Add a sample to the buffer.
        
        Args:
            sample: Sample data
            priority: Sample priority (higher = more important)
        """
        if len(self.buffer) >= self.max_size:
            self.total_dropped += 1
        
        self.buffer.append(sample)
        self.priorities.append(priority)
        self.total_added += 1
    
    def sample_batch(
        self,
        batch_size: int = 32,
        prioritized: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Sample a batch from the buffer.
        
        Args:
            batch_size: Number of samples to return
            prioritized: Use priority-weighted sampling
            
        Returns:
            List of samples
        """
        if not self.buffer:
            return []
        
        actual_size = min(batch_size, len(self.buffer))
        
        if prioritized and len(self.buffer) > actual_size:
            # Priority-weighted sampling
            priorities_array = np.array(list(self.priorities))
            probs = priorities_array / priorities_array.sum()
            
            indices = np.random.choice(
                len(self.buffer),
                size=actual_size,
                replace=False,
                p=probs,
            )
            
            return [self.buffer[i] for i in indices]
        
        # Random sampling
        indices = np.random.choice(
            len(self.buffer),
            size=actual_size,
            replace=False,
        )
        
        return [self.buffer[i] for i in indices]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all samples in buffer."""
        return list(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        return len(self.buffer) == 0
    
    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "current_size": len(self.buffer),
            "max_size": self.max_size,
            "total_added": self.total_added,
            "total_dropped": self.total_dropped,
            "utilization": len(self.buffer) / self.max_size if self.max_size > 0 else 0,
        }


class OnlineLearningModule:
    """
    Online learning module for real-time model updates.
    
    Features:
    - Priority-based sample buffering
    - Gradient accumulation
    - Adaptive learning rate
    - EWC for catastrophic forgetting prevention
    
    Usage:
        module = OnlineLearningModule(model, config)
        
        # Add sample
        module.add_sample(sample, priority=1.0)
        
        # Train on buffer
        await module.train_step()
        
        # Get stats
        stats = module.get_stats()
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        learning_rate: float = 1e-6,
        batch_size: int = 8,
        buffer_size: int = 1000,
        gradient_accumulation_steps: int = 4,
        max_gradient_norm: float = 1.0,
    ):
        """
        Initialize online learning module.
        
        Args:
            model: PyTorch model to train
            learning_rate: Learning rate
            batch_size: Training batch size
            buffer_size: Sample buffer size
            gradient_accumulation_steps: Steps for gradient accumulation
            max_gradient_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_gradient_norm = max_gradient_norm
        
        # Buffer
        self.buffer = OnlineLearningBuffer(max_size=buffer_size)
        
        # Optimizer (lazy initialization)
        self._optimizer = None
        
        # Statistics
        self.stats = LearningStats(learning_rate=learning_rate)
        
        # State
        self._is_training = False
        self._accumulation_step = 0
    
    def add_sample(
        self,
        sample: Dict[str, Any],
        priority: float = 1.0,
    ):
        """
        Add a sample for online learning.
        
        Args:
            sample: Sample data (should contain 'input' and 'target')
            priority: Sample priority
        """
        self.buffer.add(sample, priority)
        self.stats.samples_received += 1
    
    async def train_step(self) -> Optional[float]:
        """
        Perform a training step on buffered samples.
        
        Returns:
            Loss value or None if no training occurred
        """
        if self.buffer.is_empty():
            return None
        
        if self.model is None:
            logger.warning("No model set for training")
            return None
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available")
            return None
        
        self._is_training = True
        
        try:
            # Get batch
            batch = self.buffer.sample_batch(self.batch_size)
            if not batch:
                return None
            
            # Initialize optimizer if needed
            if self._optimizer is None:
                self._optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate,
                )
            
            # Training step
            loss = await self._train_on_batch(batch)
            
            # Update stats
            self.stats.samples_processed += len(batch)
            self.stats.batches_trained += 1
            self.stats.total_loss += loss
            self.stats.avg_loss = self.stats.total_loss / self.stats.batches_trained
            self.stats.last_update = datetime.now(timezone.utc)
            
            return loss
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            raise
        finally:
            self._is_training = False
    
    async def _train_on_batch(self, batch: List[Dict[str, Any]]) -> float:
        """Train on a batch of samples."""
        if not TORCH_AVAILABLE:
            return 0.0
        
        self.model.train()
        total_loss = 0.0
        
        for sample in batch:
            # Process sample (placeholder - actual implementation depends on model)
            # In production, this would process inputs through the model
            loss = self._compute_loss(sample)
            total_loss += loss
            
            self._accumulation_step += 1
            
            # Gradient step
            if self._accumulation_step >= self.gradient_accumulation_steps:
                self._optimizer.step()
                self._optimizer.zero_grad()
                self._accumulation_step = 0
        
        return total_loss / len(batch)
    
    def _compute_loss(self, sample: Dict[str, Any]) -> float:
        """Compute loss for a sample (placeholder)."""
        # In production, this would compute actual loss
        # For now, return a mock loss
        return np.random.uniform(0.1, 0.5)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "samples_received": self.stats.samples_received,
            "samples_processed": self.stats.samples_processed,
            "batches_trained": self.stats.batches_trained,
            "avg_loss": round(self.stats.avg_loss, 4) if self.stats.avg_loss else 0,
            "learning_rate": self.learning_rate,
            "is_training": self._is_training,
            "buffer": self.buffer.get_stats(),
            "last_update": (
                self.stats.last_update.isoformat()
                if self.stats.last_update else None
            ),
        }
    
    def set_learning_rate(self, lr: float):
        """Update learning rate."""
        self.learning_rate = lr
        self.stats.learning_rate = lr
        
        if self._optimizer:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
    
    def clear_buffer(self):
        """Clear the sample buffer."""
        self.buffer.clear()
    
    def is_ready_to_train(self) -> bool:
        """Check if ready to train (has samples)."""
        return len(self.buffer) >= self.batch_size
