"""
持续学习框架 (Continuous Learning Framework)

模块功能描述:
    支持增量学习、在线学习和终身学习。

主要功能:
    - 新数据增量学习
    - 任务增量和类增量学习
    - 灾难性遗忘防止
    - 动态架构扩展
    - 跨任务性能监控

主要组件:
    - ContinuousLearner: 持续学习器主类
    - LearningState: 学习状态数据类
    - ExperienceReplay: 经验回放缓冲区

最后修改日期: 2024-12-07
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import numpy as np
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)


@dataclass
class LearningState:
    """
    持续学习过程状态数据类

    功能描述:
        记录持续学习过程的各项状态信息。

    属性说明:
        - total_samples_seen: 已见样本总数
        - total_updates: 更新总次数
        - current_task_id: 当前任务ID
        - task_boundaries: 任务边界记录
        - performance_history: 性能历史记录
    """
    total_samples_seen: int = 0
    total_updates: int = 0
    current_task_id: int = 0
    task_boundaries: List[int] = field(default_factory=list)
    performance_history: List[Dict[str, float]] = field(default_factory=list)

    def record_task_boundary(self) -> None:
        """
        记录任务边界

        当从一个任务切换到另一个任务时调用。
        """
        self.task_boundaries.append(self.total_samples_seen)
        self.current_task_id += 1


class ContinuousLearner:
    """
    持续学习系统

    功能描述:
        实现模型的持续学习能力，防止灾难性遗忘。

    主要特性:
        - 新数据增量学习
        - 任务增量和类增量学习
        - 灾难性遗忘防止
        - 动态架构扩展
        - 跨任务性能监控

    参数:
        - model: PyTorch 模型
        - memory_size: 经验回放缓冲区大小
        - ewc_lambda: EWC 正则化系数
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_class: type = optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
        memory_size: int = 1000,
        replay_batch_size: int = 32,
        ewc_lambda: float = 0.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 1e-4}
        self.optimizer = optimizer_class(model.parameters(), **self.optimizer_kwargs)

        self.memory_size = memory_size
        self.replay_batch_size = replay_batch_size
        self.ewc_lambda = ewc_lambda

        # Experience replay buffer
        self.replay_buffer: deque = deque(maxlen=memory_size)

        # Learning state
        self.state = LearningState()

        # EWC parameters (for preventing catastrophic forgetting)
        self.ewc_params: Dict[str, torch.Tensor] = {}
        self.fisher_info: Dict[str, torch.Tensor] = {}

        # Task-specific heads (for task-incremental learning)
        self.task_heads: Dict[int, nn.Module] = {}

        # Callbacks
        self.callbacks: List[Callable] = []

    def add_callback(self, callback: Callable) -> None:
        """Add a learning callback"""
        self.callbacks.append(callback)

    def _call_callbacks(self, event: str, **kwargs) -> None:
        """Call registered callbacks"""
        for callback in self.callbacks:
            callback(event, **kwargs)

    def learn_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Learn from a single batch

        Args:
            inputs: Input tensor
            targets: Target tensor
            task_id: Optional task identifier

        Returns:
            Training metrics
        """
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        if task_id is not None and task_id in self.task_heads:
            features = self.model(inputs)
            outputs = self.task_heads[task_id](features)
        else:
            outputs = self.model(inputs)

        # Calculate loss
        loss = nn.functional.cross_entropy(outputs, targets)

        # Add EWC regularization
        if self.ewc_lambda > 0 and self.fisher_info:
            ewc_loss = self._compute_ewc_loss()
            loss = loss + self.ewc_lambda * ewc_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update state
        self.state.total_samples_seen += inputs.size(0)
        self.state.total_updates += 1

        # Store in replay buffer
        self._store_experience(inputs, targets, task_id)

        # Calculate accuracy
        _, predicted = outputs.max(1)
        accuracy = (predicted == targets).float().mean().item()

        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'samples_seen': self.state.total_samples_seen
        }

        self._call_callbacks('batch_complete', metrics=metrics)

        return metrics

    def learn_dataset(
        self,
        dataset: Dataset,
        epochs: int = 1,
        batch_size: int = 32,
        task_id: Optional[int] = None,
        validation_dataset: Optional[Dataset] = None
    ) -> Dict[str, Any]:
        """
        Learn from a complete dataset

        Args:
            dataset: Training dataset
            epochs: Number of epochs
            batch_size: Batch size
            task_id: Optional task identifier
            validation_dataset: Optional validation dataset

        Returns:
            Training results
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(epochs):
            epoch_losses = []
            epoch_accuracies = []

            for inputs, targets in dataloader:
                metrics = self.learn_batch(inputs, targets, task_id)
                epoch_losses.append(metrics['loss'])
                epoch_accuracies.append(metrics['accuracy'])

            avg_loss = np.mean(epoch_losses)
            avg_accuracy = np.mean(epoch_accuracies)

            history['train_loss'].append(avg_loss)
            history['train_accuracy'].append(avg_accuracy)

            # Validation
            if validation_dataset:
                val_metrics = self.evaluate(validation_dataset, task_id)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])

            # Experience replay
            if len(self.replay_buffer) > self.replay_batch_size:
                self._replay_experience()

            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}"
            )

            self._call_callbacks('epoch_complete', epoch=epoch, history=history)

        return history

    def learn_task(
        self,
        dataset: Dataset,
        task_id: int,
        epochs: int = 1,
        batch_size: int = 32,
        add_head: bool = False,
        num_classes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Learn a new task (task-incremental learning)

        Args:
            dataset: Task dataset
            task_id: Task identifier
            epochs: Number of epochs
            batch_size: Batch size
            add_head: Whether to add a new task head
            num_classes: Number of classes for new head

        Returns:
            Training results
        """
        # Add task-specific head if needed
        if add_head and num_classes:
            self._add_task_head(task_id, num_classes)

        # Record task boundary
        if task_id != self.state.current_task_id:
            # Update EWC parameters before new task
            if self.ewc_lambda > 0:
                self._update_ewc_params(dataset)

            self.state.record_task_boundary()

        # Learn the task
        result = self.learn_dataset(dataset, epochs, batch_size, task_id)

        self._call_callbacks('task_complete', task_id=task_id, result=result)

        return result

    def _add_task_head(self, task_id: int, num_classes: int) -> None:
        """Add a new task-specific head"""
        # Get feature dimension from model
        # This assumes the model has a `feature_dim` attribute
        feature_dim = getattr(self.model, 'feature_dim', 512)

        head = nn.Linear(feature_dim, num_classes).to(self.device)
        self.task_heads[task_id] = head

        # Add head parameters to optimizer
        self.optimizer.add_param_group({'params': head.parameters()})

        logger.info(f"Added task head for task {task_id} with {num_classes} classes")

    def _store_experience(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: Optional[int]
    ) -> None:
        """Store experience in replay buffer"""
        # Store a subset to manage memory
        indices = torch.randperm(inputs.size(0))[:max(1, inputs.size(0) // 4)]

        for idx in indices:
            self.replay_buffer.append({
                'input': inputs[idx].detach().cpu(),
                'target': targets[idx].detach().cpu(),
                'task_id': task_id
            })

    def _replay_experience(self) -> None:
        """Replay experiences from buffer"""
        if len(self.replay_buffer) < self.replay_batch_size:
            return

        # Sample from buffer using modern numpy random generator with seed for reproducibility
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(
            len(self.replay_buffer),
            self.replay_batch_size,
            replace=False
        )

        experiences = [self.replay_buffer[i] for i in indices]

        inputs = torch.stack([e['input'] for e in experiences]).to(self.device)
        targets = torch.stack([e['target'] for e in experiences]).to(self.device)

        # Learn from replay
        self.model.train()
        outputs = self.model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute Elastic Weight Consolidation loss"""
        loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.ewc_params:
                loss += (
                    self.fisher_info[name] *
                    (param - self.ewc_params[name]).pow(2)
                ).sum()

        return loss

    def _update_ewc_params(self, dataset: Dataset) -> None:
        """Update EWC parameters after learning a task"""
        # Store current parameters
        self.ewc_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        # Compute Fisher information
        self.model.eval()
        fisher = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        # Normalize Fisher information
        for name in fisher:
            fisher[name] /= len(dataloader)

        # Accumulate Fisher information
        for name in fisher:
            if name in self.fisher_info:
                self.fisher_info[name] = 0.5 * (self.fisher_info[name] + fisher[name])
            else:
                self.fisher_info[name] = fisher[name]

    def evaluate(
        self,
        dataset: Dataset,
        task_id: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if task_id is not None and task_id in self.task_heads:
                    features = self.model(inputs)
                    outputs = self.task_heads[task_id](features)
                else:
                    outputs = self.model(inputs)

                loss = nn.functional.cross_entropy(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                _, predicted = outputs.max(1)
                total_correct += (predicted == targets).sum().item()
                total_samples += inputs.size(0)

        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current learning state"""
        return {
            'total_samples_seen': self.state.total_samples_seen,
            'total_updates': self.state.total_updates,
            'current_task_id': self.state.current_task_id,
            'task_boundaries': self.state.task_boundaries,
            'replay_buffer_size': len(self.replay_buffer),
            'num_task_heads': len(self.task_heads)
        }


class OnlineLearner:
    """
    Online Learning System

    Features:
    - Single-sample updates
    - Concept drift detection
    - Adaptive learning rate
    - Streaming data support
    """

    def __init__(
        self,
        model: nn.Module,
        initial_lr: float = 1e-3,
        momentum: float = 0.9,
        drift_threshold: float = 0.1,
        window_size: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

        self.optimizer = optim.SGD(
            model.parameters(),
            lr=initial_lr,
            momentum=momentum,
            weight_decay=1e-4
        )

        self.drift_threshold = drift_threshold
        self.window_size = window_size

        # Performance windows for drift detection
        self.recent_losses = deque(maxlen=window_size)
        self.baseline_losses = deque(maxlen=window_size)

        # Drift detection state
        self.drift_detected = False
        self.drift_count = 0

        # Statistics
        self.samples_processed = 0
        self.total_loss = 0.0

    def update(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update model with a single sample

        Args:
            input_data: Single input tensor
            target: Single target tensor

        Returns:
            Update metrics
        """
        self.model.train()

        # Ensure proper shape
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)

        input_data = input_data.to(self.device)
        target = target.to(self.device)

        # Forward pass
        output = self.model(input_data)
        loss = nn.functional.cross_entropy(output, target)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update statistics
        self.samples_processed += 1
        loss_value = loss.item()
        self.total_loss += loss_value
        self.recent_losses.append(loss_value)

        # Check for concept drift
        drift = self._detect_drift()

        # Adaptive learning rate
        if drift:
            self._handle_drift()

        # Calculate accuracy
        _, predicted = output.max(1)
        correct = (predicted == target).item()

        return {
            'loss': loss_value,
            'correct': correct,
            'samples_processed': self.samples_processed,
            'drift_detected': drift,
            'current_lr': self.current_lr
        }

    def _detect_drift(self) -> bool:
        """Detect concept drift using loss monitoring"""
        if len(self.recent_losses) < self.window_size:
            return False

        if len(self.baseline_losses) < self.window_size:
            # Initialize baseline
            self.baseline_losses.extend(self.recent_losses)
            return False

        # Compare recent and baseline losses
        recent_mean = np.mean(list(self.recent_losses))
        baseline_mean = np.mean(list(self.baseline_losses))

        if baseline_mean > 0:
            drift_ratio = (recent_mean - baseline_mean) / baseline_mean

            if drift_ratio > self.drift_threshold:
                self.drift_detected = True
                self.drift_count += 1
                return True

        return False

    def _handle_drift(self) -> None:
        """Handle detected concept drift"""
        # Increase learning rate temporarily
        self.current_lr = min(self.current_lr * 2, self.initial_lr * 10)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

        # Reset baseline
        self.baseline_losses.clear()
        self.baseline_losses.extend(self.recent_losses)

        logger.info(
            f"Concept drift detected! Increased LR to {self.current_lr}"
        )

    def reset_drift_state(self) -> None:
        """Reset drift detection state"""
        self.drift_detected = False
        self.recent_losses.clear()
        self.baseline_losses.clear()
        self.current_lr = self.initial_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'samples_processed': self.samples_processed,
            'average_loss': self.total_loss / max(1, self.samples_processed),
            'drift_count': self.drift_count,
            'current_lr': self.current_lr,
            'drift_detected': self.drift_detected
        }
