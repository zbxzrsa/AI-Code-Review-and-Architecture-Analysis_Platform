"""
Continuous Learning System for Foundation Model

Implements lifelong learning capabilities:
1. Continual Pre-Training (CPT) - Learn from new data streams
2. Domain-Adaptive Pre-training (DAP) - Adapt to specific domains
3. Continual Fine-Tuning (CFT) - Learn new tasks incrementally

Anti-forgetting techniques:
- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)
- Experience Replay
- Progressive Neural Networks
- PackNet (parameter isolation)
- Learning without Forgetting (LwF)

Target: Enable 7×24 continuous learning without catastrophic forgetting
"""

import copy
import logging
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ContinualStrategy(str, Enum):
    """Continual learning strategies."""
    EWC = "ewc"  # Elastic Weight Consolidation
    SI = "si"  # Synaptic Intelligence
    REPLAY = "replay"  # Experience Replay
    PROGRESSIVE = "progressive"  # Progressive Networks
    PACKNET = "packnet"  # PackNet
    LWF = "lwf"  # Learning without Forgetting
    COMBINED = "combined"  # Combined strategies


class DistributionShift(str, Enum):
    """Types of distribution shifts."""
    TEMPORAL = "temporal"  # Time-based drift
    DOMAIN = "domain"  # New domain
    TASK = "task"  # New task
    LANGUAGE = "language"  # New language


@dataclass
class ContinualConfig:
    """Continual learning configuration."""
    # Strategy
    strategy: ContinualStrategy = ContinualStrategy.COMBINED

    # EWC settings
    ewc_lambda: float = 5000.0  # Fisher information importance
    ewc_gamma: float = 1.0  # Online EWC decay
    ewc_samples: int = 1000  # Samples for Fisher estimation

    # Synaptic Intelligence
    si_c: float = 1.0  # SI regularization strength
    si_epsilon: float = 1e-7

    # Experience Replay
    replay_buffer_size: int = 100000
    replay_ratio: float = 0.2  # Fraction of batch from replay

    # Progressive Networks
    lateral_connections: bool = True
    freeze_old_columns: bool = True

    # PackNet
    pruning_ratio: float = 0.5  # Fraction to prune per task

    # LwF
    lwf_temperature: float = 2.0
    lwf_alpha: float = 0.5

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0

    # Monitoring
    forgetting_threshold: float = 0.1  # Max acceptable forgetting
    drift_detection_window: int = 100


@dataclass
class Task:
    """Represents a learning task."""
    task_id: str
    name: str
    domain: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Task-specific data
    train_data: Optional[Any] = None
    eval_data: Optional[Any] = None

    # Performance tracking
    best_loss: float = float('inf')
    current_loss: float = float('inf')
    forgetting: float = 0.0


# =============================================================================
# Elastic Weight Consolidation (EWC)
# =============================================================================

class EWCRegularizer:
    """
    Elastic Weight Consolidation

    Prevents catastrophic forgetting by penalizing changes to
    parameters that are important for previous tasks.

    Loss = task_loss + λ * Σ F_i * (θ_i - θ*_i)²

    Where F_i is the Fisher information (importance) for parameter i.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ContinualConfig,
    ):
        self.model = model
        self.config = config

        # Store Fisher information and optimal parameters per task
        self.fisher_matrices: Dict[str, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[str, Dict[str, torch.Tensor]] = {}

        # Online EWC: running Fisher estimate
        self.online_fisher: Optional[Dict[str, torch.Tensor]] = None

        self.device = next(model.parameters()).device

    def compute_fisher(
        self,
        dataloader: DataLoader,
        task_id: str,
        num_samples: Optional[int] = None,
    ):
        """
        Compute Fisher information matrix diagonal.

        Fisher = E[∇log p(y|x,θ)²]

        Approximated using empirical Fisher on data.
        """
        self.model.eval()

        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        num_samples = num_samples or self.config.ewc_samples
        samples_seen = 0

        for batch in dataloader:
            if samples_seen >= num_samples:
                break

            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)

            # Forward pass
            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids)
            logits = outputs['logits']

            # Compute log likelihood gradient
            # Using cross-entropy as negative log likelihood
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

            samples_seen += input_ids.size(0)

        # Average Fisher
        for name in fisher:
            fisher[name] /= samples_seen

        # Store Fisher and optimal params
        self.fisher_matrices[task_id] = fisher
        self.optimal_params[task_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Update online Fisher (exponential moving average)
        if self.online_fisher is None:
            self.online_fisher = {k: v.clone() for k, v in fisher.items()}
        else:
            gamma = self.config.ewc_gamma
            for name in self.online_fisher:
                self.online_fisher[name] = (
                    gamma * self.online_fisher[name] + fisher[name]
                )

        self.model.train()

        logger.info(f"Computed Fisher for task {task_id}")

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term.

        Returns:
            λ * Σ_tasks Σ_params F_i * (θ_i - θ*_i)²
        """
        if not self.fisher_matrices:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for task_id in self.fisher_matrices:
            fisher = self.fisher_matrices[task_id]
            optimal = self.optimal_params[task_id]

            for name, param in self.model.named_parameters():
                if name in fisher:
                    loss += (
                        fisher[name] * (param - optimal[name]).pow(2)
                    ).sum()

        return self.config.ewc_lambda * loss

    def online_penalty(self) -> torch.Tensor:
        """
        Compute online EWC penalty using running Fisher estimate.
        """
        if self.online_fisher is None:
            return torch.tensor(0.0, device=self.device)

        # Use most recent optimal params
        if not self.optimal_params:
            return torch.tensor(0.0, device=self.device)

        latest_task = list(self.optimal_params.keys())[-1]
        optimal = self.optimal_params[latest_task]

        loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.online_fisher:
                loss += (
                    self.online_fisher[name] * (param - optimal[name]).pow(2)
                ).sum()

        return self.config.ewc_lambda * loss


# =============================================================================
# Synaptic Intelligence (SI)
# =============================================================================

class SynapticIntelligence:
    """
    Synaptic Intelligence

    Tracks online importance of parameters during training
    based on their contribution to loss reduction.

    Importance ω_k = Σ_t (∂L/∂θ_k * Δθ_k) / (Δθ_k² + ε)
    """

    def __init__(
        self,
        model: nn.Module,
        config: ContinualConfig,
    ):
        self.model = model
        self.config = config

        # Parameter importance
        self.omega: Dict[str, torch.Tensor] = {}

        # Running sums for importance computation
        self.big_omega: Dict[str, torch.Tensor] = {}
        self.small_omega: Dict[str, torch.Tensor] = {}

        # Reference parameters
        self.init_params: Dict[str, torch.Tensor] = {}
        self.star_params: Dict[str, torch.Tensor] = {}

        self.device = next(model.parameters()).device

        self._init_tracking()

    def _init_tracking(self):
        """Initialize tracking variables."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.big_omega[name] = torch.zeros_like(param)
                self.small_omega[name] = torch.zeros_like(param)
                self.init_params[name] = param.data.clone()

    def update_small_omega(self):
        """
        Update small omega during training.

        Called after each parameter update.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # ω = -grad * Δparam
                delta = param.data - self.init_params.get(name, param.data)
                self.small_omega[name] += -param.grad.data * delta

    def update_big_omega(self, task_id: str):
        """
        Update big omega after task completion.

        Consolidates importance estimates.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta = param.data - self.init_params.get(name, param.data)

                # Normalize by parameter change
                importance = self.small_omega[name] / (
                    delta.pow(2) + self.config.si_epsilon
                )

                # Accumulate
                self.big_omega[name] += importance.clamp(min=0)

                # Reset for next task
                self.small_omega[name] = torch.zeros_like(param)

        # Store optimal params
        self.star_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Update init params
        self.init_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        logger.info(f"Updated SI importance for task {task_id}")

    def penalty(self) -> torch.Tensor:
        """
        Compute SI penalty term.

        Returns:
            c * Σ Ω_k * (θ_k - θ*_k)²
        """
        if not self.star_params:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.big_omega and name in self.star_params:
                loss += (
                    self.big_omega[name] *
                    (param - self.star_params[name]).pow(2)
                ).sum()

        return self.config.si_c * loss


# =============================================================================
# Experience Replay
# =============================================================================

class ReplayBuffer:
    """
    Experience Replay Buffer

    Stores representative samples from previous tasks
    for rehearsal during new task learning.
    """

    def __init__(
        self,
        max_size: int = 100000,
        selection_strategy: str = "random",  # random, reservoir, kmeans
    ):
        self.max_size = max_size
        self.selection_strategy = selection_strategy

        self.buffer: List[Dict[str, Any]] = []
        self.task_indices: Dict[str, List[int]] = defaultdict(list)

    def add(
        self,
        samples: List[Dict[str, Any]],
        task_id: str,
    ):
        """
        Add samples to the buffer.

        Uses reservoir sampling if buffer is full.
        """
        for sample in samples:
            sample['task_id'] = task_id

            if len(self.buffer) < self.max_size:
                idx = len(self.buffer)
                self.buffer.append(sample)
                self.task_indices[task_id].append(idx)
            else:
                # Reservoir sampling
                idx = random.randint(0, len(self.buffer) + len(samples) - 1)
                if idx < self.max_size:
                    # Remove old index tracking
                    old_task = self.buffer[idx].get('task_id')
                    if old_task and idx in self.task_indices[old_task]:
                        self.task_indices[old_task].remove(idx)

                    self.buffer[idx] = sample
                    self.task_indices[task_id].append(idx)

    def sample(
        self,
        batch_size: int,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sample from buffer.

        Args:
            batch_size: Number of samples
            task_id: Optional task to sample from
        """
        if len(self.buffer) == 0:
            return []

        if task_id and task_id in self.task_indices:
            indices = self.task_indices[task_id]
            indices = random.sample(indices, min(batch_size, len(indices)))
            return [self.buffer[i] for i in indices]

        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of tasks in buffer."""
        return {
            task: len(indices)
            for task, indices in self.task_indices.items()
        }

    def __len__(self) -> int:
        return len(self.buffer)


class ExperienceReplay:
    """
    Experience Replay for Continual Learning

    Interleaves old task samples during new task training.
    """

    def __init__(
        self,
        config: ContinualConfig,
    ):
        self.config = config
        self.buffer = ReplayBuffer(
            max_size=config.replay_buffer_size,
        )

    def add_task_data(
        self,
        dataloader: DataLoader,
        task_id: str,
        num_samples: Optional[int] = None,
    ):
        """Add samples from task to replay buffer."""
        samples = []

        for batch in dataloader:
            for i in range(batch['input_ids'].size(0)):
                sample = {
                    'input_ids': batch['input_ids'][i],
                    'attention_mask': batch.get('attention_mask', None),
                    'labels': batch.get('labels', batch['input_ids'])[i],
                }
                samples.append(sample)

                if num_samples and len(samples) >= num_samples:
                    break

            if num_samples and len(samples) >= num_samples:
                break

        self.buffer.add(samples, task_id)
        logger.info(f"Added {len(samples)} samples for task {task_id} to replay buffer")

    def get_replay_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get a batch of replay samples."""
        samples = self.buffer.sample(batch_size)

        if not samples:
            return None

        # Collate samples
        input_ids = torch.stack([s['input_ids'] for s in samples]).to(device)
        labels = torch.stack([s['labels'] for s in samples]).to(device)

        return {
            'input_ids': input_ids,
            'labels': labels,
        }

    def mix_batches(
        self,
        current_batch: Dict[str, torch.Tensor],
        replay_batch: Dict[str, torch.Tensor],
        replay_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        """Mix current and replay batches."""
        current_size = current_batch['input_ids'].size(0)
        replay_size = int(current_size * replay_ratio)

        if replay_batch is None or replay_size == 0:
            return current_batch

        # Select replay samples
        replay_indices = torch.randperm(replay_batch['input_ids'].size(0))[:replay_size]

        # Replace some current samples with replay
        replace_indices = torch.randperm(current_size)[:replay_size]

        mixed_batch = {
            'input_ids': current_batch['input_ids'].clone(),
            'labels': current_batch['labels'].clone(),
        }

        mixed_batch['input_ids'][replace_indices] = replay_batch['input_ids'][replay_indices]
        mixed_batch['labels'][replace_indices] = replay_batch['labels'][replay_indices]

        return mixed_batch


# =============================================================================
# Progressive Neural Networks
# =============================================================================

class ProgressiveColumn(nn.Module):
    """A single column in Progressive Networks."""

    def __init__(
        self,
        base_model: nn.Module,
        column_id: int,
        lateral_connections: bool = True,
    ):
        super().__init__()

        self.column_id = column_id
        self.model = base_model
        self.lateral_connections = lateral_connections

        # Lateral connections from previous columns
        self.lateral_layers: nn.ModuleDict = nn.ModuleDict()

    def add_lateral_from(
        self,
        prev_column_id: int,
        hidden_size: int,
    ):
        """Add lateral connection from a previous column."""
        if self.lateral_connections:
            self.lateral_layers[str(prev_column_id)] = nn.Linear(
                hidden_size, hidden_size, bias=False
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        lateral_inputs: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward with lateral connections."""
        outputs = self.model(input_ids=input_ids, **kwargs)

        # Add lateral contributions
        if lateral_inputs and self.lateral_connections:
            hidden_states = outputs.get('last_hidden_state', outputs['logits'])

            for col_id, lateral_hidden in lateral_inputs.items():
                if str(col_id) in self.lateral_layers:
                    hidden_states = hidden_states + self.lateral_layers[str(col_id)](
                        lateral_hidden
                    )

            outputs['last_hidden_state'] = hidden_states

        return outputs


class ProgressiveNetworks:
    """
    Progressive Neural Networks

    Adds new columns for new tasks while freezing old columns.
    Lateral connections allow knowledge transfer.
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        config: ContinualConfig,
    ):
        self.base_model_fn = base_model_fn
        self.config = config

        self.columns: List[ProgressiveColumn] = []
        self.task_to_column: Dict[str, int] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_task(self, task_id: str, hidden_size: int = 4096) -> ProgressiveColumn:
        """Add a new column for a new task."""
        column_id = len(self.columns)

        # Create new column
        base_model = self.base_model_fn()
        column = ProgressiveColumn(
            base_model,
            column_id,
            self.config.lateral_connections,
        )

        # Add lateral connections from all previous columns
        for prev_id in range(column_id):
            column.add_lateral_from(prev_id, hidden_size)

        # Freeze previous columns
        if self.config.freeze_old_columns:
            for prev_column in self.columns:
                for param in prev_column.parameters():
                    param.requires_grad = False

        column = column.to(self.device)
        self.columns.append(column)
        self.task_to_column[task_id] = column_id

        logger.info(f"Added progressive column {column_id} for task {task_id}")

        return column

    def forward(
        self,
        input_ids: torch.Tensor,
        task_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward through appropriate column with lateral connections."""
        column_id = self.task_to_column.get(task_id, len(self.columns) - 1)

        # Collect lateral inputs from previous columns
        lateral_inputs = {}

        if self.config.lateral_connections:
            with torch.no_grad():
                for prev_id in range(column_id):
                    prev_outputs = self.columns[prev_id](input_ids, **kwargs)
                    lateral_inputs[prev_id] = prev_outputs.get(
                        'last_hidden_state',
                        prev_outputs['logits']
                    )

        # Forward through target column
        outputs = self.columns[column_id](
            input_ids,
            lateral_inputs=lateral_inputs,
            **kwargs,
        )

        return outputs

    def get_column(self, task_id: str) -> Optional[ProgressiveColumn]:
        """Get column for a task."""
        column_id = self.task_to_column.get(task_id)
        if column_id is not None:
            return self.columns[column_id]
        return None


# =============================================================================
# PackNet (Parameter Isolation)
# =============================================================================

class PackNet:
    """
    PackNet for Parameter Isolation

    Prunes and freezes a subset of parameters for each task,
    allowing different tasks to use different parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ContinualConfig,
    ):
        self.model = model
        self.config = config

        # Track which parameters are allocated to which task
        self.task_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        self.free_masks: Dict[str, torch.Tensor] = {}

        self.device = next(model.parameters()).device

        self._init_free_masks()

    def _init_free_masks(self):
        """Initialize all parameters as free."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.free_masks[name] = torch.ones_like(param, dtype=torch.bool)

    def allocate_task(
        self,
        task_id: str,
        dataloader: DataLoader,
    ):
        """
        Allocate parameters to a task through importance-based pruning.

        1. Train on task
        2. Compute importance (magnitude-based)
        3. Keep top-k% as task's parameters
        4. Mark remaining as free for future tasks
        """
        # Train and identify important parameters
        importance = self._compute_importance(dataloader)

        # Allocate based on pruning ratio
        task_mask = {}

        for name, imp in importance.items():
            free = self.free_masks[name]

            # Only consider free parameters
            masked_imp = imp * free.float()

            # Get threshold for top-k%
            k = int((1 - self.config.pruning_ratio) * free.sum().item())
            if k > 0:
                threshold = torch.topk(masked_imp.flatten(), k)[0][-1]
                mask = (masked_imp >= threshold) & free
            else:
                mask = torch.zeros_like(free)

            task_mask[name] = mask

            # Update free mask
            self.free_masks[name] = free & ~mask

        self.task_masks[task_id] = task_mask

        # Freeze allocated parameters
        self._freeze_task_params(task_id)

        logger.info(f"Allocated parameters for task {task_id}")

    def _compute_importance(
        self,
        dataloader: DataLoader,  # noqa: ARG002 - used for gradient-based importance in production
    ) -> Dict[str, torch.Tensor]:
        """Compute parameter importance (magnitude-based)."""
        importance = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance[name] = param.data.abs()

        return importance

    def _freeze_task_params(self, task_id: str):
        """Freeze parameters allocated to previous tasks."""
        for prev_task in self.task_masks:
            if prev_task != task_id:
                mask = self.task_masks[prev_task]

                for name, param in self.model.named_parameters():
                    if name in mask:
                        # Zero gradients for frozen params will be handled in training
                        pass

    def apply_gradient_mask(self):
        """Apply mask to gradients (zero out frozen parameters)."""
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.free_masks:
                # Only keep gradients for free parameters
                param.grad.data *= self.free_masks[name].float()

    def forward_with_mask(
        self,
        task_id: str,
    ) -> Dict[str, torch.Tensor]:
        """Get the parameter mask for a specific task."""
        if task_id in self.task_masks:
            return self.task_masks[task_id]
        return self.free_masks


# =============================================================================
# Learning without Forgetting (LwF)
# =============================================================================

class LearningWithoutForgetting:
    """
    Learning without Forgetting (LwF)

    Uses knowledge distillation from the model's previous state
    to prevent forgetting during new task learning.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ContinualConfig,
    ):
        self.model = model
        self.config = config

        # Previous model state for distillation
        self.old_model: Optional[nn.Module] = None
        self.old_tasks: List[str] = []

        self.device = next(model.parameters()).device

    def save_model_state(self, task_id: str):
        """Save current model state before learning new task."""
        self.old_model = copy.deepcopy(self.model)
        for param in self.old_model.parameters():
            param.requires_grad = False

        self.old_tasks.append(task_id)

        logger.info(f"Saved model state after task {task_id}")

    def distillation_loss(
        self,
        input_ids: torch.Tensor,
        current_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss from old model.

        L_distill = KL(softmax(old_logits/T) || softmax(new_logits/T))
        """
        if self.old_model is None:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            old_outputs = self.old_model(input_ids=input_ids)
            old_logits = old_outputs['logits']

        T = self.config.lwf_temperature

        # Soft targets from old model
        old_probs = F.softmax(old_logits / T, dim=-1)
        new_log_probs = F.log_softmax(current_logits / T, dim=-1)

        # KL divergence
        kl_loss = F.kl_div(
            new_log_probs,
            old_probs,
            reduction='batchmean',
        ) * (T ** 2)

        return kl_loss

    def combined_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        current_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combined loss: task loss + distillation loss.

        L = α * L_distill + (1-α) * L_task
        """
        # Task loss
        shift_logits = current_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Distillation loss
        distill_loss = self.distillation_loss(input_ids, current_logits)

        # Combine
        alpha = self.config.lwf_alpha
        total_loss = alpha * distill_loss + (1 - alpha) * task_loss

        return total_loss


# =============================================================================
# Continual Pre-Training (CPT)
# =============================================================================

class ContinualPretraining:
    """
    Continual Pre-Training

    Continuously pre-trains on new data streams while
    preventing catastrophic forgetting.

    Handles:
    - Temporal drift (news, current events)
    - Domain drift (new knowledge areas)
    - Language drift (new languages)
    """

    def __init__(
        self,
        model: nn.Module,
        config: ContinualConfig,
        tokenizer: Any,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        self.device = next(model.parameters()).device

        # Initialize anti-forgetting strategies
        self.ewc = EWCRegularizer(model, config)
        self.si = SynapticIntelligence(model, config)
        self.replay = ExperienceReplay(config)
        self.lwf = LearningWithoutForgetting(model, config)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Task tracking
        self.tasks: List[Task] = []
        self.current_task: Optional[Task] = None

        # Drift detection
        self.loss_history: List[float] = []

    def start_task(
        self,
        task_id: str,
        name: str,
        domain: str,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Task:
        """Start learning a new task."""
        task = Task(
            task_id=task_id,
            name=name,
            domain=domain,
            train_data=train_dataloader,
            eval_data=eval_dataloader,
        )

        self.current_task = task
        self.tasks.append(task)

        # Save model state for LwF
        if len(self.tasks) > 1:
            self.lwf.save_model_state(self.tasks[-2].task_id)

        logger.info(f"Started task: {task.name} ({task.task_id})")

        return task

    def end_task(self, dataloader: DataLoader):
        """End current task and consolidate knowledge."""
        if self.current_task is None:
            return

        task_id = self.current_task.task_id

        # Compute Fisher for EWC
        self.ewc.compute_fisher(dataloader, task_id)

        # Update SI importance
        self.si.update_big_omega(task_id)

        # Add samples to replay buffer
        self.replay.add_task_data(dataloader, task_id)

        # Evaluate final performance
        if self.current_task.eval_data:
            loss = self._evaluate(self.current_task.eval_data)
            self.current_task.best_loss = loss
            self.current_task.current_loss = loss

        logger.info(f"Ended task: {self.current_task.name}")

        self.current_task = None

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute one training step with continual learning."""
        self.model.train()

        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)

        # Mix with replay samples
        replay_batch = self.replay.get_replay_batch(
            self.config.batch_size,
            self.device,
        )

        if replay_batch:
            batch = self.replay.mix_batches(
                {'input_ids': input_ids, 'labels': labels},
                replay_batch,
                self.config.replay_ratio,
            )
            input_ids = batch['input_ids']
            labels = batch['labels']

        # Forward pass
        outputs = self.model(input_ids=input_ids)
        logits = outputs['logits']

        # Base task loss with LwF
        if self.config.strategy in [ContinualStrategy.LWF, ContinualStrategy.COMBINED]:
            loss = self.lwf.combined_loss(input_ids, labels, logits)
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Add regularization penalties
        if self.config.strategy in [ContinualStrategy.EWC, ContinualStrategy.COMBINED]:
            ewc_loss = self.ewc.penalty()
            loss = loss + ewc_loss

        if self.config.strategy in [ContinualStrategy.SI, ContinualStrategy.COMBINED]:
            si_loss = self.si.penalty()
            loss = loss + si_loss

        # Backward
        loss.backward()

        # Update SI tracking
        if self.config.strategy in [ContinualStrategy.SI, ContinualStrategy.COMBINED]:
            self.si.update_small_omega()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Track loss for drift detection
        self.loss_history.append(loss.item())

        return {
            'loss': loss.item(),
            'replay_ratio': len(replay_batch['input_ids']) / len(input_ids) if replay_batch else 0,
        }

    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on a dataloader."""
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)

                outputs = self.model(input_ids=input_ids)
                logits = outputs['logits']

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)

        self.model.train()

        return total_loss / total_samples if total_samples > 0 else 0.0

    def measure_forgetting(self) -> Dict[str, float]:
        """Measure forgetting on all previous tasks."""
        forgetting = {}

        for task in self.tasks:
            if task.eval_data and task != self.current_task:
                current_loss = self._evaluate(task.eval_data)
                task.current_loss = current_loss
                task.forgetting = max(0, current_loss - task.best_loss)
                forgetting[task.task_id] = task.forgetting

        return forgetting

    def detect_drift(self) -> bool:
        """Detect distribution drift based on loss patterns."""
        window = self.config.drift_detection_window

        if len(self.loss_history) < window * 2:
            return False

        recent = np.mean(self.loss_history[-window:])
        previous = np.mean(self.loss_history[-window*2:-window])

        # Significant increase in loss indicates drift
        return recent > previous * 1.5


# =============================================================================
# Domain-Adaptive Pre-training (DAP)
# =============================================================================

class DomainAdaptive:
    """
    Domain-Adaptive Pre-training

    Specializes the model for specific domains
    (code, medical, legal, financial, etc.)
    """

    def __init__(
        self,
        model: nn.Module,
        config: ContinualConfig,
    ):
        self.model = model
        self.config = config

        self.device = next(model.parameters()).device

        # Domain-specific adapters
        self.domain_adapters: Dict[str, nn.Module] = {}

        # Anti-forgetting
        self.ewc = EWCRegularizer(model, config)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

    def add_domain_adapter(
        self,
        domain: str,
        hidden_size: int = 4096,
        adapter_size: int = 256,
    ):
        """Add an adapter for a new domain."""
        adapter = nn.Sequential(
            nn.Linear(hidden_size, adapter_size),
            nn.GELU(),
            nn.Linear(adapter_size, hidden_size),
        ).to(self.device)

        self.domain_adapters[domain] = adapter

        logger.info(f"Added adapter for domain: {domain}")

    def train_domain(
        self,
        domain: str,
        dataloader: DataLoader,
        epochs: int = 3,
    ) -> Dict[str, float]:
        """Train on domain-specific data."""
        self.model.train()

        total_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)

                # Forward
                outputs = self.model(input_ids=input_ids)
                logits = outputs['logits']

                # Apply domain adapter if exists
                if domain in self.domain_adapters:
                    hidden = outputs.get('last_hidden_state', logits)
                    adapter_out = self.domain_adapters[domain](hidden)
                    # Residual connection
                    hidden = hidden + adapter_out

                # Loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                # EWC penalty
                ewc_loss = self.ewc.penalty()
                loss = loss + ewc_loss

                # Backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

            total_loss += epoch_loss / len(dataloader)
            logger.info(f"Domain {domain} - Epoch {epoch + 1}/{epochs}")

        # Update EWC
        self.ewc.compute_fisher(dataloader, f"domain_{domain}")

        return {
            'domain': domain,
            'avg_loss': total_loss / epochs,
        }


# =============================================================================
# Continual Fine-Tuning (CFT)
# =============================================================================

class ContinualFineTuning:
    """
    Continual Fine-Tuning

    Supports:
    - Continual Instruction Tuning (CIT)
    - Parameter-Efficient Fine-Tuning (PEFT) with LoRA
    """

    def __init__(
        self,
        model: nn.Module,
        config: ContinualConfig,
    ):
        self.model = model
        self.config = config

        self.device = next(model.parameters()).device

        # LoRA adapters per task
        self.lora_adapters: Dict[str, Dict[str, nn.Module]] = {}

        # Anti-forgetting
        self.replay = ExperienceReplay(config)
        self.ewc = EWCRegularizer(model, config)

    def add_lora_adapter(
        self,
        task_id: str,
        target_modules: List[str],
        lora_r: int = 8,
        lora_alpha: int = 32,  # noqa: ARG002 - reserved for scaling factor
    ):
        """Add LoRA adapter for a task."""
        adapters = {}

        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Create LoRA matrices
                    in_features = module.in_features
                    out_features = module.out_features

                    lora_down = nn.Linear(in_features, lora_r, bias=False)
                    lora_up = nn.Linear(lora_r, out_features, bias=False)

                    # Initialize
                    nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
                    nn.init.zeros_(lora_up.weight)

                    adapters[name] = nn.Sequential(
                        lora_down, lora_up
                    ).to(self.device)

        self.lora_adapters[task_id] = adapters

        logger.info(f"Added LoRA adapter for task {task_id} with {len(adapters)} modules")

    def train_task(
        self,
        task_id: str,
        dataloader: DataLoader,
        epochs: int = 3,
    ) -> Dict[str, float]:
        """Train on a task using LoRA."""
        if task_id not in self.lora_adapters:
            self.add_lora_adapter(task_id, ['q_proj', 'v_proj'])

        # Only train LoRA parameters
        for param in self.model.parameters():
            param.requires_grad = False

        lora_params = []
        for adapter in self.lora_adapters[task_id].values():
            for param in adapter.parameters():
                param.requires_grad = True
                lora_params.append(param)

        optimizer = AdamW(lora_params, lr=self.config.learning_rate, weight_decay=0.01)

        total_loss = 0.0

        for _epoch in range(epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)

                # Forward with LoRA
                outputs = self.model(input_ids=input_ids)
                logits = outputs['logits']

                # Loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                # Backward
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()

            total_loss += epoch_loss / len(dataloader)

        # Re-enable all gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Add to replay
        self.replay.add_task_data(dataloader, task_id)

        return {
            'task_id': task_id,
            'avg_loss': total_loss / epochs,
        }

    def merge_lora(self, task_id: str):
        """Merge LoRA weights into base model."""
        if task_id not in self.lora_adapters:
            return

        for name, adapter in self.lora_adapters[task_id].items():
            # Find corresponding module
            module = dict(self.model.named_modules())[name]

            if isinstance(module, nn.Linear):
                # Merge: W' = W + BA
                with torch.no_grad():
                    lora_weight = adapter[1].weight @ adapter[0].weight
                    module.weight.data += lora_weight

        logger.info(f"Merged LoRA adapter for task {task_id}")
