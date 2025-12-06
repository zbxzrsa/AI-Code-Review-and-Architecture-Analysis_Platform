"""
Multi-Task Learning and Transfer Learning Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import logging
import copy
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TaskConfig:
    """Configuration for a task"""
    name: str
    output_dim: int
    loss_fn: str = 'cross_entropy'
    weight: float = 1.0
    metrics: List[str] = field(default_factory=lambda: ['accuracy'])


class MultiTaskHead(nn.Module):
    """Task-specific output head"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_dim:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.head = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiTaskLearner(nn.Module):
    """
    Multi-Task Learning System
    
    Features:
    - Shared encoder with task-specific heads
    - Dynamic task weighting
    - Gradient surgery for conflicting gradients
    - Task sampling strategies
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        tasks: List[TaskConfig],
        feature_dim: int,
        use_gradient_surgery: bool = False,
        dynamic_weighting: bool = True
    ):
        super().__init__()
        
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.use_gradient_surgery = use_gradient_surgery
        self.dynamic_weighting = dynamic_weighting
        
        # Task heads
        self.task_heads = nn.ModuleDict()
        self.task_configs = {}
        
        for task in tasks:
            self.task_heads[task.name] = MultiTaskHead(
                input_dim=feature_dim,
                output_dim=task.output_dim
            )
            self.task_configs[task.name] = task
        
        # Task weights (learnable if dynamic)
        if dynamic_weighting:
            self.log_task_weights = nn.Parameter(
                torch.zeros(len(tasks))
            )
        else:
            self.task_weights = {task.name: task.weight for task in tasks}
        
        # Loss functions
        self.loss_fns = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'mse': nn.MSELoss(),
            'bce': nn.BCEWithLogitsLoss(),
            'l1': nn.L1Loss()
        }
        
        # Task performance tracking
        self.task_losses: Dict[str, List[float]] = {task.name: [] for task in tasks}
    
    def forward(
        self,
        x: torch.Tensor,
        task_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor
            task_name: Specific task (None for all tasks)
            
        Returns:
            Dictionary of task outputs
        """
        # Encode
        features = self.encoder(x)
        
        if task_name:
            return {task_name: self.task_heads[task_name](features)}
        
        # All tasks
        outputs = {}
        for name, head in self.task_heads.items():
            outputs[name] = head(features)
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for all tasks
        
        Args:
            outputs: Task outputs
            targets: Task targets
            
        Returns:
            Total loss and per-task losses
        """
        task_losses = {}
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Get weights
        if self.dynamic_weighting:
            weights = F.softmax(self.log_task_weights, dim=0)
            weight_dict = {
                name: weights[i].item()
                for i, name in enumerate(self.task_heads.keys())
            }
        else:
            weight_dict = self.task_weights
        
        for task_name in outputs:
            if task_name not in targets:
                continue
            
            config = self.task_configs[task_name]
            loss_fn = self.loss_fns[config.loss_fn]
            
            task_loss = loss_fn(outputs[task_name], targets[task_name])
            task_losses[task_name] = task_loss.item()
            
            # Track loss
            self.task_losses[task_name].append(task_loss.item())
            
            # Weighted sum
            total_loss = total_loss + weight_dict[task_name] * task_loss
        
        return total_loss, task_losses
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dictionary with 'input' and task targets
            optimizer: Optimizer
            
        Returns:
            Training metrics
        """
        self.train()
        
        inputs = batch['input']
        targets = {k: v for k, v in batch.items() if k != 'input'}
        
        if self.use_gradient_surgery:
            return self._train_step_with_surgery(inputs, targets, optimizer)
        
        optimizer.zero_grad()
        
        outputs = self.forward(inputs)
        loss, task_losses = self.compute_loss(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        return {'total_loss': loss.item(), **task_losses}
    
    def _train_step_with_surgery(
        self,
        inputs: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Training step with gradient surgery"""
        task_grads = {}
        task_losses = {}
        
        # Compute gradients for each task
        for task_name in targets:
            optimizer.zero_grad()
            
            outputs = self.forward(inputs, task_name)
            config = self.task_configs[task_name]
            loss_fn = self.loss_fns[config.loss_fn]
            
            loss = loss_fn(outputs[task_name], targets[task_name])
            loss.backward()
            
            task_losses[task_name] = loss.item()
            
            # Store gradients
            task_grads[task_name] = {
                name: param.grad.clone()
                for name, param in self.encoder.named_parameters()
                if param.grad is not None
            }
        
        # Apply gradient surgery
        final_grads = self._project_conflicting_gradients(task_grads)
        
        # Apply final gradients
        optimizer.zero_grad()
        for name, param in self.encoder.named_parameters():
            if name in final_grads:
                param.grad = final_grads[name]
        
        optimizer.step()
        
        return {'total_loss': sum(task_losses.values()), **task_losses}
    
    def _project_conflicting_gradients(
        self,
        task_grads: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Project out conflicting gradient components"""
        task_names = list(task_grads.keys())
        param_names = list(task_grads[task_names[0]].keys())
        
        final_grads = {}
        
        for param_name in param_names:
            grads = [task_grads[t][param_name] for t in task_names]
            
            # Simple PCGrad-style projection
            projected = grads[0].clone()
            
            for i in range(1, len(grads)):
                g_i = grads[i]
                
                # Check for conflict
                dot = (projected * g_i).sum()
                
                if dot < 0:
                    # Project out conflicting component
                    projected = projected - (dot / (g_i.norm() ** 2 + 1e-8)) * g_i
            
            final_grads[param_name] = projected
        
        return final_grads
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights"""
        if self.dynamic_weighting:
            weights = F.softmax(self.log_task_weights, dim=0)
            return {
                name: weights[i].item()
                for i, name in enumerate(self.task_heads.keys())
            }
        return self.task_weights
    
    def freeze_encoder(self) -> None:
        """Freeze encoder for head-only training"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder"""
        for param in self.encoder.parameters():
            param.requires_grad = True


class TransferLearning:
    """
    Transfer Learning System
    
    Features:
    - Pre-trained model adaptation
    - Domain adaptation
    - Fine-tuning strategies
    - Feature extraction modes
    """
    
    def __init__(
        self,
        source_model: nn.Module,
        target_output_dim: int,
        freeze_strategy: str = 'partial',
        num_frozen_layers: int = 0
    ):
        """
        Initialize Transfer Learning
        
        Args:
            source_model: Pre-trained model
            target_output_dim: Output dimension for target task
            freeze_strategy: 'none', 'partial', 'encoder', or 'all'
            num_frozen_layers: Number of layers to freeze (for partial)
        """
        self.source_model = copy.deepcopy(source_model)
        self.target_output_dim = target_output_dim
        self.freeze_strategy = freeze_strategy
        self.num_frozen_layers = num_frozen_layers
        
        self.target_model = None
        self.feature_dim = self._get_feature_dim()
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension from source model"""
        if hasattr(self.source_model, 'feature_dim'):
            return self.source_model.feature_dim
        elif hasattr(self.source_model, 'get_feature_dim'):
            return self.source_model.get_feature_dim()
        else:
            # Try to infer from last layer
            modules = list(self.source_model.modules())
            for m in reversed(modules):
                if isinstance(m, nn.Linear):
                    return m.in_features
            return 512  # Default
    
    def create_target_model(
        self,
        head_hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ) -> nn.Module:
        """
        Create model for target task
        
        Args:
            head_hidden_dim: Hidden dimension for new head
            dropout: Dropout rate
            
        Returns:
            Adapted model
        """
        # Clone encoder
        encoder = copy.deepcopy(self.source_model)
        
        # Remove original output layer if present
        if hasattr(encoder, 'output_layer'):
            delattr(encoder, 'output_layer')
        if hasattr(encoder, 'fc'):
            encoder.fc = nn.Identity()
        if hasattr(encoder, 'classifier'):
            encoder.classifier = nn.Identity()
        
        # Create new head
        if head_hidden_dim:
            head = nn.Sequential(
                nn.Linear(self.feature_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, self.target_output_dim)
            )
        else:
            head = nn.Linear(self.feature_dim, self.target_output_dim)
        
        # Combine
        class TargetModel(nn.Module):
            def __init__(self, enc, hd, feat_dim):
                super().__init__()
                self.encoder = enc
                self.head = hd
                self.feature_dim = feat_dim
            
            def forward(self, x):
                features = self.encoder(x)
                if hasattr(self.encoder, 'extract_features'):
                    features = self.encoder.extract_features(x)
                return self.head(features)
            
            def extract_features(self, x):
                if hasattr(self.encoder, 'extract_features'):
                    return self.encoder.extract_features(x)
                return self.encoder(x)
        
        self.target_model = TargetModel(encoder, head, self.feature_dim)
        
        # Apply freeze strategy
        self._apply_freeze_strategy()
        
        return self.target_model
    
    def _apply_freeze_strategy(self) -> None:
        """Apply freezing strategy to target model"""
        if self.freeze_strategy == 'none':
            return
        
        elif self.freeze_strategy == 'all':
            # Freeze entire model including head
            for param in self.target_model.parameters():
                param.requires_grad = False
        
        elif self.freeze_strategy == 'encoder':
            # Freeze only encoder, keep head trainable
            for param in self.target_model.encoder.parameters():
                param.requires_grad = False
        
        elif self.freeze_strategy == 'partial':
            # Freeze first n layers
            layers_frozen = 0
            for name, child in self.target_model.encoder.named_children():
                if layers_frozen < self.num_frozen_layers:
                    for param in child.parameters():
                        param.requires_grad = False
                    layers_frozen += 1
    
    def fine_tune(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 32,
        warmup_epochs: int = 2,
        gradual_unfreeze: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the target model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            lr: Learning rate
            batch_size: Batch size
            warmup_epochs: Warmup epochs before unfreezing
            gradual_unfreeze: Whether to gradually unfreeze layers
            
        Returns:
            Training history
        """
        if self.target_model is None:
            self.create_target_model()
        
        device = next(self.target_model.parameters()).device
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Optimizer with different LRs for encoder and head
        encoder_params = list(self.target_model.encoder.parameters())
        head_params = list(self.target_model.head.parameters())
        
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': lr * 0.1, 'weight_decay': 1e-4},
            {'params': head_params, 'lr': lr, 'weight_decay': 1e-4}
        ])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        history = {'train_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Gradual unfreezing
            if gradual_unfreeze and epoch >= warmup_epochs:
                self._gradual_unfreeze(epoch - warmup_epochs)
            
            # Training
            self.target_model.train()
            epoch_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = self.target_model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            
            # Validation
            if val_dataset:
                val_acc = self.evaluate(val_dataset)
                history['val_accuracy'].append(val_acc)
            
            scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        return history
    
    def _gradual_unfreeze(self, unfreeze_epoch: int) -> None:
        """Gradually unfreeze layers"""
        layers = list(self.target_model.encoder.named_children())
        n_layers = len(layers)
        
        # Unfreeze one layer per epoch from the end
        layers_to_unfreeze = min(unfreeze_epoch + 1, n_layers)
        
        for i, (name, child) in enumerate(reversed(layers)):
            if i < layers_to_unfreeze:
                for param in child.parameters():
                    param.requires_grad = True
    
    def evaluate(self, dataset: Dataset) -> float:
        """Evaluate model accuracy"""
        if self.target_model is None:
            return 0.0
        
        device = next(self.target_model.parameters()).device
        self.target_model.eval()
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = self.target_model(inputs)
                _, predicted = outputs.max(1)
                
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        return correct / total
    
    def extract_features(
        self,
        dataset: Dataset,
        batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features using the encoder
        
        Args:
            dataset: Input dataset
            batch_size: Batch size
            
        Returns:
            Features and labels
        """
        if self.target_model is None:
            self.create_target_model()
        
        device = next(self.target_model.parameters()).device
        self.target_model.eval()
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                features = self.target_model.extract_features(inputs)
                
                all_features.append(features.detach().cpu())
                all_labels.append(labels)
        
        return torch.cat(all_features), torch.cat(all_labels)
